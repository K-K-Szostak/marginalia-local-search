from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent
LIBRARY_DB = Path(os.getenv("MARGINALIA_LIBRARY_DB", str(BASE / "unified_library.sqlite"))).resolve()
CLEAN_DB = Path(os.getenv("MARGINALIA_CLEAN_DB", str(BASE / "clean_text.sqlite"))).resolve()
CLEAN_CACHE_DB = BASE / "clean_text_cache.sqlite"
CLEAN_CACHE_PROFILE = "clean-text-v3-page-ocr"
REPORT = BASE / "clean_text_report.json"
sys.path.insert(0, str(BASE / "vendor"))
import pymupdf
from progress_output import progress


SCHEMA = """
CREATE TABLE IF NOT EXISTS clean_document_blocks(
  id INTEGER PRIMARY KEY,
  attachment_id INTEGER NOT NULL,
  item_id INTEGER NOT NULL,
  title TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  block_type TEXT NOT NULL,
  section_title TEXT,
  page_start INTEGER NOT NULL,
  page_end INTEGER NOT NULL,
  text TEXT NOT NULL,
  char_count INTEGER NOT NULL,
  source TEXT NOT NULL,
  quality_flags TEXT NOT NULL DEFAULT '[]',
  UNIQUE(attachment_id,ordinal)
);
CREATE INDEX IF NOT EXISTS clean_blocks_attachment ON clean_document_blocks(attachment_id,ordinal);
CREATE INDEX IF NOT EXISTS clean_blocks_item ON clean_document_blocks(item_id);
CREATE TABLE IF NOT EXISTS clean_extraction_status(
  attachment_id INTEGER PRIMARY KEY,
  item_id INTEGER NOT NULL,
  status TEXT NOT NULL,
  source TEXT NOT NULL,
  raw_pages INTEGER NOT NULL,
  clean_blocks INTEGER NOT NULL,
  raw_chars INTEGER NOT NULL,
  clean_chars INTEGER NOT NULL,
  removed_running_elements INTEGER NOT NULL,
  headings INTEGER NOT NULL,
  paragraphs INTEGER NOT NULL,
  error TEXT
);
CREATE TABLE IF NOT EXISTS clean_text_info(key TEXT PRIMARY KEY,value TEXT NOT NULL);
"""


def plain(value):
    value = html.unescape(str(value or "").replace("\x00", "").replace("\u00ad", ""))
    return re.sub(r"[ \t]+", " ", value).strip()


def canonical_margin(value):
    value = plain(value).casefold()
    value = re.sub(r"\b(?:page|strona)\s*\d+\b", " page ", value)
    value = re.sub(r"\b[ivxlcdm]+\b", " # ", value)
    value = re.sub(r"\d+", "#", value)
    return re.sub(r"\W+", "", value, flags=re.UNICODE)


def join_lines(lines):
    value = ""
    for line in (plain(line) for line in lines):
        if not line:
            continue
        if value.endswith("-") and line[:1].islower():
            value = value[:-1] + line
        else:
            value += (" " if value else "") + line
    return plain(value)


def collapse_letter_spacing(value):
    parts = re.split(r"\s{2,}", str(value or "").strip())
    collapsed = []
    for part in parts:
        part = re.sub(
            r"(?<!\w)(?:[A-Za-zÀ-ž]\s+){2,}[A-Za-zÀ-ž](?!\w)",
            lambda match: re.sub(r"\s+", "", match.group()),
            plain(part),
        )
        if part:
            collapsed.append(part)
    value = " ".join(collapsed)
    return re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", value)


def remove_exact_repetition(value):
    words = plain(value).split()
    if len(words) >= 4 and len(words) % 2 == 0 and words[:len(words)//2] == words[len(words)//2:]:
        return " ".join(words[:len(words)//2])
    return plain(value)


def sentence_end(value):
    return bool(re.search(r"[.!?][’”\"')\]]?$", value.rstrip()))


def heading_words(value):
    return bool(re.match(
        r"^(?:\d+(?:\.\d+){0,4}\.?\s+)?(?:abstract|introduction|background|literature review|methods?|methodology|results?|findings?|discussion|conclusions?|references|bibliography|appendix|chapter|section|article|contents|table of contents|acknowledg(?:e)?ments?)\b",
        plain(value), re.I,
    ))


def numbered_heading(value):
    return bool(re.match(r"^(?:chapter\s+)?(?:[IVXLCDM]+|[A-Z]|\d+(?:\.\d+){0,4})[.:]\s+[A-ZÀ-Ž]", plain(value)))


def weighted_median(values):
    if not values:
        return 10.0
    values = sorted(values)
    total = sum(weight for _, weight in values)
    position = total / 2
    running = 0
    for value, weight in values:
        running += weight
        if running >= position:
            return value
    return values[-1][0]


def pdf_pages(path):
    document = pymupdf.open(path)
    try:
        if document.needs_pass and not document.authenticate(""):
            raise ValueError("Encrypted PDF requires a password")
        pages = []
        for page_number, page in enumerate(document, 1):
            page_data = page.get_text("dict", sort=True)
            blocks = []
            for source_block in page_data.get("blocks", []):
                if source_block.get("type") != 0:
                    continue
                lines = []
                spans = []
                line_records = []
                for source_line in source_block.get("lines", []):
                    line_spans = source_line.get("spans", [])
                    line_text = collapse_letter_spacing("".join(span.get("text", "") for span in line_spans))
                    if plain(line_text):
                        lines.append(line_text)
                        spans.extend(line_spans)
                        line_chars = max(1, sum(len(span.get("text", "")) for span in line_spans))
                        line_size = sum(float(span.get("size", 0)) * len(span.get("text", "")) for span in line_spans) / line_chars
                        line_bold = sum(
                            len(span.get("text", "")) for span in line_spans
                            if "bold" in span.get("font", "").casefold() or int(span.get("flags", 0)) & 16
                        ) / line_chars
                        line_records.append({
                            "text": line_text,
                            "bbox": tuple(float(x) for x in source_line.get("bbox", source_block.get("bbox", (0, 0, 0, 0)))),
                            "size": line_size,
                            "bold_ratio": line_bold,
                        })
                if not lines or not spans:
                    continue
                groups = []
                for record in line_records:
                    if groups:
                        previous = groups[-1][-1]
                        gap = record["bbox"][1] - previous["bbox"][3]
                        line_height = max(1.0, previous["bbox"][3] - previous["bbox"][1])
                        font_change = abs(record["size"] - previous["size"]) > max(.8, previous["size"] * .12)
                        style_change = abs(record["bold_ratio"] - previous["bold_ratio"]) > .62
                        base_left = min(item["bbox"][0] for item in groups[-1])
                        paragraph_indent = record["bbox"][0] - base_left > max(9, record["size"] * .8) and sentence_end(previous["text"])
                        vertical_break = gap > max(5.0, line_height * .58)
                        list_break = bool(re.match(r"^(?:[-–—•▪◦]|\(?\d+[.)]|\(?[a-z][.)])\s+", record["text"], re.I))
                        if font_change or style_change or paragraph_indent or vertical_break or list_break:
                            groups.append([])
                    else:
                        groups.append([])
                    groups[-1].append(record)
                for group in groups:
                    text = join_lines(item["text"] for item in group)
                    if not text:
                        continue
                    total_chars = max(1, sum(len(item["text"]) for item in group))
                    x0 = min(item["bbox"][0] for item in group); y0 = min(item["bbox"][1] for item in group)
                    x1 = max(item["bbox"][2] for item in group); y1 = max(item["bbox"][3] for item in group)
                    blocks.append({
                        "text": text, "lines": [item["text"] for item in group], "bbox": (x0, y0, x1, y1),
                        "size": sum(item["size"] * len(item["text"]) for item in group) / total_chars,
                        "bold_ratio": sum(item["bold_ratio"] * len(item["text"]) for item in group) / total_chars,
                        "italic_ratio": 0.0,
                    })
            native_text = (page.get_text("text") or "").replace("\x00", "").strip()
            pages.append({
                "number": page_number, "width": float(page.rect.width), "height": float(page.rect.height),
                "blocks": blocks, "native_text": native_text,
            })
        return pages
    finally:
        document.close()


def repeated_margins(pages):
    counts = Counter()
    for page in pages:
        seen = set()
        for block in page["blocks"]:
            top, bottom = block["bbox"][1], block["bbox"][3]
            if top <= page["height"] * .18 or bottom >= page["height"] * .86:
                key = canonical_margin(block["text"])
                if len(key) >= 5:
                    seen.add(key)
        counts.update(seen)
    # A running title may occur only inside one chapter of a long book, so a
    # global percentage threshold would miss it. Repetition in the page margin
    # on three pages is already strong layout evidence.
    threshold = 2 if len(pages) <= 40 else 3
    return {key for key, count in counts.items() if count >= threshold}


def body_font_size(pages, repeated):
    sizes = []
    for page in pages:
        for block in page["blocks"]:
            top, bottom = block["bbox"][1], block["bbox"][3]
            if top < page["height"] * .12 or bottom > page["height"] * .9:
                continue
            if canonical_margin(block["text"]) in repeated:
                continue
            if len(block["text"]) >= 35:
                sizes.append((round(block["size"], 2), len(block["text"])))
    return weighted_median(sizes)


def is_page_number(text):
    return bool(re.fullmatch(r"\s*(?:page\s*)?[ivxlcdm\d]+\s*", text, re.I))


def reliable_text_heading(text):
    text = plain(text)
    article_sentence = bool(re.match(r"^Article\s+\d+[a-z]?(?:\([^)]*\))?\s+", text, re.I)) and bool(
        re.search(r"\b(?:shall|must|may|calls?|defines?|provides?|requires?|means|includes?|is|are|has|have)\b", text, re.I)
    )
    common = heading_words(text) and len(text) <= 120 and len(text.split()) <= 16 and not article_sentence
    legal_sentence = bool(re.match(r"^\d+(?:\.\d+)*[.:]?\s+", text)) and bool(
        re.search(r"\b(?:shall|must|may|applies?|provides?|requires?|means|includes?|is|are|has|have)\b", text, re.I)
    )
    numbered = (
        numbered_heading(text)
        and len(text) <= 95
        and len(text.split()) <= 11
        and len(re.findall(r"[.!?]", text)) <= 1
        and not legal_sentence
        and not bool(re.search(r"[;,]\s*$", text))
    )
    return not sentence_end(text) and (common or numbered)


def classify(block, body_size, page_height, page_width):
    text = block["text"]
    top, bottom = block["bbox"][1], block["bbox"][3]
    short = len(text) <= 160
    terminal = sentence_end(text)
    large = block["size"] >= body_size * 1.16
    alpha_count = sum(character.isalpha() for character in text)
    structural = reliable_text_heading(text)
    if re.search(r"contents lists available at\s+sciencedirect", text, re.I):
        return "metadata"
    toc_entry = len(text) <= 240 and (text.count("Chapter ") >= 2 or bool(re.search(r"(?:\.{3,}|-{3,}).*\d+\s*$", text)))
    table_row = (len(re.findall(r"\b\d+(?:[.,]\d+)?\b", text)) >= 2 and len(text.split()) <= 14) or bool(re.fullmatch(r"[A-Z]{2,4}\s+\d+\s+[A-Z]{2,4}", text))
    citation = bool(re.match(r"^[A-Z]\.[ \t]+[A-ZÀ-Ž]", text)) and bool(re.search(r"\b(?:18|19|20)\d{2}\b", text))
    numbered_footnote = bool(re.match(r"^\d+\s+[A-ZÀ-Ža-zà-ž]", text)) and len(text) > 75
    # Override the broad first-pass signals with conservative structural
    # signals. Legal provisions and bibliography continuations are prose, not
    # section headings, even when they happen to use a larger font.
    article_refs = len(re.findall(r"\bArticle\s+\d+", text, re.I))
    table_row = table_row or article_refs >= 2 or bool(
        re.fullmatch(r"Article\s+\d+(?:\s*/+)?", text, re.I)
    )
    author_lead = bool(re.match(r"^[A-Z]\.[ \t]+[^\W\d_][\w'’-]+", text, re.UNICODE))
    citation = citation or (author_lead and len(text) > 45)
    numbered_footnote = numbered_footnote or (
        bool(re.match(r"^\d+\s+[^\W\d_]", text, re.UNICODE))
        and (len(text) > 45 or bool(re.search(r"[,;:]|\b(?:18|19|20)\d{2}\b", text)))
    )
    if toc_entry:
        return "toc_entry"
    if table_row:
        return "table_row"
    if citation or numbered_footnote:
        return "footnote"
    first_alpha = next((character for character in text if character.isalpha()), "")
    block_width = block["bbox"][2] - block["bbox"][0]
    centered_title = block_width <= page_width * .75 and abs((block["bbox"][0] + block["bbox"][2]) / 2 - page_width / 2) <= page_width * .1
    title_start = bool(first_alpha and first_alpha.isupper()) or centered_title
    sentence_lead = bool(re.match(
        r"^(?:because|although|however|therefore|thus|instead|moreover|furthermore|"
        r"it\s+(?:is|can|may|must|should)|this\s+(?:is|means|shows)|"
        r"the\s+(?:court|commission|board|group|member state))\b",
        text, re.I,
    ))
    provision_sentence = bool(re.match(r"^\d+(?:\.\d+)*[.:]?\s+", text)) and bool(
        re.search(r"\b(?:shall|must|may|applies?|provides?|requires?|means|includes?|is|are|has|have)\b", text, re.I)
    )
    article_sentence = bool(re.match(r"^Article\s+\d+[a-z]?(?:\([^)]*\))?\s+", text, re.I)) and bool(
        re.search(r"\b(?:shall|must|may|calls?|defines?|provides?|requires?|means|includes?|is|are|has|have)\b", text, re.I)
    )
    enumerated_clause = bool(re.match(r"^\d+(?:\.\d+)*[.:]\s+", text)) and bool(re.search(r"[;,]\s*$", text))
    if enumerated_clause:
        return "list_item"
    layout_title = (
        large and title_start and len(text) <= 100 and len(text.split()) <= 12
        and not sentence_lead and not author_lead and not provision_sentence and not article_sentence
    )
    if alpha_count >= 3 and short and not terminal and (layout_title or structural):
        return "heading"
    if re.match(r"^(?:[•▪◦–—-]|\(?\d+[.)]|\(?[a-z][.)])\s+", text, re.I):
        return "list_item"
    if block["size"] <= body_size * .82 or (top >= page_height * .82 and block["size"] < body_size * .95):
        return "footnote"
    return "paragraph"


def can_merge(previous, current):
    if current["block_type"] != "paragraph" or previous["block_type"] not in {"paragraph", "list_item"}:
        return False
    if previous["block_type"] == "list_item" and re.search(r"[.;:]\s*$", previous["text"]):
        return False
    if previous["page_end"] == current["page_start"]:
        vertical_gap = current["bbox"][1] - previous["bbox"][3]
        same_left = abs(previous["bbox"][0] - current["bbox"][0]) <= 18
        same_right = abs(previous["bbox"][2] - current["bbox"][2]) <= 28
        same_font = abs(previous["size"] - current["size"]) <= max(.7, previous["size"] * .08)
        tight_gap = vertical_gap <= max(previous["size"], current["size"]) * .72
        layout_continuation = same_font and (same_left or same_right) and vertical_gap <= max(previous["size"], current["size"]) * 1.7
    else:
        tight_gap = False
        layout_continuation = current["page_start"] == previous["page_end"] + 1
    textual_continuation = not sentence_end(previous["text"]) or current["text"][:1].islower()
    # Some PDFs split one paragraph after every sentence. A very small gap
    # with matching margins and font is stronger evidence than punctuation.
    return layout_continuation and (textual_continuation or tight_gap) and len(previous["text"]) + len(current["text"]) < 4_500


def compact_for_comparison(value):
    return re.sub(r"[^\w]+", "", plain(value).casefold(), flags=re.UNICODE)


def remove_repeated_extract_fragments(blocks):
    """Drop pull quotes/callouts duplicated verbatim in nearby body prose."""
    remove = set()
    normalized = [compact_for_comparison(block["text"]) for block in blocks]
    for index, block in enumerate(blocks):
        if block["block_type"] == "paragraph" and 35 <= len(block["text"]) <= 320:
            needle = normalized[index]
            if len(needle) >= 28:
                for other_index, other in enumerate(blocks):
                    if other_index == index or other["block_type"] != "paragraph":
                        continue
                    if abs(other["page_start"] - block["page_start"]) > 2:
                        continue
                    if len(normalized[other_index]) >= len(needle) * 1.35 and needle in normalized[other_index]:
                        remove.add(index)
                        break
        if block["block_type"] == "heading" and index + 1 < len(blocks):
            following = blocks[index + 1]
            if following["block_type"] != "paragraph" or following["page_start"] != block["page_start"]:
                continue
            combined = compact_for_comparison(block["text"] + " " + following["text"])
            if not 35 <= len(combined) <= 360:
                continue
            for other_index, other in enumerate(blocks):
                if other_index in {index, index + 1} or other["block_type"] != "paragraph":
                    continue
                if abs(other["page_start"] - block["page_start"]) > 2:
                    continue
                if len(normalized[other_index]) >= len(combined) * 1.2 and combined in normalized[other_index]:
                    remove.update({index, index + 1})
                    break
    return [block for index, block in enumerate(blocks) if index not in remove]


def refresh_sections(blocks):
    section = ""
    for block in blocks:
        if block["block_type"] == "heading":
            section = block["text"]
        block["section_title"] = section
    return blocks


def reliable_cache_heading(text):
    """Conservative heading detection when font/layout data is unavailable."""
    text = plain(text)
    if not text or sentence_end(text) or re.search(r"[;,:]\s*$", text):
        return False
    if text.isupper() and len(text) <= 120 and len(text.split()) <= 14:
        return True
    if re.fullmatch(r"(?:Article|Chapter|Section)\s+(?:\d+[a-z]?|[IVXLCDM]+)", text, re.I):
        return True
    if len(text) > 70 or len(text.split()) > 9:
        return False
    return reliable_text_heading(text)


def clean_layout_pages(pages):
    repeated = repeated_margins(pages)
    body_size = body_font_size(pages, repeated)
    contents_pages = set()
    for index, page in enumerate(pages):
        if any(canonical_margin(block["text"]) in {"contents", "tableofcontents"} for block in page["blocks"]):
            contents_pages.add(page["number"])
            for following in pages[index + 1:index + 4]:
                signals = sum(
                    block["text"].count("Chapter ") >= 2 or bool(re.search(r"(?:\.{3,}|-{3,}).*\d+\s*$", block["text"]))
                    for block in following["blocks"]
                )
                if not signals:
                    break
                contents_pages.add(following["number"])
    blocks = []; section = ""; removed = 0
    for page in pages:
        for source in page["blocks"]:
            top, bottom = source["bbox"][1], source["bbox"][3]
            margin = top <= page["height"] * .14 or bottom >= page["height"] * .86
            if (top <= page["height"] * .18 or bottom >= page["height"] * .86) and canonical_margin(source["text"]) in repeated:
                removed += 1
                continue
            if margin and is_page_number(source["text"]):
                removed += 1
                continue
            source["text"] = remove_exact_repetition(collapse_letter_spacing(source["text"]))
            block_type = classify(source, body_size, page["height"], page["width"])
            if page["number"] in contents_pages and canonical_margin(source["text"]) not in {"contents", "tableofcontents"}:
                block_type = "toc_entry"
            if blocks and blocks[-1]["block_type"] == "heading" and block_type == "paragraph" and blocks[-1]["page_end"] == page["number"]:
                previous = blocks[-1]
                gap = source["bbox"][1] - previous["bbox"][3]
                centered = abs((source["bbox"][0] + source["bbox"][2]) / 2 - page["width"] / 2) <= page["width"] * .12
                combined_length = len(previous["text"]) + 1 + len(source["text"])
                if len(source["text"]) <= 70 and combined_length <= 140 and not sentence_end(source["text"]) and centered and gap <= max(previous["size"], source["size"]) * 1.6 and abs(previous["size"] - source["size"]) <= 1.2:
                    previous["text"] = join_lines([previous["text"], source["text"]])
                    previous["section_title"] = previous["text"]
                    section = previous["text"]
                    continue
            if block_type == "heading":
                section = source["text"]
            current = {
                "block_type": block_type, "section_title": section if block_type != "heading" else source["text"],
                "page_start": page["number"], "page_end": page["number"], "text": source["text"],
                "bbox": source["bbox"], "size": source["size"], "quality_flags": [],
            }
            if blocks and can_merge(blocks[-1], current):
                previous = blocks[-1]
                previous["text"] = join_lines([previous["text"], current["text"]])
                previous["page_end"] = current["page_end"]
                previous["bbox"] = (previous["bbox"][0], previous["bbox"][1], current["bbox"][2], current["bbox"][3])
                continue
            blocks.append(current)
    blocks = refresh_sections(remove_repeated_extract_fragments(blocks))
    return blocks, {"body_font_size": round(body_size, 2), "running_patterns": len(repeated), "removed_running_elements": removed}


def cache_page_lines(text, repeated):
    paragraph = []
    blank_pending = False
    for raw in str(text or "").splitlines():
        line = plain(raw)
        if not line:
            blank_pending = True
            continue
        if (
            canonical_margin(line) in repeated
            or is_page_number(line)
            or bool(re.match(r"^(?:[▼►]\w?\s*)?0?\d{4,}[A-Z]\d+\s+[—-]\s+[A-Z]{2}\s+[—-]\s+\d{1,2}\.\d{1,2}\.\d{4}", line))
        ):
            continue
        article_refs = len(re.findall(r"\bArticle\s+\d+", line, re.I))
        if article_refs >= 2 or bool(re.fullmatch(r"[A-Z]{2,4}\s+\d+\s+[A-Z]{2,4}", line)):
            if paragraph:
                yield "paragraph", join_lines(paragraph); paragraph = []
            yield "table_row", line
            blank_pending = False
            continue
        author_lead = bool(re.match(r"^[A-Z]\.[ \t]+[^\W\d_][\w'’-]+", line, re.UNICODE))
        if author_lead and len(line) > 45:
            if paragraph:
                yield "paragraph", join_lines(paragraph); paragraph = []
            yield "footnote", line
            blank_pending = False
            continue
        if re.search(r"contents lists available at\s+sciencedirect", line, re.I):
            if paragraph:
                yield "paragraph", join_lines(paragraph); paragraph = []
            yield "metadata", line
            blank_pending = False
            continue
        if reliable_cache_heading(line):
            if paragraph:
                yield "paragraph", join_lines(paragraph); paragraph = []
            yield "heading", line
            blank_pending = False
            continue
        list_line = bool(re.match(r"^(?:[-–—•▪◦]|\(?\d+[.)]|\(?[a-z][.)])\s+", line, re.I))
        if list_line:
            if paragraph:
                yield "paragraph", join_lines(paragraph); paragraph = []
            yield "list_item", line
            blank_pending = False
            continue
        if blank_pending and paragraph:
            previous = paragraph[-1]
            # Blank OCR lines are unreliable. They become a paragraph break
            # only when the preceding sentence is complete and the next line
            # looks like the beginning of a new sentence.
            continuation = not sentence_end(previous) or line[:1].islower()
            if not continuation:
                yield "paragraph", join_lines(paragraph); paragraph = []
        paragraph.append(line)
        blank_pending = False
        length = sum(len(value) + 1 for value in paragraph)
        if length >= 1_200 or (length >= 450 and sentence_end(line)):
            yield "paragraph", join_lines(paragraph); paragraph = []
    if paragraph:
        yield "paragraph", join_lines(paragraph)


def clean_cache_pages(raw_pages):
    counts = Counter()
    for _, text in raw_pages:
        lines = [plain(line) for line in str(text or "").splitlines() if plain(line)]
        for line in lines[:2] + lines[-2:]:
            key = canonical_margin(line)
            if len(key) >= 5:
                counts[key] += 1
    threshold = max(3, round(len(raw_pages) * .18))
    repeated = {key for key, count in counts.items() if count >= threshold}
    blocks = []; section = ""; removed = 0
    for page_number, text in raw_pages:
        before = len([line for line in str(text or "").splitlines() if plain(line)])
        for block_type, value in cache_page_lines(text, repeated):
            if block_type == "heading":
                section = value
            current = {"block_type": block_type, "section_title": section, "page_start": page_number,
              "page_end": page_number, "text": value, "bbox": (0, 0, 0, 0), "size": 0,
              "quality_flags": ["heuristic-no-layout"]}
            if (
                blocks and not sentence_end(blocks[-1]["text"])
                and block_type == "paragraph" and blocks[-1]["block_type"] in {"paragraph", "list_item"}
                and page_number <= blocks[-1]["page_end"] + 1
                and not (blocks[-1]["block_type"] == "list_item" and re.search(r"[.;:]\s*$", blocks[-1]["text"]))
                and len(blocks[-1]["text"]) + len(value) < 4_500
            ):
                blocks[-1]["text"] = join_lines([blocks[-1]["text"], value]); blocks[-1]["page_end"] = page_number
            else:
                blocks.append(current)
        after = sum(1 for block in blocks if block["page_start"] == page_number)
        removed += max(0, before - after)
    blocks = refresh_sections(remove_repeated_extract_fragments(blocks))
    return blocks, {"body_font_size": None, "running_patterns": len(repeated), "removed_running_elements": removed}


def clean_hybrid_pdf_pages(pages, raw_pages):
    """Keep PDF layout on good pages and substitute OCR only on pages where OCR won."""
    raw_by_page = {int(page_number): str(text or "") for page_number, text in raw_pages}
    ocr_page_numbers = {
        page["number"] for page in pages
        if plain(raw_by_page.get(page["number"], "")) != plain(page.get("native_text", ""))
        and len(raw_by_page.get(page["number"], "")) > len(page.get("native_text", ""))
    }
    if not ocr_page_numbers:
        blocks, metrics = clean_layout_pages(pages)
        metrics["ocr_pages"] = 0
        return blocks, metrics, "pdf-layout"

    layout_pages = [
        {**page, "blocks": [] if page["number"] in ocr_page_numbers else page["blocks"]}
        for page in pages
    ]
    layout_blocks, layout_metrics = clean_layout_pages(layout_pages)
    ocr_blocks, ocr_metrics = clean_cache_pages([
        (page_number, raw_by_page[page_number]) for page_number in sorted(ocr_page_numbers)
    ])
    for block in ocr_blocks:
        block["quality_flags"] = sorted(set(block.get("quality_flags", [])) | {"ocr-page"})
    blocks = sorted(
        [*layout_blocks, *ocr_blocks],
        key=lambda block: (block["page_start"], block["page_end"]),
    )
    blocks = refresh_sections(remove_repeated_extract_fragments(blocks))
    metrics = {
        "body_font_size": layout_metrics["body_font_size"],
        "running_patterns": layout_metrics["running_patterns"] + ocr_metrics["running_patterns"],
        "removed_running_elements": layout_metrics["removed_running_elements"] + ocr_metrics["removed_running_elements"],
        "ocr_pages": len(ocr_page_numbers),
    }
    return blocks, metrics, "pdf-layout+page-ocr"


def rebuild_fts(db):
    db.executescript("""
    DROP TABLE IF EXISTS clean_document_search;
    CREATE VIRTUAL TABLE clean_document_search USING fts5(
      block_id UNINDEXED,attachment_id UNINDEXED,item_id UNINDEXED,page_start UNINDEXED,page_end UNINDEXED,
      parent_title,section_title,block_type,block_text,tokenize='unicode61 remove_diacritics 2'
    );
    INSERT INTO clean_document_search
    SELECT id,attachment_id,item_id,page_start,page_end,title,section_title,block_type,text
    FROM clean_document_blocks;
    """)


def main():
    parser = argparse.ArgumentParser(description="Create layout-aware, paragraph-structured text without replacing raw PDF extraction.")
    parser.add_argument("--attachment", action="append", type=int, default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--after-attachment", type=int, default=0)
    parser.add_argument("--existing-source", choices=("pdf-layout", "pdf-layout+page-ocr", "text-heuristic", "text-heuristic-ocr"))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--defer-fts", action="store_true")
    parser.add_argument("--fts-only", action="store_true")
    args = parser.parse_args()
    source = sqlite3.connect(f"file:{LIBRARY_DB.as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    clean = sqlite3.connect(CLEAN_DB)
    clean.executescript(SCHEMA)
    if args.fts_only:
        rebuild_fts(clean)
        clean.execute("INSERT OR REPLACE INTO clean_text_info VALUES('completed_at',CURRENT_TIMESTAMP)")
        clean.commit(); clean.close(); source.close()
        print("Full-text index rebuilt", flush=True)
        return
    if args.reset:
        clean.execute("DELETE FROM clean_document_blocks"); clean.execute("DELETE FROM clean_extraction_status"); clean.commit()
    query = """
      SELECT a.id,a.item_id,i.title,a.local_path,coalesce(s.status,'') status,a.sha256
      FROM attachments a JOIN items i ON i.id=a.item_id LEFT JOIN extraction_status s ON s.attachment_id=a.id
      WHERE EXISTS(SELECT 1 FROM document_pages p WHERE p.attachment_id=a.id)
    """
    params = []
    if args.attachment:
        query += " AND a.id IN (" + ",".join("?" for _ in args.attachment) + ")"; params.extend(args.attachment)
    if args.after_attachment:
        query += " AND a.id>?"; params.append(args.after_attachment)
    query += " ORDER BY a.id"
    targets = source.execute(query, params).fetchall()
    if args.existing_source:
        allowed = {
            row[0] for row in clean.execute(
                "SELECT attachment_id FROM clean_extraction_status WHERE source=?", (args.existing_source,)
            )
        }
        targets = [row for row in targets if row["id"] in allowed]
    if args.limit:
        targets = targets[:args.limit]
    cache = sqlite3.connect(CLEAN_CACHE_DB)
    cache.execute("""CREATE TABLE IF NOT EXISTS clean_cache(
      content_hash TEXT NOT NULL, profile TEXT NOT NULL, method TEXT NOT NULL,
      blocks_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
      PRIMARY KEY(content_hash,profile))""")
    report = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "targets": len(targets), "processed": 0,
      "reused": 0, "ocr_pages_merged": 0,
      "errors": [], "blocks": 0, "paragraphs": 0, "headings": 0, "removed_running_elements": 0}
    next_id = clean.execute("SELECT coalesce(max(id),0)+1 FROM clean_document_blocks").fetchone()[0]
    for number, row in enumerate(targets, 1):
        display_name = Path(row["local_path"]).name if row["local_path"] else (plain(row["title"]) or f"attachment {row['id']}")
        progress(f"Structuring text: {display_name}", number, len(targets))
        raw_pages = source.execute("SELECT page_number,text FROM document_pages WHERE attachment_id=? ORDER BY page_number", (row["id"],)).fetchall()
        raw_chars = sum(len(page["text"] or "") for page in raw_pages)
        raw_text_hash = hashlib.sha256(
            "\n\f\n".join(page["text"] or "" for page in raw_pages).encode("utf-8")
        ).hexdigest()
        content_hash = hashlib.sha256(
            ((row["sha256"] or "") + "\0" + raw_text_hash).encode("utf-8")
        ).hexdigest()
        path = BASE / row["local_path"] if row["local_path"] else None
        try:
            cached = cache.execute(
                "SELECT method,blocks_json,metrics_json FROM clean_cache WHERE content_hash=? AND profile=?",
                (content_hash, CLEAN_CACHE_PROFILE),
            ).fetchone()
            if cached:
                method, blocks_json, metrics_json = cached
                blocks, metrics = json.loads(blocks_json), json.loads(metrics_json)
                report["reused"] += 1
            else:
                if path and path.is_file() and path.suffix.casefold() == ".pdf":
                    pages = pdf_pages(path)
                    blocks, metrics, method = clean_hybrid_pdf_pages(pages, raw_pages)
                else:
                    blocks, metrics = clean_cache_pages(raw_pages); method = "text-heuristic"
                cache.execute("INSERT OR REPLACE INTO clean_cache VALUES(?,?,?,?,?)",(
                    content_hash,CLEAN_CACHE_PROFILE,method,
                    json.dumps(blocks,ensure_ascii=False),json.dumps(metrics,ensure_ascii=False)))
                cache.commit()
            clean.execute("DELETE FROM clean_document_blocks WHERE attachment_id=?", (row["id"],))
            for ordinal, block in enumerate(blocks, 1):
                clean.execute("INSERT INTO clean_document_blocks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    next_id,row["id"],row["item_id"],plain(row["title"]) or "Untitled",ordinal,block["block_type"],block["section_title"],
                    block["page_start"],block["page_end"],block["text"],len(block["text"]),method,
                    json.dumps(block["quality_flags"],ensure_ascii=False),
                )); next_id += 1
            headings = sum(block["block_type"] == "heading" for block in blocks)
            paragraphs = sum(block["block_type"] == "paragraph" for block in blocks)
            clean_chars = sum(len(block["text"]) for block in blocks)
            clean.execute("INSERT OR REPLACE INTO clean_extraction_status VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                row["id"],row["item_id"],"ok",method,len(raw_pages),len(blocks),raw_chars,clean_chars,
                metrics["removed_running_elements"],headings,paragraphs,None,
            ))
            report["processed"] += 1; report["blocks"] += len(blocks); report["paragraphs"] += paragraphs
            report["headings"] += headings; report["removed_running_elements"] += metrics["removed_running_elements"]
            report["ocr_pages_merged"] += int(metrics.get("ocr_pages", 0))
            cache_note = "reused cached cleanup" if cached else "new cleanup"
            progress(
                f"Structured {display_name}: {len(raw_pages):,} pages · {len(blocks):,} blocks · "
                f"{paragraphs:,} paragraphs · {headings:,} headings · {method} · {cache_note}",
                number, len(targets),
            )
        except Exception as exc:
            clean.execute("INSERT OR REPLACE INTO clean_extraction_status VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                row["id"],row["item_id"],"error","",len(raw_pages),0,raw_chars,0,0,0,0,str(exc)[:1000],
            ))
            report["errors"].append({"attachment_id":row["id"],"error":str(exc)})
            progress(f"Could not structure {display_name}: {exc}", number, len(targets))
        clean.commit()
        print(f"{number}/{len(targets)} attachment {row['id']}: {len(blocks) if 'blocks' in locals() else 0} clean blocks", flush=True)
    if not args.defer_fts:
        progress(f"Building clean full-text index from {report['blocks']:,} text blocks", len(targets), len(targets))
        rebuild_fts(clean)
        clean.execute("INSERT OR REPLACE INTO clean_text_info VALUES('completed_at',CURRENT_TIMESTAMP)")
        progress(f"Clean full-text index ready: {report['paragraphs']:,} paragraphs indexed", len(targets), len(targets))
    clean.execute("INSERT OR REPLACE INTO clean_text_info VALUES('source_database',?)", (str(LIBRARY_DB),))
    clean.commit(); clean.close(); source.close(); cache.close()
    report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
