from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("MARGINALIA_LIBRARY_DB", str(BASE / "unified_library.sqlite"))).resolve()
sys.path.insert(0,str(BASE / "vendor"))
import pymupdf
from extraction_cache import load_document, open_cache, profile as cache_profile, save_document
from ocr_support import (
    DEFAULT_MIN_TEXT_CHARS, DEFAULT_RENDER_DPI, available_languages, find_tesseract,
    needs_ocr, ocr_page, select_languages,
)
from progress_output import progress


def clean_text(value):
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


def normalized(value):
    return re.sub(r"\W+", "", clean_text(value).casefold(), flags=re.UNICODE)


def point_pair(value):
    if hasattr(value,"x"):
        return [float(value.x),float(value.y)]
    if isinstance(value,(list,tuple)) and len(value) >= 2 and all(isinstance(x,(int,float)) for x in value[:2]):
        return [float(value[0]),float(value[1])]
    if isinstance(value,(list,tuple)):
        return [point_pair(v) for v in value]
    return [0.0,0.0]


def deref(value):
    try:
        return value.get_object()
    except Exception:
        return value


def annotation_regions(annotation, page_height):
    quads = deref(annotation.get("/QuadPoints"))
    boxes = []
    if quads and len(quads) >= 8:
        for offset in range(0, len(quads) - 7, 8):
            values = [float(x) for x in quads[offset:offset + 8]]
            xs, ys = values[0::2], values[1::2]
            boxes.append((min(xs), page_height - max(ys), max(xs), page_height - min(ys)))
    rect = deref(annotation.get("/Rect"))
    if not boxes and rect and len(rect) >= 4:
        x0,y0,x1,y1 = map(float,rect[:4])
        boxes.append((min(x0,x1),page_height-max(y0,y1),max(x0,x1),page_height-min(y0,y1)))
    return boxes


def words_in_regions(words, regions):
    selected = []
    for word in words:
        wx0,wx1,wt,wb = float(word["x0"]),float(word["x1"]),float(word["top"]),float(word["bottom"])
        for x0,top,x1,bottom in regions:
            overlap_x = max(0,min(wx1,x1)-max(wx0,x0))
            overlap_y = max(0,min(wb,bottom)-max(wt,top))
            if overlap_x * overlap_y > 0:
                selected.append(word)
                break
    selected.sort(key=lambda w:(round(float(w["top"]),1),float(w["x0"])))
    return " ".join(w["text"] for w in selected)


db = sqlite3.connect(DB_PATH)
db.row_factory = sqlite3.Row
db.execute("PRAGMA foreign_keys=ON")
db.executescript("""
CREATE TABLE IF NOT EXISTS document_pages(
 id INTEGER PRIMARY KEY,
 attachment_id INTEGER NOT NULL REFERENCES attachments(id),
 item_id INTEGER NOT NULL REFERENCES items(id),
 page_number INTEGER NOT NULL,
 text TEXT NOT NULL,
 char_count INTEGER NOT NULL,
 UNIQUE(attachment_id,page_number)
);
CREATE INDEX IF NOT EXISTS idx_document_pages_attachment_page
ON document_pages(attachment_id,page_number);

CREATE TABLE IF NOT EXISTS embedded_pdf_annotations(
 id INTEGER PRIMARY KEY,
 attachment_id INTEGER NOT NULL REFERENCES attachments(id),
 item_id INTEGER NOT NULL REFERENCES items(id),
 page_number INTEGER NOT NULL,
 annotation_type TEXT NOT NULL,
 text TEXT,
 comment TEXT,
 author_name TEXT,
 color_json TEXT,
 position_json TEXT,
 fingerprint TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_embedded_annotations_attachment_page
ON embedded_pdf_annotations(attachment_id,page_number);

CREATE TABLE IF NOT EXISTS extraction_status(
 attachment_id INTEGER PRIMARY KEY REFERENCES attachments(id),
 status TEXT NOT NULL,
 page_count INTEGER,
 extracted_chars INTEGER,
 embedded_annotation_count INTEGER,
 error TEXT
);
""")

# Rebuild so reruns are deterministic and recover cleanly after interruption.
db.execute("DELETE FROM document_pages")
db.execute("DELETE FROM embedded_pdf_annotations")
db.execute("DELETE FROM extraction_status")
db.commit()

pdfs = db.execute("""
SELECT a.id,a.item_id,a.local_path,a.sha256,i.title,i.metadata_json
FROM attachments a JOIN items i ON i.id=a.item_id
WHERE lower(a.content_type)='application/pdf' OR lower(a.local_path) LIKE '%.pdf'
ORDER BY a.id
""").fetchall()

ocr_enabled = os.getenv("MARGINALIA_OCR", "1").strip().casefold() not in {"0", "false", "no", "off"}
ocr_command = find_tesseract() if ocr_enabled else None
ocr_installed_languages = available_languages(ocr_command) if ocr_command else set()
default_language_selection = select_languages(ocr_installed_languages) if ocr_command else None
ocr_min_chars = max(0, int(os.getenv("MARGINALIA_OCR_MIN_CHARS", str(DEFAULT_MIN_TEXT_CHARS))))
ocr_dpi = max(72, int(os.getenv("MARGINALIA_OCR_DPI", str(DEFAULT_RENDER_DPI))))
cache = open_cache(Path(os.getenv("MARGINALIA_EXTRACTION_CACHE", str(BASE / "pdf_extraction_cache.sqlite"))).resolve())
ocr_pages = ocr_characters = ocr_failures = 0
ocr_pages_reused = reused_documents = extracted_documents = 0
ocr_language_documents = Counter()
ocr_missing_languages = Counter()
extraction_profiles = set()
if ocr_enabled and not ocr_command:
    print("WARNING: Tesseract was not found; image-only PDF pages will remain unrecognized.", flush=True)
elif ocr_command:
    print(
        f"OCR enabled: {ocr_command} (automatic languages from installed data: "
        f"{'+'.join(sorted(ocr_installed_languages))}; fallback: {default_language_selection.languages}; "
        f"threshold: {ocr_min_chars} characters)", flush=True,
    )

page_id = embedded_id = 1
failures = []
for index,row in enumerate(pdfs,1):
    path = BASE / row["local_path"] if row["local_path"] else None
    metadata = json.loads(row["metadata_json"] or "{}")
    language_sample = "\n".join(str(metadata.get(key) or "") for key in (
        "title", "abstractNote", "publicationTitle", "bookTitle", "websiteTitle",
    ))
    language_selection = select_languages(
        ocr_installed_languages, metadata.get("language") or "", language_sample or row["title"] or "",
    ) if ocr_command else None
    ocr_languages = language_selection.languages if language_selection else ""
    extraction_profile = cache_profile(bool(ocr_command), ocr_languages, ocr_min_chars, ocr_dpi)
    extraction_profiles.add(extraction_profile)
    if language_selection:
        ocr_language_documents[ocr_languages] += 1
        if language_selection.missing:
            ocr_missing_languages[language_selection.missing] += 1
            if ocr_missing_languages[language_selection.missing] == 1:
                progress(
                    f"OCR language data '{language_selection.missing}' is not installed"
                    f"{' (detected automatically)' if language_selection.detected else ''}; "
                    f"using {ocr_languages}",
                    index, len(pdfs),
                )
    progress(f"Opening PDF: {path.name if path else 'missing attachment'}", index, len(pdfs))
    if not path or not path.is_file():
        db.execute("INSERT INTO extraction_status VALUES(?,?,?,?,?,?)",(row["id"],"missing",None,0,0,"Local PDF is absent"))
        continue
    savepoint = f"attachment_{int(row['id'])}"
    db.execute(f"SAVEPOINT {savepoint}")
    try:
        cached = load_document(cache, row["sha256"], extraction_profile) if row["sha256"] else None
        if cached:
            cached_document, cached_pages, cached_annotations = cached
            for cached_number, cached_page in enumerate(cached_pages, 1):
                progress(f"Reusing extracted text: {path.name} · page {cached_number} of {len(cached_pages)}", index, len(pdfs))
                text = clean_text(cached_page["text"])
                db.execute("INSERT INTO document_pages VALUES(?,?,?,?,?,?)",(
                    page_id,row["id"],row["item_id"],cached_page["page_number"],text,len(text)))
                page_id += 1
            embedded_count = 0
            for cached_annotation in cached_annotations:
                position = json.loads(cached_annotation["position_json"] or "{}")
                fingerprint = hashlib.sha256(json.dumps([
                    row["id"], cached_annotation["page_number"], cached_annotation["annotation_type"],
                    normalized(cached_annotation["text"]), normalized(cached_annotation["comment"]), position,
                ], sort_keys=True).encode()).hexdigest()
                db.execute("INSERT OR IGNORE INTO embedded_pdf_annotations VALUES(?,?,?,?,?,?,?,?,?,?,?)",(
                    embedded_id,row["id"],row["item_id"],cached_annotation["page_number"],
                    cached_annotation["annotation_type"],cached_annotation["text"],cached_annotation["comment"],
                    cached_annotation["author_name"],cached_annotation["color_json"],cached_annotation["position_json"],fingerprint))
                if db.execute("SELECT changes()").fetchone()[0]:
                    embedded_id += 1; embedded_count += 1
            db.execute("INSERT INTO extraction_status VALUES(?,?,?,?,?,?)",(
                row["id"],"ok",cached_document["page_count"],cached_document["extracted_chars"],
                embedded_count,"Reused content-addressed extraction cache"))
            reused_documents += 1
            ocr_pages_reused += cached_document["ocr_pages"]
            progress(
                f"Reused cached PDF: {path.name} · {len(cached_pages):,} pages · "
                f"{cached_document['extracted_chars']:,} characters",
                index, len(pdfs),
            )
            db.execute(f"RELEASE SAVEPOINT {savepoint}")
            continue
        document = pymupdf.open(path)
        password_state = ""
        if document.needs_pass:
            if not document.authenticate(""):
                raise ValueError("Encrypted PDF requires a password")
            password_state = "decrypted"
        extracted_chars = embedded_count = document_ocr_pages = document_ocr_characters = 0
        cached_page_texts = []
        cached_annotation_rows = []
        page_count = document.page_count
        try:
            for page_index,page in enumerate(document):
                progress(f"Extracting text: {path.name} · page {page_index + 1} of {page_count}", index, len(pdfs))
                try:
                    text = clean_text(page.get_text("text") or "")
                except Exception:
                    text = ""
                if ocr_command and needs_ocr(text, ocr_min_chars):
                    try:
                        progress(f"Reading {path.name} with OCR · page {page_index + 1} of {page_count}", index, len(pdfs))
                        recognized = clean_text(ocr_page(page, ocr_command, ocr_languages, ocr_dpi))
                        if len(recognized) > len(text):
                            text = recognized
                            ocr_pages += 1
                            ocr_characters += len(recognized)
                            document_ocr_pages += 1
                            document_ocr_characters += len(recognized)
                    except Exception as exc:
                        ocr_failures += 1
                        failures.append({
                            "attachment_id": row["id"], "file": str(path),
                            "page": page_index + 1, "stage": "ocr", "error": str(exc),
                        })
                db.execute("INSERT INTO document_pages VALUES(?,?,?,?,?,?)",(page_id,row["id"],row["item_id"],page_index+1,text,len(text)))
                page_id += 1
                extracted_chars += len(text)
                cached_page_texts.append(text)
                for ann in list(page.annots() or []):
                    subtype = clean_text((ann.type or (0,"unknown"))[1]).casefold().replace(" ","")
                    if subtype not in {"highlight","underline","strikeout","squiggly","text","freetext","ink","caret"}:
                        continue
                    info = ann.info or {}
                    comment = clean_text(info.get("content"))
                    author = clean_text(info.get("title"))
                    try:
                        selected = clean_text(ann.get_text("text") or "")
                    except Exception:
                        selected = ""
                    color = ann.colors
                    vertices = ann.vertices or []
                    position = {"rect": list(ann.rect),
                                "vertices": [point_pair(v) for v in vertices]}
                    fingerprint = hashlib.sha256(json.dumps([row["id"],page_index+1,subtype,normalized(selected),normalized(comment),position],sort_keys=True).encode()).hexdigest()
                    db.execute("INSERT OR IGNORE INTO embedded_pdf_annotations VALUES(?,?,?,?,?,?,?,?,?,?,?)",(
                        embedded_id,row["id"],row["item_id"],page_index+1,subtype,selected,comment,author,
                        json.dumps(color) if color is not None else None,json.dumps(position),fingerprint))
                    if db.execute("SELECT changes()").fetchone()[0]:
                        embedded_id += 1; embedded_count += 1
                    cached_annotation_rows.append({
                        "page_number": page_index + 1, "annotation_type": subtype,
                        "text": selected, "comment": comment, "author_name": author,
                        "color_json": json.dumps(color) if color is not None else None,
                        "position_json": json.dumps(position),
                    })
        finally:
            document.close()
        db.execute("INSERT INTO extraction_status VALUES(?,?,?,?,?,?)",(row["id"],"ok",page_count,extracted_chars,embedded_count,password_state))
        if row["sha256"]:
            save_document(cache,row["sha256"],extraction_profile,cached_page_texts,cached_annotation_rows,
                          document_ocr_pages,document_ocr_characters)
        extracted_documents += 1
        progress(
            f"Finished PDF: {path.name} · {page_count:,} pages · {extracted_chars:,} characters · "
            f"{embedded_count:,} embedded annotations · {document_ocr_pages:,} OCR pages",
            index, len(pdfs),
        )
        db.execute(f"RELEASE SAVEPOINT {savepoint}")
    except Exception as exc:
        db.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        db.execute(f"RELEASE SAVEPOINT {savepoint}")
        failures.append({"attachment_id":row["id"],"file":str(path),"error":str(exc)})
        db.execute("INSERT INTO extraction_status VALUES(?,?,?,?,?,?)",(row["id"],"error",None,0,0,str(exc)[:1000]))
        progress(f"Could not extract {path.name}: {exc}", index, len(pdfs))
    if index % 10 == 0:
        db.commit()
        print(f"{index}/{len(pdfs)} PDFs",flush=True)

db.executescript("""
DROP TABLE IF EXISTS document_search;
CREATE VIRTUAL TABLE document_search USING fts5(
 page_id UNINDEXED,
 attachment_id UNINDEXED,
 item_id UNINDEXED,
 page_number UNINDEXED,
 parent_title,
 authors,
 page_text,
 tokenize='unicode61 remove_diacritics 2'
);
INSERT INTO document_search
SELECT p.id,p.attachment_id,p.item_id,p.page_number,i.title,
       json_extract(i.creators_json,'$'),p.text
FROM document_pages p JOIN items i ON i.id=p.item_id;

DROP TABLE IF EXISTS embedded_annotation_search;
CREATE VIRTUAL TABLE embedded_annotation_search USING fts5(
 embedded_annotation_id UNINDEXED,
 attachment_id UNINDEXED,
 item_id UNINDEXED,
 page_number UNINDEXED,
 parent_title,
 annotation_type,
 annotation_text,
 comment,
 tokenize='unicode61 remove_diacritics 2'
);
INSERT INTO embedded_annotation_search
SELECT e.id,e.attachment_id,e.item_id,e.page_number,i.title,e.annotation_type,e.text,e.comment
FROM embedded_pdf_annotations e JOIN items i ON i.id=e.item_id;
PRAGMA optimize;
""")
db.execute("INSERT OR REPLACE INTO app_schema VALUES(?,?)",("full_document_search","document_search"))
db.execute("INSERT OR REPLACE INTO app_schema VALUES(?,?)",("embedded_annotation_search","embedded_annotation_search"))
db.commit()
summary = {
    "pdf_records":len(pdfs),
    "processed":db.execute("SELECT count(*) FROM extraction_status WHERE status='ok'").fetchone()[0],
    "missing":db.execute("SELECT count(*) FROM extraction_status WHERE status='missing'").fetchone()[0],
    "errors":db.execute("SELECT count(*) FROM extraction_status WHERE status='error'").fetchone()[0],
    "pages":db.execute("SELECT count(*) FROM document_pages").fetchone()[0],
    "characters":db.execute("SELECT coalesce(sum(char_count),0) FROM document_pages").fetchone()[0],
    "embedded_pdf_annotations":db.execute("SELECT count(*) FROM embedded_pdf_annotations").fetchone()[0],
    "ocr_enabled": bool(ocr_command),
    "ocr_engine": str(ocr_command) if ocr_command else None,
    "ocr_languages": dict(sorted(ocr_language_documents.items())),
    "ocr_installed_languages": sorted(ocr_installed_languages),
    "ocr_missing_languages": dict(sorted(ocr_missing_languages.items())),
    "ocr_pages": ocr_pages,
    "ocr_pages_reused": ocr_pages_reused,
    "ocr_characters": ocr_characters,
    "ocr_failures": ocr_failures,
    "documents_extracted": extracted_documents,
    "documents_reused": reused_documents,
    "extraction_profiles": sorted(extraction_profiles),
    "integrity":db.execute("PRAGMA integrity_check").fetchone()[0],
    "failures":failures,
}
db.close()
cache.close()
report_path = Path(os.getenv("MARGINALIA_EXTRACTION_REPORT", str(BASE / "pdf_extraction_report.json"))).resolve()
report_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(summary,indent=2,ensure_ascii=False))
