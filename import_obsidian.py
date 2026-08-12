from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from progress_output import progress
from source_manager import filesystem_path


BASE = Path(__file__).resolve().parent
DB = Path(os.getenv("MARGINALIA_OBSIDIAN_DB", str(BASE / "obsidian_notes.sqlite"))).resolve()
CONFIG = BASE / "obsidian_source.json"
IGNORED_DIRECTORIES = {".obsidian", ".trash", ".git", "__pycache__", "do not read"}


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS obsidian_notes(
  note_id TEXT PRIMARY KEY,
  relative_path TEXT NOT NULL UNIQUE,
  folder TEXT NOT NULL,
  filename TEXT NOT NULL,
  title TEXT NOT NULL,
  markdown TEXT NOT NULL,
  plain_text TEXT NOT NULL,
  frontmatter_json TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  aliases_json TEXT NOT NULL,
  authors_json TEXT NOT NULL,
  created_at TEXT,
  modified_at TEXT,
  checksum TEXT NOT NULL,
  imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS obsidian_notes_folder ON obsidian_notes(folder);
CREATE INDEX IF NOT EXISTS obsidian_notes_title ON obsidian_notes(title COLLATE NOCASE);
CREATE TABLE IF NOT EXISTS obsidian_sections(
  section_id TEXT PRIMARY KEY,
  note_id TEXT NOT NULL REFERENCES obsidian_notes(note_id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  heading TEXT,
  heading_path TEXT,
  heading_level INTEGER NOT NULL DEFAULT 0,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  markdown TEXT NOT NULL,
  plain_text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS obsidian_sections_note ON obsidian_sections(note_id,ordinal);
CREATE TABLE IF NOT EXISTS obsidian_links(
  id INTEGER PRIMARY KEY,
  note_id TEXT NOT NULL REFERENCES obsidian_notes(note_id) ON DELETE CASCADE,
  target TEXT NOT NULL,
  display_text TEXT,
  heading TEXT,
  is_embed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS obsidian_links_note ON obsidian_links(note_id);
CREATE INDEX IF NOT EXISTS obsidian_links_target ON obsidian_links(target COLLATE NOCASE);
CREATE TABLE IF NOT EXISTS obsidian_assets(
  relative_path TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  content_type TEXT,
  byte_size INTEGER NOT NULL,
  modified_at TEXT
);
CREATE TABLE IF NOT EXISTS obsidian_import_info(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS obsidian_search USING fts5(
  section_id UNINDEXED,
  note_id UNINDEXED,
  title,
  heading,
  path,
  tags,
  authors,
  text,
  tokenize='unicode61 remove_diacritics 2'
);
"""


def read_text(path: Path) -> str:
    with open(filesystem_path(path), encoding="utf-8", errors="replace") as note_file:
        return note_file.read()


def file_stat(path: Path):
    return os.stat(filesystem_path(path))


def iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def clean_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [clean_scalar(part) for part in re.split(r",\s*", body)]
    if value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    return value


def parse_frontmatter(markdown: str):
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, markdown, 0
    end = next((index for index in range(1, min(len(lines), 400)) if lines[index].strip() == "---"), None)
    if end is None:
        return {}, markdown, 0
    metadata = {}
    current_key = None
    for line in lines[1:end]:
        list_item = re.match(r"^\s*-\s+(.+)$", line)
        if list_item and current_key:
            if not isinstance(metadata.get(current_key), list):
                metadata[current_key] = []
            metadata[current_key].append(clean_scalar(list_item.group(1)))
            continue
        match = re.match(r"^([\w -]+):\s*(.*)$", line)
        if not match:
            continue
        current_key = match.group(1).strip()
        metadata[current_key] = clean_scalar(match.group(2))
    return metadata, "\n".join(lines[end + 1 :]), end + 1


def values(metadata: dict, *keys: str) -> list[str]:
    folded = {str(key).casefold(): value for key, value in metadata.items()}
    output = []
    for key in keys:
        value = folded.get(key.casefold())
        if value is None:
            continue
        candidates = value if isinstance(value, list) else re.split(r"[,;]", str(value))
        for candidate in candidates:
            candidate = str(candidate).strip().strip("#")
            if candidate and candidate not in output:
                output.append(candidate)
    return output


def markdown_plain_text(markdown: str) -> str:
    text = html.unescape(re.sub(r"```.*?```", " ", markdown, flags=re.S))
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[\[([^]|#]+)(?:#[^]|]+)?(?:\|([^]]+))?\]\]", lambda m: m.group(2) or m.group(1), text)
    text = re.sub(r"\[\[([^]|#]+)(?:#[^]|]+)?(?:\|([^]]+))?\]\]", lambda m: m.group(2) or m.group(1), text)
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*(?:[-*+] |\d+[.)] )", "", text)
    text = re.sub(r"[*_~>|]", " ", text)
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", text)]
    return "\n\n".join(part for part in paragraphs if part)


def title_for(path: Path, body: str, metadata: dict) -> str:
    explicit = values(metadata, "title")
    if explicit:
        return explicit[0]
    for line in body.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match and markdown_plain_text(match.group(1)):
            return markdown_plain_text(match.group(1))
    return path.stem


def sections_for(note_id: str, body: str, line_offset: int):
    lines = body.splitlines()
    headings = []
    starts = []
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            starts.append((index, len(match.group(1)), markdown_plain_text(match.group(2))))
    boundaries = [(0, 0, "")] if not starts or starts[0][0] else []
    boundaries.extend(starts)
    if not boundaries:
        boundaries = [(0, 0, "")]
    for ordinal, (start, level, heading) in enumerate(boundaries):
        end = boundaries[ordinal + 1][0] if ordinal + 1 < len(boundaries) else len(lines)
        if level:
            headings[:] = headings[: level - 1]
            while len(headings) < level - 1:
                headings.append("")
            headings.append(heading)
        elif ordinal == 0:
            headings = []
        markdown = "\n".join(lines[start:end]).strip()
        plain = markdown_plain_text(markdown)
        if not plain:
            continue
        section_id = hashlib.sha1(f"{note_id}\0{ordinal}\0{heading}".encode("utf-8")).hexdigest()
        yield {
            "section_id": section_id,
            "ordinal": ordinal,
            "heading": heading,
            "heading_path": " › ".join(value for value in headings if value),
            "heading_level": level,
            "start_line": line_offset + start + 1,
            "end_line": line_offset + end,
            "markdown": markdown,
            "plain_text": plain,
        }


def links_for(markdown: str):
    pattern = re.compile(r"(!)?\[\[([^]|#]+)(?:#([^]|]+))?(?:\|([^]]+))?\]\]")
    for match in pattern.finditer(markdown):
        yield {
            "target": match.group(2).strip(),
            "heading": (match.group(3) or "").strip(),
            "display_text": (match.group(4) or match.group(2)).strip(),
            "is_embed": bool(match.group(1)),
        }


def inline_tags(body: str) -> list[str]:
    without_code = re.sub(r"```.*?```|`[^`]*`", " ", body, flags=re.S)
    found = re.findall(r"(?<![\w/])#([\w\-]+(?:/[\w\-]+)*)", without_code, flags=re.UNICODE)
    return list(dict.fromkeys(found))


def ignored(path: Path, vault: Path) -> bool:
    return any(part.casefold() in IGNORED_DIRECTORIES for part in path.relative_to(vault).parts[:-1])


def main():
    parser = argparse.ArgumentParser(description="Import an Obsidian vault into Marginalia without modifying the vault.")
    parser.add_argument("--vault", type=Path)
    parser.add_argument("--database", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--no-save-source", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    database = (args.database or DB).resolve()
    configured = {}
    if CONFIG.is_file():
        configured = json.loads(CONFIG.read_text(encoding="utf-8"))
    vault = (args.vault or Path(configured.get("vault_path", ""))).expanduser().resolve()
    if not vault.is_dir():
        raise SystemExit("Pass --vault with the path to an Obsidian vault copy.")

    if not args.no_save_source:
        CONFIG.write_text(json.dumps({"vault_path": str(vault)}, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = []
    excluded = 0
    markdown_paths = sorted(vault.rglob("*.md"), key=lambda value: value.as_posix().casefold())
    for path_number, path in enumerate(markdown_paths, 1):
        progress(f"Reading Obsidian note: {path.relative_to(vault).as_posix()}", path_number, len(markdown_paths))
        if ignored(path, vault):
            excluded += 1
            continue
        raw = read_text(path)
        metadata, body, offset = parse_frontmatter(raw)
        relative = path.relative_to(vault).as_posix()
        note_id = hashlib.sha1(relative.casefold().encode("utf-8")).hexdigest()
        stat = file_stat(path)
        tags = list(dict.fromkeys(values(metadata, "tags", "tag") + inline_tags(body)))
        aliases = values(metadata, "aliases", "alias")
        authors = values(metadata, "authors", "author")
        created = (values(metadata, "created", "date created", "date_created") or [iso_time(min(stat.st_ctime, stat.st_mtime))])[0]
        modified = (values(metadata, "modified", "updated", "date modified", "date_modified") or [iso_time(stat.st_mtime)])[0]
        sections = list(sections_for(note_id, body, offset))
        notes.append({
            "note_id": note_id,
            "relative_path": relative,
            "folder": path.parent.relative_to(vault).as_posix() if path.parent != vault else "",
            "filename": path.name,
            "title": title_for(path, body, metadata),
            "markdown": body,
            "plain_text": markdown_plain_text(body),
            "frontmatter_json": json.dumps(metadata, ensure_ascii=False),
            "tags": tags,
            "aliases": aliases,
            "authors": authors,
            "created_at": str(created),
            "modified_at": str(modified),
            "checksum": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "sections": sections,
            "links": list(links_for(body)),
        })

    db = sqlite3.connect(database)
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(SCHEMA)
    existing = {row[0]: row[1] for row in db.execute("SELECT note_id,checksum FROM obsidian_notes")}
    current = {note["note_id"] for note in notes}
    deleted = set(existing) - current
    for note_id in deleted:
        db.execute("DELETE FROM obsidian_search WHERE note_id=?", (note_id,))
        db.execute("DELETE FROM obsidian_notes WHERE note_id=?", (note_id,))
    changed = [note for note in notes if existing.get(note["note_id"]) != note["checksum"]]
    added = sum(note["note_id"] not in existing for note in changed)
    updated = len(changed) - added
    for changed_number, note in enumerate(changed, 1):
        progress(f"Indexing Obsidian note: {note['relative_path']}", changed_number, len(changed))
        db.execute("DELETE FROM obsidian_search WHERE note_id=?", (note["note_id"],))
        db.execute("DELETE FROM obsidian_notes WHERE note_id=?", (note["note_id"],))
        db.execute(
            """INSERT INTO obsidian_notes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
            (
                note["note_id"], note["relative_path"], note["folder"], note["filename"], note["title"],
                note["markdown"], note["plain_text"], note["frontmatter_json"],
                json.dumps(note["tags"], ensure_ascii=False), json.dumps(note["aliases"], ensure_ascii=False),
                json.dumps(note["authors"], ensure_ascii=False), note["created_at"], note["modified_at"], note["checksum"],
            ),
        )
        for section in note["sections"]:
            db.execute(
                "INSERT INTO obsidian_sections VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    section["section_id"], note["note_id"], section["ordinal"], section["heading"],
                    section["heading_path"], section["heading_level"], section["start_line"], section["end_line"],
                    section["markdown"], section["plain_text"],
                ),
            )
            db.execute(
                "INSERT INTO obsidian_search VALUES(?,?,?,?,?,?,?,?)",
                (
                    section["section_id"], note["note_id"], note["title"], section["heading_path"],
                    note["relative_path"], " ".join(note["tags"]), " ".join(note["authors"]), section["plain_text"],
                ),
            )
        db.executemany(
            "INSERT INTO obsidian_links(note_id,target,display_text,heading,is_embed) VALUES(?,?,?,?,?)",
            ((note["note_id"], link["target"], link["display_text"], link["heading"], int(link["is_embed"])) for link in note["links"]),
        )

    db.execute("DELETE FROM obsidian_assets")
    assets = 0
    asset_paths = sorted((path for path in vault.rglob("*") if path.is_file() and path.suffix.casefold() != ".md"))
    for asset_number, path in enumerate(asset_paths, 1):
        if ignored(path, vault):
            continue
        relative = path.relative_to(vault).as_posix()
        progress(f"Cataloguing Obsidian asset: {relative}", asset_number, len(asset_paths))
        stat = file_stat(path)
        db.execute(
            "INSERT INTO obsidian_assets VALUES(?,?,?,?,?)",
            (relative, path.name, mimetypes.guess_type(path.name)[0] or "application/octet-stream", stat.st_size, iso_time(stat.st_mtime)),
        )
        assets += 1
    link_count = db.execute("SELECT count(*) FROM obsidian_links").fetchone()[0]
    section_count = db.execute("SELECT count(*) FROM obsidian_sections").fetchone()[0]
    info = {
        "vault_path": str(vault),
        "imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "notes": str(len(notes)),
        "sections": str(section_count),
        "links": str(link_count),
        "assets": str(assets),
        "excluded_notes": str(excluded),
        "added": str(added),
        "updated": str(updated),
        "deleted": str(len(deleted)),
        "unchanged": str(len(notes) - len(changed)),
    }
    db.executemany("INSERT OR REPLACE INTO obsidian_import_info VALUES(?,?)", info.items())
    db.commit()
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.close()
    progress(
        f"Obsidian ready: {len(notes):,} notes · {section_count:,} sections · "
        f"{link_count:,} links · {assets:,} assets · {excluded:,} excluded",
        len(notes), len(notes),
    )
    print(json.dumps({**info, "integrity": integrity}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
