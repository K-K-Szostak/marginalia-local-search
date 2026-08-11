from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


BASE = Path(__file__).resolve().parent
DB = Path(os.getenv("MARGINALIA_LIBRARY_DB", str(BASE / "unified_library.sqlite"))).resolve()
REPORT = BASE / "build_report.json"

DB.unlink(missing_ok=True)
db = sqlite3.connect(DB)
db.executescript("""
PRAGMA foreign_keys=ON;
CREATE TABLE library_info(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE items(
 id INTEGER PRIMARY KEY, canonical_source_item_id INTEGER NOT NULL, zotero_key TEXT NOT NULL,
 item_type TEXT NOT NULL, title TEXT, date_created TEXT, date_added TEXT, date_modified TEXT,
 doi TEXT, isbn TEXT, url TEXT, metadata_json TEXT NOT NULL, creators_json TEXT NOT NULL,
 tags_json TEXT NOT NULL, collections_json TEXT NOT NULL, source_count INTEGER NOT NULL
);
CREATE TABLE item_sources(
 item_id INTEGER NOT NULL REFERENCES items(id), source_item_id INTEGER NOT NULL,
 source_key TEXT NOT NULL, date_added TEXT, annotation_count INTEGER NOT NULL,
 metadata_json TEXT NOT NULL, PRIMARY KEY(item_id,source_item_id)
);
CREATE TABLE attachments(
 id INTEGER PRIMARY KEY, item_id INTEGER NOT NULL REFERENCES items(id),
 canonical_source_attachment_id INTEGER NOT NULL, zotero_key TEXT NOT NULL,
 title TEXT, content_type TEXT, original_path TEXT, local_path TEXT,
 sha256 TEXT, size_bytes INTEGER, date_added TEXT, date_modified TEXT,
 annotation_count INTEGER NOT NULL, source_count INTEGER NOT NULL
);
CREATE TABLE attachment_sources(
 attachment_id INTEGER NOT NULL REFERENCES attachments(id), source_attachment_id INTEGER NOT NULL,
 source_key TEXT NOT NULL, original_path TEXT, sha256 TEXT, annotation_count INTEGER NOT NULL,
 PRIMARY KEY(attachment_id,source_attachment_id)
);
CREATE TABLE annotations(
 id INTEGER PRIMARY KEY, attachment_id INTEGER NOT NULL REFERENCES attachments(id),
 source_annotation_id INTEGER NOT NULL, zotero_key TEXT NOT NULL, annotation_type INTEGER NOT NULL,
 author_name TEXT, text TEXT, comment TEXT, color TEXT, page_label TEXT,
 sort_index TEXT, position_json TEXT, is_external INTEGER NOT NULL,
 date_added TEXT, date_modified TEXT
);
CREATE TABLE notes(
 id INTEGER PRIMARY KEY, source_note_item_id INTEGER NOT NULL, zotero_key TEXT NOT NULL,
 item_id INTEGER REFERENCES items(id), attachment_id INTEGER REFERENCES attachments(id),
 title TEXT, note_html TEXT, date_added TEXT, date_modified TEXT
);
CREATE TABLE duplicate_groups(
 item_id INTEGER NOT NULL REFERENCES items(id), match_method TEXT NOT NULL,
 match_key TEXT NOT NULL, source_item_ids_json TEXT NOT NULL
);
CREATE INDEX idx_items_title ON items(title);
CREATE INDEX idx_annotations_attachment ON annotations(attachment_id);
""")
stats = {
    "source_bibliographic_items": 0, "source_standalone_attachments": 0,
    "unified_items": 0, "duplicate_item_groups": 0, "source_attachments": 0,
    "unified_attachments": 0, "merged_attachment_groups": 0, "annotations": 0,
    "notes": 0, "missing_files": 0,
    "source_changes": {"added": 0, "modified": 0, "deleted": 0, "unchanged": 0},
    "duplicate_items_removed": 0, "duplicate_attachments_removed": 0,
}
for key, value in {
    "format": "Empty Zotero library for Obsidian-only mode v1",
    "source": "No Zotero source selected",
    "rules": "Zotero is optional",
    "statistics": json.dumps(stats),
}.items():
    db.execute("INSERT INTO library_info VALUES(?,?)", (key, value))
db.commit()
integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
db.close()
REPORT.write_text(json.dumps({"integrity_check": integrity, **stats}, indent=2), encoding="utf-8")
print(REPORT.read_text(encoding="utf-8"))
