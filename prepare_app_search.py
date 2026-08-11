from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from library_paths import zotero_root
from progress_output import progress

BASE = Path(__file__).resolve().parent
DB = Path(os.getenv("MARGINALIA_LIBRARY_DB", str(BASE / "unified_library.sqlite"))).resolve()
SOURCE_DB = zotero_root() / "zotero.sqlite"
db = sqlite3.connect(DB)
db.execute("PRAGMA foreign_keys=ON")

db.executescript("""
CREATE TABLE IF NOT EXISTS app_schema(
 key TEXT PRIMARY KEY,
 value TEXT NOT NULL
);

DROP TABLE IF EXISTS zotero_collections;
CREATE TABLE zotero_collections(
 collection_id INTEGER PRIMARY KEY,
 name TEXT NOT NULL,
 parent_id INTEGER,
 path TEXT NOT NULL,
 depth INTEGER NOT NULL
);
DROP TABLE IF EXISTS item_collections;
CREATE TABLE item_collections(
 item_id INTEGER NOT NULL,
 collection_id INTEGER NOT NULL,
 PRIMARY KEY(item_id,collection_id)
);
CREATE INDEX idx_item_collections_collection ON item_collections(collection_id,item_id);
DROP TABLE IF EXISTS collection_descendants;
CREATE TABLE collection_descendants(
 ancestor_id INTEGER NOT NULL,
 descendant_id INTEGER NOT NULL,
 PRIMARY KEY(ancestor_id,descendant_id)
);

DROP VIEW IF EXISTS annotation_context;
CREATE VIEW annotation_context AS
SELECT
 a.id AS annotation_id,
 a.attachment_id,
 at.item_id,
 i.title AS parent_title,
 i.item_type,
 i.date_created,
 i.creators_json,
 at.title AS attachment_title,
 at.local_path,
 CASE a.annotation_type
   WHEN 1 THEN 'highlight'
   WHEN 2 THEN 'note'
   WHEN 3 THEN 'image'
   WHEN 4 THEN 'ink'
   WHEN 5 THEN 'underline'
   WHEN 6 THEN 'text'
   ELSE 'unknown'
 END AS annotation_type,
 a.text,
 a.comment,
 a.color,
 a.page_label,
 a.position_json,
 a.is_external,
 a.date_added,
 a.date_modified
FROM annotations a
JOIN attachments at ON at.id=a.attachment_id
JOIN items i ON i.id=at.item_id;

DROP VIEW IF EXISTS app_items;
CREATE VIEW app_items AS
SELECT
 i.*,
 (SELECT COUNT(*) FROM attachments a WHERE a.item_id=i.id) AS attachment_count,
 (SELECT COUNT(*) FROM attachments a JOIN annotations n ON n.attachment_id=a.id
  WHERE a.item_id=i.id) AS annotation_count,
 (SELECT COUNT(*) FROM notes n WHERE n.item_id=i.id) AS note_count
FROM items i;

DROP TABLE IF EXISTS item_search;
CREATE VIRTUAL TABLE item_search USING fts5(
 item_id UNINDEXED,
 title,
 authors,
 metadata,
 tags,
 collections,
 tokenize='unicode61 remove_diacritics 2'
);

DROP TABLE IF EXISTS annotation_search;
CREATE VIRTUAL TABLE annotation_search USING fts5(
 annotation_id UNINDEXED,
 item_id UNINDEXED,
 attachment_id UNINDEXED,
 parent_title,
 authors,
 annotation_text,
 comment,
 page_label,
 tokenize='unicode61 remove_diacritics 2'
);

DROP TABLE IF EXISTS note_search;
CREATE VIRTUAL TABLE note_search USING fts5(
 note_id UNINDEXED,
 item_id UNINDEXED,
 title,
 note_text,
 tokenize='unicode61 remove_diacritics 2'
);
""")

if SOURCE_DB.is_file():
    source = sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    collection_rows = list(source.execute("SELECT collectionID,collectionName,parentCollectionID FROM collections"))
    collection_map = {r["collectionID"]: r for r in collection_rows}
    def collection_path(collection_id):
        names=[]; seen=set(); current=collection_map.get(collection_id)
        while current and current["collectionID"] not in seen:
            seen.add(current["collectionID"]); names.append(current["collectionName"])
            current=collection_map.get(current["parentCollectionID"])
        return " / ".join(reversed(names))
    for row in collection_rows:
        path=collection_path(row["collectionID"])
        db.execute("INSERT INTO zotero_collections VALUES(?,?,?,?,?)",(
            row["collectionID"],row["collectionName"],row["parentCollectionID"],path,max(0,path.count(" / "))))
        current=row["collectionID"]; seen=set()
        while current and current not in seen:
            seen.add(current)
            db.execute("INSERT OR IGNORE INTO collection_descendants VALUES(?,?)",(current,row["collectionID"]))
            parent=collection_map.get(current)
            current=parent["parentCollectionID"] if parent else None
    unified_by_source={r[1]:r[0] for r in db.execute("SELECT item_id,source_item_id FROM item_sources")}
    for membership in source.execute("SELECT collectionID,itemID FROM collectionItems"):
        item_id=unified_by_source.get(membership["itemID"])
        if item_id:
            db.execute("INSERT OR IGNORE INTO item_collections VALUES(?,?)",(item_id,membership["collectionID"]))
    source.close()

search_items = db.execute("SELECT id,title,creators_json,metadata_json,tags_json,collections_json FROM items").fetchall()
search_annotations = db.execute("SELECT annotation_id,item_id,attachment_id,parent_title,creators_json,text,comment,page_label FROM annotation_context").fetchall()
search_notes = db.execute("SELECT id,item_id,title,note_html FROM notes").fetchall()
search_total = len(search_items) + len(search_annotations) + len(search_notes)
for number, r in enumerate(search_items, 1):
    progress(f"Indexing publication: {r[1] or 'Untitled publication'}", number, search_total)
    creators = json.loads(r[2] or "[]")
    authors = " ; ".join(" ".join(x for x in (c.get("firstName", ""), c.get("lastName", "")) if x) for c in creators)
    metadata = json.loads(r[3] or "{}")
    metadata_text = "\n".join(f"{k}: {v}" for k,v in metadata.items() if v)
    tags = " ; ".join(json.loads(r[4] or "[]"))
    collections = " ; ".join(json.loads(r[5] or "[]"))
    db.execute("INSERT INTO item_search VALUES(?,?,?,?,?,?)",(r[0],r[1],authors,metadata_text,tags,collections))

for number, r in enumerate(search_annotations, 1):
    progress(f"Indexing annotation from: {r[3] or 'Untitled publication'}", len(search_items) + number, search_total)
    creators = json.loads(r[4] or "[]")
    authors = " ; ".join(" ".join(x for x in (c.get("firstName", ""), c.get("lastName", "")) if x) for c in creators)
    db.execute("INSERT INTO annotation_search VALUES(?,?,?,?,?,?,?,?)",(r[0],r[1],r[2],r[3],authors,r[5],r[6],r[7]))

for number, r in enumerate(search_notes, 1):
    progress(f"Indexing Zotero note: {r[2] or 'Untitled note'}", len(search_items) + len(search_annotations) + number, search_total)
    db.execute("INSERT INTO note_search VALUES(?,?,?,?)",r)

values = {
    "schema_version": "1",
    "search_engine": "SQLite FTS5",
    "annotation_only_search": "annotation_search",
    "bibliographic_search": "item_search",
    "notes_search": "note_search",
    "full_document_search": "not_yet_indexed",
}
db.executemany("INSERT OR REPLACE INTO app_schema VALUES(?,?)",values.items())
db.commit()

result = {
    "integrity": db.execute("PRAGMA integrity_check").fetchone()[0],
    "items_indexed": db.execute("SELECT count(*) FROM item_search").fetchone()[0],
    "annotations_indexed": db.execute("SELECT count(*) FROM annotation_search").fetchone()[0],
    "notes_indexed": db.execute("SELECT count(*) FROM note_search").fetchone()[0],
    "zotero_folders": db.execute("SELECT count(*) FROM zotero_collections").fetchone()[0],
    "folder_memberships": db.execute("SELECT count(*) FROM item_collections").fetchone()[0],
}
db.close()
print(json.dumps(result,indent=2))
