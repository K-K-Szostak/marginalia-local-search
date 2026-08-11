from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

from library_paths import zotero_root
from progress_output import progress

ROOT = zotero_root()
SOURCE_DB = ROOT / "zotero.sqlite"
OUT_DIR = Path(__file__).resolve().parent
OUT_DB = Path(os.getenv("MARGINALIA_LIBRARY_DB", str(OUT_DIR / "unified_library.sqlite"))).resolve()
HASH_CACHE_PATH = OUT_DIR / "attachment_hash_cache.json"
SYNC_STATE_PATH = OUT_DIR / "zotero_sync_state.json"
try:
    HASH_CACHE = json.loads(HASH_CACHE_PATH.read_text(encoding="utf-8")) if HASH_CACHE_PATH.is_file() else {}
except (OSError, ValueError, TypeError):
    HASH_CACHE = {}
CURRENT_HASH_CACHE = {}
try:
    LINKED_ATTACHMENT_MAP = json.loads((ROOT / "linked_attachment_map.json").read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError):
    LINKED_ATTACHMENT_MAP = {}


def norm(value: str | None) -> str:
    value = unicodedata.normalize("NFKD", value or "").casefold()
    return re.sub(r"[^a-z0-9]+", "", value)


def year(value: str | None) -> str:
    m = re.search(r"(?:18|19|20|21)\d{2}", value or "")
    return m.group(0) if m else ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_sha256(path: Path, cache_key: str | None = None) -> str:
    # Snapshot generation paths change on every refresh. Zotero's source item
    # identity is stable and lets unchanged hardlinks reuse their digest.
    resolved = str(cache_key or path.resolve())
    stat = path.stat()
    signature = f"{stat.st_size}:{stat.st_mtime_ns}"
    cached = HASH_CACHE.get(resolved, {})
    digest = cached.get("sha256") if cached.get("signature") == signature else sha256(path)
    CURRENT_HASH_CACHE[resolved] = {"signature": signature, "sha256": digest}
    return digest


def unique(values):
    seen, result = set(), []
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


src = sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro", uri=True)
src.row_factory = sqlite3.Row

item_types = {r[0]: r[1] for r in src.execute("SELECT itemTypeID,typeName FROM itemTypes")}
fields = {r[0]: r[1] for r in src.execute("SELECT fieldID,fieldName FROM fields")}
creator_types = {r[0]: r[1] for r in src.execute("SELECT creatorTypeID,creatorType FROM creatorTypes")}

metadata = defaultdict(dict)
for r in src.execute("SELECT itemID,fieldID,value FROM itemData JOIN itemDataValues USING(valueID)"):
    metadata[r[0]][fields[r[1]]] = r[2]

creators = defaultdict(list)
for r in src.execute("""
    SELECT ic.itemID,ic.orderIndex,ic.creatorTypeID,c.firstName,c.lastName
    FROM itemCreators ic JOIN creators c USING(creatorID)
    ORDER BY ic.itemID,ic.orderIndex
"""):
    creators[r[0]].append({"type": creator_types[r[2]], "firstName": r[3] or "", "lastName": r[4] or ""})

tags = defaultdict(list)
for r in src.execute("SELECT itemID,name FROM itemTags JOIN tags USING(tagID)"):
    tags[r[0]].append(r[1])

collections = defaultdict(list)
for r in src.execute("SELECT itemID,collectionName FROM collectionItems JOIN collections USING(collectionID)"):
    collections[r[0]].append(r[1])

annotations = defaultdict(list)
for r in src.execute("""
    SELECT a.*,i.key,i.dateAdded,i.dateModified
    FROM itemAnnotations a JOIN items i ON i.itemID=a.itemID
    ORDER BY a.parentItemID,a.sortIndex
"""):
    annotations[r["parentItemID"]].append(dict(r))

attachments = {}
children = defaultdict(list)
for r in src.execute("""
    SELECT a.*,i.key,i.dateAdded,i.dateModified
    FROM itemAttachments a JOIN items i ON i.itemID=a.itemID
"""):
    row = dict(r)
    p = row.get("path") or ""
    if p.startswith("storage:"):
        row["source_file"] = str(ROOT / "storage" / row["key"] / p[8:])
    elif p and row.get("linkMode") in (2, 3):
        managed = LINKED_ATTACHMENT_MAP.get(str(row["itemID"]), "")
        row["source_file"] = str(ROOT / managed) if managed else ""
    else:
        row["source_file"] = ""
    attachments[row["itemID"]] = row
    if row["parentItemID"]:
        children[row["parentItemID"]].append(row["itemID"])

deleted = {r[0] for r in src.execute("SELECT itemID FROM deletedItems")}
source_versions = {
    str(r["itemID"]): r["dateModified"] or ""
    for r in src.execute("SELECT itemID,dateModified FROM items")
    if r["itemID"] not in deleted
}
try:
    previous_versions = json.loads(SYNC_STATE_PATH.read_text(encoding="utf-8")) if SYNC_STATE_PATH.is_file() else {}
except (OSError, ValueError, TypeError):
    previous_versions = {}
added_source_records = set(source_versions) - set(previous_versions)
deleted_source_records = set(previous_versions) - set(source_versions)
modified_source_records = {
    key for key in set(source_versions) & set(previous_versions)
    if source_versions[key] != previous_versions[key]
}
bib = {}
for r in src.execute("SELECT * FROM items ORDER BY dateAdded,itemID"):
    iid = r["itemID"]
    typ = item_types[r["itemTypeID"]]
    if iid in deleted or typ in {"attachment", "note", "annotation"}:
        continue
    bib[iid] = {
        "itemID": iid, "key": r["key"], "type": typ,
        "dateAdded": r["dateAdded"], "dateModified": r["dateModified"],
        "metadata": dict(metadata[iid]), "creators": creators[iid],
        "tags": tags[iid], "collections": collections[iid],
        "attachment_ids": children[iid],
    }

# Keep standalone attachments in the unified model as synthetic parent records.
for aid, a in attachments.items():
    if a["parentItemID"] is None and aid not in deleted:
        bib[-aid] = {
            "itemID": -aid, "key": "standalone:" + a["key"], "type": "standaloneAttachment",
            "dateAdded": a["dateAdded"], "dateModified": a["dateModified"],
            "metadata": dict(metadata[aid]), "creators": creators[aid], "tags": tags[aid],
            "collections": collections[aid], "attachment_ids": [aid],
        }


def annotation_count(item):
    return sum(len(annotations[a]) for a in item["attachment_ids"])


def dedupe_key(item):
    if item["type"] == "standaloneAttachment":
        return ("standalone", str(item["itemID"]))
    m = item["metadata"]
    doi = norm(m.get("DOI"))
    if doi:
        return ("doi", doi)
    isbn = norm(m.get("ISBN"))
    if isbn:
        return ("isbn", isbn)
    title = norm(m.get("title"))
    if not title:
        return ("unique", str(item["itemID"]))
    first_author = ""
    if item["creators"]:
        first_author = norm(item["creators"][0].get("lastName") or item["creators"][0].get("firstName"))
    item_year = year(m.get("date"))
    if not item_year and not first_author:
        return ("unique", str(item["itemID"]))
    return ("title", title, item_year, first_author)


groups = defaultdict(list)
for item in bib.values():
    groups[dedupe_key(item)].append(item)

if OUT_DB.exists():
    OUT_DB.unlink()

out = sqlite3.connect(OUT_DB)
out.executescript("""
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

stats = {"source_bibliographic_items": len([x for x in bib.values() if x['type'] != 'standaloneAttachment']),
         "source_standalone_attachments": len([x for x in bib.values() if x['type'] == 'standaloneAttachment']),
         "unified_items": 0, "duplicate_item_groups": 0,
         "source_attachments": 0, "unified_attachments": 0, "merged_attachment_groups": 0,
         "annotations": 0, "missing_files": 0}

next_item = next_attachment = next_annotation = 1
source_item_to_unified = {}
source_attachment_to_unified = {}
sorted_groups = sorted(groups.items(), key=lambda x: min(i["itemID"] for i in x[1]))
for group_number, (match, members) in enumerate(sorted_groups, 1):
    # Requested precedence: most annotations, then oldest record.
    members.sort(key=lambda i: (-annotation_count(i), i["dateAdded"], i["itemID"]))
    canonical = members[0]
    progress(f"Organizing publication: {canonical['metadata'].get('title') or canonical['key']}", group_number, len(sorted_groups))
    merged_meta = dict(canonical["metadata"])
    for member in sorted(members, key=lambda i: (i["dateAdded"], i["itemID"])):
        for k, v in member["metadata"].items():
            if not merged_meta.get(k) and v:
                merged_meta[k] = v
    merged_creators = canonical["creators"] or next((m["creators"] for m in members if m["creators"]), [])
    merged_tags = unique(t for m in members for t in m["tags"])
    merged_cols = unique(c for m in members for c in m["collections"])
    iid = next_item; next_item += 1
    out.execute("INSERT INTO items VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        iid, canonical["itemID"], canonical["key"], canonical["type"], merged_meta.get("title"),
        merged_meta.get("date"), canonical["dateAdded"], max(m["dateModified"] for m in members),
        merged_meta.get("DOI"), merged_meta.get("ISBN"), merged_meta.get("url"),
        json.dumps(merged_meta, ensure_ascii=False), json.dumps(merged_creators, ensure_ascii=False),
        json.dumps(merged_tags, ensure_ascii=False), json.dumps(merged_cols, ensure_ascii=False), len(members)))
    for m in members:
        if m["itemID"] > 0:
            source_item_to_unified[m["itemID"]] = iid
        out.execute("INSERT INTO item_sources VALUES(?,?,?,?,?,?)", (iid,m["itemID"],m["key"],m["dateAdded"],annotation_count(m),json.dumps(m["metadata"],ensure_ascii=False)))
    out.execute("INSERT INTO duplicate_groups VALUES(?,?,?,?)", (iid,match[0],json.dumps(match[1:],ensure_ascii=False),json.dumps([m["itemID"] for m in members])))
    stats["unified_items"] += 1
    if len(members) > 1: stats["duplicate_item_groups"] += 1

    candidates = [attachments[a] for m in members for a in m["attachment_ids"] if a in attachments]
    stats["source_attachments"] += len(candidates)
    attachment_groups = defaultdict(list)
    for a in candidates:
        p = Path(a["source_file"]) if a["source_file"] else None
        digest = ""
        if p and p.is_file():
            progress(f"Checking attachment: {p.name}", group_number, len(sorted_groups))
            try: digest = cached_sha256(p, f"zotero:{a['itemID']}:{a['key']}")
            except OSError: pass
        else:
            stats["missing_files"] += 1
        a["sha256"] = digest
        # Exact file equality is a safe duplicate test; missing files remain distinct.
        key = ("hash", digest) if digest else ("source", str(a["itemID"]))
        attachment_groups[key].append(a)

    for _, amembers in attachment_groups.items():
        amembers.sort(key=lambda a: (-len(annotations[a["itemID"]]), a["dateAdded"], a["itemID"]))
        ac = amembers[0]
        aid = next_attachment; next_attachment += 1
        source_path = Path(ac["source_file"]) if ac["source_file"] else None
        local_path, size = "", None
        if source_path and source_path.is_file():
            # source_path already belongs to Marginalia's private snapshot.
            # Index that safe copy directly instead of duplicating it again.
            local_path = str(source_path.relative_to(OUT_DIR))
            size = source_path.stat().st_size
        title = metadata[ac["itemID"]].get("title")
        merged_anns = []
        seen_anns = set()
        for a in amembers:
            source_attachment_to_unified[a["itemID"]] = aid
            for ann in annotations[a["itemID"]]:
                marker = (ann["type"],ann["text"] or "",ann["comment"] or "",ann["pageLabel"] or "",ann["position"] or "")
                if marker not in seen_anns:
                    seen_anns.add(marker); merged_anns.append(ann)
        out.execute("INSERT INTO attachments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            aid,iid,ac["itemID"],ac["key"],title,ac["contentType"],ac["path"],local_path,
            ac["sha256"],size,ac["dateAdded"],max(a["dateModified"] for a in amembers),len(merged_anns),len(amembers)))
        for a in amembers:
            out.execute("INSERT INTO attachment_sources VALUES(?,?,?,?,?,?)", (aid,a["itemID"],a["key"],a["path"],a["sha256"],len(annotations[a["itemID"]])))
        for ann in merged_anns:
            out.execute("INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                next_annotation,aid,ann["itemID"],ann["key"],ann["type"],ann["authorName"],ann["text"],
                ann["comment"],ann["color"],ann["pageLabel"],ann["sortIndex"],ann["position"],ann["isExternal"],
                ann["dateAdded"],ann["dateModified"]))
            next_annotation += 1
        stats["annotations"] += len(merged_anns)
        stats["unified_attachments"] += 1
        if len(amembers) > 1: stats["merged_attachment_groups"] += 1
    progress(
        f"Added publication: {merged_meta.get('title') or canonical['key']} · "
        f"{len(candidates):,} attachment(s) · {sum(annotation_count(member) for member in members):,} source annotation(s)",
        group_number, len(sorted_groups),
    )

note_count = 0
note_rows = src.execute("""
    SELECT n.*,i.key,i.dateAdded,i.dateModified
    FROM itemNotes n JOIN items i ON i.itemID=n.itemID
    LEFT JOIN deletedItems d ON d.itemID=n.itemID WHERE d.itemID IS NULL
""").fetchall()
for note_number, r in enumerate(note_rows, 1):
    progress(f"Adding Zotero note: {r['title'] or r['key']}", note_number, len(note_rows))
    parent = r["parentItemID"]
    out.execute("INSERT INTO notes VALUES(?,?,?,?,?,?,?,?,?)", (
        note_count + 1,r["itemID"],r["key"],source_item_to_unified.get(parent),
        source_attachment_to_unified.get(parent),r["title"],r["note"],r["dateAdded"],r["dateModified"]))
    note_count += 1
stats["notes"] = note_count
progress(
    f"Unified library ready: {stats['unified_items']:,} publications · "
    f"{stats['unified_attachments']:,} attachments · {stats['annotations']:,} annotations · {note_count:,} notes",
    len(sorted_groups), len(sorted_groups),
)
stats["source_changes"] = {
    "added": len(added_source_records),
    "modified": len(modified_source_records),
    "deleted": len(deleted_source_records),
    "unchanged": len(source_versions) - len(added_source_records) - len(modified_source_records),
}

stats["duplicate_items_removed"] = stats["source_bibliographic_items"] + stats["source_standalone_attachments"] - stats["unified_items"]
stats["duplicate_attachments_removed"] = stats["source_attachments"] - stats["unified_attachments"]
for k,v in {"format":"Unified Zotero knowledge base v1","source":str(SOURCE_DB),"rules":"prefer annotated; otherwise oldest; merge annotations; fill missing metadata","statistics":json.dumps(stats,ensure_ascii=False)}.items():
    out.execute("INSERT INTO library_info VALUES(?,?)",(k,v))
out.commit()
integrity = out.execute("PRAGMA integrity_check").fetchone()[0]
fk_errors = out.execute("PRAGMA foreign_key_check").fetchall()
out.close(); src.close()
report = OUT_DIR / "build_report.json"
report.write_text(json.dumps({"integrity_check":integrity,"foreign_key_errors":len(fk_errors),**stats},indent=2,ensure_ascii=False),encoding="utf-8")
HASH_CACHE_PATH.write_text(json.dumps(CURRENT_HASH_CACHE,indent=2,ensure_ascii=False),encoding="utf-8")
SYNC_STATE_PATH.write_text(json.dumps(source_versions,indent=2,ensure_ascii=False),encoding="utf-8")
print(report.read_text(encoding="utf-8"))
