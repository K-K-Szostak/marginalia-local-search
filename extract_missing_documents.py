from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

from library_paths import zotero_root
from progress_output import progress

BASE = Path(__file__).resolve().parent
ROOT = zotero_root()
DB_PATH = Path(os.getenv("MARGINALIA_LIBRARY_DB", str(BASE / "unified_library.sqlite"))).resolve()
ZOTERO_DB = ROOT / "zotero.sqlite"
sys.path.insert(0, str(BASE / "vendor"))
import pymupdf


def clean(value):
    return str(value or "").replace("\x00", "").strip()


def cache_pages(text: str, count: int) -> list[str]:
    text = clean(text)
    if not text:
        return []
    count = max(1, min(int(count or 1), 2000, len(text)))
    bounds = [0]
    for index in range(1, count):
        expected = round(len(text) * index / count)
        left = text.rfind("\n", max(bounds[-1] + 1, expected - 600), expected + 1)
        right = text.find("\n", expected, min(len(text), expected + 600))
        choices = [point + 1 for point in (left, right) if point >= bounds[-1]]
        boundary = min(choices, key=lambda point: abs(point - expected)) if choices else expected
        bounds.append(max(bounds[-1] + 1, boundary))
    bounds.append(len(text))
    return [clean(text[bounds[i]:bounds[i + 1]]) for i in range(count)]


db = sqlite3.connect(DB_PATH)
db.row_factory = sqlite3.Row
zotero = sqlite3.connect(f"file:{ZOTERO_DB.as_posix()}?mode=ro", uri=True) if ZOTERO_DB.is_file() else None
next_page_id = db.execute("SELECT coalesce(max(id),0)+1 FROM document_pages").fetchone()[0]
targets = db.execute("""
    SELECT a.id,a.item_id,a.canonical_source_attachment_id,a.zotero_key,a.local_path
    FROM attachments a
    WHERE (lower(a.content_type)='application/pdf' OR lower(a.local_path) LIKE '%.pdf')
      AND NOT EXISTS(SELECT 1 FROM document_pages p WHERE p.attachment_id=a.id)
    ORDER BY a.id
""").fetchall()

recovered_pdf = recovered_cache = unavailable = errors = 0
for number, row in enumerate(targets, 1):
    pages: list[str] = []
    source = ""
    path = BASE / row["local_path"] if row["local_path"] else None
    progress(f"Recovering text: {path.name if path else row['zotero_key']}", number, len(targets))
    if path and path.is_file():
        try:
            document = pymupdf.open(path)
            try:
                if document.needs_pass and not document.authenticate(""):
                    raise ValueError("Encrypted PDF requires a password")
                pages = [clean(page.get_text("text") or "") for page in document]
            finally:
                document.close()
            if pages:
                source = "pdf"; recovered_pdf += 1
        except Exception as exc:
            db.execute("INSERT OR REPLACE INTO extraction_status VALUES(?,?,?,?,?,?)",
                       (row["id"], "error", None, 0, 0, str(exc)[:1000]))
            errors += 1
    if not pages:
        cache = ROOT / "storage" / row["zotero_key"] / ".zotero-ft-cache"
        if cache.is_file() and cache.stat().st_size:
            text = cache.read_text(encoding="utf-8", errors="replace")
            page_count = 0
            if zotero:
                info = zotero.execute("SELECT indexedPages,totalPages FROM fulltextItems WHERE itemID=?",
                                      (row["canonical_source_attachment_id"],)).fetchone()
                if info: page_count = info[1] or info[0] or 0
            if not page_count:
                annotation_page = db.execute("""
                    SELECT coalesce(max(CAST(json_extract(position_json,'$.pageIndex') AS INTEGER)+1),0)
                    FROM annotations WHERE attachment_id=?
                """, (row["id"],)).fetchone()[0]
                page_count = annotation_page or 1
            pages = cache_pages(text, page_count)
            if pages:
                source = "zotero-cache"; recovered_cache += 1
    if pages:
        for page_number, text in enumerate(pages, 1):
            db.execute("INSERT INTO document_pages(id,attachment_id,item_id,page_number,text,char_count) VALUES(?,?,?,?,?,?)",
                       (next_page_id, row["id"], row["item_id"], page_number, text, len(text)))
            next_page_id += 1
        status = "ok" if source == "pdf" else "cache"
        note = "" if source == "pdf" else "Recovered from Zotero full-text cache; original PDF unavailable"
        db.execute("INSERT OR REPLACE INTO extraction_status VALUES(?,?,?,?,?,?)",
                   (row["id"], status, len(pages), sum(len(text) for text in pages), 0, note))
    else:
        unavailable += 1
        db.execute("INSERT OR REPLACE INTO extraction_status VALUES(?,?,?,?,?,?)",
                   (row["id"], "missing", None, 0, 0, "Neither local PDF nor Zotero text cache is available"))
    if number % 25 == 0:
        db.commit(); print(f"{number}/{len(targets)} missing documents checked", flush=True)

db.executescript("""
DELETE FROM document_search;
INSERT INTO document_search
SELECT p.id,p.attachment_id,p.item_id,p.page_number,i.title,
       json_extract(i.creators_json,'$'),p.text
FROM document_pages p JOIN items i ON i.id=p.item_id
WHERE coalesce((SELECT status FROM extraction_status s WHERE s.attachment_id=p.attachment_id),'') <> 'cache'
   OR (
       NOT EXISTS(
           SELECT 1 FROM attachments a2 JOIN extraction_status s2 ON s2.attachment_id=a2.id
           WHERE a2.item_id=p.item_id AND s2.status='ok'
             AND EXISTS(SELECT 1 FROM document_pages p2 WHERE p2.attachment_id=a2.id)
       )
       AND p.attachment_id=(
           SELECT a3.id FROM attachments a3 JOIN extraction_status s3 ON s3.attachment_id=a3.id
           WHERE a3.item_id=p.item_id AND s3.status='cache'
             AND EXISTS(SELECT 1 FROM document_pages p3 WHERE p3.attachment_id=a3.id)
           ORDER BY a3.annotation_count DESC,a3.id LIMIT 1
       )
   );
PRAGMA optimize;
""")
db.commit()
summary = {
    "checked": len(targets),
    "recovered_from_pdf": recovered_pdf,
    "recovered_from_zotero_cache": recovered_cache,
    "still_unavailable": unavailable,
    "errors": errors,
    "total_pages": db.execute("SELECT count(*) FROM document_pages").fetchone()[0],
    "search_rows": db.execute("SELECT count(*) FROM document_search").fetchone()[0],
    "integrity": db.execute("PRAGMA integrity_check").fetchone()[0],
}
db.close()
if zotero: zotero.close()
(BASE / "missing_extraction_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
