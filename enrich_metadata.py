from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from datetime import datetime,timezone
from pathlib import Path

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/"vendor"))
import pymupdf
from progress_output import progress

DB=Path(os.getenv("MARGINALIA_LIBRARY_DB",str(BASE/"unified_library.sqlite"))).resolve()
CACHE_DB=BASE/"metadata_enrichment_cache.sqlite"
GENERIC=re.compile(r"^(?:paper|full.?text(?: pdf)?|article|document|manuscript|download|untitled|main|submission|accepted.?manuscript|attachment|[a-f0-9]{8,})(?:\.pdf)?$",re.I)

def clean(value):
    value=re.sub(r"<[^>]*>"," ",str(value or ""))
    return re.sub(r"\s+"," ",value).strip(" ._-\t\r\n")

def bad_title(value):
    value=clean(value)
    return not value or bool(GENERIC.fullmatch(value)) or value.casefold().endswith(".pdf")

def pdf_title(document):
    title=clean((document.metadata or {}).get("title"))
    if not bad_title(title) and len(title)>8:
        return title,"pdf_metadata"
    if not document.page_count:
        return None,None
    page=document[0]
    candidates=[]
    try:
        data=page.get_text("dict")
        cutoff=page.rect.height*.55
        for block in data.get("blocks",[]):
            for line in block.get("lines",[]):
                if line.get("bbox",[0,0,0,9999])[1]>cutoff: continue
                spans=line.get("spans",[])
                text=clean(" ".join(s.get("text","") for s in spans))
                size=max([float(s.get("size",0)) for s in spans] or [0])
                y=float(line.get("bbox",[0,0,0,0])[1])
                if 12<=len(text)<=350 and size>=11 and not re.match(r"^(doi:|https?://|abstract$|arxiv|working paper|journal of)",text,re.I):
                    candidates.append((size,-y,text))
    except Exception:
        candidates=[]
    if not candidates: return None,None
    max_size=max(x[0] for x in candidates)
    lines=[x for x in sorted(candidates,key=lambda x:-x[1]) if x[0]>=max_size-1.2]
    title=clean(" ".join(x[2] for x in lines[:4]))
    return (title,"first_page_typography") if not bad_title(title) and len(title)>8 else (None,None)

def pdf_authors(document):
    raw=clean((document.metadata or {}).get("author"))
    if not raw or raw.casefold() in {"anonymous","unknown","author"}: return []
    parts=[clean(x) for x in re.split(r"\s*;\s*|\s+and\s+",raw) if clean(x)]
    if len(parts)==1 and raw.count(",")>=2:
        parts=[clean(x) for x in raw.split(",") if clean(x)]
    result=[]
    for name in parts[:30]:
        if "," in name:
            last,first=[clean(x) for x in name.split(",",1)]
        else:
            bits=name.split(); first=" ".join(bits[:-1]); last=bits[-1] if bits else ""
        if first or last: result.append({"type":"author","firstName":first,"lastName":last})
    return result

db=sqlite3.connect(DB)
db.row_factory=sqlite3.Row
cache=sqlite3.connect(CACHE_DB)
cache.row_factory=sqlite3.Row
cache.execute("""CREATE TABLE IF NOT EXISTS enrichment_cache(
 content_hash TEXT PRIMARY KEY,title TEXT,title_source TEXT,authors_json TEXT NOT NULL)""")
db.execute("""CREATE TABLE IF NOT EXISTS metadata_enrichments(
 id INTEGER PRIMARY KEY,item_id INTEGER NOT NULL,field TEXT NOT NULL,old_value TEXT,new_value TEXT NOT NULL,
 source TEXT NOT NULL,applied_at TEXT NOT NULL,UNIQUE(item_id,field,new_value))""")
items=db.execute("SELECT id,title,creators_json,metadata_json FROM items ORDER BY id").fetchall()
changed_titles=changed_authors=0
for item_number, item in enumerate(items, 1):
    progress(f"Checking metadata: {clean(item['title']) or 'Untitled publication'}", item_number, len(items))
    creators=json.loads(item["creators_json"] or "[]")
    need_title=bad_title(item["title"]); need_authors=not creators
    if not need_title and not need_authors: continue
    paths=db.execute("SELECT local_path,sha256 FROM attachments WHERE item_id=? AND lower(local_path) LIKE '%.pdf' AND local_path!='' ORDER BY annotation_count DESC,id",(item["id"],)).fetchall()
    if not paths: continue
    path=BASE/paths[0][0]
    if not path.is_file(): continue
    content_hash=paths[0][1] or ""
    cached=cache.execute("SELECT title,title_source,authors_json FROM enrichment_cache WHERE content_hash=?",(content_hash,)).fetchone() if content_hash else None
    if cached:
        title,source=cached["title"],cached["title_source"]
        found=json.loads(cached["authors_json"] or "[]")
    else:
        try: document=pymupdf.open(path)
        except Exception: continue
        try:
            title,source=pdf_title(document)
            found=pdf_authors(document)
        finally: document.close()
        if content_hash:
            cache.execute("INSERT OR REPLACE INTO enrichment_cache VALUES(?,?,?,?)",(
                content_hash,title,source,json.dumps(found,ensure_ascii=False)))
            cache.commit()
    if need_title and title:
        metadata=json.loads(item["metadata_json"] or "{}"); metadata["title"]=title
        db.execute("UPDATE items SET title=?,metadata_json=? WHERE id=?",(title,json.dumps(metadata,ensure_ascii=False),item["id"]))
        db.execute("INSERT OR IGNORE INTO metadata_enrichments(item_id,field,old_value,new_value,source,applied_at) VALUES(?,?,?,?,?,?)",(item["id"],"title",item["title"],title,source,datetime.now(timezone.utc).isoformat()))
        changed_titles+=1
    if need_authors and found:
        value=json.dumps(found,ensure_ascii=False)
        db.execute("UPDATE items SET creators_json=? WHERE id=?",(value,item["id"]))
        db.execute("INSERT OR IGNORE INTO metadata_enrichments(item_id,field,old_value,new_value,source,applied_at) VALUES(?,?,?,?,?,?)",(item["id"],"creators",item["creators_json"],value,"pdf_metadata",datetime.now(timezone.utc).isoformat()))
        changed_authors+=1

db.commit()
print(json.dumps({"titles_enriched":changed_titles,"author_lists_enriched":changed_authors,"enrichment_records":db.execute('select count(*) from metadata_enrichments').fetchone()[0]},indent=2))
db.close()
cache.close()
