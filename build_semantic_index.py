from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sqlite3
import time
import urllib.request
from pathlib import Path

import numpy as np
from progress_output import progress as live_progress
from semantic_models import index_path as semantic_model_index_path, progress_path as semantic_model_progress_path, register_model as register_embedding_model


BASE = Path(__file__).resolve().parent
LIBRARY_DB = BASE / "unified_library.sqlite"
CLEAN_DB = BASE / "clean_text.sqlite"
INDEX_DB = BASE / "semantic_index.sqlite"
OBSIDIAN_DB = BASE / "obsidian_notes.sqlite"
PROGRESS = BASE / "semantic_index_progress.json"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIMENSIONS = 0
MODEL_IDENTITY = ""
TARGET_CHARS = 4_000
MAX_CHARS = 12_000


SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_chunks(
  id INTEGER PRIMARY KEY,
  chunk_key TEXT NOT NULL UNIQUE,
  content_hash TEXT NOT NULL,
  model TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  kind TEXT NOT NULL,
  label TEXT NOT NULL,
  item_id INTEGER,
  attachment_id INTEGER,
  page TEXT,
  section_title TEXT,
  annotation_id INTEGER,
  embedded_annotation_id INTEGER,
  note_id INTEGER,
  obsidian_note_id TEXT,
  obsidian_section_id TEXT,
  vault_path TEXT,
  title TEXT,
  text TEXT NOT NULL,
  embedding BLOB NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS semantic_chunks_item ON semantic_chunks(item_id);
CREATE INDEX IF NOT EXISTS semantic_chunks_scope ON semantic_chunks(kind);
CREATE INDEX IF NOT EXISTS semantic_chunks_model ON semantic_chunks(model);
CREATE TABLE IF NOT EXISTS semantic_index_info(key TEXT PRIMARY KEY,value TEXT NOT NULL);
"""


def plain_text(value):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", str(value or "")))).strip()


def paragraph_text(value):
    parts = []
    for part in re.split(r"(?:\r?\n){2,}", html.unescape(re.sub(r"<[^>]*>", " ", str(value or "")))):
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            parts.append(part)
    return "\n\n".join(parts)


def page_from_annotation(position_json, page_label):
    try:
        page_index = json.loads(position_json or "{}").get("pageIndex")
        if isinstance(page_index, int) and page_index >= 0:
            return str(page_index + 1)
    except Exception:
        pass
    match = re.search(r"\d+", str(page_label or ""))
    return match.group() if match else str(page_label or "")


def selected_document_attachments(library, clean):
    """Prefer layout extraction; retain one cache-only version per publication."""
    attachment_info = {
        row[0]: row for row in library.execute(
            "SELECT id,item_id,annotation_count,date_added FROM attachments"
        )
    }
    by_item = {}
    for row in clean.execute("""
        SELECT attachment_id,item_id,source,clean_chars,clean_blocks
        FROM clean_extraction_status WHERE status='ok'
    """):
        by_item.setdefault(row[1], []).append(row)
    selected = set()
    for candidates in by_item.values():
        layout = [row for row in candidates if row[2] == "pdf-layout"]
        if layout:
            selected.update(row[0] for row in layout)
            continue

        def preference(row):
            info = attachment_info.get(row[0])
            annotation_count = int(info[2] if info else 0)
            date_added = str(info[3] if info else "")
            return annotation_count > 0, row[3], row[4], date_added == "", -row[0]

        selected.add(max(candidates, key=preference)[0])
    return selected


def chunk(key, kind, label, item_id, attachment_id, page, title, text, section_title="", **ids):
    text = paragraph_text(text)[:MAX_CHARS]
    title = plain_text(title) or "Untitled"
    section_title = plain_text(section_title)
    if not text:
        return None
    embedding_parts = [f"Title: {title}", f"Type: {label}"]
    if section_title:
        embedding_parts.append(f"Section: {section_title}")
    embedding_parts.append(text)
    embedded_text = "\n".join(embedding_parts)[:MAX_CHARS]
    digest = hashlib.sha256((EMBED_MODEL + "\0" + MODEL_IDENTITY + "\0" + str(EMBED_DIMENSIONS) + "\0" + embedded_text).encode("utf-8")).hexdigest()
    return {
        "chunk_key": key,
        "content_hash": digest,
        "kind": kind,
        "label": label,
        "item_id": item_id,
        "attachment_id": attachment_id,
        "page": str(page or ""),
        "section_title": section_title,
        "annotation_id": ids.get("annotation_id"),
        "embedded_annotation_id": ids.get("embedded_annotation_id"),
        "note_id": ids.get("note_id"),
        "obsidian_note_id": ids.get("obsidian_note_id"),
        "obsidian_section_id": ids.get("obsidian_section_id"),
        "vault_path": ids.get("vault_path"),
        "title": title,
        "text": text,
        "embedded_text": embedded_text,
    }


def document_section_chunks(library, clean):
    """Group complete clean paragraphs by section, never by page boundaries."""
    selected = selected_document_attachments(library, clean)
    current_attachment = current_item = None
    current_title = current_section = ""
    buffers = {}
    ordinal = 0

    def flush_kind(kind):
        nonlocal ordinal
        state = buffers.get(kind)
        if not state or not state["parts"]:
            return None
        ordinal += 1
        body = "\n\n".join(state["parts"])
        page = str(state["start_page"]) if state["start_page"] == state["end_page"] else f"{state['start_page']}–{state['end_page']}"
        labels = {
            "document": "PDF section",
            "document_footnote": "PDF footnotes",
            "document_table": "PDF table",
        }
        value = chunk(
            f"{kind}:{current_attachment}:{ordinal}", kind, labels[kind],
            current_item, current_attachment, page, current_title, body, current_section,
        )
        buffers[kind] = {"parts": [], "chars": 0, "start_page": None, "end_page": None}
        return value

    def flush_all():
        values = []
        for kind in ("document", "document_footnote", "document_table"):
            value = flush_kind(kind)
            if value:
                values.append(value)
        return values

    rows = clean.execute("""
        SELECT attachment_id,item_id,title,block_type,section_title,page_start,page_end,text
        FROM clean_document_blocks ORDER BY attachment_id,ordinal
    """)
    for attachment_id, item_id, title, block_type, section_title, page_start, page_end, text in rows:
        if attachment_id not in selected:
            continue
        if current_attachment != attachment_id:
            for value in flush_all():
                yield value
            current_attachment = attachment_id
            current_item = item_id
            current_title = title
            current_section = ""
            buffers = {}
            ordinal = 0
        if block_type == "heading":
            for value in flush_all():
                yield value
            current_section = plain_text(text)
            continue
        if block_type in {"toc_entry", "metadata"}:
            continue
        kind = "document_footnote" if block_type == "footnote" else "document_table" if block_type == "table_row" else "document"
        section = plain_text(section_title)
        if section != current_section:
            for value in flush_all():
                yield value
            current_section = section
        state = buffers.setdefault(kind, {"parts": [], "chars": 0, "start_page": None, "end_page": None})
        proposed = state["chars"] + len(text) + (2 if state["parts"] else 0)
        if state["parts"] and proposed > TARGET_CHARS:
            value = flush_kind(kind)
            if value:
                yield value
            state = buffers[kind]
        if state["start_page"] is None:
            state["start_page"] = page_start
        state["end_page"] = page_end
        state["parts"].append(plain_text(text))
        state["chars"] += len(text) + (2 if len(state["parts"]) > 1 else 0)
    for value in flush_all():
        yield value


def metadata_text(row):
    try:
        metadata = json.loads(row[5] or "{}")
    except Exception:
        metadata = {}
    try:
        creators = json.loads(row[6] or "[]")
    except Exception:
        creators = []
    authors = ", ".join(
        " ".join(value for value in (creator.get("firstName", ""), creator.get("lastName", "")) if value)
        for creator in creators
    )
    fields = [
        row[1], authors, row[2], metadata.get("abstractNote"), metadata.get("publicationTitle"),
        metadata.get("bookTitle"), metadata.get("conferenceName"), metadata.get("shortTitle"),
        metadata.get("rights"), metadata.get("place"), metadata.get("publisher"), metadata.get("series"),
    ]
    return plain_text(". ".join(str(value) for value in fields if value))


def library_chunks(library, clean):
    for row in library.execute("SELECT id,title,date_created,doi,isbn,metadata_json,creators_json FROM items ORDER BY id"):
        value = chunk(
            f"metadata:{row[0]}", "metadata", "Title & metadata", row[0], None, None,
            row[1], metadata_text(row),
        )
        if value:
            yield value

    yield from document_section_chunks(library, clean)

    for row in library.execute("""
        SELECT n.id,a.item_id,n.attachment_id,n.page_label,n.position_json,i.title,n.text,n.comment
        FROM annotations n JOIN attachments a ON a.id=n.attachment_id JOIN items i ON i.id=a.item_id
        WHERE length(trim(coalesce(n.text,'')||' '||coalesce(n.comment,'')))>0 ORDER BY n.id
    """):
        value = chunk(
            f"annotation:{row[0]}", "annotation", "My annotation", row[1], row[2],
            page_from_annotation(row[4], row[3]), row[5],
            "\n\n".join(value for value in (row[6], row[7]) if value), annotation_id=row[0],
        )
        if value:
            yield value

    for row in library.execute("""
        SELECT e.id,e.item_id,e.attachment_id,e.page_number,i.title,e.text,e.comment
        FROM embedded_pdf_annotations e JOIN items i ON i.id=e.item_id
        WHERE length(trim(coalesce(e.text,'')||' '||coalesce(e.comment,'')))>0 ORDER BY e.id
    """):
        value = chunk(
            f"embedded:{row[0]}", "embedded_annotation", "Embedded PDF annotation", row[1], row[2],
            row[3], row[4], "\n\n".join(value for value in (row[5], row[6]) if value),
            embedded_annotation_id=row[0],
        )
        if value:
            yield value

    for row in library.execute("""
        SELECT n.id,n.item_id,n.attachment_id,coalesce(i.title,n.title),n.note_html
        FROM notes n LEFT JOIN items i ON i.id=n.item_id
        WHERE length(trim(n.note_html))>0 ORDER BY n.id
    """):
        value = chunk(
            f"note:{row[0]}", "note", "Zotero Note", row[1], row[2], None,
            row[3], plain_text(row[4]), note_id=row[0],
        )
        if value:
            yield value


def obsidian_chunks(obsidian):
    """Chunk Markdown sections by complete paragraphs, preserving the vault hierarchy."""
    rows=obsidian.execute("""SELECT s.section_id,s.note_id,s.heading_path,s.plain_text,
      n.title,n.relative_path,n.tags_json,n.aliases_json,n.authors_json
      FROM obsidian_sections s JOIN obsidian_notes n ON n.note_id=s.note_id
      ORDER BY n.relative_path COLLATE NOCASE,s.ordinal""")
    for section_id,note_id,heading,text,title,path,tags_json,aliases_json,authors_json in rows:
        try: tags=json.loads(tags_json or "[]")
        except Exception: tags=[]
        try: aliases=json.loads(aliases_json or "[]")
        except Exception: aliases=[]
        try: authors=json.loads(authors_json or "[]")
        except Exception: authors=[]
        prefix=". ".join(value for value in (
          "Tags: "+", ".join(tags) if tags else "",
          "Aliases: "+", ".join(aliases) if aliases else "",
          "Authors: "+", ".join(authors) if authors else "",
        ) if value)
        paragraphs=[part.strip() for part in re.split(r"\n\s*\n",text or "") if part.strip()]
        parts=[]; current=[]; chars=0
        for paragraph in paragraphs:
            if current and chars+len(paragraph)+2>TARGET_CHARS:
                parts.append("\n\n".join(current)); current=[]; chars=0
            current.append(paragraph); chars+=len(paragraph)+(2 if len(current)>1 else 0)
        if current: parts.append("\n\n".join(current))
        for part_index,body in enumerate(parts):
            value=chunk(f"obsidian:{section_id}:{part_index}","obsidian","Obsidian note",None,None,None,
              title,body,heading,obsidian_note_id=note_id,obsidian_section_id=section_id,vault_path=path)
            if value: yield value

def embed_batch(texts):
    request_body = {"model": EMBED_MODEL, "input": texts, "keep_alive": -1}
    if EMBED_DIMENSIONS:
        request_body["dimensions"] = EMBED_DIMENSIONS
    payload = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_BASE_URL + "/api/embed", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.load(response)
    vectors = data.get("embeddings") or []
    if len(vectors) != len(texts):
        raise RuntimeError(f"Ollama returned {len(vectors)} embeddings for {len(texts)} texts")
    normalized = []
    for vector in vectors:
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        normalized.append(vector / norm if norm else vector)
    return normalized


def ollama_model_identity() -> str:
    """Identify the concrete model behind a mutable Ollama tag."""
    try:
        with urllib.request.urlopen(OLLAMA_BASE_URL + "/api/tags", timeout=10) as response:
            records = json.load(response).get("models", [])
        record = next(
            value for value in records
            if (value.get("name") or value.get("model")) == EMBED_MODEL
        )
        if record.get("digest"):
            return str(record["digest"])
        stable = {key: record.get(key) for key in ("name", "model", "size", "modified_at")}
        return hashlib.sha256(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest()
    except Exception as exc:
        raise RuntimeError(f"Could not identify installed Ollama model {EMBED_MODEL}: {exc}") from exc


def embedding_runtime() -> dict:
    role = os.getenv("MARGINALIA_EMBED_SERVICE_ROLE", "embedding-service")
    try:
        with urllib.request.urlopen(OLLAMA_BASE_URL + "/api/ps", timeout=5) as response:
            payload = json.load(response)
        record = next(
            (entry for entry in payload.get("models", []) if (entry.get("name") or entry.get("model")) == EMBED_MODEL),
            None,
        )
        if not record:
            return {"device": "UNKNOWN", "detail": "Embedding device could not be determined", "service": role}
        size = max(0, int(record.get("size") or 0))
        size_vram = max(0, int(record.get("size_vram") or 0))
        device = "GPU" if size_vram > 0 else "CPU"
        partial = bool(size_vram and size and size_vram < size * .9)
        detail = f"Embedding device: {device}"
        if partial:
            detail += " (partial GPU offload)"
        detail += f" · {'isolated bulk-build service' if role == 'gpu-preferred' else 'isolated CPU fallback'}"
        return {
            "device": device, "detail": detail, "service": role,
            "size_bytes": size, "size_vram_bytes": size_vram, "base_url": OLLAMA_BASE_URL,
        }
    except Exception as exc:
        return {
            "device": "UNKNOWN", "detail": f"Embedding device detection failed: {exc}",
            "service": role, "base_url": OLLAMA_BASE_URL,
        }


def write_progress(**values):
    current = {"model": EMBED_MODEL, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **values}
    temporary = PROGRESS.with_name(f"{PROGRESS.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    # On Windows a reader can briefly prevent os.replace(). Progress is
    # telemetry, so a sharing violation must never abort the index build.
    for attempt in range(20):
        try:
            temporary.replace(PROGRESS)
            return
        except PermissionError:
            if attempt < 19:
                time.sleep(0.1)
    temporary.unlink(missing_ok=True)


def ensure_schema(index):
    index.executescript(SCHEMA)
    columns = {row[1] for row in index.execute("PRAGMA table_info(semantic_chunks)")}
    if "section_title" not in columns:
        index.execute("ALTER TABLE semantic_chunks ADD COLUMN section_title TEXT")
    for name in ("obsidian_note_id","obsidian_section_id","vault_path"):
        if name not in columns:
            index.execute(f"ALTER TABLE semantic_chunks ADD COLUMN {name} TEXT")
    index.commit()


def build_streaming(args, library, clean, index, batch_size):
    """Build without retaining the complete corpus or changed-chunk queue in RAM."""
    obsidian = None
    if OBSIDIAN_DB.is_file():
        obsidian = sqlite3.connect(f"file:{OBSIDIAN_DB.as_posix()}?mode=ro", uri=True)

    def chunks():
        yield from library_chunks(library, clean)
        if obsidian is not None:
            yield from obsidian_chunks(obsidian)

    index.execute("CREATE TEMP TABLE IF NOT EXISTS build_seen(chunk_key TEXT PRIMARY KEY)")
    index.execute("DELETE FROM build_seen")
    counts = {}
    total_chunks = pending_count = 0
    for value in chunks():
        total_chunks += 1
        counts[value["kind"]] = counts.get(value["kind"], 0) + 1
        index.execute("INSERT OR IGNORE INTO build_seen VALUES(?)", (value["chunk_key"],))
        existing = index.execute(
            "SELECT content_hash FROM semantic_chunks WHERE model=? AND chunk_key=?",
            (EMBED_MODEL, value["chunk_key"]),
        ).fetchone()
        pending_count += int(not existing or existing[0] != value["content_hash"])
    index.commit()
    existing_count = index.execute(
        "SELECT count(*) FROM semantic_chunks WHERE model=?", (EMBED_MODEL,)
    ).fetchone()[0]
    pending_total = min(pending_count, args.limit) if args.limit else pending_count
    print(json.dumps({"chunks": total_chunks, "pending": pending_total, "kinds": counts}, indent=2), flush=True)
    if args.plan_only:
        if obsidian is not None: obsidian.close()
        library.close(); clean.close(); index.close()
        return
    if not pending_total:
        live_progress(f"AI index is already current: {total_chunks:,} passages need no changes", 1, 1)

    sql = """INSERT INTO semantic_chunks(
      chunk_key,content_hash,model,dimensions,kind,label,item_id,attachment_id,page,section_title,
      annotation_id,embedded_annotation_id,note_id,obsidian_note_id,obsidian_section_id,vault_path,title,text,embedding,updated_at)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
      ON CONFLICT(chunk_key) DO UPDATE SET content_hash=excluded.content_hash,model=excluded.model,
      dimensions=excluded.dimensions,kind=excluded.kind,label=excluded.label,item_id=excluded.item_id,
      attachment_id=excluded.attachment_id,page=excluded.page,section_title=excluded.section_title,
      annotation_id=excluded.annotation_id,embedded_annotation_id=excluded.embedded_annotation_id,
      note_id=excluded.note_id,obsidian_note_id=excluded.obsidian_note_id,
      obsidian_section_id=excluded.obsidian_section_id,vault_path=excluded.vault_path,
      title=excluded.title,text=excluded.text,embedding=excluded.embedding,updated_at=CURRENT_TIMESTAMP"""
    started = time.time()
    completed = 0
    runtime = None
    total_batches = (pending_total + batch_size - 1) // batch_size
    write_progress(status="running", total_chunks=total_chunks, pending=pending_total,
                   completed=0, indexed=existing_count)

    def changed_chunks():
        emitted = 0
        for value in chunks():
            existing = index.execute(
                "SELECT content_hash FROM semantic_chunks WHERE model=? AND chunk_key=?",
                (EMBED_MODEL, value["chunk_key"]),
            ).fetchone()
            if existing and existing[0] == value["content_hash"]:
                continue
            if args.limit and emitted >= args.limit:
                break
            emitted += 1
            yield value

    iterator = iter(changed_chunks())
    while completed < pending_total:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(iterator))
            except StopIteration:
                break
        if not batch:
            break
        batch_number = completed // batch_size + 1
        first_title = batch[0].get("title") or batch[0].get("label") or batch[0]["kind"]
        live_progress(
            f"Starting batch {batch_number:,} of {total_batches:,}; passages {completed + 1:,}-{completed + len(batch):,} of {pending_total:,}; {first_title}",
            completed, pending_total,
        )
        vectors = embed_batch([value["embedded_text"] for value in batch])
        if runtime is None:
            runtime = embedding_runtime()
            print("RUNTIME\t" + json.dumps(runtime, ensure_ascii=False), flush=True)
        for value, vector in zip(batch, vectors):
            index.execute(sql, (
                value["chunk_key"], value["content_hash"], EMBED_MODEL, len(vector), value["kind"],
                value["label"], value["item_id"], value["attachment_id"], value["page"],
                value["section_title"], value["annotation_id"], value["embedded_annotation_id"],
                value["note_id"], value["obsidian_note_id"], value["obsidian_section_id"], value["vault_path"],
                value["title"], value["text"], vector.tobytes(),
            ))
        index.commit()
        completed += len(batch)
        elapsed = max(.001, time.time() - started)
        rate = completed / elapsed
        eta = int((pending_total - completed) / rate) if rate else 0
        live_progress(
            f"Completed batch {batch_number:,} of {total_batches:,}; {completed:,} of {pending_total:,} passages; {rate:.1f}/s; ETA {eta // 60}m {eta % 60}s",
            completed, pending_total,
        )
        write_progress(status="running", total_chunks=total_chunks, pending=pending_total,
                       completed=completed, indexed=existing_count + completed,
                       rate=round(rate, 2), eta_seconds=eta,
                       device=(runtime or {}).get("device", ""),
                       runtime_detail=(runtime or {}).get("detail", ""))

    if not args.limit:
        index.execute("""DELETE FROM semantic_chunks
          WHERE model=? AND NOT EXISTS(
            SELECT 1 FROM build_seen WHERE build_seen.chunk_key=semantic_chunks.chunk_key
          )""", (EMBED_MODEL,))
        clean_completed = ""
        try:
            row = clean.execute("SELECT value FROM clean_text_info WHERE key='completed_at'").fetchone()
            clean_completed = row[0] if row else ""
        except sqlite3.Error:
            pass
        info = {
            "model": EMBED_MODEL, "model_identity": MODEL_IDENTITY,
            "dimensions": str(index.execute("SELECT coalesce(max(dimensions),0) FROM semantic_chunks").fetchone()[0]),
            "requested_dimensions": str(EMBED_DIMENSIONS or "model default"),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clean_text_completed_at": clean_completed,
            "chunking": "clean paragraphs grouped by section; pages are locators only",
            "target_chars": str(TARGET_CHARS),
            "build_device": str((runtime or {}).get("device") or "not-needed"),
            "build_service": str((runtime or {}).get("service") or os.getenv("MARGINALIA_EMBED_SERVICE_ROLE", "")),
        }
        index.executemany("INSERT OR REPLACE INTO semantic_index_info VALUES(?,?)", info.items())
        index.commit()
    final_count = index.execute("SELECT count(*) FROM semantic_chunks WHERE model=?", (EMBED_MODEL,)).fetchone()[0]
    if obsidian is not None: obsidian.close()
    library.close(); clean.close(); index.close()
    status = "complete" if not args.limit else "partial"
    write_progress(status=status, total_chunks=total_chunks, pending=pending_total, completed=completed,
                   indexed=final_count, elapsed_seconds=round(time.time() - started, 1), kinds=counts,
                   device=(runtime or {}).get("device", ""), runtime_detail=(runtime or {}).get("detail", ""))
    print(json.dumps({"status": status, "indexed": final_count, "changed": completed,
                      "elapsed_seconds": round(time.time() - started, 1), "kinds": counts}, indent=2), flush=True)


def main():
    global EMBED_MODEL, EMBED_DIMENSIONS, MODEL_IDENTITY, INDEX_DB, PROGRESS
    parser = argparse.ArgumentParser(description="Build local semantic retrieval from clean sections and paragraphs.")
    parser.add_argument("--batch-size", type=int, default=0, help="Defaults to 64 for Qwen 0.6B and a conservative 8 for other models.")
    parser.add_argument("--limit", type=int, default=0, help="Index only this many changed chunks.")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Count chunks without calling Ollama.")
    parser.add_argument("--model", default=EMBED_MODEL, help="Ollama embedding model to use.")
    parser.add_argument("--dimensions", type=int, default=0, help="Optional output vector dimensions supported by the model.")
    parser.add_argument("--output", type=Path, help="Separate SQLite index path. Defaults are model-specific.")
    parser.add_argument("--progress", type=Path, help="Model-specific progress JSON path.")
    args = parser.parse_args()

    EMBED_MODEL = args.model.strip()
    EMBED_DIMENSIONS = max(0, args.dimensions)
    register_embedding_model(EMBED_MODEL)
    batch_size=max(1,args.batch_size or (64 if EMBED_MODEL == "qwen3-embedding:0.6b" else 8))
    INDEX_DB = (args.output or semantic_model_index_path(EMBED_MODEL)).resolve()
    PROGRESS = (args.progress or semantic_model_progress_path(EMBED_MODEL)).resolve()
    print(json.dumps({"model":EMBED_MODEL,"dimensions":EMBED_DIMENSIONS or "model default","batch_size":batch_size,"index":str(INDEX_DB),"progress":str(PROGRESS)},indent=2),flush=True)

    library = sqlite3.connect(f"file:{LIBRARY_DB.as_posix()}?mode=ro", uri=True)
    library.row_factory = sqlite3.Row
    clean = sqlite3.connect(f"file:{CLEAN_DB.as_posix()}?mode=ro", uri=True)
    clean.row_factory = sqlite3.Row
    index = sqlite3.connect(INDEX_DB)
    index.execute("PRAGMA secure_delete=ON")
    ensure_schema(index)
    if args.plan_only:
        row=index.execute("SELECT value FROM semantic_index_info WHERE key='model_identity'").fetchone()
        MODEL_IDENTITY=row[0] if row else "unknown-model-version"
    else:
        MODEL_IDENTITY=ollama_model_identity()
    if args.reset:
        index.execute("DELETE FROM semantic_chunks")
        index.commit()
    build_streaming(args, library, clean, index, batch_size)
    return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        write_progress(status="interrupted")
        raise
    except Exception as exc:
        write_progress(status="error", error=str(exc))
        raise
