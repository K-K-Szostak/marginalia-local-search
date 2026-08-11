from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_VERSION = "pdf-text-v3-auto-language"


def open_cache(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS cached_documents(
      content_hash TEXT NOT NULL,
      profile TEXT NOT NULL,
      page_count INTEGER NOT NULL,
      extracted_chars INTEGER NOT NULL,
      ocr_pages INTEGER NOT NULL,
      ocr_characters INTEGER NOT NULL,
      PRIMARY KEY(content_hash,profile)
    );
    CREATE TABLE IF NOT EXISTS cached_pages(
      content_hash TEXT NOT NULL,
      profile TEXT NOT NULL,
      page_number INTEGER NOT NULL,
      text TEXT NOT NULL,
      PRIMARY KEY(content_hash,profile,page_number)
    );
    CREATE TABLE IF NOT EXISTS cached_embedded_annotations(
      content_hash TEXT NOT NULL,
      profile TEXT NOT NULL,
      ordinal INTEGER NOT NULL,
      page_number INTEGER NOT NULL,
      annotation_type TEXT NOT NULL,
      text TEXT,
      comment TEXT,
      author_name TEXT,
      color_json TEXT,
      position_json TEXT,
      PRIMARY KEY(content_hash,profile,ordinal)
    );
    """)
    return db


def profile(ocr_enabled: bool, languages: str, minimum_chars: int, dpi: int) -> str:
    return f"{SCHEMA_VERSION}|ocr={int(ocr_enabled)}|lang={languages}|min={minimum_chars}|dpi={dpi}"


def load_document(db: sqlite3.Connection, content_hash: str, extraction_profile: str):
    document = db.execute(
        "SELECT * FROM cached_documents WHERE content_hash=? AND profile=?",
        (content_hash, extraction_profile),
    ).fetchone()
    if not document:
        return None
    pages = db.execute(
        "SELECT page_number,text FROM cached_pages WHERE content_hash=? AND profile=? ORDER BY page_number",
        (content_hash, extraction_profile),
    ).fetchall()
    annotations = db.execute(
        "SELECT * FROM cached_embedded_annotations WHERE content_hash=? AND profile=? ORDER BY ordinal",
        (content_hash, extraction_profile),
    ).fetchall()
    if len(pages) != document["page_count"]:
        return None
    return document, pages, annotations


def save_document(
    db: sqlite3.Connection, content_hash: str, extraction_profile: str,
    pages: list[str], annotations: list[dict], ocr_pages: int, ocr_characters: int,
) -> None:
    db.execute("DELETE FROM cached_pages WHERE content_hash=? AND profile=?", (content_hash, extraction_profile))
    db.execute("DELETE FROM cached_embedded_annotations WHERE content_hash=? AND profile=?", (content_hash, extraction_profile))
    db.executemany(
        "INSERT INTO cached_pages VALUES(?,?,?,?)",
        ((content_hash, extraction_profile, number, text) for number, text in enumerate(pages, 1)),
    )
    db.executemany(
        "INSERT INTO cached_embedded_annotations VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            (content_hash, extraction_profile, ordinal, value["page_number"], value["annotation_type"],
             value["text"], value["comment"], value["author_name"], value["color_json"], value["position_json"])
            for ordinal, value in enumerate(annotations, 1)
        ),
    )
    db.execute(
        "INSERT OR REPLACE INTO cached_documents VALUES(?,?,?,?,?,?)",
        (content_hash, extraction_profile, len(pages), sum(len(text) for text in pages), ocr_pages, ocr_characters),
    )
    db.commit()
