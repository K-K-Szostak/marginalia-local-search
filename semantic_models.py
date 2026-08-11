from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "embedding_models.json"
REGISTRY_LOCK_PATH = ROOT / ".embedding_models.lock"
ACTIVE_MODEL_PATH = ROOT / "active_embedding_model.json"
DEFAULT_MODEL = "qwen3-embedding:0.6b"
RECOMMENDED_MODELS = (DEFAULT_MODEL, "qwen3-embedding:8b")
LEGACY_INDEXES = {
    DEFAULT_MODEL: ROOT / "semantic_index.sqlite",
    "qwen3-embedding:8b": ROOT / "semantic_index_8b.sqlite",
}
MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?")


def valid_model_name(model: str) -> bool:
    return bool(MODEL_PATTERN.fullmatch(str(model or "").strip()))


def model_token(model: str) -> str:
    model = str(model).strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-").lower()[:48] or "model"
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


def index_path(model: str) -> Path:
    model = str(model).strip()
    return LEGACY_INDEXES.get(model, ROOT / f"semantic_index_{model_token(model)}.sqlite")


def progress_path(model: str) -> Path:
    model = str(model).strip()
    if model == DEFAULT_MODEL:
        return ROOT / "semantic_index_progress.json"
    if model == "qwen3-embedding:8b":
        return ROOT / "semantic_index_8b_progress.json"
    return ROOT / f"semantic_index_{model_token(model)}_progress.json"


def setup_path(model: str) -> Path:
    model = str(model).strip()
    if model == "qwen3-embedding:8b":
        return ROOT / "semantic_index_8b_setup.json"
    return ROOT / f"semantic_index_{model_token(model)}_setup.json"


def _read_registry() -> list[str]:
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    values = payload.get("models", []) if isinstance(payload, dict) else payload
    return [str(value).strip() for value in values if valid_model_name(str(value))]


@contextmanager
def _registry_lock(timeout: float = 10):
    """Serialize registry updates made by the web process and index workers."""
    deadline = time.monotonic() + timeout
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(REGISTRY_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode("ascii"))
        except FileExistsError:
            try:
                if time.time() - REGISTRY_LOCK_PATH.stat().st_mtime > 120:
                    REGISTRY_LOCK_PATH.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Embedding model registry is busy")
            time.sleep(.05)
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        REGISTRY_LOCK_PATH.unlink(missing_ok=True)


def register_model(model: str) -> None:
    model = str(model or "").strip()
    if not valid_model_name(model):
        raise ValueError("Invalid Ollama model name")
    with _registry_lock():
        models = list(dict.fromkeys([*_read_registry(), model]))
        temporary = REGISTRY_PATH.with_name(f"{REGISTRY_PATH.name}.{os.getpid()}.new")
        temporary.write_text(json.dumps({"models": models}, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, REGISTRY_PATH)


def unregister_model(model: str, delete_files: bool = False) -> None:
    model = str(model or "").strip()
    if model in RECOMMENDED_MODELS:
        raise ValueError("Recommended models cannot be removed from the model list")
    with _registry_lock():
        models = [value for value in _read_registry() if value != model]
        temporary = REGISTRY_PATH.with_name(f"{REGISTRY_PATH.name}.{os.getpid()}.new")
        temporary.write_text(json.dumps({"models": models}, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, REGISTRY_PATH)
    if delete_files:
        for path in (index_path(model), progress_path(model), setup_path(model)):
            path.unlink(missing_ok=True)


def active_model() -> str:
    try:
        value = json.loads(ACTIVE_MODEL_PATH.read_text(encoding="utf-8")).get("model", "")
    except (OSError, ValueError, TypeError, AttributeError):
        value = ""
    return value if valid_model_name(value) else DEFAULT_MODEL


def set_active_model(model: str) -> None:
    model = str(model or "").strip()
    if not valid_model_name(model):
        raise ValueError("Invalid Ollama model name")
    register_model(model)
    temporary = ACTIVE_MODEL_PATH.with_name(f"{ACTIVE_MODEL_PATH.name}.{os.getpid()}.new")
    temporary.write_text(json.dumps({"model": model}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, ACTIVE_MODEL_PATH)


def indexed_models() -> list[str]:
    """Discover every completed or partial semantic index, including legacy files."""
    models = list(dict.fromkeys([*RECOMMENDED_MODELS, *_read_registry()]))
    candidates = set(LEGACY_INDEXES.values()) | set(ROOT.glob("semantic_index_*.sqlite"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            database = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5)
            try:
                info = dict(database.execute("SELECT key,value FROM semantic_index_info"))
            finally:
                database.close()
            model = str(info.get("model") or "").strip()
            if valid_model_name(model):
                models.append(model)
        except sqlite3.Error:
            continue
    return list(dict.fromkeys(models))
