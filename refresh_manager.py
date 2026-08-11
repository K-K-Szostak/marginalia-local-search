from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from local_network import loopback_url

from source_manager import (
    ROOT, cleanup_snapshot_generations, discard_snapshot_generation, load_config,
    publish_source_pointers, reserve_snapshot_generation, snapshot_sources,
)
from semantic_models import DEFAULT_MODEL as DEFAULT_EMBED_MODEL, index_path as semantic_model_index_path, indexed_models as registered_embedding_models, progress_path as semantic_model_progress_path, register_model as register_embedding_model, set_active_model as set_active_embedding_model, valid_model_name


LOCK = threading.Lock()
BUILD_OLLAMA_LOCK = threading.Lock()
BUILD_OLLAMA_PROCESS = None
AI_SETUP_EVENT = threading.Event()
STATE_PATH = ROOT / "refresh_state.json"
STATE = {
    "running": False, "phase": "idle", "message": "Ready", "current": 0,
    "total": 0, "started_at": None, "finished_at": None, "error": None,
    "warnings": [], "snapshot": None, "activity": "", "detail_current": 0,
    "detail_total": 0, "detail_eta": "", "activity_log": [], "stage_labels": [], "stage_started_at": None,
    "source_warning": "", "source_blocked": False, "resume_required": False, "resume_phase": "",
    "semantic_device": "", "semantic_runtime_detail": "", "semantic_service": "", "semantic_base_url": "",
    "ai_setup_required": False, "ai_setup_skipped": False, "ai_setup_model": "",
    "semantic_model": "", "semantic_queue": [], "semantic_completed": [],
    "semantic_model_current": 0, "semantic_model_total": 0,
    "semantic_generation_root": "",
}


def _persist_state_unlocked() -> None:
    temporary = STATE_PATH.with_suffix(".json.new")
    try:
        temporary.write_text(json.dumps(STATE, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, STATE_PATH)
    except OSError:
        temporary.unlink(missing_ok=True)


def _restore_state() -> None:
    try:
        persisted = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return
    if not isinstance(persisted, dict):
        return
    STATE.update({key: persisted[key] for key in STATE if key in persisted})
    if STATE.get("running"):
        interrupted_phase = str(STATE.get("phase") or "")
        interrupted_resume = (
            str(STATE.get("resume_phase") or "")
            if STATE.get("semantic_queue") else interrupted_phase
        )
        STATE.update(
            running=False, phase="interrupted", resume_required=True,
            resume_phase=interrupted_resume,
            message="The previous refresh was interrupted and will resume automatically",
            error=None, finished_at=time.time(), source_warning="", source_blocked=False,
        )
        history = list(STATE.get("activity_log") or [])
        history.append({"message": "Previous process stopped before refresh completed; automatic resume queued.",
                        "at": time.time()})
        STATE["activity_log"] = history[-14:]
        STATE["activity"] = history[-1]["message"]
        _persist_state_unlocked()


_restore_state()

STAGES = [
    ("library", "Building the unified Zotero library", "build_unified_library.py", []),
    ("metadata", "Repairing incomplete metadata", "enrich_metadata.py", []),
    ("search", "Building keyword search indexes", "prepare_app_search.py", []),
    ("documents", "Extracting document text and OCR", "extract_pdf_corpus.py", []),
    ("missing", "Recovering Zotero full-text caches", "extract_missing_documents.py", []),
    ("clean", "Structuring clean document text", "clean_extracted_text.py", ["--reset"]),
    ("obsidian", "Importing Obsidian notes", "import_obsidian.py", []),
    ("semantic", "Updating semantic search", "build_semantic_index.py", []),
]

SEMANTIC_MODELS = {model: semantic_model_index_path(model) for model in registered_embedding_models()}
ACTIVE_DATABASES = {
    "library": ROOT / "unified_library.sqlite",
    "clean": ROOT / "clean_text.sqlite",
    "obsidian": ROOT / "obsidian_notes.sqlite",
}
WORK_DATABASES = {name: path.with_name(path.stem + ".next.sqlite") for name, path in ACTIVE_DATABASES.items()}
PUBLISH_JOURNAL = ROOT / "generation_publish.json"
GENERATION_MANIFEST = ROOT / "library_generation.json"


def invalidate_semantic_indexes(paths=None) -> list[str]:
    """Hide completed semantic indexes while retaining their vectors for reuse."""
    invalidated = []
    for path in paths or SEMANTIC_MODELS.values():
        path = Path(path)
        if not path.is_file():
            continue
        try:
            database = sqlite3.connect(path, timeout=10)
            table = database.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_index_info'"
            ).fetchone()
            if table:
                database.execute(
                    "DELETE FROM semantic_index_info WHERE key IN ('completed_at','clean_text_completed_at')"
                )
                database.commit()
                invalidated.append(path.name)
            database.close()
        except sqlite3.Error:
            # A later semantic rebuild will still validate its source stamp before
            # the server exposes the index.
            continue
    return invalidated


def completed_semantic_models() -> list[str]:
    """Return models whose indexes were complete before a library refresh."""
    completed=[]
    models = list(dict.fromkeys([*registered_embedding_models(), *SEMANTIC_MODELS]))
    for model in models:
        path = Path(SEMANTIC_MODELS.get(model, semantic_model_index_path(model)))
        if not path.is_file(): continue
        database = None
        try:
            database=sqlite3.connect(f"file:{path.as_posix()}?mode=ro",uri=True,timeout=10)
            info=dict(database.execute("SELECT key,value FROM semantic_index_info"))
            if info.get("model")==model and info.get("completed_at"):
                completed.append(model)
        except sqlite3.Error:
            continue
        finally:
            if database is not None:
                database.close()
    return completed


def _database_tables(path: Path) -> set[str]:
    database = sqlite3.connect(path, timeout=30)
    try:
        database.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        integrity = database.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"{path.name} failed integrity_check: {integrity}")
        return {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    finally:
        database.close()


def _database_value(path: Path, query: str) -> str:
    database = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=30)
    try:
        row = database.execute(query).fetchone()
        return str(row[0] or "") if row else ""
    finally:
        database.close()


def _remove_sqlite_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)


def validate_work_generation(config: dict) -> list[tuple[Path, Path]]:
    """Validate every database before any active file is replaced."""
    required = {
        "library": {"items", "attachments", "document_pages", "document_search", "item_search"},
        "clean": {"clean_document_blocks", "clean_document_search", "clean_text_info"},
    }
    entries = []
    for name in ("library", "clean"):
        work = WORK_DATABASES[name]
        if not work.is_file():
            raise RuntimeError(f"Working database is missing: {work.name}")
        missing = required[name] - _database_tables(work)
        if missing:
            raise RuntimeError(f"{work.name} is incomplete; missing: {', '.join(sorted(missing))}")
        if name == "clean" and not _database_value(
            work, "SELECT value FROM clean_text_info WHERE key='completed_at'"
        ):
            raise RuntimeError(f"{work.name} has no completion marker")
        entries.append((work, ACTIVE_DATABASES[name]))
    if config.get("obsidian_path"):
        work = WORK_DATABASES["obsidian"]
        if not work.is_file():
            raise RuntimeError(f"Working database is missing: {work.name}")
        missing = {"obsidian_notes", "obsidian_sections", "obsidian_search"} - _database_tables(work)
        if missing:
            raise RuntimeError(f"{work.name} is incomplete; missing: {', '.join(sorted(missing))}")
        if not _database_value(work, "SELECT value FROM obsidian_import_info WHERE key='imported_at'"):
            raise RuntimeError(f"{work.name} has no completion marker")
        entries.append((work, ACTIVE_DATABASES["obsidian"]))
    return entries


def recover_interrupted_publication() -> bool:
    """Roll back an incomplete multi-file publication, or finish its cleanup."""
    if not PUBLISH_JOURNAL.is_file():
        return False
    try:
        payload = json.loads(PUBLISH_JOURNAL.read_text(encoding="utf-8"))
        entries = payload.get("entries") or []
    except (OSError, ValueError, TypeError) as exc:
        # A corrupt journal cannot describe the exact midpoint. Known backups
        # are always the last complete files, so restore them conservatively
        # and quarantine the journal for diagnosis instead of blocking startup.
        try:
            for active in ACTIVE_DATABASES.values():
                backup = active.with_name(active.stem + ".previous.sqlite")
                if backup.exists():
                    active.unlink(missing_ok=True)
                    os.replace(backup, active)
            manifest_backup = GENERATION_MANIFEST.with_name(GENERATION_MANIFEST.stem + ".previous.json")
            if manifest_backup.exists():
                GENERATION_MANIFEST.unlink(missing_ok=True)
                os.replace(manifest_backup, GENERATION_MANIFEST)
            damaged = PUBLISH_JOURNAL.with_name(
                f"{PUBLISH_JOURNAL.stem}.damaged-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}{PUBLISH_JOURNAL.suffix}"
            )
            os.replace(PUBLISH_JOURNAL, damaged)
            return True
        except OSError as recovery_exc:
            raise RuntimeError(f"The database publication journal is damaged and its backups could not be restored: {recovery_exc}") from exc
    phase = payload.get("phase")
    incomplete = phase == "publishing" or (phase is None and any(Path(entry["work"]).exists() for entry in entries))
    if incomplete:
        for entry in reversed(entries):
            active, backup = Path(entry["active"]), Path(entry["backup"])
            if backup.exists():
                active.unlink(missing_ok=True)
                os.replace(backup, active)
            elif not entry.get("had_active"):
                active.unlink(missing_ok=True)
        manifest = Path(payload.get("manifest") or GENERATION_MANIFEST)
        manifest_backup = Path(payload.get("manifest_backup") or str(GENERATION_MANIFEST.with_name(GENERATION_MANIFEST.stem + ".previous.json")))
        if manifest_backup.exists():
            manifest.unlink(missing_ok=True)
            os.replace(manifest_backup, manifest)
        elif not payload.get("had_manifest", True):
            manifest.unlink(missing_ok=True)
    for entry in entries:
        Path(entry["backup"]).unlink(missing_ok=True)
    for key in ("manifest_backup", "manifest_new"):
        value = payload.get(key)
        if value:
            Path(value).unlink(missing_ok=True)
    PUBLISH_JOURNAL.unlink(missing_ok=True)
    return True


def publish_work_generation(config: dict, snapshot: dict | None = None) -> list[str]:
    """Publish a validated core generation with crash-recoverable rollback."""
    recover_interrupted_publication()
    pairs = validate_work_generation(config)
    entries = []
    for work, active in pairs:
        backup = active.with_name(active.stem + ".previous.sqlite")
        backup.unlink(missing_ok=True)
        entries.append({"work": str(work), "active": str(active), "backup": str(backup),
                        "had_active": active.is_file()})
    manifest_backup = GENERATION_MANIFEST.with_name(GENERATION_MANIFEST.stem + ".previous.json")
    manifest_backup.unlink(missing_ok=True)
    manifest_new = GENERATION_MANIFEST.with_suffix(".json.new")
    manifest_new.write_text(json.dumps({
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "zotero_path": str(config.get("zotero_path") or ""),
        "obsidian_path": str(config.get("obsidian_path") or ""),
        "linked_attachment_base_path": str(config.get("linked_attachment_base_path") or ""),
        "snapshot_root": str((snapshot or {}).get("snapshot_root") or ""),
        "databases": [Path(entry["active"]).name for entry in entries],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    journal = {
        "phase": "publishing", "entries": entries,
        "manifest": str(GENERATION_MANIFEST), "manifest_new": str(manifest_new),
        "manifest_backup": str(manifest_backup), "had_manifest": GENERATION_MANIFEST.is_file(),
    }
    temporary_journal = PUBLISH_JOURNAL.with_suffix(".json.new")
    temporary_journal.write_text(json.dumps(journal, indent=2), encoding="utf-8")
    os.replace(temporary_journal, PUBLISH_JOURNAL)
    try:
        if GENERATION_MANIFEST.is_file():
            os.replace(GENERATION_MANIFEST, manifest_backup)
        for entry in entries:
            active, backup = Path(entry["active"]), Path(entry["backup"])
            if active.exists():
                _database_tables(active)
                _remove_sqlite_sidecars(active)
                os.replace(active, backup)
        for entry in entries:
            os.replace(Path(entry["work"]), Path(entry["active"]))
            _remove_sqlite_sidecars(Path(entry["work"]))
        os.replace(manifest_new, GENERATION_MANIFEST)
        journal["phase"] = "committed"
        temporary_journal.write_text(json.dumps(journal, indent=2), encoding="utf-8")
        os.replace(temporary_journal, PUBLISH_JOURNAL)
    except Exception:
        recover_interrupted_publication()
        raise
    for entry in entries:
        Path(entry["backup"]).unlink(missing_ok=True)
    manifest_backup.unlink(missing_ok=True)
    PUBLISH_JOURNAL.unlink(missing_ok=True)
    if snapshot:
        try:
            publish_source_pointers(snapshot)
            cleanup_snapshot_generations(str(snapshot.get("snapshot_root") or ""))
        except OSError:
            # The active databases contain their generation paths, so pointer
            # cleanup is optional and can be retried by a later refresh.
            pass
    if not config.get("obsidian_path"):
        for suffix in ("", "-wal", "-shm"):
            Path(str(ACTIVE_DATABASES["obsidian"]) + suffix).unlink(missing_ok=True)
    return [Path(entry["active"]).name for entry in entries]


def active_generation_matches(config: dict) -> bool:
    if PUBLISH_JOURNAL.is_file():
        return False
    try:
        manifest = json.loads(GENERATION_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return all(
        str(manifest.get(key) or "") == str(config.get(key) or "")
        for key in ("zotero_path", "obsidian_path", "linked_attachment_base_path")
    )


def snapshot_generation_is_active(snapshot: dict | None) -> bool:
    if not snapshot or not snapshot.get("snapshot_root"):
        return False
    try:
        manifest = json.loads(GENERATION_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return Path(str(manifest.get("snapshot_root") or "")).resolve() == Path(snapshot["snapshot_root"]).resolve()


def semantic_service_candidates() -> list[tuple[str, str]]:
    preferred = loopback_url("EMBED_BUILD_OLLAMA_BASE_URL", "http://127.0.0.1:11436")
    fallback = loopback_url("EMBED_OLLAMA_BASE_URL", "http://127.0.0.1:11435")
    values = [("gpu-preferred", preferred), ("cpu-fallback", fallback)]
    return [(role, url) for index, (role, url) in enumerate(values) if url and url not in {item[1] for item in values[:index]}]


def _ollama_models(base_url: str, timeout: float = 3) -> set[str]:
    with urllib.request.urlopen(base_url + "/api/tags", timeout=timeout) as response:
        payload = json.load(response)
    return {entry.get("name") or entry.get("model") for entry in payload.get("models", []) if entry.get("name") or entry.get("model")}


def ollama_executable() -> Path | None:
    configured = os.getenv("OLLAMA_EXE")
    candidates = [Path(configured)] if configured else []
    discovered = shutil.which("ollama")
    if discovered:
        candidates.append(Path(discovered))
    candidates.append(Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Programs/Ollama/ollama.exe")
    for path in candidates:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def ensure_bulk_embedding_ollama(timeout: float = 20) -> bool:
    """Start a temporary GPU-capable Ollama isolated from both Gemma and query embeddings."""
    global BUILD_OLLAMA_PROCESS
    preferred = semantic_service_candidates()[0][1]
    try:
        _ollama_models(preferred, 1)
        return True
    except Exception:
        pass
    with BUILD_OLLAMA_LOCK:
        try:
            _ollama_models(preferred, 1)
            return True
        except Exception:
            pass
        executable = ollama_executable()
        if not executable:
            return False
        environment = os.environ.copy()
        environment.update({
            "OLLAMA_HOST": urllib.parse.urlparse(preferred).netloc,
            "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_NO_CLOUD": "1",
        })
        # The query service deliberately sets these to -1. The bulk service must
        # be free to use any GPU Ollama can detect.
        for key in ("CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES"):
            if environment.get(key) == "-1":
                environment.pop(key)
        BUILD_OLLAMA_PROCESS = subprocess.Popen(
            [str(executable), "serve"], env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if BUILD_OLLAMA_PROCESS.poll() is not None:
                break
            try:
                _ollama_models(preferred, 1)
                return True
            except Exception:
                time.sleep(.25)
        release_bulk_embedding_ollama()
        return False


def release_bulk_embedding_ollama() -> None:
    """Stop only the temporary service started by this Marginalia process."""
    global BUILD_OLLAMA_PROCESS
    process, BUILD_OLLAMA_PROCESS = BUILD_OLLAMA_PROCESS, None
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def choose_semantic_service(model: str | None = None) -> dict | None:
    model = str(model or os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")).strip()
    for role, base_url in semantic_service_candidates():
        try:
            names = _ollama_models(base_url)
        except Exception:
            continue
        if model in names:
            return {"role": role, "base_url": base_url, "model": model}
    return None


def semantic_preflight(model: str | None = None) -> str:
    """Prefer GPU-capable Ollama for bulk indexing and accept the isolated CPU service as fallback."""
    model = str(model or os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")).strip()
    if choose_semantic_service(model):
        return ""
    # The temporary GPU service shares Ollama's local model store with the main
    # answer service, but is not started until stage 8 actually begins.
    main_base_url = loopback_url("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        if model in _ollama_models(main_base_url):
            return ""
    except Exception:
        pass
    any_service = False
    for _, base_url in semantic_service_candidates():
        try:
            _ollama_models(base_url)
            any_service = True
        except Exception:
            pass
    if any_service:
        return f"The required embedding model {model} is not downloaded. Run: ollama pull {model}. BM25 keyword search is ready now."
    if ollama_executable():
        return "Ollama is installed but no embedding service is running. Start Ollama, then refresh again. BM25 keyword search is ready now."
    return "Ollama is not installed. Semantic search and generated answers were skipped; BM25 keyword search is ready now."


def skip_semantic_setup() -> bool:
    """Let a waiting refresh continue without optional semantic indexing."""
    with LOCK:
        if not STATE.get("running") or STATE.get("phase") != "ai_setup":
            return False
        STATE["ai_setup_required"] = False
        STATE["ai_setup_skipped"] = True
        _persist_state_unlocked()
    AI_SETUP_EVENT.set()
    return True


def select_semantic_setup_model(model: str) -> bool:
    """Continue a paused refresh with the embedding model chosen by the user."""
    model=str(model or "").strip()
    if not valid_model_name(model) or semantic_preflight(model):
        return False
    with LOCK:
        if not STATE.get("running") or STATE.get("phase") != "ai_setup" or STATE.get("ai_setup_skipped"):
            return False
        STATE["ai_setup_model"] = model
        STATE["ai_setup_required"] = False
        _persist_state_unlocked()
    set_active_embedding_model(model)
    AI_SETUP_EVENT.set()
    return True


def await_semantic_setup(model: str) -> str | None:
    """Pause before semantic indexing until its model exists or the user skips AI."""
    reason = semantic_preflight(model)
    if not reason:
        return model
    AI_SETUP_EVENT.clear()
    update(phase="ai_setup", message="Choose optional local AI setup",
           activity=reason, ai_setup_required=True, ai_setup_model=model)
    while True:
        if AI_SETUP_EVENT.wait(1):
            current=state()
            update(ai_setup_required=False)
            if current.get("ai_setup_skipped"):
                return None
            selected=str(current.get("ai_setup_model") or model).strip()
            if not semantic_preflight(selected):
                return selected
            AI_SETUP_EVENT.clear()
            update(ai_setup_required=True)
        if not semantic_preflight(model):
            update(ai_setup_required=False)
            return model


def configured_stages(config: dict) -> list[tuple[str, str, str, list[str]]]:
    """Return only the stages needed by the selected source folders."""
    zotero_enabled = bool(config.get("zotero_path"))
    obsidian_enabled = bool(config.get("obsidian_path"))
    stages = []
    for stage in STAGES:
        if stage[0] == "library" and not zotero_enabled:
            stages.append(("library", "Preparing Obsidian-only library", "create_empty_library.py", []))
        elif stage[0] != "obsidian" or obsidian_enabled:
            stages.append(stage)
    return stages


def state() -> dict:
    with LOCK:
        return dict(STATE)


def update(**values):
    with LOCK:
        STATE.update(values)
        _persist_state_unlocked()


def update_activity(message: str, current: int | None = None, total: int | None = None):
    """Update the current operation and retain a short, user-visible audit trail."""
    text = " ".join(str(message or "Working…").splitlines()).strip()[:300]
    with LOCK:
        history = list(STATE.get("activity_log") or [])
        entry = {"message": text, "at": time.time()}
        last_message = history[-1].get("message") if history and isinstance(history[-1], dict) else (str(history[-1]) if history else "")
        if not history or last_message != text:
            history.append(entry)
        else:
            history[-1] = entry
        STATE["activity"] = text
        STATE["activity_log"] = history[-14:]
        if current is not None:
            STATE["detail_current"] = max(0, int(current))
        if total is not None:
            STATE["detail_total"] = max(0, int(total))
        eta = re.search(r"\bETA\s+([0-9]+m\s+[0-9]+s)\b", text, flags=re.IGNORECASE)
        if eta:
            STATE["detail_eta"] = eta.group(1)
        _persist_state_unlocked()


def worker_command(script: str, arguments: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--run-stage", script, *arguments]
    return [sys.executable, str(ROOT / script), *arguments]


def run_stage(name: str, message: str, script: str, arguments: list[str], number: int, total: int,
              snapshot: dict | None = None):
    update(phase=name, message=message, activity=f"Starting: {message}", current=number, total=total,
           detail_current=0, detail_total=0, detail_eta="", stage_started_at=time.time())
    update_activity(f"Starting: {message}", 0, 0)
    log_path = ROOT / "refresh.log"
    environment = os.environ.copy()
    # Windows otherwise inherits a legacy console encoding (often cp1252),
    # which crashes progress output for titles containing characters such as ł.
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    if name != "semantic":
        environment["MARGINALIA_LIBRARY_DB"] = str(WORK_DATABASES["library"])
        environment["MARGINALIA_CLEAN_DB"] = str(WORK_DATABASES["clean"])
        environment["MARGINALIA_OBSIDIAN_DB"] = str(WORK_DATABASES["obsidian"])
        if snapshot and snapshot.get("zotero_root"):
            environment["ZOTERO_LIBRARY_DIR"] = str(snapshot["zotero_root"])
        if name == "obsidian" and snapshot and snapshot.get("obsidian_root"):
            arguments = [*arguments, "--vault", str(snapshot["obsidian_root"]), "--no-save-source"]
    else:
        ensure_bulk_embedding_ollama()
        requested_model = arguments[arguments.index("--model") + 1] if "--model" in arguments else os.getenv("EMBED_MODEL",DEFAULT_EMBED_MODEL).strip()
        register_embedding_model(requested_model)
        SEMANTIC_MODELS[requested_model]=semantic_model_index_path(requested_model)
        if "--model" not in arguments:
            arguments=[*arguments,"--model",requested_model]
        if "--output" not in arguments:
            arguments=[*arguments,"--output",str(semantic_model_index_path(requested_model))]
        if "--progress" not in arguments:
            arguments=[*arguments,"--progress",str(semantic_model_progress_path(requested_model))]
        service = choose_semantic_service(requested_model)
        if not service:
            release_bulk_embedding_ollama()
            raise RuntimeError("No Ollama service with the required embedding model is available")
        environment["OLLAMA_BASE_URL"] = service["base_url"]
        environment["MARGINALIA_EMBED_SERVICE_ROLE"] = service["role"]
        update(
            semantic_device="detecting", semantic_runtime_detail="Loading the embedding model…",
            semantic_service=service["role"], semantic_base_url=service["base_url"],
        )
    if not getattr(sys, "frozen", False):
        development_vendor = ROOT.parent / "vendor"
        if development_vendor.is_dir():
            environment["PYTHONPATH"] = str(development_vendor) + os.pathsep + environment.get("PYTHONPATH", "")
    bundled_tesseract = ROOT / "tools" / "tesseract" / "tesseract.exe"
    if bundled_tesseract.is_file():
        environment["TESSERACT_CMD"] = str(bundled_tesseract)
        environment["TESSDATA_PREFIX"] = str(bundled_tesseract.parent / "tessdata")
    try:
        process = subprocess.Popen(
            worker_command(script, arguments), cwd=ROOT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        if name == "semantic":
            release_bulk_embedding_ollama()
        raise
    with log_path.open("a", encoding="utf-8") as log:
        for line in process.stdout or []:
            line = line.rstrip()
            log.write(f"[{name}] {line}\n")
            if line.startswith("PROGRESS\t"):
                parts = line.split("\t", 3)
                activity = parts[1] if len(parts) > 1 else "Working…"
                detail_current = None
                detail_total = None
                if len(parts) > 2 and parts[2].isdigit():
                    detail_current = int(parts[2])
                if len(parts) > 3 and parts[3].isdigit():
                    detail_total = int(parts[3])
                update_activity(activity, detail_current, detail_total)
            elif name == "semantic" and line.startswith("RUNTIME\t"):
                try:
                    runtime = json.loads(line.split("\t", 1)[1])
                except (ValueError, TypeError):
                    runtime = {}
                device = str(runtime.get("device") or "UNKNOWN").upper()
                detail = str(runtime.get("detail") or f"Embedding on {device}")
                update(semantic_device=device, semantic_runtime_detail=detail)
                update_activity(detail)
                if device == "CPU":
                    warning = (
                        "Semantic indexing is running on CPU. It will complete normally, but the initial build "
                        "may take substantially longer because Ollama did not offload the embedding model to a GPU."
                    )
                    current_warnings = state()["warnings"]
                    if warning not in current_warnings:
                        update(warnings=[*current_warnings, warning])
    code = process.wait()
    if name == "semantic":
        release_bulk_embedding_ollama()
    if code:
        raise RuntimeError(f"{message} failed (exit code {code}). See refresh.log")


def _run_semantic_resume(config: dict, model: str | None = None) -> None:
    """Resume the persisted semantic-model queue after the core generation was published."""
    stages = configured_stages(config)
    semantic = next(stage for stage in stages if stage[0] == "semantic")
    number = stages.index(semantic) + 2
    total = len(stages) + 1
    current = state()
    queue = [value for value in current.get("semantic_queue", []) if valid_model_name(value)]
    requested = str(model or current.get("semantic_model") or current.get("ai_setup_model") or DEFAULT_EMBED_MODEL).strip()
    if requested and requested not in queue:
        queue.insert(0, requested)
    _run_semantic_queue(config, semantic, queue, number, total, None, require_setup=True)
    update_activity("Everything is ready for searching.", 0, 0)
    update(running=False, phase="complete", message="Library refresh complete",
           current=total, total=total, finished_at=time.time(), resume_required=False,
           resume_phase="", semantic_model="", semantic_queue=[], semantic_generation_root="")


def _run_semantic_queue(config: dict, stage, models: list[str], number: int, total: int,
                        snapshot: dict | None, require_setup: bool = False) -> None:
    """Build models in order while persisting enough state for an exact restart."""
    queue = list(dict.fromkeys(model for model in models if valid_model_name(model)))
    original_total = max(len(queue), int(state().get("semantic_model_total") or 0))
    completed = list(state().get("semantic_completed") or [])
    first = True
    while queue:
        requested = queue[0]
        update(phase="semantic", semantic_model=requested, semantic_queue=queue,
               resume_phase="semantic", resume_required=True,
               semantic_model_current=original_total-len(queue)+1, semantic_model_total=original_total)
        selected = requested
        setup_validated = first and require_setup
        if setup_validated:
            selected = await_semantic_setup(requested)
            if not selected:
                update(warnings=state()["warnings"] + ["Semantic search was skipped by the user; BM25 is ready."],
                       semantic_queue=[], semantic_model="", resume_required=False)
                return
            if selected != requested:
                queue[0] = selected
                requested = selected
                update(semantic_model=selected, semantic_queue=queue)
        first = False
        reason = "" if setup_validated else semantic_preflight(requested)
        if reason:
            update(warnings=state()["warnings"] + [f"{requested} index needs an update: {reason}"])
        else:
            current_stage = (stage[0], f"Updating semantic search ({requested})", stage[2], ["--model", requested])
            try:
                run_stage(*current_stage, number, total, snapshot=snapshot)
                completed.append(requested)
            except Exception as exc:
                update(warnings=state()["warnings"] + [f"{requested} index update was skipped: {exc}"])
        queue.pop(0)
        update(semantic_queue=queue, semantic_model=queue[0] if queue else "",
               semantic_completed=list(dict.fromkeys(completed)))
        try:
            run_stage(*semantic, number, total)
        except Exception as exc:
            update(warnings=state()["warnings"] + [f"Semantic search was skipped: {exc}"])
    update_activity("Everything is ready for searching.", 0, 0)
    update(running=False, phase="complete", message="Library refresh complete",
           current=total, total=total, finished_at=time.time(), resume_required=False,
           resume_phase="")


def rebuild_additional_semantic_models(stage, models: list[str], default_model: str,
                                       number: int, total: int, snapshot: dict) -> None:
    """Refresh every previously ready non-default embedding index."""
    for model in dict.fromkeys(models):
        if model == default_model:
            continue
        reason = semantic_preflight(model)
        if reason:
            update(warnings=state()["warnings"] + [f"{model} index needs an update: {reason}"])
            continue
        try:
            run_stage(stage[0], f"Updating semantic search ({model})", stage[2],
                      ["--model", model], number, total, snapshot=snapshot)
        except Exception as exc:
            update(warnings=state()["warnings"] + [f"{model} index update was skipped: {exc}"])


def _run(resume_semantic: bool = False, semantic_model: str | None = None):
    snapshot = None
    published = False
    try:
        config = load_config()
        recover_interrupted_publication()
        if resume_semantic and active_generation_matches(config):
            _run_semantic_resume(config, semantic_model)
            return
        stages = configured_stages(config)
        total = len(stages) + 1
        stage_labels = ["Copying source libraries safely", *[stage[1] for stage in stages]]
        update(running=True, phase="snapshot", message="Copying source libraries safely",
               activity="Preparing private copies…", current=1, total=total, detail_current=0,
               detail_total=0, detail_eta="", activity_log=[{"message": "Preparing private copies…", "at": time.time()}],
               stage_labels=stage_labels, stage_started_at=time.time(), started_at=time.time(),
               finished_at=None, error=None, warnings=[], source_warning="", source_blocked=False,
               semantic_device="", semantic_runtime_detail="", semantic_service="", semantic_base_url="",
               ai_setup_required=False, ai_setup_skipped=False, ai_setup_model="",
               semantic_model="", semantic_queue=[], semantic_completed=[],
               semantic_model_current=0, semantic_model_total=0, semantic_generation_root="")
        previously_completed_models = completed_semantic_models()
        def copy_progress(message, current, total, kind=None):
            if kind == "zotero_locked":
                update(source_warning=message, source_blocked=True)
            elif kind == "zotero_copying":
                update(source_warning="", source_blocked=False)
            update_activity(message, current, total)

        snapshot = {"snapshot_root": str(reserve_snapshot_generation())}
        snapshot = snapshot_sources(copy_progress, Path(snapshot["snapshot_root"]), config)
        update(snapshot=snapshot, source_warning="", source_blocked=False)
        for number, stage in enumerate(stages, 2):
            if stage[0] == "semantic":
                queue = list(dict.fromkeys([
                    os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL).strip(), *previously_completed_models,
                ]))
                # Persist the exact semantic hand-off before publishing. If the
                # process stops inside publication, restart resumes this queue
                # only when the matching snapshot became the active generation.
                update(phase="publishing", semantic_queue=queue,
                       semantic_model=queue[0] if queue else "", resume_required=bool(queue),
                       resume_phase="semantic", semantic_model_current=0,
                       semantic_model_total=len(queue),
                       semantic_generation_root=str(snapshot.get("snapshot_root") or ""))
                published_databases = publish_work_generation(config, snapshot)
                published = True
                update_activity(f"Published complete database generation: {', '.join(published_databases)}", 0, 0)
                invalidated = invalidate_semantic_indexes(
                    [semantic_model_index_path(model) for model in previously_completed_models]
                )
                update(semantic_queue=queue, semantic_model=queue[0] if queue else "",
                       resume_required=bool(queue), resume_phase="semantic",
                       semantic_model_current=0, semantic_model_total=len(queue))
                if invalidated:
                    update_activity("Published the new library; semantic indexes are queued for refresh.", 0, 0)
                _run_semantic_queue(config, stage, queue, number, len(stages) + 1, snapshot, require_setup=True)
                continue
            run_stage(*stage, number, len(stages) + 1, snapshot=snapshot)
        update_activity("Everything is ready for searching.", 0, 0)
        update(running=False, phase="complete", message="Library refresh complete",
               current=total, total=total, finished_at=time.time(), resume_required=False,
               resume_phase="", semantic_model="", semantic_queue=[], semantic_generation_root="")
    except Exception as exc:
        published = published or snapshot_generation_is_active(snapshot)
        if not published:
            discard_snapshot_generation(snapshot)
        update_activity(f"Refresh stopped: {exc}")
        update(running=False, phase="failed", message="Refresh failed", error=str(exc),
               finished_at=time.time(), resume_required=False)


def start() -> bool:
    with LOCK:
        if STATE["running"]:
            return False
        config = load_config()
        stages = configured_stages(config)
        total = len(stages) + 1
        # New state files remember the exact interrupted phase. For state files
        # written by older versions, current == total identifies stage 8/semantic.
        resume_queue = list(STATE.get("semantic_queue") or [])
        resume_model = str(STATE.get("semantic_model") or STATE.get("ai_setup_model") or "").strip()
        queued_generation={"snapshot_root":STATE.get("semantic_generation_root")} if STATE.get("semantic_generation_root") else None
        resume_semantic = bool(STATE.get("resume_required")) and (
            STATE.get("resume_phase") in {"semantic", "ai_setup"}
            or (STATE.get("phase") == "interrupted" and STATE.get("current") == total)
        ) and active_generation_matches(config) and (
            not queued_generation or snapshot_generation_is_active(queued_generation)
        )
        STATE.update(running=True, phase="snapshot", message="Copying source libraries safely",
                     activity="Preparing private copies…", current=1, total=total,
                     detail_current=0, detail_total=0, detail_eta="",
                     activity_log=[{"message": "Preparing private copies…", "at": time.time()}],
                     stage_labels=["Copying source libraries safely", *[stage[1] for stage in stages]],
                     stage_started_at=time.time(), started_at=time.time(), finished_at=None,
                     error=None, warnings=[], source_warning="", source_blocked=False,
                     resume_required=False, resume_phase="", semantic_device="", semantic_runtime_detail="",
                     semantic_service="", semantic_base_url="", ai_setup_required=False, ai_setup_skipped=False,
                     ai_setup_model=resume_model if resume_semantic else "",
                     semantic_model=resume_model if resume_semantic else "",
                     semantic_queue=resume_queue if resume_semantic else [], semantic_completed=[],
                     semantic_model_current=STATE.get("semantic_model_current",0) if resume_semantic else 0,
                     semantic_model_total=STATE.get("semantic_model_total",0) if resume_semantic else 0,
                     semantic_generation_root=STATE.get("semantic_generation_root","") if resume_semantic else "")
        _persist_state_unlocked()
    threading.Thread(target=_run, args=(resume_semantic, resume_model or None), daemon=True, name="marginalia-refresh").start()
    return True


def start_semantic(model: str | None = None) -> bool:
    """Build only a semantic index for an already-published library generation."""
    model=str(model or os.getenv("EMBED_MODEL",DEFAULT_EMBED_MODEL)).strip()
    if not valid_model_name(model):
        return False
    with LOCK:
        if STATE["running"]:
            return False
        config = load_config()
        if not active_generation_matches(config):
            return False
        try:
            register_embedding_model(model); SEMANTIC_MODELS[model]=semantic_model_index_path(model)
        except OSError:
            return False
        stages = configured_stages(config)
        semantic = next(stage for stage in stages if stage[0] == "semantic")
        total = len(stages) + 1
        number = stages.index(semantic) + 2
        STATE.update(
            running=True, phase="semantic", message="Updating semantic search",
            activity="Preparing semantic indexing…", current=number, total=total,
            detail_current=0, detail_total=0, detail_eta="", activity_log=[],
            stage_labels=["Copying source libraries safely", *[stage[1] for stage in stages]],
            stage_started_at=time.time(), started_at=time.time(), finished_at=None,
            error=None, warnings=[], source_warning="", source_blocked=False,
            resume_required=False, resume_phase="", semantic_device="",
            semantic_runtime_detail="", semantic_service="", semantic_base_url="",
            ai_setup_required=False, ai_setup_skipped=False, ai_setup_model="",
            semantic_model=model, semantic_queue=[model], semantic_completed=[],
            semantic_model_current=1, semantic_model_total=1,
        )
        _persist_state_unlocked()
    threading.Thread(target=_run, args=(True, model), daemon=True,
                     name="marginalia-semantic-refresh").start()
    return True


def resume_required() -> bool:
    with LOCK:
        return bool(STATE.get("resume_required"))


def report_recovery_failure(error: Exception) -> None:
    """Keep the UI available when an interrupted publication needs attention."""
    message = f"Could not recover the interrupted database publication: {error}"
    update_activity(message, 0, 0)
    update(running=False, phase="failed", message="Refresh recovery needs attention",
           error=message, finished_at=time.time(), resume_required=False)
