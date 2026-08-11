from __future__ import annotations

import json
import html
import hashlib
import heapq
import mimetypes
import os
import posixpath
import re
import shutil
import subprocess
import socket
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr_support import find_tesseract
from refresh_manager import active_generation_matches, recover_interrupted_publication, release_bulk_embedding_ollama, report_recovery_failure, resume_required as refresh_resume_required, select_semantic_setup_model, skip_semantic_setup, start as start_refresh, start_semantic, state as refresh_state
from semantic_models import DEFAULT_MODEL as DEFAULT_EMBED_MODEL, active_model as persisted_active_embedding_model, index_path as semantic_model_index_path, indexed_models as registered_embedding_models, progress_path as semantic_model_progress_path, register_model as register_embedding_model, set_active_model as persist_active_embedding_model, setup_path as semantic_model_setup_path, unregister_model as unregister_embedding_model, valid_model_name
from source_manager import choose_folder_via_helper, configured as sources_configured, load_config, save_config

APP = Path(__file__).resolve().parent
BASE = APP.parent
DB = BASE / "unified_library.sqlite"
CLEAN_DB = BASE / "clean_text.sqlite"
OBSIDIAN_DB = BASE / "obsidian_notes.sqlite"
HISTORY_DB = BASE / "search_history.sqlite"
HISTORY_MAX_ENTRIES = 100
HISTORY_RETENTION_DAYS = 180
GENERATION_MANIFEST = BASE / "library_generation.json"
HOST = os.getenv("MARGINALIA_HOST", "127.0.0.1")
INSTALL_ID = hashlib.sha256(str(BASE).casefold().encode("utf-8")).hexdigest()[:16]
DEFAULT_PORT = 20000 + (int(INSTALL_ID[:8], 16) % 20000)
PORT = int(os.getenv("MARGINALIA_PORT", str(DEFAULT_PORT)))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
EMBED_OLLAMA_BASE_URL = os.getenv("EMBED_OLLAMA_BASE_URL", "http://127.0.0.1:11435").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("LLM_MODEL", "gemma4:12b"))
EMBED_MODEL = os.getenv("EMBED_MODEL", persisted_active_embedding_model()).strip()
EMBED_MODELS = {model: semantic_model_index_path(model) for model in registered_embedding_models()}
EMBED_TIMEOUTS = {
    "qwen3-embedding:0.6b": int(os.getenv("OLLAMA_EMBED_TIMEOUT_06B", "180")),
    "qwen3-embedding:8b": int(os.getenv("OLLAMA_EMBED_TIMEOUT_8B", "600")),
}
if not valid_model_name(EMBED_MODEL): EMBED_MODEL=DEFAULT_EMBED_MODEL
EMBED_MODELS.setdefault(EMBED_MODEL,semantic_model_index_path(EMBED_MODEL))
ASK_CACHE = {}
ASK_CACHE_LOCK = threading.Lock()
ANSWER_MODELS_CACHE = {"key":None,"models":[]}
OLLAMA_CALL_LOCK = threading.Lock()
EMBED_OLLAMA_CALL_LOCK = threading.Lock()
EMBED_OLLAMA_START_LOCK = threading.Lock()
EMBED_OLLAMA_PROCESS = None
ANSWER_OLLAMA_START_LOCK = threading.Lock()
ANSWER_OLLAMA_PROCESS = None
MODEL_INSTALL_LOCK = threading.Lock()
MODEL_INSTALL = {"running":False,"model":"","status":"idle","completed":0,"total":0,"error":"",
  "started_during_ai_setup":False,"index_started":False}
RECOMMENDED_ANSWER_MODEL = "gemma4:12b"
AI_PREFS_PATH = BASE / "ai_preferences.json"


def ai_preferences():
    try:
        value=json.loads(AI_PREFS_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError,TypeError):
        return {}


def update_ai_preferences(**values):
    current=ai_preferences(); current.update(values)
    temporary=AI_PREFS_PATH.with_name(f"{AI_PREFS_PATH.name}.{os.getpid()}.new")
    temporary.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding="utf-8")
    os.replace(temporary,AI_PREFS_PATH)


def model_install_state():
    with MODEL_INSTALL_LOCK:
        return dict(MODEL_INSTALL)


def install_embedding_model(model):
    try:
        ensure_embedding_ollama()
        request=urllib.request.Request(EMBED_OLLAMA_BASE_URL+"/api/pull",
          data=json.dumps({"model":model,"stream":True}).encode(),headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(request,timeout=3600) as response:
            for raw_line in response:
                if not raw_line.strip(): continue
                event=json.loads(raw_line.decode("utf-8"))
                if event.get("error"): raise RuntimeError(str(event["error"]))
                with MODEL_INSTALL_LOCK:
                    MODEL_INSTALL.update(status=str(event.get("status") or "Downloading model"),
                      completed=int(event.get("completed") or MODEL_INSTALL["completed"]),
                      total=int(event.get("total") or MODEL_INSTALL["total"]))
        if not ollama_model_supports(model,"embedding",EMBED_OLLAMA_BASE_URL):
            raise ValueError(f"{model} is not an embedding model")
        persist_active_embedding_model(model)
        update_ai_preferences(embedding_enabled=True,embedding_model=model)
        with MODEL_INSTALL_LOCK:
            during_setup=bool(MODEL_INSTALL.get("started_during_ai_setup"))
            MODEL_INSTALL.update(status="Model downloaded",error="")
        if sources_configured():
            if during_setup and refresh_state().get("phase")=="ai_setup" and select_semantic_setup_model(model):
                with MODEL_INSTALL_LOCK: MODEL_INSTALL.update(running=False,status="indexing",index_started=True)
                return
            while refresh_state().get("running"):
                time.sleep(1)
            if during_setup and refresh_state().get("ai_setup_skipped"):
                with MODEL_INSTALL_LOCK: MODEL_INSTALL.update(running=False,status="complete",index_started=False)
                return
            started=False
            status=semantic_index_status(model)
            installed_digest=ollama_model_digest(EMBED_OLLAMA_BASE_URL,model)
            if not status.get("ready") or not installed_digest or status.get("model_identity")!=installed_digest:
                started=start_semantic(model)
            with MODEL_INSTALL_LOCK:
                MODEL_INSTALL.update(running=False,status="indexing" if started else "complete",index_started=started)
        else:
            with MODEL_INSTALL_LOCK: MODEL_INSTALL.update(running=False,status="complete",index_started=False)
    except Exception as exc:
        with MODEL_INSTALL_LOCK:
            MODEL_INSTALL.update(running=False,status="failed",error=str(exc))


def start_embedding_model_install(model):
    model=str(model or "").strip()
    if not valid_model_name(model): return False,"Invalid Ollama model name"
    if not embedding_ollama_executable(): return False,"Ollama is not installed"
    with MODEL_INSTALL_LOCK:
        if MODEL_INSTALL["running"]: return False,"Another model download is already running"
        try:
            register_embedding_model(model); EMBED_MODELS[model]=semantic_model_index_path(model)
        except (OSError,ValueError) as exc:
            return False,str(exc)
        MODEL_INSTALL.update(running=True,model=model,status="Starting download",completed=0,total=0,error="",
          started_during_ai_setup=refresh_state().get("phase")=="ai_setup",index_started=False)
    threading.Thread(target=install_embedding_model,args=(model,),daemon=True,
      name="marginalia-model-install").start()
    return True,""


def start_answer_model_install(model):
    model=str(model or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?",model):
        return False,"Invalid Ollama model name"
    if not embedding_ollama_executable(): return False,"Ollama is not installed"
    with MODEL_INSTALL_LOCK:
        if MODEL_INSTALL["running"]: return False,"Another model download is already running"
        MODEL_INSTALL.update(running=True,model=model,status="Starting download",completed=0,total=0,error="",
          started_during_ai_setup=False,index_started=False)
    threading.Thread(target=install_answer_model,args=(model,),daemon=True,name="marginalia-answer-install").start()
    return True,""


def install_answer_model(model):
    try:
        if not ensure_answer_ollama(): raise RuntimeError("Ollama could not be started")
        request=urllib.request.Request(OLLAMA_BASE_URL+"/api/pull",
          data=json.dumps({"model":model,"stream":True}).encode(),headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(request,timeout=7200) as response:
            for raw_line in response:
                if not raw_line.strip(): continue
                event=json.loads(raw_line.decode("utf-8"))
                if event.get("error"): raise RuntimeError(str(event["error"]))
                with MODEL_INSTALL_LOCK:
                    MODEL_INSTALL.update(status=str(event.get("status") or "Downloading model"),
                      completed=int(event.get("completed") or MODEL_INSTALL["completed"]),
                      total=int(event.get("total") or MODEL_INSTALL["total"]))
        if not ollama_model_supports(model,"completion",OLLAMA_BASE_URL):
            raise ValueError(f"{model} cannot generate answers")
        update_ai_preferences(answer_enabled=True,answer_model=model)
        with MODEL_INSTALL_LOCK: MODEL_INSTALL.update(running=False,status="complete",error="")
    except Exception as exc:
        with MODEL_INSTALL_LOCK: MODEL_INSTALL.update(running=False,status="failed",error=str(exc))


def ollama_names(base_url,timeout=5):
    with urllib.request.urlopen(base_url+"/api/tags",timeout=timeout) as response:
        models=json.loads(response.read()).get("models",[])
    return {model.get("name") or model.get("model") for model in models}


def ollama_model_digest(base_url, model, timeout=5):
    with urllib.request.urlopen(base_url+"/api/tags",timeout=timeout) as response:
        records=json.loads(response.read()).get("models",[])
    record=next((value for value in records if (value.get("name") or value.get("model"))==model),None)
    return str((record or {}).get("digest") or "")


def ollama_model_supports(model, capability, base_url, timeout=10):
    request=urllib.request.Request(base_url+"/api/show",data=json.dumps({"model":model}).encode(),
      headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        info=json.loads(response.read())
    return capability in set(info.get("capabilities") or [])


def available_answer_models(names=None,timeout=5):
    names=names if names is not None else ollama_names(OLLAMA_BASE_URL,timeout)
    key=tuple(sorted(name for name in names if name))
    if ANSWER_MODELS_CACHE["key"]==key: return ANSWER_MODELS_CACHE["models"]
    models=[]
    for name in key:
        try:
            payload=json.dumps({"model":name}).encode()
            request=urllib.request.Request(OLLAMA_BASE_URL+"/api/show",data=payload,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(request,timeout=timeout) as response: info=json.loads(response.read())
            capabilities=info.get("capabilities") or []
            model_info=info.get("model_info") or {}
            context_lengths=[int(value) for key,value in model_info.items() if key.endswith(".context_length") and value]
            if "completion" in capabilities: models.append({"model":name,"capabilities":capabilities,
              "context_length":max(context_lengths) if context_lengths else 16384})
        except Exception:
            continue
    ANSWER_MODELS_CACHE.update({"key":key,"models":models})
    return models


def selected_answer_model(value=None):
    requested=str(value or OLLAMA_MODEL).strip()
    available={entry["model"] for entry in available_answer_models()}
    if requested not in available:
        raise ValueError(f"{requested} is not an installed Ollama completion model")
    preferences=ai_preferences()
    if not preferences.get("answer_enabled") or preferences.get("answer_model")!=requested:
        update_ai_preferences(answer_enabled=True,answer_model=requested)
    return requested


def answer_request_status(requested, enabled):
    requested=str(requested or OLLAMA_MODEL).strip()
    if not enabled:
        return requested,False,""
    try:
        if not ensure_answer_ollama(): raise ConnectionError("Ollama could not be started")
        return selected_answer_model(requested),True,""
    except ValueError:
        return requested,False,(f"The answer model {requested} is not downloaded. "
          f"Search results are still available; run: ollama pull {requested}.")
    except Exception:
        return requested,False,("The answer-model Ollama service is unavailable. "
          "Search results are still available; start Ollama to generate an answer.")


def answer_model_context(model):
    for entry in available_answer_models():
        if entry["model"]==model: return int(entry.get("context_length") or 16384)
    return 16384


def embedding_ollama_executable():
    configured=os.getenv("OLLAMA_EXE")
    candidates=[Path(configured)] if configured else []
    discovered=shutil.which("ollama")
    if discovered: candidates.append(Path(discovered))
    candidates.append(Path(os.getenv("LOCALAPPDATA",Path.home()/"AppData/Local"))/"Programs/Ollama/ollama.exe")
    for path in candidates:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def ensure_answer_ollama(timeout=20):
    """Start the answer-model Ollama service if Windows did not start it."""
    global ANSWER_OLLAMA_PROCESS
    try:
        ollama_names(OLLAMA_BASE_URL,1); return True
    except Exception:
        pass
    with ANSWER_OLLAMA_START_LOCK:
        try:
            ollama_names(OLLAMA_BASE_URL,1); return True
        except Exception:
            pass
        executable=embedding_ollama_executable()
        if not executable: return False
        environment=os.environ.copy()
        environment.update({"OLLAMA_HOST":urllib.parse.urlparse(OLLAMA_BASE_URL).netloc,
          "OLLAMA_KEEP_ALIVE":"-1","OLLAMA_MAX_LOADED_MODELS":"1","OLLAMA_NUM_PARALLEL":"1",
          "OLLAMA_NO_CLOUD":"1"})
        ANSWER_OLLAMA_PROCESS=subprocess.Popen([str(executable),"serve"],env=environment,
          stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
          creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if ANSWER_OLLAMA_PROCESS.poll() is not None: break
            try:
                ollama_names(OLLAMA_BASE_URL,1); return True
            except Exception:
                time.sleep(.25)
        process,ANSWER_OLLAMA_PROCESS=ANSWER_OLLAMA_PROCESS,None
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        return False


def release_answer_ollama():
    """Stop only the answer service started by this Marginalia process."""
    global ANSWER_OLLAMA_PROCESS
    process,ANSWER_OLLAMA_PROCESS=ANSWER_OLLAMA_PROCESS,None
    if process is not None and process.poll() is None:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=5)


def ensure_embedding_ollama(timeout=20):
    """Start a CPU-only Ollama runner dedicated to query embeddings."""
    global EMBED_OLLAMA_PROCESS
    try:
        ollama_names(EMBED_OLLAMA_BASE_URL,1); return True
    except Exception:
        pass
    with EMBED_OLLAMA_START_LOCK:
        try:
            ollama_names(EMBED_OLLAMA_BASE_URL,1); return True
        except Exception:
            pass
        executable=embedding_ollama_executable()
        if not executable: raise RuntimeError("Ollama executable was not found for the dedicated embedding service")
        environment=os.environ.copy()
        environment.update({"OLLAMA_HOST":urllib.parse.urlparse(EMBED_OLLAMA_BASE_URL).netloc,
          "OLLAMA_KEEP_ALIVE":"-1","OLLAMA_MAX_LOADED_MODELS":"2","OLLAMA_NUM_PARALLEL":"1",
          "OLLAMA_NO_CLOUD":"1","CUDA_VISIBLE_DEVICES":"-1","GGML_VK_VISIBLE_DEVICES":"-1"})
        creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0)
        EMBED_OLLAMA_PROCESS=subprocess.Popen([str(executable),"serve"],env=environment,
          stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=creationflags)
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if EMBED_OLLAMA_PROCESS.poll() is not None: break
            try:
                ollama_names(EMBED_OLLAMA_BASE_URL,1); return True
            except Exception:
                time.sleep(.25)
        process,EMBED_OLLAMA_PROCESS=EMBED_OLLAMA_PROCESS,None
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        raise RuntimeError(f"The dedicated embedding Ollama did not start at {EMBED_OLLAMA_BASE_URL}")


def warm_default_embedding_model():
    try:
        ensure_embedding_ollama()
        model=persisted_active_embedding_model()
        payload={"model":model,"input":["warmup"],"keep_alive":-1}
        request=urllib.request.Request(EMBED_OLLAMA_BASE_URL+"/api/embed",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
        with EMBED_OLLAMA_CALL_LOCK:
            with urllib.request.urlopen(request,timeout=EMBED_TIMEOUTS.get(model,600)) as response: response.read()
    except Exception:
        pass


def selected_embed_model(value=None):
    value=str(value or "").strip()
    return value if valid_model_name(value) and value in EMBED_MODELS else EMBED_MODEL


def select_active_embedding_model(model):
    global EMBED_MODEL
    persist_active_embedding_model(model)
    update_ai_preferences(embedding_enabled=True,embedding_model=model)
    EMBED_MODEL=model
    EMBED_MODELS[model]=semantic_model_index_path(model)


def semantic_db_path(embed_model=None):
    model=selected_embed_model(embed_model)
    return EMBED_MODELS.setdefault(model,semantic_model_index_path(model))


def semantic_progress_path(embed_model=None):
    return semantic_model_progress_path(selected_embed_model(embed_model))


class SingleInstanceServer(ThreadingHTTPServer):
    allow_reuse_address=False


def matching_instance(port):
    """Return True only when this exact installation already owns the port."""
    try:
        with urllib.request.urlopen(
            f"http://{HOST}:{port}/api/app-instance", timeout=.75
        ) as response:
            payload=json.loads(response.read())
        return payload.get("install_id")==INSTALL_ID
    except Exception:
        return False


def connect():
    db = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def clean_connect():
    db = sqlite3.connect(f"file:{CLEAN_DB.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def obsidian_connect():
    db = sqlite3.connect(f"file:{OBSIDIAN_DB.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def history_connect():
    db=sqlite3.connect(HISTORY_DB,timeout=30)
    db.row_factory=sqlite3.Row
    db.execute("PRAGMA secure_delete=ON")
    return db


def prune_history(db):
    cutoff=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(time.time()-HISTORY_RETENTION_DAYS*86400))
    db.execute("DELETE FROM search_history WHERE updated_at < ?",(cutoff,))
    db.execute("DELETE FROM search_history WHERE id IN (SELECT id FROM search_history ORDER BY updated_at DESC LIMIT -1 OFFSET ?)",(HISTORY_MAX_ENTRIES,))


def initialize_history():
    db=history_connect()
    db.execute("""CREATE TABLE IF NOT EXISTS search_history (
      id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
      query TEXT NOT NULL, scope TEXT NOT NULL, method TEXT NOT NULL,
      embedding_model TEXT, answer_model TEXT, exact INTEGER NOT NULL DEFAULT 0,
      generate_answer INTEGER NOT NULL DEFAULT 0, answer_mode TEXT NOT NULL DEFAULT 'fast',
      result_count INTEGER NOT NULL DEFAULT 0, source_count INTEGER NOT NULL DEFAULT 0,
      retrieval_seconds REAL, answer_seconds REAL, snapshot_json TEXT NOT NULL
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS search_history_updated ON search_history(updated_at DESC)")
    prune_history(db)
    db.commit(); db.close()


def history_summary(row):
    return {key:row[key] for key in ("id","created_at","updated_at","query","scope","method","embedding_model",
      "answer_model","exact","generate_answer","answer_mode","result_count","source_count","retrieval_seconds","answer_seconds")}


def current_generation_token():
    try:
        manifest = json.loads(GENERATION_MANIFEST.read_text(encoding="utf-8"))
        identity = {key: manifest.get(key) for key in (
            "completed_at", "snapshot_root", "zotero_path", "obsidian_path",
            "linked_attachment_base_path",
        )}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    except (OSError, ValueError, TypeError):
        return ""


def obsidian_info():
    if not OBSIDIAN_DB.is_file():
        return {"ready":False,"notes":0,"sections":0,"assets":0}
    try:
        db=obsidian_connect(); info=dict(db.execute("SELECT key,value FROM obsidian_import_info")); db.close()
        return {"ready":True,**{key:int(value) if key in {"notes","sections","links","assets","excluded_notes"} else value for key,value in info.items()}}
    except (sqlite3.Error,ValueError):
        return {"ready":False,"notes":0,"sections":0,"assets":0}


def obsidian_asset_record(relative, note_path=""):
    """Resolve an Obsidian asset deterministically in the note's folder context."""
    requested=str(relative or "").replace("\\","/").lstrip("/")
    note_path=str(note_path or "").replace("\\","/").lstrip("/")
    if not requested or requested.startswith("../"):
        return None, "missing"
    candidates=[posixpath.normpath(requested)]
    note_folder=posixpath.dirname(note_path)
    if note_folder:
        contextual=posixpath.normpath(posixpath.join(note_folder,requested))
        if not contextual.startswith("../"):
            candidates.insert(0,contextual)
    candidates=list(dict.fromkeys(candidates))
    db=obsidian_connect()
    try:
        for candidate in candidates:
            row=db.execute(
                "SELECT relative_path,content_type FROM obsidian_assets WHERE relative_path=? COLLATE NOCASE",
                (candidate,),
            ).fetchone()
            if row:
                return row, "exact"
        filename=requested.rsplit("/",1)[-1]
        rows=db.execute(
            "SELECT relative_path,content_type FROM obsidian_assets WHERE filename=? COLLATE NOCASE ORDER BY relative_path",
            (filename,),
        ).fetchall()
        if len(rows)==1:
            return rows[0], "filename"
        return None, "ambiguous" if rows else "missing"
    finally:
        db.close()


def allowed_attachment_path(local_path):
    """Resolve a database attachment path only inside a managed private copy."""
    if not local_path:
        return None
    path=(BASE/str(local_path)).resolve()
    roots=((BASE/"files").resolve(),(BASE/"source_snapshots").resolve())
    return path if any(root in path.parents for root in roots) and path.is_file() else None


def ollama_status(timeout=5):
    """Return local Ollama availability without sending any library content."""
    executable=embedding_ollama_executable()
    installed=bool(executable)
    answer_names=set(); answer_models=[]; answer_available=False; answer_detail=""
    embed_names=set(); embedding_available=False; embedding_detail=""
    try:
        answer_names=ollama_names(OLLAMA_BASE_URL,timeout)
        answer_models=available_answer_models(answer_names,timeout)
        answer_available=True
    except Exception as exc:
        answer_detail=str(exc)
    try:
        embed_names=ollama_names(EMBED_OLLAMA_BASE_URL,timeout)
        embedding_available=True
    except Exception as exc:
        embedding_detail=str(exc)
    choices=[]
    for model in EMBED_MODELS:
        identity=ollama_model_digest(EMBED_OLLAMA_BASE_URL,model,timeout) if model in embed_names else ""
        choices.append({"model":model,"installed":model in embed_names,
          "semantic_index":semantic_index_status(model,identity)})
    installed_answers={entry["model"] for entry in answer_models}
    return {"available":answer_available,"answer_available":answer_available,
      "embedding_available":embedding_available,"installed":installed,"executable":str(executable or ""),
      "model":OLLAMA_MODEL,"model_installed":OLLAMA_MODEL in installed_answers,"answer_models":answer_models,
      "answer_detail":answer_detail,"embedding_detail":embedding_detail,
      "embedding_model":EMBED_MODEL,"embedding_model_installed":EMBED_MODEL in embed_names,
      "embedding_models":choices,"embedding_base_url":EMBED_OLLAMA_BASE_URL,
      "semantic_index":semantic_index_status(EMBED_MODEL),"model_install":model_install_state(),"local":True}


def ollama_chat(messages, num_predict=480, timeout=3600, num_ctx=16384, model=None):
    model=selected_answer_model(model)
    payload={"model":model,"stream":False,"think":False,"messages":messages,"keep_alive":-1,
      "options":{"temperature":0.1,"num_ctx":num_ctx,"num_predict":num_predict}}
    request=urllib.request.Request(OLLAMA_BASE_URL+"/api/chat",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    with OLLAMA_CALL_LOCK:
        with urllib.request.urlopen(request,timeout=timeout) as response:
            response_data=json.loads(response.read())
    answer=(response_data.get("message") or {}).get("content","").strip()
    if not answer: raise ValueError("Ollama returned an empty answer")
    return answer


def ollama_chat_stream(messages, num_predict=480, timeout=3600, num_ctx=16384, model=None, prompt_tokens_estimate=None):
    """Yield progress, answer text, and final Ollama timings."""
    model=selected_answer_model(model)
    payload={"model":model,"stream":True,"think":False,"messages":messages,"keep_alive":-1,
      "options":{"temperature":0.1,"num_ctx":num_ctx,"num_predict":num_predict}}
    request=urllib.request.Request(OLLAMA_BASE_URL+"/api/chat",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    received=False
    with OLLAMA_CALL_LOCK:
        prompt_detail=f" The prompt is estimated at {int(prompt_tokens_estimate):,} tokens." if prompt_tokens_estimate else ""
        yield {"type":"status","message":f"{model} has the Ollama slot. Preparing a {num_ctx:,}-token context.{prompt_detail} Reading the supplied sources…"}
        with urllib.request.urlopen(request,timeout=timeout) as response:
            yield {"type":"status","message":f"{model} is processing the prompt. The first words appear after all supplied passages have been read…"}
            for raw_line in response:
                if not raw_line.strip(): continue
                event=json.loads(raw_line.decode("utf-8"))
                if event.get("error"): raise RuntimeError(str(event["error"]))
                content=(event.get("message") or {}).get("content","")
                if content:
                    received=True
                    yield {"type":"token","text":content}
                if event.get("done"):
                    yield {"type":"metrics","load_duration":event.get("load_duration",0),
                      "prompt_eval_duration":event.get("prompt_eval_duration",0),"eval_duration":event.get("eval_duration",0),
                      "prompt_eval_count":event.get("prompt_eval_count",0),"eval_count":event.get("eval_count",0)}
    if not received: raise ValueError("Ollama returned an empty answer")


def ollama_embed_query(question, embed_model=None, timeout=None):
    embed_model=selected_embed_model(embed_model)
    timeout=timeout or EMBED_TIMEOUTS.get(embed_model,600)
    instructed=("Instruct: Retrieve academically relevant passages that answer the research question. "
      "The sources and the question may use different languages.\nQuery: "+question)
    ensure_embedding_ollama()
    payload={"model":embed_model,"input":[instructed],"keep_alive":-1}
    request=urllib.request.Request(EMBED_OLLAMA_BASE_URL+"/api/embed",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    try:
        with EMBED_OLLAMA_CALL_LOCK:
            with urllib.request.urlopen(request,timeout=timeout) as response:
                data=json.loads(response.read())
    except (TimeoutError, socket.timeout) as exc:
        raise TimeoutError(
          f"{embed_model} did not finish embedding the query within {timeout} seconds. "
          "Ollama may still be loading or swapping the local model; try the search again."
        ) from exc
    vectors=data.get("embeddings") or []
    if len(vectors)!=1: raise ValueError("Ollama did not return a query embedding")
    vector=np.asarray(vectors[0],dtype=np.float32)
    norm=float(np.linalg.norm(vector))
    return vector/norm if norm else vector


def semantic_index_status(embed_model=None, installed_identity=""):
    embed_model=selected_embed_model(embed_model); semantic_db=semantic_db_path(embed_model)
    progress={}; setup={}
    try:
        progress=json.loads(semantic_progress_path(embed_model).read_text(encoding="utf-8"))
    except (OSError,ValueError):
        pass
    try:
        setup=json.loads(semantic_model_setup_path(embed_model).read_text(encoding="utf-8-sig"))
    except (OSError,ValueError):
        pass
    if not semantic_db.is_file():
        return {"ready":False,"chunks":0,"model":embed_model,"path":semantic_db.name,"progress":progress,"setup":setup}
    try:
        db=sqlite3.connect(f"file:{semantic_db.as_posix()}?mode=ro",uri=True)
        info=dict(db.execute("SELECT key,value FROM semantic_index_info"))
        chunks=db.execute("SELECT count(*) FROM semantic_chunks WHERE model=?",(embed_model,)).fetchone()[0]
        db.close()
        clean_completed=""
        if CLEAN_DB.is_file():
            try:
                clean_db=sqlite3.connect(f"file:{CLEAN_DB.as_posix()}?mode=ro",uri=True)
                row=clean_db.execute("SELECT value FROM clean_text_info WHERE key='completed_at'").fetchone()
                clean_completed=row[0] if row else ""
                clean_db.close()
            except sqlite3.Error:
                clean_completed=""
        source_current=bool(clean_completed and info.get("clean_text_completed_at")==clean_completed)
        model_current=not installed_identity or info.get("model_identity")==installed_identity
        ready=bool(chunks and info.get("model")==embed_model and info.get("completed_at") and source_current and model_current)
        return {"ready":ready,"chunks":chunks,"model":embed_model,"dimensions":int(info.get("dimensions") or 0),
          "completed_at":info.get("completed_at",""),"source_current":source_current,
          "model_identity":info.get("model_identity",""),
          "model_current":model_current,
          "path":semantic_db.name,"progress":progress,"setup":setup}
    except sqlite3.Error as exc:
        return {"ready":False,"chunks":0,"model":embed_model,"path":semantic_db.name,"progress":progress,"setup":setup,"detail":str(exc)}


def semantic_ranked_candidates(embed_model, query_vector, allowed_kinds, allowed_items,
                               selected_folders, candidate_count):
    """Score the SQLite index in bounded batches instead of retaining every vector in RAM."""
    semantic_db=semantic_db_path(embed_model)
    if not semantic_db.is_file():
        raise RuntimeError(f"The semantic index for {embed_model} has not been built yet")
    db=sqlite3.connect(f"file:{semantic_db.as_posix()}?mode=ro",uri=True); db.row_factory=sqlite3.Row
    try:
        columns={row[1] for row in db.execute("PRAGMA table_info(semantic_chunks)")}
        optional=lambda name: name if name in columns else f"NULL AS {name}"
        cursor=db.execute(f"""SELECT id,kind,label,item_id,attachment_id,page,section_title,
          annotation_id,embedded_annotation_id,note_id,{optional('obsidian_note_id')},
          {optional('obsidian_section_id')},{optional('vault_path')},title,text,dimensions,embedding
          FROM semantic_chunks WHERE model=? ORDER BY id""",(embed_model,))
        adjustments={"annotation":.015,"embedded_annotation":.012,"metadata":-.015,
          "document_footnote":-.012,"document_table":-.025,"note":-.005}
        heap=[]; sequence=0; found=False
        while rows:=cursor.fetchmany(512):
            batch_records=[]; batch_vectors=[]; batch_adjustments=[]
            for row in rows:
                found=True
                kind=row["kind"]
                folder=(row["vault_path"] or "").rsplit("/",1)[0] if "/" in (row["vault_path"] or "") else ""
                if kind not in allowed_kinds: continue
                if kind=="obsidian":
                    if selected_folders is not None and folder not in selected_folders: continue
                elif allowed_items is not None and row["item_id"] not in allowed_items:
                    continue
                dimensions=int(row["dimensions"] or 0)
                vector=np.frombuffer(row["embedding"],dtype=np.float32)
                if dimensions!=len(query_vector) or len(vector)!=dimensions:
                    raise RuntimeError("Embedding dimensions do not match the semantic index")
                record={key:row[key] for key in ("id","kind","label","item_id","attachment_id","page","section_title","annotation_id","embedded_annotation_id","note_id","obsidian_note_id","obsidian_section_id","vault_path","title","text")}
                batch_records.append(record); batch_vectors.append(vector); batch_adjustments.append(adjustments.get(kind,0))
            if not batch_vectors:
                continue
            matrix=np.vstack(batch_vectors)
            norms=np.linalg.norm(matrix,axis=1)
            similarities=(matrix @ query_vector) / np.where(norms, norms, 1.0)
            for record,similarity,adjustment in zip(batch_records,similarities,batch_adjustments):
                similarity=float(similarity); adjusted=similarity+adjustment
                entry=(adjusted,sequence,similarity,record); sequence+=1
                if len(heap)<candidate_count: heapq.heappush(heap,entry)
                elif adjusted>heap[0][0]: heapq.heapreplace(heap,entry)
        if not found: raise RuntimeError("The semantic index is empty")
        return [(record,similarity,adjusted) for adjusted,_,similarity,record in sorted(heap,reverse=True)]
    finally:
        db.close()


ASK_STOPWORDS={
    "a","about","according","all","an","and","answer","are","as","at","be","been","being","by","can","could","did","do","does","explain","for","from","give","how","i","in","is","it","library","me","my","of","on","please","say","says","should","tell","that","the","their","them","there","these","this","to","was","were","what","when","where","which","who","why","with","would",
    "a","aby","ale","biblioteka","bibliotece","biblioteki","by","był","była","było","co","czy","dla","do","gdzie","i","jak","jaka","jakie","jest","która","które","mi","mnie","moja","moje","mojej","na","o","od","opisz","oraz","po","powiedz","przez","się","są","ta","ten","te","to","w","we","według","z","za","ze"
}


def retrieval_terms(question):
    terms=[]; seen=set()
    for term in re.findall(r"[\wÀ-ž]+",question,flags=re.UNICODE):
        folded=term.casefold()
        if len(term)>2 and folded not in ASK_STOPWORDS and folded not in seen:
            terms.append(term); seen.add(folded)
    return " ".join(terms[:10])


def expand_retrieval_terms(question):
    answer=ollama_chat([
      {"role":"system","content":"Convert a research question into 2 to 6 concise English search keywords likely to occur verbatim in academic sources. Translate if needed. Return keywords only, separated by spaces. No explanation, punctuation, or quotation marks."},
      {"role":"user","content":question}],num_predict=40,timeout=180)
    return " ".join(re.findall(r"[\wÀ-ž]+",answer,flags=re.UNICODE)[:8])


def plain_text(value):
    return re.sub(r"\s+"," ",html.unescape(re.sub(r"<[^>]*>"," ",str(value or "")))).strip()


def safe_snippet(value):
    value = str(value or "").replace("<mark>","\x01").replace("</mark>","\x02")
    value = html.escape(plain_text(value))
    return value.replace("\x01","<mark>").replace("\x02","</mark>")


def clean_document_status(attachment_id):
    if not CLEAN_DB.is_file():
        return None
    try:
        clean=clean_connect()
        row=clean.execute("""SELECT source,raw_pages,clean_blocks FROM clean_extraction_status
          WHERE attachment_id=? AND status='ok' AND clean_blocks>0""",(attachment_id,)).fetchone()
        clean.close()
        return dict(row) if row else None
    except sqlite3.Error:
        return None


def canonical_locator_text(value):
    value=plain_text(value).replace("\u00ad","").casefold()
    value=re.sub(r"(?<=\w)-\s+(?=\w)","",value)
    return re.sub(r"[^\w]+"," ",value,flags=re.UNICODE).strip()


def find_clean_text_page(clean,attachment_id,needle,fallback=0):
    """Locate a result in cleaned blocks and return the block's real start page."""
    rows=clean.execute("""SELECT page_start,page_end,text FROM clean_document_blocks
      WHERE attachment_id=? ORDER BY ordinal""",(attachment_id,)).fetchall()
    wanted=canonical_locator_text(needle)
    if wanted:
        candidates=[wanted]
        words=wanted.split()
        if len(words)>18:
            candidates.extend((" ".join(words[:18])," ".join(words[-18:])))
        for row in rows:
            haystack=canonical_locator_text(row[2])
            if any(candidate and candidate in haystack for candidate in candidates):
                return int(row[0])
    if fallback:
        for row in rows:
            if int(row[0]) <= fallback <= int(row[1]):
                return int(row[0])
    return fallback if fallback>0 else (int(rows[0][0]) if rows else 0)


def clean_document_pages(attachment_id,start,limit):
    """Return cleaned semantic blocks grouped by their original start page."""
    clean=clean_connect()
    status=clean.execute("""SELECT source,raw_pages FROM clean_extraction_status
      WHERE attachment_id=? AND status='ok' AND clean_blocks>0""",(attachment_id,)).fetchone()
    if not status:
        clean.close(); return None
    total=max(0,int(status[1] or 0))
    end=min(total,start+limit-1)
    block_rows=clean.execute("""SELECT ordinal,block_type,section_title,page_start,page_end,text
      FROM clean_document_blocks WHERE attachment_id=? AND page_start BETWEEN ? AND ?
      ORDER BY ordinal""",(attachment_id,start,end)).fetchall()
    clean.close()
    grouped={page:[] for page in range(start,end+1)}
    for row in block_rows:
        grouped[int(row[3])].append({
          "ordinal":row[0],"block_type":row[1],"section_title":row[2] or "",
          "page_start":row[3],"page_end":row[4],"text":row[5] or ""})
    pages=[]
    for page_number in range(start,end+1):
        blocks=grouped[page_number]
        pages.append({"page_number":page_number,"text":"\n\n".join(block["text"] for block in blocks),
          "char_count":sum(len(block["text"]) for block in blocks),"blocks":blocks})
    return {"attachment_id":attachment_id,"start":start,"total_pages":total,"pages":pages,
      "has_more":end<total,"text_source":"clean","cleaning_method":status[0]}


def creator_names(value):
    try: creators=json.loads(value or "[]")
    except Exception: creators=[]
    authors=[]
    preferred=[c for c in creators if c.get("type") in {"author","bookAuthor"}] or creators
    for creator in preferred:
        name=plain_text(" ".join(x for x in (creator.get("firstName",""),creator.get("lastName","")) if x))
        if name and not any(x["name"].casefold()==name.casefold() for x in authors):
            authors.append({"name":name,"type":creator.get("type","author")})
    return authors


def publication_date(value):
    value=plain_text(value)
    match=re.search(r"((?:18|19|20|21)\d{2})[-/](\d{1,2})[-/](\d{1,2})",value)
    if match: return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    match=re.search(r"(?:18|19|20|21)\d{2}",value)
    return match.group(0) if match else value


def annotation_page(position_json, page_label=None):
    try:
        page_index=json.loads(position_json or "{}").get("pageIndex")
        if isinstance(page_index,int) and page_index >= 0:
            return page_index + 1
    except Exception:
        pass
    match=re.search(r"\d+",str(page_label or ""))
    return int(match.group()) if match else 0


def find_text_page(db, attachment_id, value):
    terms=[]; seen=set()
    for term in re.findall(r"[\wÀ-ž]+",plain_text(value),flags=re.UNICODE):
        folded=term.casefold()
        if len(term)>2 and folded not in seen:
            terms.append(term); seen.add(folded)
        if len(terms)>=12: break
    if not terms: return 0
    query=" OR ".join('"'+term.replace('"','""')+'"' for term in terms)
    try:
        row=db.execute("""
            SELECT page_number FROM document_search
            WHERE document_search MATCH ? AND attachment_id=?
            ORDER BY bm25(document_search) LIMIT 1
        """,(query,attachment_id)).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def search_filters(value):
    value=value or {}
    try: collection_id=max(0,int(value.get("collection_id") or 0))
    except (TypeError,ValueError): collection_id=0
    raw_collections=value.get("zotero_collections") if isinstance(value,dict) else None
    if raw_collections is None:
        zotero_collections=None
    elif isinstance(raw_collections,list):
        zotero_collections=[str(entry) for entry in raw_collections]
    else:
        try:
            parsed=json.loads(str(raw_collections))
            zotero_collections=[str(entry) for entry in parsed] if isinstance(parsed,list) else None
        except (ValueError,TypeError):
            zotero_collections=None
    def day(name):
        candidate=plain_text(value.get(name,""))
        return candidate if re.fullmatch(r"\d{4}-\d{2}-\d{2}",candidate) else ""
    raw_folders=value.get("obsidian_folders") if isinstance(value,dict) else None
    if raw_folders is None:
        obsidian_folders=None
    elif isinstance(raw_folders,list):
        obsidian_folders=[plain_text(folder) for folder in raw_folders if isinstance(folder,str)]
    else:
        try:
            parsed=json.loads(str(raw_folders))
            obsidian_folders=[plain_text(folder) for folder in parsed if isinstance(folder,str)] if isinstance(parsed,list) else None
        except (ValueError,TypeError):
            obsidian_folders=None
    return {"collection_id":collection_id,"zotero_collections":zotero_collections,"published_from":day("published_from"),
            "published_to":day("published_to"),"added_from":day("added_from"),"added_to":day("added_to"),
            "obsidian_folders":obsidian_folders}


def zotero_filters_active(filters):
    filters=search_filters(filters)
    return bool(filters["collection_id"] or filters["zotero_collections"] is not None or any(filters[k] for k in ("published_from","published_to","added_from","added_to")))


def date_interval(value):
    value=plain_text(value)
    match=re.search(r"((?:18|19|20|21)\d{2})(?:[-/](\d{1,2}))?(?:[-/](\d{1,2}))?",value)
    if not match: return None
    year=int(match.group(1)); month=int(match.group(2) or 0); day=int(match.group(3) or 0)
    if 1 <= month <= 12:
        if 1 <= day <= 31: return (f"{year:04d}-{month:02d}-{day:02d}",f"{year:04d}-{month:02d}-{day:02d}")
        last=29 if month==2 else 30 if month in {4,6,9,11} else 31
        return (f"{year:04d}-{month:02d}-01",f"{year:04d}-{month:02d}-{last:02d}")
    return (f"{year:04d}-01-01",f"{year:04d}-12-31")


def filtered_item_ids(db, filters):
    filters=search_filters(filters)
    active=filters["collection_id"] or filters["zotero_collections"] is not None or any(filters[k] for k in ("published_from","published_to","added_from","added_to"))
    if not active: return None
    collection_items=None
    if filters["zotero_collections"] is not None:
        selected={int(value) for value in filters["zotero_collections"] if str(value).isdigit()}
        collection_items=set()
        if selected:
            marks=",".join("?" for _ in selected)
            collection_items.update(row[0] for row in db.execute(
              f"SELECT DISTINCT item_id FROM item_collections WHERE collection_id IN ({marks})",sorted(selected)))
        if "__unfiled__" in filters["zotero_collections"]:
            collection_items.update(row[0] for row in db.execute(
              "SELECT id FROM items WHERE id NOT IN (SELECT DISTINCT item_id FROM item_collections)"))
    elif filters["collection_id"]:
        collection_items={r[0] for r in db.execute("""
            SELECT DISTINCT ic.item_id FROM item_collections ic
            JOIN collection_descendants cd ON cd.descendant_id=ic.collection_id
            WHERE cd.ancestor_id=?
        """,(filters["collection_id"],))}
    allowed=set()
    for row in db.execute("SELECT id,date_created,date_added FROM items"):
        if collection_items is not None and row[0] not in collection_items: continue
        if filters["published_from"] or filters["published_to"]:
            interval=date_interval(row[1])
            if not interval: continue
            if filters["published_from"] and interval[1] < filters["published_from"]: continue
            if filters["published_to"] and interval[0] > filters["published_to"]: continue
        if filters["added_from"] or filters["added_to"]:
            interval=date_interval(row[2])
            if not interval: continue
            if filters["added_from"] and interval[1] < filters["added_from"]: continue
            if filters["added_to"] and interval[0] > filters["added_to"]: continue
        allowed.add(row[0])
    return allowed


def semantic_search(question,scope="all",limit=12,filters=None,embed_model=None):
    embed_model=selected_embed_model(embed_model)
    query_vector=ollama_embed_query(question,embed_model)
    scopes={
      "all":{"document","document_footnote","document_table","annotation","embedded_annotation","metadata","note","obsidian"},
      "documents":{"document","document_footnote","document_table"},
      "annotations":{"annotation","embedded_annotation"},
      "metadata":{"metadata"},
      "notes":{"note"},
      "obsidian":{"obsidian"},
    }
    allowed_kinds=scopes.get(scope,scopes["all"])
    filters=search_filters(filters)
    library=connect(); allowed_items=filtered_item_ids(library,filters)
    selected_folders=filters.get("obsidian_folders")
    try:
        ranked=semantic_ranked_candidates(embed_model,query_vector,allowed_kinds,allowed_items,
          selected_folders,max(160,limit*2))
    except Exception:
        library.close()
        raise
    if not ranked:
        library.close(); return []

    chosen=[]; per_item={}; seen_text=set()
    for record,similarity,rank_score in ranked:
        item_id=record["item_id"] if record["kind"]!="obsidian" else "obsidian:"+str(record.get("obsidian_note_id") or "")
        digest=hash(record["text"].casefold())
        if digest in seen_text or per_item.get(item_id,0)>=2: continue
        chosen.append((record,similarity,rank_score)); seen_text.add(digest)
        per_item[item_id]=per_item.get(item_id,0)+1
        if len(chosen)>=limit: break
    if len(chosen)<limit:
        for record,similarity,rank_score in ranked:
            digest=hash(record["text"].casefold())
            if digest in seen_text: continue
            chosen.append((record,similarity,rank_score)); seen_text.add(digest)
            if len(chosen)>=limit: break

    item_ids=sorted({record["item_id"] for record,_,_ in chosen if record.get("item_id")})
    item_context={}
    if item_ids:
        marks=",".join("?" for _ in item_ids)
        for row in library.execute(f"SELECT id,title,date_created,date_added,creators_json FROM items WHERE id IN ({marks})",item_ids):
            item_context[row[0]]={"title":plain_text(row[1]),"published":publication_date(row[2]),
              "date_added":row[3],"authors":creator_names(row[4])}
    library.close()
    obsidian_context={}
    obsidian_ids=sorted({record.get("obsidian_note_id") for record,_,_ in chosen if record.get("obsidian_note_id")})
    if obsidian_ids and OBSIDIAN_DB.is_file():
        obs=obsidian_connect(); marks=",".join("?" for _ in obsidian_ids)
        for row in obs.execute(f"SELECT note_id,title,relative_path,folder,tags_json,authors_json,created_at,modified_at FROM obsidian_notes WHERE note_id IN ({marks})",obsidian_ids):
            obsidian_context[row[0]]={"title":row[1],"path":row[2],"folder":row[3],"tags":json.loads(row[4] or "[]"),
              "authors":[{"name":name,"type":"author"} for name in json.loads(row[5] or "[]")],"created":row[6],"modified":row[7]}
        obs.close()
    output=[]
    for record,similarity,rank_score in chosen:
        info=obsidian_context.get(record.get("obsidian_note_id"),{}) if record["kind"]=="obsidian" else item_context.get(record.get("item_id"),{})
        source_kind="document" if record["kind"].startswith("document") else record["kind"]
        # A semantic result is one indexed chunk. Return that chunk in full so the
        # result card, citation popup, and evidence supplied to Gemma all refer to
        # the same visible passage instead of an arbitrary prefix of it.
        excerpt=plain_text(record["text"])
        result={"kind":source_kind,"semantic_kind":record["kind"],"semantic_chunk_id":record["id"],"label":record["label"],
          "item_id":record["item_id"],"attachment_id":record["attachment_id"],"page":record["page"],
          "section_title":record["section_title"],"annotation_id":record["annotation_id"],
          "embedded_annotation_id":record["embedded_annotation_id"],"note_id":record["note_id"],
          "obsidian_note_id":record.get("obsidian_note_id"),"obsidian_section_id":record.get("obsidian_section_id"),
          "vault_path":record.get("vault_path"),
          "title":info.get("title") or plain_text(record["title"]) or "Untitled",
          "snippet":safe_snippet(excerpt),"score":1-rank_score,"semantic_similarity":round(similarity,4),
          "published":info.get("published"),"date_added":info.get("date_added"),"authors":info.get("authors"),
          "created":info.get("created"),"modified":info.get("modified"),"tags":info.get("tags"),
          "_context":record["text"]}
        output.append(result)
    return output


def fts_query(text,exact=False):
    terms = re.findall(r"[\wÀ-ž]+",text,flags=re.UNICODE)
    if exact and terms:
        return '"'+' '.join(terms).replace('"','""')+'"'
    return " AND ".join('"'+term.replace('"','""')+'"' for term in terms[:20])


def search_library(query, scope="all", limit=40, exact=False, offset=0, item_filter=None, obsidian_filter=None, filters=None):
    match = fts_query(query,exact)
    if not match:
        return [],False
    fetch_limit=offset+limit+1
    db = connect(); results = []
    try:
        clauses=[]; filter_values=[]
        if item_filter is not None: clauses.append("CAST(item_id AS INTEGER)=?"); filter_values.append(item_filter)
        allowed=filtered_item_ids(db,filters)
        zotero_empty=allowed is not None and not allowed
        if allowed is not None and allowed:
            clauses.append("CAST(item_id AS INTEGER) IN ("+",".join("?" for _ in allowed)+")"); filter_values.extend(sorted(allowed))
        item_clause=" AND "+" AND ".join(clauses) if clauses else ""
        def args(): return (match,*filter_values,fetch_limit)
        if not zotero_empty and scope in {"all","documents"}:
            for r in db.execute(f"""
                SELECT item_id,attachment_id,page_number,parent_title,
                  snippet(document_search,6,'<mark>','</mark>',' … ',34) snippet,
                  bm25(document_search,0,0,0,0,3,1,8) score
                FROM document_search WHERE document_search MATCH ?{item_clause}
                ORDER BY score,item_id,attachment_id,page_number LIMIT ?
            """,args()):
                results.append({"kind":"document","label":"PDF text","item_id":r[0],"attachment_id":r[1],"page":r[2],"title":plain_text(r[3]) or "Untitled","snippet":safe_snippet(r[4]),"score":r[5]})
        if not zotero_empty and scope in {"all","annotations"}:
            for r in db.execute(f"""
                SELECT item_id,attachment_id,page_label,parent_title,
                  snippet(annotation_search,5,'<mark>','</mark>',' … ',34),
                  snippet(annotation_search,6,'<mark>','</mark>',' … ',24),
                  bm25(annotation_search,0,0,0,2,1,8,5,0) score,annotation_id,
                  (SELECT position_json FROM annotation_context c
                   WHERE c.annotation_id=CAST(annotation_search.annotation_id AS INTEGER)) position_json
                FROM annotation_search WHERE annotation_search MATCH ?{item_clause}
                ORDER BY score,item_id,attachment_id,page_label,annotation_id LIMIT ?
            """,args()):
                snippet = safe_snippet(r[4] or r[5] or "Annotation match")
                if r[5] and r[5] != r[4]: snippet += "<br><span class=\"comment\">Note: " + safe_snippet(r[5]) + "</span>"
                results.append({"kind":"annotation","label":"My annotation","item_id":r[0],"attachment_id":r[1],"page":annotation_page(r[8],r[2]) or r[2],"title":plain_text(r[3]) or "Untitled","snippet":snippet,"score":r[6]-0.3,"annotation_id":r[7]})
            for r in db.execute(f"""
                SELECT item_id,attachment_id,page_number,parent_title,
                  snippet(embedded_annotation_search,6,'<mark>','</mark>',' … ',34),
                  snippet(embedded_annotation_search,7,'<mark>','</mark>',' … ',24),
                  bm25(embedded_annotation_search) score,embedded_annotation_id
                FROM embedded_annotation_search WHERE embedded_annotation_search MATCH ?{item_clause}
                ORDER BY score,item_id,attachment_id,page_number,embedded_annotation_id LIMIT ?
            """,args()):
                results.append({"kind":"embedded_annotation","label":"Embedded PDF annotation","item_id":r[0],"attachment_id":r[1],"page":r[2],"title":plain_text(r[3]) or "Untitled","snippet":safe_snippet(r[4] or r[5] or "Embedded annotation match"),"score":r[6]-0.2,"embedded_annotation_id":r[7]})
        if not zotero_empty and scope in {"all","metadata"}:
            for r in db.execute(f"""
                SELECT item_id,title,
                  snippet(item_search,3,'<mark>','</mark>',' … ',32),
                  bm25(item_search,0,8,4,2,3,3) score
                FROM item_search WHERE item_search MATCH ?{item_clause} ORDER BY score,item_id LIMIT ?
            """,args()):
                results.append({"kind":"metadata","label":"Title & metadata","item_id":r[0],"attachment_id":None,"page":None,"title":plain_text(r[1]) or "Untitled","snippet":safe_snippet(r[2] or "Metadata match"),"score":r[3]})
        if not zotero_empty and scope in {"all","notes"}:
            for r in db.execute(f"""
                SELECT note_id,item_id,title,snippet(note_search,3,'<mark>','</mark>',' … ',34),bm25(note_search) score,
                  EXISTS(SELECT 1 FROM notes n WHERE n.id=CAST(note_search.note_id AS INTEGER) AND n.note_html LIKE '%data-annotation%') generated
                FROM note_search WHERE note_search MATCH ?{item_clause} ORDER BY score,note_id LIMIT ?
            """,args()):
                generated=bool(r[5])
                results.append({"kind":"annotation_note" if generated else "note","label":"Zotero Note","generated_from_annotations":generated,"item_id":r[1],"attachment_id":None,"page":None,"title":plain_text(r[2]) or "Zotero Note","snippet":safe_snippet(r[3]),"score":r[4],"note_id":r[0]})
        item_ids=sorted({r["item_id"] for r in results if r.get("item_id")})
        context={}
        if item_ids:
            marks=",".join("?" for _ in item_ids)
            for row in db.execute(f"SELECT id,title,date_created,date_added,creators_json FROM items WHERE id IN ({marks})",item_ids):
                context[row[0]]={"title":plain_text(row[1]),"published":publication_date(row[2]),"date_added":row[3],"authors":creator_names(row[4])}
        for result in results:
            info=context.get(result.get("item_id"),{})
            if info.get("title"): result["title"]=info["title"]
            result.update({k:info.get(k) for k in ("published","date_added","authors")})
    finally:
        db.close()
    filters=search_filters(filters)
    allow_obsidian=scope in {"all","obsidian"} and OBSIDIAN_DB.is_file()
    selected_folders=filters.get("obsidian_folders")
    if allow_obsidian and selected_folders != []:
        obs=obsidian_connect()
        clauses=[]; values=[]
        if obsidian_filter:
            clauses.append("note_id=?"); values.append(obsidian_filter)
        if selected_folders is not None:
            clauses.append("note_id IN (SELECT note_id FROM obsidian_notes WHERE folder IN ("+",".join("?" for _ in selected_folders)+"))")
            values.extend(selected_folders)
        where=" AND "+" AND ".join(clauses) if clauses else ""
        for row in obs.execute(f"""SELECT note_id,section_id,title,heading,path,
          snippet(obsidian_search,7,'<mark>','</mark>',' … ',42),bm25(obsidian_search,0,0,4,2,1,1,1,6)
          FROM obsidian_search WHERE obsidian_search MATCH ?{where} ORDER BY 7 LIMIT ?""",
          (match,*values,fetch_limit)):
            note=obs.execute("SELECT folder,tags_json,authors_json,created_at,modified_at FROM obsidian_notes WHERE note_id=?",(row[0],)).fetchone()
            results.append({"kind":"obsidian","label":"Obsidian note","item_id":None,"attachment_id":None,"page":None,
              "obsidian_note_id":row[0],"obsidian_section_id":row[1],"title":plain_text(row[2]) or "Untitled note",
              "section_title":plain_text(row[3]),"vault_path":row[4],"snippet":safe_snippet(row[5]),"score":row[6]-0.1,
              "folder":note[0],"tags":json.loads(note[1] or "[]"),
              "authors":[{"name":name,"type":"author"} for name in json.loads(note[2] or "[]")],
              "created":note[3],"modified":note[4]})
        obs.close()
    results.sort(key=lambda x:(x["score"],x["kind"],x.get("item_id") or 0,str(x.get("page") or "")))
    if scope=="all" and item_filter is None and obsidian_filter is None:
        zotero_results=[result for result in results if result["kind"]!="obsidian"]
        obsidian_results=[result for result in results if result["kind"]=="obsidian"]
        if zotero_results and obsidian_results:
            blended=[]
            while zotero_results or obsidian_results:
                blended.extend(zotero_results[:5]); del zotero_results[:5]
                if obsidian_results: blended.append(obsidian_results.pop(0))
            results=blended
    has_more=len(results)>offset+limit
    return results[offset:offset+limit],has_more


def result_identity(source):
    return (source.get("kind"),source.get("item_id"),source.get("attachment_id"),source.get("page"),
      source.get("annotation_id"),source.get("embedded_annotation_id"),source.get("note_id"),
      source.get("obsidian_note_id"),source.get("obsidian_section_id"),source.get("section_title"))


def semantic_ready(embed_model=None):
    embed_model=selected_embed_model(embed_model); status=semantic_index_status(embed_model)
    if not status.get("ready"):
        raise RuntimeError(f"The semantic index for {embed_model} is not ready yet. Build that model's independent index first.")


def semantic_unavailable_reason(embed_model=None):
    """Explain why semantic retrieval cannot run, or return an empty string."""
    embed_model=selected_embed_model(embed_model)
    status=semantic_index_status(embed_model)
    if not status.get("ready"):
        return f"The semantic index for {embed_model} is not ready."
    try:
        ensure_embedding_ollama(timeout=5)
        names=ollama_names(EMBED_OLLAMA_BASE_URL,2)
    except Exception:
        return "The local embedding service is unavailable."
    if embed_model not in names:
        return f"The embedding model {embed_model} is not downloaded."
    try:
        installed_identity=ollama_model_digest(EMBED_OLLAMA_BASE_URL,embed_model,2)
    except Exception:
        installed_identity=""
    if installed_identity and status.get("model_identity")!=installed_identity:
        return f"The embedding model {embed_model} was updated. Rebuild its semantic index."
    return ""


def hybrid_search(query,scope="all",limit=40,offset=0,filters=None,embed_model=None):
    """Fuse lexical, literal-phrase, and dense ranks without comparing raw scores."""
    embed_model=selected_embed_model(embed_model); semantic_ready(embed_model)
    target=offset+limit+1
    candidate_limit=min(500,max(100,target*3))
    lexical,_=search_library(query,scope,candidate_limit,False,filters=filters)
    semantic=semantic_search(query,scope,candidate_limit,filters,embed_model)
    words=re.findall(r"\w+",query,flags=re.UNICODE)
    phrase=[]
    if 1 <= len(words) <= 6:
        phrase,_=search_library(query,scope,min(candidate_limit,160),True,filters=filters)
    rankings=((phrase,2.8,"exact phrase"),(lexical,1.35,"BM25"),(semantic,1.0,"embeddings"))
    fused={}
    for ranking,weight,label in rankings:
        for rank,source in enumerate(ranking,1):
            key=result_identity(source)
            state=fused.setdefault(key,{"source":dict(source),"rrf":0.0,"methods":[]})
            state["rrf"]+=weight/(40+rank)
            if label not in state["methods"]: state["methods"].append(label)
            if label=="exact phrase" or (source.get("_context") and not state["source"].get("_context")):
                preserved={key:value for key,value in state["source"].items() if key.startswith("_")}
                state["source"].update(source); state["source"].update(preserved)
                if source.get("_context"): state["source"]["_context"]=source["_context"]
    ordered=sorted(fused.values(),key=lambda value:value["rrf"],reverse=True)
    results=[]
    for state in ordered:
        source=state["source"]; source["score"]=-state["rrf"]
        source["retrieval_sources"]=state["methods"]
        results.append(source)
    return results[offset:offset+limit],len(results)>offset+limit


def search_by_method(query,scope="all",method="hybrid",limit=40,exact=False,offset=0,filters=None,embed_model=None):
    embed_model=selected_embed_model(embed_model)
    method=method if method in {"bm25","semantic","hybrid"} else "hybrid"
    if method=="bm25":
        results,has_more=search_library(query,scope,limit,exact,offset=offset,filters=filters)
        return results,has_more,"Exact phrase" if exact else "Traditional BM25"
    if method=="semantic":
        semantic_ready(embed_model); target=offset+limit+1
        candidates=semantic_search(query,scope,target,filters,embed_model)
        return candidates[offset:offset+limit],len(candidates)>offset+limit,"Semantic embeddings"
    results,has_more=hybrid_search(query,scope,limit,offset,filters,embed_model)
    return results,has_more,"Hybrid: exact phrase + BM25 + embeddings"


def hydrate_source_context(sources):
    library=connect(); clean=clean_connect() if CLEAN_DB.is_file() else None
    obsidian=obsidian_connect() if OBSIDIAN_DB.is_file() else None
    try:
        for source in sources:
            if source.get("_context"): continue
            context=""; kind=source.get("kind")
            if kind=="document" and source.get("attachment_id"):
                if clean:
                    page_match=re.search(r"\d+",str(source.get("page") or "")); page=int(page_match.group()) if page_match else 0
                    row=clean.execute("""SELECT text FROM clean_document_blocks WHERE attachment_id=?
                      ORDER BY CASE WHEN page_start<=? AND page_end>=? THEN 0 ELSE 1 END,abs(page_start-?) LIMIT 1""",
                      (source["attachment_id"],page,page,page)).fetchone()
                    if row: context=row[0] or ""
                if not context:
                    row=library.execute("SELECT text FROM document_pages WHERE attachment_id=? AND page_number=?",(source["attachment_id"],source.get("page"))).fetchone()
                    if row: context=row[0] or ""
            elif kind=="annotation" and source.get("annotation_id"):
                row=library.execute("SELECT text,comment FROM annotation_context WHERE annotation_id=?",(source["annotation_id"],)).fetchone()
                if row: context="\n\n".join(value for value in row if value)
            elif kind=="embedded_annotation" and source.get("embedded_annotation_id"):
                row=library.execute("SELECT text,comment FROM embedded_pdf_annotations WHERE id=?",(source["embedded_annotation_id"],)).fetchone()
                if row: context="\n\n".join(value for value in row if value)
            elif kind in {"note","annotation_note"} and source.get("note_id"):
                row=library.execute("SELECT note_html FROM notes WHERE id=?",(source["note_id"],)).fetchone()
                if row: context=plain_text(row[0])
            elif kind=="metadata" and source.get("item_id"):
                row=library.execute("SELECT title,date_created,metadata_json,creators_json FROM items WHERE id=?",(source["item_id"],)).fetchone()
                if row: context=plain_text(". ".join(str(value) for value in row if value))
            elif kind=="obsidian" and obsidian and source.get("obsidian_section_id"):
                row=obsidian.execute("SELECT plain_text FROM obsidian_sections WHERE section_id=?",(source["obsidian_section_id"],)).fetchone()
                if row: context=row[0] or ""
            source["_context"]=context or plain_text(source.get("snippet"))
    finally:
        library.close()
        if clean: clean.close()
        if obsidian: obsidian.close()
    return sources


def retrieve_ask_sources(question,scope,exact,filters,method="hybrid",embed_model=None,limit=10):
    sources,_,label=search_by_method(question,scope,method,limit,exact,0,filters,embed_model)
    return hydrate_source_context(sources),label


def public_ask_sources(sources):
    return [{key:value for key,value in source.items() if not key.startswith("_")} for source in sources]


def store_ask_retrieval(question,sources,method,embed_model=None,answer_mode="fast",answer_model=None):
    token=uuid.uuid4().hex
    now=time.time()
    with ASK_CACHE_LOCK:
        expired=[key for key,value in ASK_CACHE.items() if now-value["created"]>3600]
        for key in expired: ASK_CACHE.pop(key,None)
        while len(ASK_CACHE)>=20:
            oldest=min(ASK_CACHE,key=lambda key:ASK_CACHE[key]["created"])
            ASK_CACHE.pop(oldest,None)
        ASK_CACHE[token]={"created":now,"question":question,"sources":sources,"method":method,
          "embedding_model":selected_embed_model(embed_model),"answer_mode":"full" if answer_mode=="full" else "fast",
          "answer_model":selected_answer_model(answer_model)}
    return token


def get_ask_retrieval(token):
    with ASK_CACHE_LOCK:
        value=ASK_CACHE.get(token)
    if not value or time.time()-value["created"]>3600:
        raise ValueError("The retrieved evidence has expired. Run Ask Library again.")
    return value


def prepare_ask_answer(question,sources,answer_mode="fast",answer_model=None):
    answer_model=selected_answer_model(answer_model)
    maximum_context=answer_model_context(answer_model)
    full=answer_mode=="full"
    context_parts=[]; used_sources=[]; remaining=min(24000,max(3000,(maximum_context-2048)*2)) if not full else None
    for source in sources:
        passage=plain_text(source.get("_context") or source.get("snippet"))
        if not full: passage=passage[:3200]
        header=f"[Source {len(used_sources)+1}: {source['title']}, {source['label']}, page {source.get('page') or 'n/a'}]\n"
        if not full:
            available=remaining-len(header)
            if available<300: break
            passage=passage[:available]
        context_parts.append(header+passage)
        used_source=dict(source)
        used_source["citation_text"]=passage
        used_sources.append(used_source)
        if not full: remaining-=len(header)+len(passage)+2
    context="\n\n".join(context_parts)
    messages=[
      {"role":"system","content":"You answer questions about the user's personal research library. The passages were found by the user's selected local retrieval method. Use only the supplied library passages. Cite factual claims with source numbers such as [1] or [2]. Distinguish the user's annotations from ordinary document text when the labels say so. If the passages do not support an answer, say so plainly. Do not invent sources. Answer in the language of the question. Be concise and keep the answer under 350 words unless the user explicitly asks for a detailed treatment."},
      {"role":"user","content":f"Question: {question}\n\nLibrary passages:\n{context}"}]
    prompt_tokens_estimate=(sum(len(message["content"]) for message in messages)+1)//2
    num_ctx=min(16384,maximum_context)
    if full:
        # Choose only as much KV cache as this model and request need, and never
        # silently truncate Full-mode input.
        estimated_tokens=prompt_tokens_estimate+2048
        candidates=sorted({value for value in (4096,8192,16384,32768,65536,131072,262144,maximum_context) if value<=maximum_context})
        for candidate in candidates:
            if estimated_tokens<=candidate:
                num_ctx=candidate; break
        else:
            raise ValueError(
              f"Full answer needs approximately {estimated_tokens:,} tokens, exceeding {answer_model}'s {maximum_context:,}-token context. "
              "No source text was truncated. Use Fast answer or narrow the search filters."
            )
    return messages,used_sources,num_ctx,prompt_tokens_estimate


def compose_ask_answer(question,sources,answer_mode="fast",answer_model=None):
    messages,used_sources,num_ctx,_=prepare_ask_answer(question,sources,answer_mode,answer_model)
    return ollama_chat(messages,num_ctx=num_ctx,model=answer_model),used_sources


def evidence_text_units(text,max_chars=800):
    """Return exact, numbered text spans; the model selects IDs, never rewrites quotes."""
    text=str(text or "")
    units=[]
    for match in re.finditer(r"[^.!?;]+(?:[.!?;]+(?=\s|$)|$)",text):
        start,end=match.span()
        while start<end and text[start].isspace(): start+=1
        while end>start and text[end-1].isspace(): end-=1
        while end-start>max_chars:
            split=text.rfind(" ",start,start+max_chars)
            if split<=start: split=min(end,start+max_chars)
            units.append({"id":len(units)+1,"start":start,"end":split,"text":text[start:split]})
            start=split
            while start<end and text[start].isspace(): start+=1
        if end>start: units.append({"id":len(units)+1,"start":start,"end":end,"text":text[start:end]})
    if not units and text: units=[{"id":1,"start":0,"end":len(text),"text":text}]
    return units


def answer_citation_occurrences(answer,source_count):
    occurrences=[]
    for match in re.finditer(r"\[((?:\d+\s*,\s*)*\d+)\]",answer or ""):
        left=max(answer.rfind("\n",0,match.start()),answer.rfind(". ",0,match.start()),
          answer.rfind("? ",0,match.start()),answer.rfind("! ",0,match.start()))+1
        right_candidates=[value for value in (answer.find("\n",match.end()),answer.find(". ",match.end()),
          answer.find("? ",match.end()),answer.find("! ",match.end())) if value>=0]
        right=min(right_candidates) if right_candidates else len(answer)
        claim=re.sub(r"\[((?:\d+\s*,\s*)*\d+)\]","",answer[left:right]).strip(" \n-*•")
        for raw_number in match.group(1).split(","):
            source_number=int(raw_number.strip())
            if 1<=source_number<=source_count:
                occurrences.append({"occurrence":len(occurrences),"source":source_number,"claim":claim})
    return occurrences


def localize_answer_evidence(answer,sources,answer_model,preferred_num_ctx=None):
    occurrences=answer_citation_occurrences(answer,len(sources))
    if not occurrences: return []
    cited_numbers=sorted({entry["source"] for entry in occurrences})
    units_by_source={}
    source_blocks=[]
    for number in cited_numbers:
        source=sources[number-1]
        text=str(source.get("citation_text") or source.get("_context") or source.get("snippet") or "")
        units=evidence_text_units(text); units_by_source[number]=(text,units)
        rendered="\n".join(f"U{unit['id']}: {unit['text']}" for unit in units)
        source_blocks.append(f"SOURCE {number}: {source.get('title') or 'Library source'}\n{rendered}")
    claim_lines="\n".join(f"O{entry['occurrence']} | SOURCE {entry['source']} | CLAIM: {entry['claim']}" for entry in occurrences)
    messages=[
      {"role":"system","content":"You locate exact supporting evidence for claims in already-selected library sources. Each source is divided into immutable numbered units. Return JSON only. Never quote or rewrite source text. For every occurrence choose the smallest contiguous unit range that directly supports its claim. Use at most 3 units unless more are strictly necessary. If the source does not support the claim, use null for both unit IDs and confidence low."},
      {"role":"user","content":f"CITATION OCCURRENCES:\n{claim_lines}\n\nSOURCES:\n\n"+"\n\n".join(source_blocks)+
       "\n\nReturn exactly: {\"evidence\":[{\"occurrence\":0,\"source\":1,\"start_unit\":1,\"end_unit\":2,\"confidence\":\"high\"}]}"}]
    prompt_chars=sum(len(message["content"]) for message in messages)
    required=(prompt_chars+1)//2+2048; maximum=answer_model_context(answer_model)
    candidates=sorted({value for value in (4096,8192,16384,32768,65536,131072,262144,maximum) if value<=maximum})
    num_ctx=int(preferred_num_ctx) if preferred_num_ctx and required<=int(preferred_num_ctx)<=maximum else next((value for value in candidates if required<=value),None)
    if num_ctx is None: raise ValueError(f"Evidence localization needs approximately {required:,} tokens, exceeding {answer_model}'s context")
    raw=ollama_chat(messages,num_predict=max(480,min(2400,len(occurrences)*120)),num_ctx=num_ctx,model=answer_model)
    fenced=re.sub(r"^```(?:json)?\s*|\s*```$","",raw.strip(),flags=re.I)
    try: parsed=json.loads(fenced)
    except json.JSONDecodeError:
        match=re.search(r"\{.*\}",fenced,flags=re.S)
        if not match: raise ValueError("The answer model did not return valid evidence positions")
        parsed=json.loads(match.group(0))
    expected={(entry["occurrence"],entry["source"]):entry for entry in occurrences}; localized=[]
    for choice in parsed.get("evidence",[]):
        try:
            occurrence=int(choice.get("occurrence")); source_number=int(choice.get("source"))
            if (occurrence,source_number) not in expected: continue
            start_unit=int(choice.get("start_unit")); end_unit=int(choice.get("end_unit"))
            text,units=units_by_source[source_number]
            if not (1<=start_unit<=end_unit<=len(units)): continue
            start=units[start_unit-1]["start"]; end=units[end_unit-1]["end"]
            localized.append({"occurrence":occurrence,"source_index":source_number-1,"start":start,"end":end,
              "confidence":str(choice.get("confidence") or "medium").lower()})
        except (TypeError,ValueError,KeyError):
            continue
    return localized


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, value, status=200):
        body = json.dumps(value,ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def send_stream_event(self, value):
        try:
            self.wfile.write((json.dumps(value,ensure_ascii=False)+"\n").encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
            return False

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(url.query)
        if url.path == "/api/app-instance":
            return self.send_json({"app":"marginalia-local","install_id":INSTALL_ID})
        if url.path == "/api/setup/status":
            config=load_config()
            zotero_enabled=bool(config.get("zotero_path"))
            obsidian_enabled=bool(config.get("obsidian_path"))
            library_ready=(
                DB.is_file()
                and (not zotero_enabled or CLEAN_DB.is_file())
                and (not obsidian_enabled or OBSIDIAN_DB.is_file())
                and active_generation_matches(config)
            )
            try: ollama_ready=bool(ollama_names(OLLAMA_BASE_URL,1))
            except Exception: ollama_ready=False
            return self.send_json({
                "configured":sources_configured(),"library_ready":library_ready,
                "zotero_path":config.get("zotero_path",""),"obsidian_path":config.get("obsidian_path",""),
                "linked_attachment_base_path":config.get("linked_attachment_base_path",""),
                "tesseract_ready":bool(find_tesseract()),"ollama_ready":ollama_ready,
                "refresh":refresh_state(),
            })
        if url.path == "/api/setup/pick-folder":
            kind=params.get("kind",[""])[0]
            if kind not in {"zotero","obsidian","linked"}: return self.send_json({"error":"Invalid folder type"},400)
            try: return self.send_json({"path":choose_folder_via_helper(kind)})
            except Exception as exc:
                return self.send_json({"error":"Folder picker failed: "+str(exc)},500)
        if url.path == "/api/refresh/status":
            return self.send_json(refresh_state())
        if url.path == "/api/history":
            db=history_connect(); prune_history(db); db.commit(); rows=db.execute("SELECT * FROM search_history ORDER BY updated_at DESC").fetchall(); db.close()
            return self.send_json({"history":[history_summary(row) for row in rows]})
        if url.path.startswith("/api/history/"):
            history_id=url.path.rsplit("/",1)[-1]
            db=history_connect(); row=db.execute("SELECT * FROM search_history WHERE id=?",(history_id,)).fetchone(); db.close()
            if not row: return self.send_json({"error":"History entry not found"},404)
            try: snapshot=json.loads(row["snapshot_json"])
            except json.JSONDecodeError: return self.send_json({"error":"Saved history entry is damaged"},500)
            saved_generation=str(snapshot.get("_library_generation") or "")
            snapshot["_source_current"]=bool(saved_generation and saved_generation==current_generation_token())
            if not snapshot["_source_current"]:
                for source in snapshot.get("answer_sources", []):
                    if isinstance(source, dict):
                        for key in ("attachment_id", "annotation_id", "embedded_annotation_id", "note_id",
                                    "obsidian_note_id", "obsidian_section_id"):
                            source[key]=None
            return self.send_json({"entry":history_summary(row),"snapshot":snapshot,
              "source_current":snapshot["_source_current"]})
        if url.path == "/api/stats":
            if not DB.is_file():
                return self.send_json({"items":0,"pdfs":0,"pages":0,"annotations":0,"notes":0,"obsidian":{"notes":0}})
            db=connect()
            obsidian=obsidian_info()
            data={
                "items":db.execute("SELECT count(*) FROM items").fetchone()[0],
                "pdfs":db.execute("SELECT count(*) FROM extraction_status WHERE status IN ('ok','cache')").fetchone()[0],
                "pages":db.execute("SELECT count(*) FROM document_pages").fetchone()[0],
                "annotations":db.execute("SELECT count(*) FROM annotations").fetchone()[0] + db.execute("SELECT count(*) FROM embedded_pdf_annotations").fetchone()[0],
                "notes":db.execute("SELECT count(*) FROM notes").fetchone()[0],
                "llm_configured":True,
                "llm_model":OLLAMA_MODEL,
                "embedding_model":EMBED_MODEL,
                "embedding_models":[semantic_index_status(model) for model in EMBED_MODELS],
                "semantic_index":semantic_index_status(EMBED_MODEL),
                "obsidian":obsidian,
            }
            db.close(); return self.send_json(data)
        if url.path == "/api/llm-status":
            return self.send_json(ollama_status())
        if url.path == "/api/collections":
            if not DB.is_file(): return self.send_json({"collections":[],"unfiled_count":0})
            db=connect(); folders=[]
            for row in db.execute("""
                SELECT c.collection_id,c.name,c.parent_id,c.path,c.depth,
                  count(DISTINCT ic.item_id) item_count
                FROM zotero_collections c
                LEFT JOIN collection_descendants cd ON cd.ancestor_id=c.collection_id
                LEFT JOIN item_collections ic ON ic.collection_id=cd.descendant_id
                GROUP BY c.collection_id ORDER BY c.path COLLATE NOCASE
            """):
                folders.append(dict(row))
            unfiled=db.execute("SELECT count(*) FROM items WHERE id NOT IN (SELECT DISTINCT item_id FROM item_collections)").fetchone()[0]
            db.close(); return self.send_json({"collections":folders,"unfiled_count":unfiled})
        if url.path == "/api/obsidian/folders":
            if not OBSIDIAN_DB.is_file(): return self.send_json({"folders":[],"ready":False})
            db=obsidian_connect(); note_folders=[row[0] for row in db.execute("SELECT folder FROM obsidian_notes")]; paths={""}
            for folder in note_folders:
                parts=folder.split("/") if folder else []
                paths.update("/".join(parts[:index]) for index in range(1,len(parts)+1))
            folders=[]
            for path in sorted(paths,key=str.casefold):
                count=sum(1 for folder in note_folders if folder==path or (path and folder.startswith(path+"/"))) if path else sum(1 for folder in note_folders if not folder)
                folders.append({"path":path,"name":path.rsplit("/",1)[-1] if path else "Vault root","depth":path.count("/") if path else 0,"note_count":count})
            db.close(); return self.send_json({"folders":folders,"ready":True})
        if url.path == "/api/search":
            q=params.get("q",[""])[0]; scope=params.get("scope",["all"])[0]
            method=params.get("method",["hybrid"])[0].lower()
            if method not in {"bm25","semantic","hybrid"}: method="hybrid"
            requested_method=method; fallback_reason=""
            embed_model=selected_embed_model(params.get("embedding_model",[EMBED_MODEL])[0])
            exact=method=="bm25" and params.get("exact",["0"])[0] in {"1","true","yes"}
            if method!="bm25":
                fallback_reason=semantic_unavailable_reason(embed_model)
                if fallback_reason:
                    method="bm25"; exact=False
            generate_answer=params.get("generate_answer",["0"])[0] in {"1","true","yes"}
            answer_mode="full" if params.get("answer_mode",["fast"])[0]=="full" else "fast"
            requested_answer_model=params.get("answer_model",[OLLAMA_MODEL])[0]
            answer_model,generate_answer,answer_warning=answer_request_status(requested_answer_model,generate_answer)
            try: offset=max(0,int(params.get("offset",["0"])[0]))
            except ValueError: offset=0
            filters=search_filters({key:params.get(key,[None])[0] for key in ("collection_id","zotero_collections","obsidian_folders","published_from","published_to","added_from","added_to")})
            try:
                results,has_more,retrieval_method=search_by_method(q,scope,method,40,exact,offset,filters,embed_model)
            except Exception as exc:
                return self.send_json({"error":"Library retrieval could not be completed.","detail":str(exc),
                  "method":method,"embedding_model":embed_model},502)
            retrieval_token=None
            if generate_answer and offset==0 and results:
                selected=results if answer_mode=="full" else results[:10]
                answer_sources=hydrate_source_context([dict(source) for source in selected])
                retrieval_token=store_ask_retrieval(q,answer_sources,retrieval_method,embed_model,answer_mode,answer_model)
            return self.send_json({"query":q,"scope":scope,"method":method,"requested_method":requested_method,
              "fallback_reason":fallback_reason,"exact":exact,"offset":offset,
              "filters":filters,"results":public_ask_sources(results),"has_more":has_more,
              "generate_answer":generate_answer,"answer_warning":answer_warning,"retrieval_token":retrieval_token,
              "retrieval_method":retrieval_method,"model":answer_model,"embedding_model":embed_model,
              "answer_mode":answer_mode,"answer_source_count":len(answer_sources) if retrieval_token else 0})
        if url.path == "/api/search/publication":
            q=params.get("q",[""])[0]; scope=params.get("scope",["all"])[0]
            exact=params.get("exact",["0"])[0] in {"1","true","yes"}
            obsidian_note_id=plain_text(params.get("obsidian_note_id",[""])[0])
            item_id=None
            if not obsidian_note_id:
                try: item_id=int(params.get("item_id",["0"])[0])
                except ValueError: return self.send_json({"error":"Invalid source"},400)
            filters=search_filters({key:params.get(key,[None])[0] for key in ("zotero_collections","obsidian_folders","published_from","published_to","added_from","added_to")})
            results,_=search_library(q,scope,limit=5000,exact=exact,item_filter=item_id,obsidian_filter=obsidian_note_id or None,filters=filters)
            return self.send_json({"item_id":item_id,"obsidian_note_id":obsidian_note_id,"results":results,"count":len(results)})
        if url.path == "/api/author":
            name=plain_text(params.get("name",[""])[0])
            if not name: return self.send_json({"error":"Author name required"},400)
            db=connect(); papers=[]
            for row in db.execute("SELECT id,title,item_type,date_created,date_added,creators_json FROM items ORDER BY coalesce(date_created,'' ) DESC,title"):
                authors=creator_names(row[5])
                if any(a["name"].casefold()==name.casefold() for a in authors):
                    papers.append({"kind":"author_paper","label":"Publication","item_id":row[0],"attachment_id":None,"page":None,
                      "title":plain_text(row[1]) or "Untitled","snippet":"","published":publication_date(row[3]),"date_added":row[4],"authors":authors,"score":0})
            db.close(); return self.send_json({"author":name,"results":papers,"count":len(papers)})
        if url.path == "/api/context":
            def number(key):
                try: return int(params.get(key,["0"])[0])
                except ValueError: return 0
            attachment_id=number("attachment_id"); annotation_id=number("annotation_id")
            embedded_id=number("embedded_annotation_id"); note_id=number("note_id")
            obsidian_note_id=plain_text(params.get("obsidian_note_id",[""])[0])
            obsidian_section_id=plain_text(params.get("obsidian_section_id",[""])[0])
            page_raw=params.get("page",[""])[0]; page_match=re.search(r"\d+",page_raw)
            page_number=int(page_match.group()) if page_match else 0
            match_text=params.get("match_text",[""])[0]
            db=connect(); data={"page":page_raw,"reader_page":page_number,"page_text":"","annotation_text":"","comment":"","note_text":"","has_document_text":False}
            if obsidian_note_id and OBSIDIAN_DB.is_file():
                obs=obsidian_connect()
                row=obs.execute("SELECT note_id,title,relative_path,folder,markdown,tags_json,aliases_json,authors_json,created_at,modified_at FROM obsidian_notes WHERE note_id=?",(obsidian_note_id,)).fetchone()
                section=obs.execute("SELECT section_id,heading_path,start_line,end_line,plain_text FROM obsidian_sections WHERE section_id=? AND note_id=?",(obsidian_section_id,obsidian_note_id)).fetchone() if obsidian_section_id else None
                obs.close(); db.close()
                if not row: return self.send_json({"error":"Obsidian note not found"},404)
                return self.send_json({"kind":"obsidian","obsidian_note_id":row[0],"title":row[1],"vault_path":row[2],"folder":row[3],
                  "markdown":row[4],"tags":json.loads(row[5] or "[]"),"aliases":json.loads(row[6] or "[]"),
                  "authors":[{"name":name,"type":"author"} for name in json.loads(row[7] or "[]")],"created":row[8],"modified":row[9],
                  "matched_section":dict(section) if section else None})
            if annotation_id:
                row=db.execute("SELECT text,comment,page_label,attachment_id,position_json FROM annotation_context WHERE annotation_id=?",(annotation_id,)).fetchone()
                if row:
                    data.update({"annotation_text":row[0] or "","comment":row[1] or "","page":row[2] or page_raw}); attachment_id=attachment_id or row[3]
                    page_number=annotation_page(row[4],row[2]) or page_number
            if embedded_id:
                row=db.execute("SELECT text,comment,page_number,attachment_id FROM embedded_pdf_annotations WHERE id=?",(embedded_id,)).fetchone()
                if row:
                    data.update({"annotation_text":row[0] or "","comment":row[1] or "","page":row[2]}); attachment_id=attachment_id or row[3]; page_number=row[2]
            if note_id:
                row=db.execute("SELECT title,note_html FROM notes WHERE id=?",(note_id,)).fetchone()
                if row: data.update({"note_title":plain_text(row[0]),"note_text":plain_text(row[1])})
            if attachment_id:
                clean_status=clean_document_status(attachment_id)
                total_pages=(clean_status or {}).get("raw_pages") or db.execute("SELECT coalesce(max(page_number),0) FROM document_pages WHERE attachment_id=?",(attachment_id,)).fetchone()[0]
                status_row=db.execute("SELECT status FROM extraction_status WHERE attachment_id=?",(attachment_id,)).fetchone()
                extraction_source=("clean:"+clean_status["source"]) if clean_status else (status_row[0] if status_row else "")
                data.update({"has_document_text":bool(total_pages),"total_pages":total_pages,
                  "extraction_source":extraction_source,"clean_text":bool(clean_status)})
                if clean_status:
                    clean=clean_connect()
                    page_number=find_clean_text_page(clean,attachment_id,data["annotation_text"] or match_text,page_number)
                    clean.close()
                elif extraction_source=="cache":
                    located=find_text_page(db,attachment_id,data["annotation_text"] or match_text)
                    if located: page_number=located
                elif not (1 <= page_number <= total_pages):
                    page_number=find_text_page(db,attachment_id,data["annotation_text"] or match_text)
                if page_number:
                    row=db.execute("SELECT text FROM document_pages WHERE attachment_id=? AND page_number=?",(attachment_id,page_number)).fetchone()
                    if row: data["page_text"]=row[0] or ""
                data["reader_page"]=page_number
                row=db.execute("SELECT i.title,i.creators_json FROM attachments a JOIN items i ON i.id=a.item_id WHERE a.id=?",(attachment_id,)).fetchone()
                if row: data.update({"title":plain_text(row[0]),"authors":creator_names(row[1]),"attachment_id":attachment_id})
            db.close(); return self.send_json(data)
        if url.path == "/api/obsidian/asset":
            relative=urllib.parse.unquote(params.get("path",[""])[0]).replace("\\","/").lstrip("/")
            note_path=urllib.parse.unquote(params.get("note_path",[""])[0]).replace("\\","/").lstrip("/")
            if not relative or not OBSIDIAN_DB.is_file(): return self.send_error(404)
            asset,resolution=obsidian_asset_record(relative,note_path)
            if not asset: return self.send_error(409 if resolution=="ambiguous" else 404)
            obs=obsidian_connect(); vault_row=obs.execute("SELECT value FROM obsidian_import_info WHERE key='vault_path'").fetchone(); obs.close()
            if not vault_row: return self.send_error(404)
            vault=Path(vault_row[0]).resolve()
            path=(vault/asset[0]).resolve()
            if vault not in path.parents or not path.is_file(): return self.send_error(403)
            size=path.stat().st_size; self.send_response(200); self.send_header("Content-Type",asset[1] or "application/octet-stream")
            self.send_header("Content-Length",str(size)); self.end_headers()
            with path.open("rb") as file:
                while block:=file.read(1024*1024): self.wfile.write(block)
            return
        if url.path == "/api/document-pages":
            try:
                attachment_id=int(params.get("attachment_id",["0"])[0])
                start=max(1,int(params.get("start",["1"])[0]))
                limit=min(30,max(1,int(params.get("limit",["10"])[0])))
            except ValueError: return self.send_json({"error":"Invalid page request"},400)
            if CLEAN_DB.is_file():
                try:
                    cleaned=clean_document_pages(attachment_id,start,limit)
                    if cleaned is not None: return self.send_json(cleaned)
                except sqlite3.Error:
                    pass
            db=connect()
            total=db.execute("SELECT coalesce(max(page_number),0) FROM document_pages WHERE attachment_id=?",(attachment_id,)).fetchone()[0]
            pages=[dict(r) for r in db.execute("SELECT page_number,text,char_count FROM document_pages WHERE attachment_id=? AND page_number>=? ORDER BY page_number LIMIT ?",(attachment_id,start,limit))]
            db.close()
            return self.send_json({"attachment_id":attachment_id,"start":start,"total_pages":total,"pages":pages,
              "has_more":bool(pages and pages[-1]["page_number"]<total),"text_source":"raw"})
        if url.path.startswith("/api/item/"):
            try: iid=int(url.path.rsplit("/",1)[1])
            except ValueError: return self.send_json({"error":"Invalid item"},400)
            db=connect(); item=db.execute("SELECT * FROM app_items WHERE id=?",(iid,)).fetchone()
            if not item: db.close(); return self.send_json({"error":"Not found"},404)
            attachments=[dict(x) for x in db.execute("SELECT id,title,content_type,local_path,annotation_count FROM attachments WHERE item_id=?",(iid,))]
            data=dict(item); data["title"]=plain_text(data.get("title")); data["attachments"]=attachments
            db.close(); return self.send_json(data)
        if url.path.startswith("/api/file/"):
            try: aid=int(url.path.rsplit("/",1)[1])
            except ValueError: return self.send_error(400)
            db=connect(); row=db.execute("SELECT local_path FROM attachments WHERE id=?",(aid,)).fetchone(); db.close()
            if not row or not row[0]: return self.send_error(404)
            path=allowed_attachment_path(row[0])
            if not path: return self.send_error(403)
            size=path.stat().st_size; self.send_response(200)
            self.send_header("Content-Type",mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length",str(size)); self.send_header("Content-Disposition",f'inline; filename="{path.name}"')
            self.end_headers()
            with path.open("rb") as f:
                while chunk:=f.read(1024*1024): self.wfile.write(chunk)
            return
        if url.path == "/LICENSE":
            path=BASE / "LICENSE"
            if not path.is_file(): return self.send_error(404)
            body=path.read_bytes(); self.send_response(200)
            self.send_header("Content-Type","text/plain; charset=utf-8")
            self.send_header("Cache-Control","no-cache")
            self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
            return
        name = "index.html" if url.path == "/" else url.path.lstrip("/")
        path=(APP/name).resolve()
        if APP not in path.parents and path != APP: return self.send_error(403)
        if not path.is_file(): return self.send_error(404)
        body=path.read_bytes(); self.send_response(200)
        self.send_header("Content-Type",mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        if path.suffix.lower() in {".html",".js",".css"}: self.send_header("Cache-Control","no-cache")
        self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        allowed={"/api/ask","/api/ask/retrieve","/api/ask/compose","/api/history","/api/setup/configure","/api/refresh/start","/api/ai/install-embedding","/api/ai/install-answer","/api/ai/build-index","/api/ai/select-embedding","/api/ai/skip"}
        if self.path not in allowed: return self.send_error(404)
        try:
            body=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))))
        except Exception: return self.send_json({"error":"Invalid request"},400)
        if self.path=="/api/ai/install-embedding":
            model=str(body.get("model") or "")
            started,error=start_embedding_model_install(model)
            return self.send_json({"started":started,"install":model_install_state(),"error":error or None},202 if started else 409)
        if self.path=="/api/ai/install-answer":
            started,error=start_answer_model_install(body.get("model") or RECOMMENDED_ANSWER_MODEL)
            return self.send_json({"started":started,"install":model_install_state(),"error":error or None},202 if started else 409)
        if self.path=="/api/ai/build-index":
            model=str(body.get("model") or "")
            if not valid_model_name(model): return self.send_json({"error":"Invalid Ollama model name"},400)
            try:
                names=ollama_names(EMBED_OLLAMA_BASE_URL,2)
            except Exception:
                try: ensure_embedding_ollama(); names=ollama_names(EMBED_OLLAMA_BASE_URL,2)
                except Exception as exc: return self.send_json({"error":"Embedding service is unavailable","detail":str(exc)},409)
            if model not in names: return self.send_json({"error":"Install the embedding model first"},409)
            try:
                if not ollama_model_supports(model,"embedding",EMBED_OLLAMA_BASE_URL):
                    return self.send_json({"error":"The selected Ollama model does not support embeddings"},409)
            except Exception as exc:
                return self.send_json({"error":"Could not validate the embedding model","detail":str(exc)},409)
            started=start_semantic(model)
            if started:
                EMBED_MODELS[model]=semantic_model_index_path(model)
            if not started:
                current=refresh_state()
                detail=("Another refresh or index build is already running" if current.get("running")
                        else "Refresh the core library before building a semantic index")
                return self.send_json({"started":False,"error":detail,"refresh":current},409)
            return self.send_json({"started":True,"refresh":refresh_state()},202)
        if self.path=="/api/ai/select-embedding":
            model=str(body.get("model") or "").strip()
            status=semantic_index_status(model)
            if not status.get("ready"):
                return self.send_json({"error":"Build this model's semantic index before selecting it"},409)
            try:
                select_active_embedding_model(model)
            except (OSError,ValueError) as exc:
                return self.send_json({"error":"Could not save the selected model","detail":str(exc)},500)
            return self.send_json({"selected":model})
        if self.path=="/api/ai/skip":
            skipped=skip_semantic_setup()
            return self.send_json({"skipped":skipped},200 if skipped else 409)
        if self.path=="/api/setup/configure":
            if refresh_state().get("running"):
                return self.send_json({"error":"Wait for the current refresh to finish before changing source folders."},409)
            try:
                config=save_config(str(body.get("zotero_path") or ""),str(body.get("obsidian_path") or ""),
                  str(body.get("linked_attachment_base_path") or ""))
            except ValueError as exc:
                return self.send_json({"error":str(exc)},400)
            started=start_refresh() if body.get("start_refresh",True) else False
            return self.send_json({"configured":True,"config":config,"refresh_started":started})
        if self.path=="/api/refresh/start":
            if not sources_configured(): return self.send_json({"error":"Configure your source folders first"},400)
            started=start_refresh()
            return self.send_json({"started":started,"refresh":refresh_state()},202 if started else 409)
        if self.path=="/api/history":
            snapshot=body.get("snapshot") if isinstance(body.get("snapshot"),dict) else None
            if not snapshot or not str(snapshot.get("query") or "").strip(): return self.send_json({"error":"Invalid history snapshot"},400)
            snapshot=dict(snapshot); snapshot["_library_generation"]=current_generation_token()
            history_id=str(body.get("id") or uuid.uuid4().hex)
            now=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
            db=history_connect(); existing=db.execute("SELECT created_at FROM search_history WHERE id=?",(history_id,)).fetchone()
            results=snapshot.get("results") if isinstance(snapshot.get("results"),list) else []
            source_keys=set()
            for result in results:
                if isinstance(result,dict): source_keys.add((result.get("item_id"),result.get("obsidian_note_id"),result.get("title")))
            db.execute("""INSERT OR REPLACE INTO search_history
              (id,created_at,updated_at,query,scope,method,embedding_model,answer_model,exact,generate_answer,answer_mode,
               result_count,source_count,retrieval_seconds,answer_seconds,snapshot_json)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (history_id,existing[0] if existing else now,now,str(snapshot.get("query")),str(snapshot.get("scope") or "all"),
               str(snapshot.get("method") or "hybrid"),snapshot.get("embedding_model"),snapshot.get("answer_model"),
               int(bool(snapshot.get("exact"))),int(bool(snapshot.get("generate_answer"))),
               "full" if snapshot.get("answer_mode")=="full" else "fast",len(results),len(source_keys),
               snapshot.get("retrieval_seconds"),snapshot.get("answer_seconds"),json.dumps(snapshot,ensure_ascii=False)))
            prune_history(db)
            db.commit(); db.close(); return self.send_json({"id":history_id,"updated_at":now})
        if self.path=="/api/ask/compose":
            token=str(body.get("retrieval_token") or "").strip()
            if not token: return self.send_json({"error":"Missing retrieved evidence"},400)
            try:
                retrieval=get_ask_retrieval(token)
                if body.get("stream"):
                    answer_mode=retrieval.get("answer_mode","fast")
                    answer_model=retrieval.get("answer_model",OLLAMA_MODEL)
                    messages,sources,num_ctx,prompt_tokens_estimate=prepare_ask_answer(retrieval["question"],retrieval["sources"],answer_mode,answer_model)
                    self.send_response(200)
                    self.send_header("Content-Type","application/x-ndjson; charset=utf-8")
                    self.send_header("Cache-Control","no-cache, no-transform")
                    self.send_header("Connection","close")
                    self.end_headers(); self.close_connection=True
                    if not self.send_stream_event({"type":"meta","configured":True,"local":True,"model":answer_model,
                      "embedding_model":retrieval.get("embedding_model",EMBED_MODEL),"retrieval_method":retrieval["method"],
                      "retrieval_query":retrieval["question"],"sources":public_ask_sources(sources),
                      "queued":OLLAMA_CALL_LOCK.locked(),"answer_mode":answer_mode,"num_ctx":num_ctx,
                      "prompt_tokens_estimate":prompt_tokens_estimate}): return
                    stream=None
                    try:
                        if OLLAMA_CALL_LOCK.locked():
                            if not self.send_stream_event({"type":"status","message":"Another local model task is finishing…"}): return
                        stream=ollama_chat_stream(messages,num_ctx=num_ctx,model=answer_model,
                          prompt_tokens_estimate=prompt_tokens_estimate)
                        buffered=[]; buffered_chars=0; last_flush=time.monotonic(); answer_parts=[]
                        for stream_event in stream:
                            if stream_event["type"]!="token":
                                if buffered:
                                    if not self.send_stream_event({"type":"token","text":"".join(buffered)}): return
                                    buffered=[]; buffered_chars=0; last_flush=time.monotonic()
                                if not self.send_stream_event(stream_event): return
                                continue
                            text_chunk=stream_event["text"]
                            answer_parts.append(text_chunk)
                            buffered.append(text_chunk); buffered_chars+=len(text_chunk)
                            now=time.monotonic()
                            if buffered_chars>=96 or now-last_flush>=0.06:
                                if not self.send_stream_event({"type":"token","text":"".join(buffered)}): return
                                buffered=[]; buffered_chars=0; last_flush=now
                        if buffered and not self.send_stream_event({"type":"token","text":"".join(buffered)}): return
                        if not self.send_stream_event({"type":"evidence_status","message":f"{answer_model} finished writing. Locating the exact evidence behind each citation…"}): return
                        try:
                            evidence=localize_answer_evidence("".join(answer_parts),sources,answer_model,num_ctx)
                            if not self.send_stream_event({"type":"evidence","items":evidence,
                              "citation_count":len(answer_citation_occurrences("".join(answer_parts),len(sources)))}): return
                        except Exception as evidence_exc:
                            if not self.send_stream_event({"type":"evidence_error","detail":str(evidence_exc)}): return
                        self.send_stream_event({"type":"done"})
                    except Exception as exc:
                        self.send_stream_event({"type":"error","detail":str(exc)})
                    finally:
                        if stream is not None: stream.close()
                    return
                answer_mode=retrieval.get("answer_mode","fast")
                answer_model=retrieval.get("answer_model",OLLAMA_MODEL)
                answer,sources=compose_ask_answer(retrieval["question"],retrieval["sources"],answer_mode,answer_model)
                return self.send_json({"configured":True,"local":True,"model":answer_model,
                  "embedding_model":retrieval.get("embedding_model",EMBED_MODEL),"retrieval_method":retrieval["method"],
                  "retrieval_query":retrieval["question"],"answer":answer,"sources":public_ask_sources(sources),
                  "answer_mode":answer_mode})
            except urllib.error.HTTPError as exc:
                try: detail=exc.read().decode("utf-8","replace")
                except Exception: detail=str(exc)
                return self.send_json({"error":"The local Ollama model could not answer.","detail":detail,"model":OLLAMA_MODEL},502)
            except Exception as exc:
                return self.send_json({"error":"The local Ollama model could not answer. Make sure Ollama is running.","detail":str(exc),"model":OLLAMA_MODEL},502)

        question=body.get("question","").strip(); scope=body.get("scope","all")
        method=str(body.get("method","hybrid")).lower()
        if method not in {"bm25","semantic","hybrid"}: method="hybrid"
        fallback_reason=""
        answer_mode="full" if body.get("answer_mode")=="full" else "fast"
        answer_model,generate_answer,answer_warning=answer_request_status(
          body.get("answer_model"),bool(body.get("generate_answer",True)))
        embed_model=selected_embed_model(body.get("embedding_model"))
        exact=bool(body.get("exact",False)); filters=search_filters(body.get("filters"))
        if method!="bm25":
            exact=False
            fallback_reason=semantic_unavailable_reason(embed_model)
            if fallback_reason: method="bm25"
        if not question: return self.send_json({"error":"Enter a question"},400)
        try:
            sources,retrieval_method=retrieve_ask_sources(question,scope,exact,filters,method,embed_model,40 if answer_mode=="full" else 10)
        except Exception as exc:
            return self.send_json({"error":"Library retrieval could not be completed.","detail":str(exc),
              "method":method,"embedding_model":embed_model},502)
        if not sources:
            return self.send_json({"configured":True,"local":True,"model":answer_model,
              "embedding_model":embed_model,"retrieval_method":retrieval_method,"retrieval_query":question,
              "answer":"I could not find relevant passages in the selected part of your library. Try broader wording or different filters.",
              "sources":[],"generate_answer":False,"answer_warning":answer_warning,"fallback_reason":fallback_reason})
        token=store_ask_retrieval(question,sources,retrieval_method,embed_model,answer_mode,answer_model) if generate_answer else None
        retrieved={"configured":True,"local":True,"model":answer_model,"embedding_model":embed_model,
          "retrieval_method":retrieval_method,"retrieval_query":question,"retrieval_token":token,
          "sources":public_ask_sources(sources),"answer_mode":answer_mode,
          "generate_answer":generate_answer,"answer_warning":answer_warning,"fallback_reason":fallback_reason}
        if self.path=="/api/ask/retrieve": return self.send_json(retrieved)
        if not generate_answer:
            retrieved["answer"]=""
            return self.send_json(retrieved)
        try:
            answer,used_sources=compose_ask_answer(question,sources,answer_mode,answer_model)
            retrieved.update({"answer":answer,"sources":public_ask_sources(used_sources)})
            return self.send_json(retrieved)
        except urllib.error.HTTPError as exc:
            try: detail=exc.read().decode("utf-8","replace")
            except Exception: detail=str(exc)
            return self.send_json({"error":"The local Ollama model could not answer.","detail":detail,
              "model":OLLAMA_MODEL,"sources":public_ask_sources(sources)},502)
        except Exception as exc:
            return self.send_json({"error":"The local Ollama model could not answer. Make sure Ollama is running.",
              "detail":str(exc),"model":OLLAMA_MODEL,"sources":public_ask_sources(sources)},502)

    def do_DELETE(self):
        url=urllib.parse.urlparse(self.path)
        if url.path == "/api/ai/model":
            model=urllib.parse.parse_qs(url.query).get("model",[""])[0]
            if not valid_model_name(model): return self.send_json({"error":"Invalid Ollama model name"},400)
            if model==persisted_active_embedding_model():
                return self.send_json({"error":"Select another ready embedding model before removing this index"},409)
            if refresh_state().get("running"): return self.send_json({"error":"Wait for indexing to finish"},409)
            try:
                unregister_embedding_model(model,delete_files=True); EMBED_MODELS.pop(model,None)
            except (OSError,ValueError) as exc:
                return self.send_json({"error":str(exc)},400)
            return self.send_json({"removed":model,"ollama_model_kept":True})
        if url.path == "/api/history":
            db=history_connect(); db.execute("DELETE FROM search_history"); db.commit(); db.execute("VACUUM"); db.close()
            return self.send_json({"cleared":True})
        if url.path.startswith("/api/history/"):
            history_id=url.path.rsplit("/",1)[-1]
            db=history_connect(); cursor=db.execute("DELETE FROM search_history WHERE id=?",(history_id,)); db.commit(); deleted=cursor.rowcount; db.close()
            return self.send_json({"deleted":bool(deleted)},200 if deleted else 404)
        return self.send_error(404)


if __name__ == "__main__":
    try:
        recover_interrupted_publication()
    except Exception as exc:
        print(f"Refresh recovery needs attention: {exc}")
        report_recovery_failure(exc)
    initialize_history()
    try:
        server=SingleInstanceServer((HOST,PORT),Handler)
    except OSError:
        if matching_instance(PORT):
            print("This Marginalia installation is already running. Opening it.")
            if "--open" in sys.argv: webbrowser.open(f"http://{HOST}:{PORT}")
            raise SystemExit(0)
        # An unrelated local service owns the preferred port. Use a free one;
        # never open another Marginalia installation or the development app.
        server=SingleInstanceServer((HOST,0),Handler)
    active_port=server.server_port
    app_url=f"http://{HOST}:{active_port}"
    print(f"Library app: {app_url}")
    if refresh_resume_required() and sources_configured():
        print("Resuming the refresh interrupted during the previous session.")
        start_refresh()
    if "--open" in sys.argv:
        threading.Timer(.7,lambda:webbrowser.open(app_url)).start()
    preferences=ai_preferences()
    if preferences.get("answer_enabled"):
        threading.Thread(target=ensure_answer_ollama,daemon=True,name="answer-ollama-autostart").start()
    if semantic_index_status(persisted_active_embedding_model()).get("ready"):
        threading.Thread(target=warm_default_embedding_model,daemon=True,name="embedding-ollama-warmup").start()
    try:
        server.serve_forever()
    finally:
        release_bulk_embedding_ollama()
        release_answer_ollama()
        if EMBED_OLLAMA_PROCESS is not None and EMBED_OLLAMA_PROCESS.poll() is None:
            EMBED_OLLAMA_PROCESS.terminate()
