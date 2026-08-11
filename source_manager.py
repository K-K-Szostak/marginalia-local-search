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
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "app_config.json"
SNAPSHOT_ROOT = ROOT / "source_snapshots"
SNAPSHOT_GENERATIONS = SNAPSHOT_ROOT / "generations"
OBSIDIAN_IGNORED_DIRECTORIES = {".obsidian", ".trash", ".git", "__pycache__", "do not read"}


def filesystem_path(path: Path) -> str:
    """Return a Windows extended path so snapshot nesting cannot hit MAX_PATH."""
    value = str(Path(path).absolute())
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value.lstrip("\\")
    return "\\\\?\\" + value


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _validate_source_boundary(path: Path | None, label: str) -> None:
    if path and _paths_overlap(path, SNAPSHOT_ROOT):
        raise ValueError(f"{label} cannot contain Marginalia's application or managed data folder")


def discover_zotero_linked_base() -> Path | None:
    """Read Zotero's configured Linked Attachment Base Directory when available."""
    profiles = Path(os.getenv("APPDATA", "")) / "Zotero" / "Zotero" / "Profiles"
    try:
        available = profiles.is_dir()
    except OSError:
        available = False
    if not available:
        return None
    pattern = re.compile(r'user_pref\("extensions\.zotero\.baseAttachmentPath",\s*"((?:\\.|[^"\\])*)"\)')
    try:
        preference_files = list(profiles.glob("*/prefs.js"))
    except OSError:
        return None
    def modified(path):
        try: return path.stat().st_mtime
        except OSError: return 0
    for prefs in sorted(preference_files, key=modified, reverse=True):
        try:
            match = pattern.search(prefs.read_text(encoding="utf-8", errors="replace"))
            if match:
                candidate = Path(json.loads('"' + match.group(1) + '"')).expanduser().resolve()
                if candidate.is_dir():
                    return candidate
        except (OSError, ValueError):
            continue
    return None


def save_config(zotero_path: str = "", obsidian_path: str = "",
                linked_attachment_base_path: str = "") -> dict:
    zotero = Path(zotero_path).expanduser().resolve() if zotero_path else None
    obsidian = Path(obsidian_path).expanduser().resolve() if obsidian_path else None
    if not zotero and not obsidian:
        raise ValueError("Choose at least one source: Zotero or Obsidian")
    if zotero and not (zotero / "zotero.sqlite").is_file():
        raise ValueError("The selected Zotero folder does not contain zotero.sqlite")
    if obsidian and not obsidian.is_dir():
        raise ValueError("The selected Obsidian vault does not exist")
    linked_base = Path(linked_attachment_base_path).expanduser().resolve() if linked_attachment_base_path else None
    if linked_base and not linked_base.is_dir():
        raise ValueError("The selected Zotero linked-attachment base directory does not exist")
    _validate_source_boundary(zotero, "The Zotero source")
    _validate_source_boundary(obsidian, "The Obsidian vault")
    value = {
        "zotero_path": str(zotero) if zotero else "",
        "obsidian_path": str(obsidian) if obsidian else "",
        "linked_attachment_base_path": str(linked_base) if linked_base else "",
    }
    temporary = CONFIG_PATH.with_suffix(".json.new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, CONFIG_PATH)
    return value


def configured(value: dict | None = None) -> bool:
    value = value if value is not None else load_config()
    zotero = value.get("zotero_path", "")
    obsidian = value.get("obsidian_path", "")
    zotero_valid = bool(zotero and (Path(zotero) / "zotero.sqlite").is_file())
    obsidian_valid = bool(obsidian and Path(obsidian).is_dir())
    requested = bool(zotero or obsidian)
    try:
        if zotero: _validate_source_boundary(Path(zotero), "The Zotero source")
        if obsidian: _validate_source_boundary(Path(obsidian), "The Obsidian vault")
    except ValueError:
        return False
    return requested and (not zotero or zotero_valid) and (not obsidian or obsidian_valid)


def _same_file(source: Path, target: Path) -> bool:
    if not os.path.isfile(filesystem_path(target)):
        return False
    left, right = os.stat(filesystem_path(source)), os.stat(filesystem_path(target))
    return left.st_size == right.st_size and left.st_mtime_ns == right.st_mtime_ns


def source_files(source: Path, ignored_directories=None) -> list[Path]:
    ignored = {str(name).casefold() for name in (ignored_directories or set())}
    files = []
    for directory, names, filenames in os.walk(source, topdown=True, followlinks=False):
        names[:] = sorted(name for name in names if name.casefold() not in ignored)
        base = Path(directory)
        files.extend(base / name for name in sorted(filenames) if (base / name).is_file() and not (base / name).is_symlink())
    return files


def _seed_tree(source: Path, target: Path) -> None:
    """Seed a new immutable generation cheaply without modifying the old one."""
    if not source.is_dir():
        return
    for directory, names, filenames in os.walk(source, topdown=True, followlinks=False):
        relative = Path(directory).relative_to(source)
        destination_directory = target / relative
        os.makedirs(filesystem_path(destination_directory), exist_ok=True)
        names[:] = sorted(names)
        for name in sorted(filenames):
            old = Path(directory) / name
            new = destination_directory / name
            try:
                os.link(filesystem_path(old), filesystem_path(new))
            except OSError:
                shutil.copy2(filesystem_path(old), filesystem_path(new))


def _previous_generation(config: dict) -> Path | None:
    manifest_path = ROOT / "library_generation.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = Path(str(manifest.get("snapshot_root") or "")).resolve()
    except (OSError, ValueError, TypeError):
        return None
    if not candidate.is_dir() or candidate.parent != SNAPSHOT_GENERATIONS.resolve():
        return None
    if any(str(manifest.get(key) or "") != str(config.get(key) or "") for key in ("zotero_path", "obsidian_path", "linked_attachment_base_path")):
        return None
    return candidate


def mirror_tree(source: Path, target: Path, progress=None, label="files", ignored_directories=None) -> dict:
    source = source.resolve()
    target.mkdir(parents=True, exist_ok=True)
    files = source_files(source, ignored_directories)
    source_relatives = {path.relative_to(source) for path in files}
    copied = reused = deleted = skipped = 0
    for number, path in enumerate(files, 1):
        relative = path.relative_to(source)
        destination = target / relative
        os.makedirs(filesystem_path(destination.parent), exist_ok=True)
        try:
            if _same_file(path, destination):
                reused += 1
                action = "Checked"
            else:
                temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
                shutil.copy2(filesystem_path(path), filesystem_path(temporary))
                os.replace(filesystem_path(temporary), filesystem_path(destination))
                copied += 1
                action = "Copied"
        except FileNotFoundError:
            # Zotero may atomically replace or remove .zotero-ft-cache files
            # while its storage tree is being enumerated.
            if path.exists():
                raise
            source_relatives.discard(relative)
            skipped += 1
            if progress:
                progress(f"Skipped changing {label}: {relative.as_posix()}", number, len(files))
            continue
        if progress:
            progress(f"{action} {label}: {relative.as_posix()}", number, len(files))
    stale_files = sorted((path for path in target.rglob("*") if path.is_file() and path.relative_to(target) not in source_relatives), reverse=True)
    for stale_number, path in enumerate(stale_files, 1):
        if path.relative_to(target) not in source_relatives:
            relative = path.relative_to(target)
            os.unlink(filesystem_path(path))
            deleted += 1
            if progress:
                progress(f"Removed obsolete {label}: {relative.as_posix()}", stale_number, len(stale_files))
    for directory in sorted((path for path in target.rglob("*") if path.is_dir()), reverse=True):
        try:
            os.rmdir(filesystem_path(directory))
        except OSError:
            pass
    if progress:
        progress(
            f"Finished {label}: {copied:,} copied · {reused:,} unchanged · {deleted:,} removed",
            len(files), len(files),
        )
    return {"copied": copied, "reused": reused, "skipped": skipped, "deleted": deleted, "total": len(files)}


def reserve_snapshot_generation() -> Path:
    generation_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    generation_root = (SNAPSHOT_GENERATIONS / generation_id).resolve()
    generation_root.mkdir(parents=True, exist_ok=False)
    return generation_root


def snapshot_sources(progress=None, generation_root: Path | None = None, config: dict | None = None) -> dict:
    config = dict(config if config is not None else load_config())
    if not configured(config):
        raise ValueError("Choose at least one valid source before refreshing")
    generation_root = Path(generation_root).resolve() if generation_root else reserve_snapshot_generation()
    if generation_root.parent != SNAPSHOT_GENERATIONS.resolve():
        raise ValueError("Snapshot generation must be inside the managed generations folder")
    generation_root.mkdir(parents=True, exist_ok=True)
    generation_id = generation_root.name
    previous = _previous_generation(config)
    empty = {"copied": 0, "reused": 0, "skipped": 0, "deleted": 0, "total": 0}
    zotero_files = dict(empty)
    zotero_source_value = config.get("zotero_path", "")
    if zotero_source_value:
        zotero_source = Path(zotero_source_value).resolve()
        zotero_snapshot = generation_root / "zotero"
        zotero_snapshot.mkdir(parents=True, exist_ok=True)
        if previous:
            _seed_tree(previous / "zotero" / "storage", zotero_snapshot / "storage")
            _seed_tree(previous / "zotero" / "linked_attachments", zotero_snapshot / "linked_attachments")

        source_database = zotero_source / "zotero.sqlite"
        database_megabytes = source_database.stat().st_size / (1024 * 1024)
        if progress:
            progress(f"Preparing zotero.sqlite ({database_megabytes:,.1f} MB)", 0, 0)
        temporary = zotero_snapshot / "zotero.sqlite.new"
        if temporary.exists():
            temporary.unlink()
        source_db = sqlite3.connect(f"file:{source_database.as_posix()}?mode=ro", uri=True)
        destination_db = sqlite3.connect(temporary)
        try:
            lock_wait_started = None

            def database_progress(status, remaining, total):
                nonlocal lock_wait_started
                if progress:
                    if status in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                        lock_wait_started = lock_wait_started or time.monotonic()
                        waited = round(time.monotonic() - lock_wait_started)
                        progress(
                            f"Zotero is actively locking zotero.sqlite · waiting {waited}s. "
                            f"Close Zotero now so setup can continue.", 0, 0, "zotero_locked",
                        )
                        return
                    lock_wait_started = None
                    copied = max(0, total - remaining)
                    fraction = copied / total if total else 0
                    copied_megabytes = database_megabytes * fraction
                    percent = round(100 * fraction)
                    progress(
                        f"Copying zotero.sqlite: {copied:,} of {total:,} pages · "
                        f"{copied_megabytes:,.1f} of {database_megabytes:,.1f} MB · {percent}%",
                        copied, total, "zotero_copying",
                    )

            source_db.backup(destination_db, pages=512, progress=database_progress)
        finally:
            destination_db.close()
            source_db.close()
        os.replace(temporary, zotero_snapshot / "zotero.sqlite")
        if progress:
            progress(f"Finished zotero.sqlite: {database_megabytes:,.1f} MB safely copied", 1, 1)

        storage = zotero_source / "storage"
        zotero_files = mirror_tree(storage, zotero_snapshot / "storage", progress, "Zotero attachments") if storage.is_dir() else dict(empty)

        # Linked Zotero attachments live outside storage. Copy absolute linked
        # files into the immutable generation and record only their managed
        # locations, so later pipeline stages never reopen the originals.
        linked_map = {}
        snapshot_db = sqlite3.connect(f"file:{(zotero_snapshot / 'zotero.sqlite').as_posix()}?mode=ro", uri=True)
        try:
            linked_rows = snapshot_db.execute("""
                SELECT a.itemID,i.key,a.path,a.linkMode
                FROM itemAttachments a JOIN items i ON i.itemID=a.itemID
                WHERE a.linkMode IN (2,3) AND coalesce(a.path,'')!=''
                ORDER BY a.itemID
            """).fetchall()
        finally:
            snapshot_db.close()
        linked_base_value = str(config.get("linked_attachment_base_path") or "").strip()
        linked_base = Path(linked_base_value).resolve() if linked_base_value else discover_zotero_linked_base()
        linked_relatives = set()
        for linked_number, (item_id, key, stored_path, _) in enumerate(linked_rows, 1):
            stored_path = str(stored_path)
            candidate = Path(stored_path)
            if stored_path.startswith("attachments:") and linked_base:
                candidate = (linked_base / stored_path[len("attachments:"):].replace("/", os.sep)).resolve()
                if candidate != linked_base and linked_base not in candidate.parents:
                    if progress:
                        progress(f"Rejected unsafe linked attachment path: {stored_path}", linked_number, len(linked_rows))
                    continue
            if not candidate.is_absolute() or not candidate.is_file():
                if progress:
                    progress(f"Linked attachment unavailable for private copy: {stored_path}", linked_number, len(linked_rows))
                continue
            destination = zotero_snapshot / "linked_attachments" / str(key) / candidate.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_link = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
            try:
                if not _same_file(candidate, destination):
                    shutil.copy2(filesystem_path(candidate), filesystem_path(temporary_link))
                    os.replace(filesystem_path(temporary_link), filesystem_path(destination))
            except FileNotFoundError:
                temporary_link.unlink(missing_ok=True)
                if not candidate.exists():
                    if progress:
                        progress(f"Linked attachment changed during private copy: {stored_path}", linked_number, len(linked_rows))
                    continue
                raise
            linked_map[str(item_id)] = destination.relative_to(zotero_snapshot).as_posix()
            linked_relatives.add(destination.relative_to(zotero_snapshot / "linked_attachments"))
            if progress:
                progress(f"Copied linked Zotero attachment: {candidate.name}", linked_number, len(linked_rows))
        linked_root = zotero_snapshot / "linked_attachments"
        if linked_root.is_dir():
            for old in sorted((path for path in linked_root.rglob("*") if path.is_file()), reverse=True):
                if old.relative_to(linked_root) not in linked_relatives:
                    old.unlink(missing_ok=True)
            for directory in sorted((path for path in linked_root.rglob("*") if path.is_dir()), reverse=True):
                try: directory.rmdir()
                except OSError: pass
        linked_map_path = zotero_snapshot / "linked_attachment_map.json"
        linked_map_path.write_text(json.dumps(linked_map, ensure_ascii=False, indent=2), encoding="utf-8")

    obsidian_files = dict(empty)
    obsidian_source_value = config.get("obsidian_path", "")
    if obsidian_source_value:
        obsidian_source = Path(obsidian_source_value).resolve()
        obsidian_snapshot = generation_root / "obsidian"
        if previous:
            _seed_tree(previous / "obsidian", obsidian_snapshot)
        obsidian_files = mirror_tree(
            obsidian_source, obsidian_snapshot, progress, "Obsidian files",
            ignored_directories=OBSIDIAN_IGNORED_DIRECTORIES,
        )
    return {
        "generation_id": generation_id, "snapshot_root": str(generation_root),
        "zotero_root": str(generation_root / "zotero") if zotero_source_value else "",
        "obsidian_root": str(generation_root / "obsidian") if obsidian_source_value else "",
        "zotero": zotero_files, "obsidian": obsidian_files,
    }


def publish_source_pointers(snapshot: dict) -> None:
    """Publish small compatibility pointers only after the databases are active."""
    values = {
        ROOT / "zotero_source.json": ({"library_root": snapshot.get("zotero_root")} if snapshot.get("zotero_root") else None),
        ROOT / "obsidian_source.json": ({"vault_path": snapshot.get("obsidian_root")} if snapshot.get("obsidian_root") else None),
    }
    for path, value in values.items():
        if value is None:
            path.unlink(missing_ok=True)
            continue
        temporary = path.with_suffix(".json.new")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)


def cleanup_snapshot_generations(current_root: str) -> None:
    """Remove generations no longer referenced by the active databases."""
    root = SNAPSHOT_GENERATIONS.resolve()
    current = Path(current_root).resolve()
    if not root.is_dir() or current.parent != root:
        return
    for candidate in root.iterdir():
        resolved = candidate.resolve()
        if resolved != current and resolved.parent == root and resolved.is_dir():
            shutil.rmtree(resolved)
    for legacy_name in ("zotero", "obsidian"):
        legacy = (SNAPSHOT_ROOT / legacy_name).resolve()
        if legacy.parent == SNAPSHOT_ROOT.resolve() and legacy.is_dir():
            shutil.rmtree(legacy)


def discard_snapshot_generation(snapshot: dict | None) -> None:
    if not snapshot or not snapshot.get("snapshot_root"):
        return
    root = SNAPSHOT_GENERATIONS.resolve()
    candidate = Path(snapshot["snapshot_root"]).resolve()
    if candidate.parent == root and candidate.is_dir():
        shutil.rmtree(candidate)


def choose_folder(kind: str) -> str:
    import tkinter as tk
    from tkinter import filedialog

    config = load_config()
    config_key = "linked_attachment_base_path" if kind == "linked" else f"{kind}_path"
    initial = config.get(config_key, "")
    if not initial or not Path(initial).is_dir():
        initial = str(Path.home())
    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        root.update_idletasks()
        titles = {
            "zotero": "Choose your Zotero data folder",
            "obsidian": "Choose your Obsidian vault",
            "linked": "Choose Zotero's Linked Attachment Base Directory",
        }
        title = titles.get(kind, "Choose a folder")
        selected = filedialog.askdirectory(
            parent=root, title=title, initialdir=initial, mustexist=True
        )
        return selected or ""
    finally:
        root.destroy()


def choose_folder_via_helper(kind: str) -> str:
    """Run Tk in a short-lived main-thread process, as required on Windows."""
    result_path = ROOT / f".folder-picker-{uuid.uuid4().hex}.json"
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--pick-folder", kind, str(result_path)]
    else:
        command = [sys.executable, str(ROOT / "launcher.py"), "--pick-folder", kind, str(result_path)]
    try:
        completed = subprocess.run(
            command, cwd=ROOT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        if not result_path.is_file():
            raise RuntimeError(f"Folder picker closed unexpectedly (exit code {completed.returncode})")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        return str(payload.get("path") or "")
    finally:
        result_path.unlink(missing_ok=True)
