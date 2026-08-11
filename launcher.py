from __future__ import annotations

import importlib
import json
import runpy
import sys
import traceback
from pathlib import Path


ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def run_script(path: Path, arguments: list[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Marginalia component is missing: {path.name}")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(path.parent))
    sys.argv = [str(path), *arguments]
    runpy.run_path(str(path), run_name="__main__")


def pick_folder(kind: str, result_path: Path) -> None:
    payload = {}
    try:
        sys.path.insert(0, str(ROOT))
        source_manager = importlib.import_module("source_manager")
        payload["path"] = source_manager.choose_folder(kind)
    except Exception as exc:
        payload["error"] = str(exc)
    result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    if len(sys.argv) >= 4 and sys.argv[1] == "--pick-folder":
        pick_folder(sys.argv[2], Path(sys.argv[3]).resolve())
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-stage":
        run_script(ROOT / Path(sys.argv[2]).name, sys.argv[3:])
        return
    run_script(ROOT / "app" / "server.py", sys.argv[1:] or ["--open"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            (ROOT / "startup_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        raise
