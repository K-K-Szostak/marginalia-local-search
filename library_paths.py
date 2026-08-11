from __future__ import annotations

import json
import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent


def zotero_root() -> Path:
    """Return the external Zotero library root independently of Marginalia's location."""
    configured = os.getenv("ZOTERO_LIBRARY_DIR", "").strip()
    config_path = APP_ROOT / "zotero_source.json"
    if not configured and config_path.is_file():
        try:
            configured = str(json.loads(config_path.read_text(encoding="utf-8")).get("library_root", "")).strip()
        except (OSError, ValueError, TypeError):
            configured = ""
    if configured:
        return Path(configured).expanduser().resolve()

    # A standalone configuration with no Zotero source is intentional. Do not
    # fall back to a nearby personal Zotero database in that case.
    app_config = APP_ROOT / "app_config.json"
    if app_config.is_file():
        try:
            if not str(json.loads(app_config.read_text(encoding="utf-8")).get("zotero_path", "")).strip():
                return (APP_ROOT / "source_snapshots" / "zotero-disabled").resolve()
        except (OSError, ValueError, TypeError):
            pass

    candidates = (APP_ROOT.parent / "Zotero", APP_ROOT.parent)
    for candidate in candidates:
        if (candidate / "zotero.sqlite").is_file():
            return candidate.resolve()
    return candidates[0].resolve()
