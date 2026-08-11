from __future__ import annotations

import sys


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except (AttributeError, ValueError):
    pass


def progress(activity: object, current: int = 0, total: int = 0) -> None:
    """Emit a machine-readable, single-line update for the refresh manager."""
    text = " ".join(str(activity or "Working…").replace("\t", " ").splitlines()).strip()
    print(f"PROGRESS\t{text[:300]}\t{max(0, int(current))}\t{max(0, int(total))}", flush=True)
