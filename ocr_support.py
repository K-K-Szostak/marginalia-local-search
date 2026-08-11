from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MIN_TEXT_CHARS = 80
DEFAULT_RENDER_DPI = 300
AUTO_LANGUAGE_POLICY = "auto-v1"

LANGUAGE_ALIASES = {
    "ar": "ara", "arabic": "ara", "cs": "ces", "ces": "ces", "czech": "ces",
    "de": "deu", "deu": "deu", "german": "deu", "nl": "nld", "nld": "nld", "dutch": "nld",
    "en": "eng", "eng": "eng", "english": "eng", "es": "spa", "spa": "spa", "spanish": "spa",
    "fr": "fra", "fra": "fra", "fre": "fra", "french": "fra", "el": "ell", "ell": "ell", "greek": "ell",
    "it": "ita", "ita": "ita", "italian": "ita", "pl": "pol", "pol": "pol", "polish": "pol",
    "pt": "por", "por": "por", "portuguese": "por", "ru": "rus", "rus": "rus", "russian": "rus",
    "sk": "slk", "slk": "slk", "slovak": "slk", "uk": "ukr", "ukr": "ukr", "ukrainian": "ukr",
}

LANGUAGE_STOPWORDS = {
    "eng": {"and", "the", "of", "to", "in", "for", "with", "from", "on", "law", "legal", "article", "report"},
    "pol": {"oraz", "dla", "przez", "jest", "nie", "się", "w", "z", "na", "prawa", "prawny", "prawne", "raport"},
    "deu": {"und", "der", "die", "das", "von", "für", "mit", "auf", "recht", "bericht"},
    "fra": {"et", "le", "la", "les", "de", "des", "pour", "avec", "droit", "rapport"},
    "spa": {"y", "el", "la", "los", "de", "para", "con", "derecho", "informe"},
    "ita": {"e", "il", "la", "di", "per", "con", "diritto", "rapporto"},
    "nld": {"en", "de", "het", "van", "voor", "met", "recht", "rapport"},
    "por": {"e", "o", "a", "de", "para", "com", "direito", "relatório"},
}


@dataclass(frozen=True)
class LanguageSelection:
    languages: str
    detected: str | None
    missing: str | None
    mode: str


def find_tesseract() -> Path | None:
    """Locate Tesseract even when the current process has an older PATH."""
    configured = os.getenv("TESSERACT_CMD", "").strip()
    source_base = Path(__file__).resolve().parent
    packaged_base = Path(getattr(sys, "_MEIPASS", source_base))
    candidates = [
        Path(configured) if configured else None,
        source_base / "tools" / "tesseract" / "tesseract.exe",
        source_base / "runtime" / "tesseract" / "tesseract.exe",
        packaged_base / "runtime" / "tesseract" / "tesseract.exe",
        Path(shutil.which("tesseract") or "") if shutil.which("tesseract") else None,
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ]
    return next((path.resolve() for path in candidates if path and path.is_file()), None)


def available_languages(command: Path) -> set[str]:
    result = subprocess.run(
        [str(command), "--list-langs"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30, check=False,
    )
    if result.returncode:
        return set()
    return {
        line.strip() for line in result.stdout.splitlines()
        if line.strip() and not line.lower().startswith("list of available languages")
    }


def metadata_language_code(value: str) -> str | None:
    normalized = re.sub(r"[^a-z]+", "-", str(value or "").casefold()).strip("-")
    for part in normalized.split("-"):
        if part in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[part]
    return LANGUAGE_ALIASES.get(normalized)


def detected_language(metadata_language: str = "", sample_text: str = "") -> str | None:
    """Prefer Zotero's language field, then use conservative script/character signals."""
    explicit = metadata_language_code(metadata_language)
    if explicit:
        return explicit
    text = str(sample_text or "").casefold()
    if re.search(r"[ąćęłńóśźż]", text):
        return "pol"
    if re.search(r"[іїєґ]", text):
        return "ukr"
    if re.search(r"[а-яё]", text):
        return "rus"
    if re.search(r"[α-ωάέήίόύώϊϋΐΰ]", text):
        return "ell"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ara"
    if re.search(r"[äöüß]", text):
        return "deu"
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    scores = sorted(
        ((sum(word in vocabulary for word in words), language) for language, vocabulary in LANGUAGE_STOPWORDS.items()),
        reverse=True,
    )
    if scores and scores[0][0] >= 3 and (len(scores) == 1 or scores[0][0] >= scores[1][0] + 2):
        return scores[0][1]
    return None


def select_languages(
    installed: set[str], metadata_language: str = "", sample_text: str = "",
    requested: str | None = None,
) -> LanguageSelection:
    if not installed:
        raise RuntimeError("Tesseract has no installed OCR language data")
    configured = (requested if requested is not None else os.getenv("MARGINALIA_OCR_LANGUAGES", "auto")).strip()
    if configured and configured.casefold() != "auto":
        wanted = [part.strip() for part in configured.split("+") if part.strip()]
        selected = [language for language in wanted if language in installed]
        fallback = "eng" if "eng" in installed else sorted(installed)[0]
        return LanguageSelection("+".join(selected) or fallback, None, next((value for value in wanted if value not in installed), None), "manual")

    detected = detected_language(metadata_language, sample_text)
    fallback = "eng" if "eng" in installed else sorted(installed)[0]
    if not detected:
        return LanguageSelection(fallback, None, None, AUTO_LANGUAGE_POLICY)
    if detected not in installed:
        return LanguageSelection(fallback, detected, detected, AUTO_LANGUAGE_POLICY)
    languages = [detected]
    if detected != "eng" and "eng" in installed:
        languages.append("eng")
    return LanguageSelection("+".join(languages), detected, None, AUTO_LANGUAGE_POLICY)


def selected_languages(command: Path, metadata_language: str = "", sample_text: str = "") -> str:
    return select_languages(available_languages(command), metadata_language, sample_text).languages


def needs_ocr(text: str, minimum_chars: int = DEFAULT_MIN_TEXT_CHARS) -> bool:
    meaningful = sum(character.isalnum() for character in text)
    return meaningful < minimum_chars


def ocr_page(page, command: Path, languages: str, dpi: int = DEFAULT_RENDER_DPI) -> str:
    """Render and recognize one page in memory without changing its source PDF."""
    scale = max(1.0, float(dpi) / 72.0)
    pixmap = page.get_pixmap(matrix=(scale, scale), alpha=False)
    image = pixmap.tobytes("png")
    result = subprocess.run(
        [str(command), "stdin", "stdout", "-l", languages, "--psm", "3", "quiet"],
        input=image, capture_output=True, timeout=300, check=False,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"Tesseract exited with code {result.returncode}")
    return result.stdout.decode("utf-8", errors="replace").replace("\x00", "").strip()
