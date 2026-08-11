#!/bin/sh
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_ROOT="$SCRIPT_DIR/marginalia"
if [ ! -f "$APP_ROOT/launcher.py" ]; then
  APP_ROOT="$SCRIPT_DIR"
fi
cd "$APP_ROOT" || exit 1

VENV=".venv"
PYTHON="$VENV/bin/python"

wait_before_exit() {
  if [ -t 0 ]; then
    printf '\nPress Enter to close...'
    read -r _answer
  fi
}

fail() {
  printf '\n%s\n' "$1" >&2
  wait_before_exit
  exit 1
}

python_is_312() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1
}

find_python() {
  for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && python_is_312 "$candidate"; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

if [ ! -x "$PYTHON" ]; then
  printf 'Marginalia needs Python 3.12 on this computer.\n'
  printf 'The first start creates a private Python environment in:\n  %s/%s\n\n' "$APP_ROOT" "$VENV"
  BOOTSTRAP=$(find_python || true)
  if [ -z "$BOOTSTRAP" ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
      printf 'Python 3.12 was not found.\n'
      if command -v brew >/dev/null 2>&1; then
        printf 'Marginalia can install Python 3.12 and Tk through Homebrew.\n'
        printf 'Install them now? [y/N]: '
        read -r answer
        case "$answer" in
          y|Y|yes|YES)
            brew install python@3.12 python-tk@3.12 || fail 'Homebrew could not install Python 3.12.'
            BOOTSTRAP=$(find_python || true)
            ;;
        esac
      else
        printf 'Install Homebrew from https://brew.sh, then run:\n'
        printf '  brew install python@3.12 python-tk@3.12 tesseract\n'
      fi
    else
      printf 'Python 3.12 was not found. Install Python 3.12, its venv and Tk packages\n'
      printf 'with your Linux distribution package manager, then run this file again.\n'
      printf 'Tesseract is optional but required for OCR of scanned PDFs.\n'
    fi
    [ -n "$BOOTSTRAP" ] || fail 'Marginalia cannot continue without Python 3.12.'
  fi
  printf 'Creating Marginalia\047s local Python environment...\n'
  "$BOOTSTRAP" -m venv "$VENV" || fail 'The local Python environment could not be created. Install the Python venv package and try again.'
fi

python_is_312 "$PYTHON" || fail 'The existing .venv was created with a different Python version. Rename or remove it, then try again.'

if ! "$PYTHON" -c "import numpy, pymupdf; assert numpy.__version__ == '2.3.5'; assert pymupdf.__version__.startswith('1.28.2')" >/dev/null 2>&1; then
  printf '\nInstalling the pinned packages listed in requirements.txt:\n'
  cat requirements.txt
  printf '\nPackages are downloaded from the Python package index into .venv only.\n'
  "$PYTHON" -m pip install --disable-pip-version-check --only-binary=:all: -r requirements.txt || \
    fail "Marginalia's local environment could not be prepared. Check your internet connection and try again."
fi

if ! "$PYTHON" -c 'import tkinter' >/dev/null 2>&1; then
  fail 'Python Tk is missing. Install python3-tk on Linux or python-tk@3.12 with Homebrew, then try again.'
fi

if [ "${MARGINALIA_SETUP_ONLY:-0}" = "1" ]; then
  exit 0
fi

printf 'Starting Marginalia from auditable source files...\n'
printf 'Keep this window open while using the application.\n'
printf 'Press Ctrl+C here to stop Marginalia.\n\n'
"$PYTHON" launcher.py
status=$?
if [ "$status" -ne 0 ]; then
  printf '\nMarginalia stopped because of an error.\n' >&2
  printf 'Details are shown above and may also be available in startup_error.log.\n' >&2
  wait_before_exit
fi
exit "$status"
