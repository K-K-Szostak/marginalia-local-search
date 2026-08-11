@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Marginalia
cd /d "%~dp0"

set "MARGINALIA_VENV=.venv"
set "MARGINALIA_PYTHON=%MARGINALIA_VENV%\Scripts\python.exe"

if not exist "%MARGINALIA_PYTHON%" (
  echo Marginalia needs Python 3.12 on this computer.
  echo The first start creates a private Python environment in:
  rem Delayed expansion keeps parentheses and other path characters from
  rem being parsed as batch syntax inside this parenthesized block.
  echo   !CD!\%MARGINALIA_VENV%
  echo.
  set "MARGINALIA_BOOTSTRAP="
  set "MARGINALIA_BOOTSTRAP_ARGS="
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
    if not errorlevel 1 (
      set "MARGINALIA_BOOTSTRAP=py"
      set "MARGINALIA_BOOTSTRAP_ARGS=-3.12"
    )
  )
  if not defined MARGINALIA_BOOTSTRAP (
    where python >nul 2>nul
    if not errorlevel 1 (
      python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
      if not errorlevel 1 set "MARGINALIA_BOOTSTRAP=python"
    )
  )
  if not defined MARGINALIA_BOOTSTRAP (
    echo.
    echo Python 3.12 or the Windows Python launcher was not found.
    where winget >nul 2>nul
    if errorlevel 1 (
      echo Windows Package Manager ^(winget^) is not available.
      echo Install Python 3.12 manually from:
      echo   https://www.python.org/downloads/windows/
      pause
      exit /b 1
    )
    echo.
    echo Marginalia can now download and install the official Python 3.12 package
    echo using Windows Package Manager ^(winget^).
    choice /C YN /N /M "Install Python 3.12 now? [Y/N]: "
    if errorlevel 2 (
      echo Python installation skipped. Nothing was installed.
      pause
      exit /b 1
    )
    echo.
    echo Installing Python 3.12 after your confirmation...
    winget install --id Python.Python.3.12 --exact --source winget --scope user --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
      echo.
      echo Python installation did not complete successfully.
      echo You can install it manually from https://www.python.org/downloads/windows/
      pause
      exit /b 1
    )
    if exist "!LOCALAPPDATA!\Programs\Python\Python312\python.exe" set "MARGINALIA_BOOTSTRAP=!LOCALAPPDATA!\Programs\Python\Python312\python.exe"
    if not defined MARGINALIA_BOOTSTRAP if exist "!ProgramFiles!\Python312\python.exe" set "MARGINALIA_BOOTSTRAP=!ProgramFiles!\Python312\python.exe"
    if not defined MARGINALIA_BOOTSTRAP (
      where py >nul 2>nul
      if not errorlevel 1 (
        py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
        if not errorlevel 1 (
          set "MARGINALIA_BOOTSTRAP=py"
          set "MARGINALIA_BOOTSTRAP_ARGS=-3.12"
        )
      )
    )
    if not defined MARGINALIA_BOOTSTRAP (
      echo.
      echo Python was installed, but this window cannot locate it yet.
      echo Close this window and run Start Marginalia.cmd again.
      pause
      exit /b 1
    )
  )
  echo Creating Marginalia's local Python environment...
  "!MARGINALIA_BOOTSTRAP!" !MARGINALIA_BOOTSTRAP_ARGS! -m venv "%MARGINALIA_VENV%"
  if errorlevel 1 (
    echo.
    echo Marginalia's local environment could not be created.
    pause
    exit /b 1
  )
)

"%MARGINALIA_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
if errorlevel 1 (
  echo.
  echo The existing .venv was created with a different Python version.
  echo Rename or remove the .venv folder, then run Start Marginalia.cmd again.
  pause
  exit /b 1
)

"%MARGINALIA_PYTHON%" -c "import numpy, pymupdf; assert numpy.__version__ == '2.3.5'; assert pymupdf.__version__.startswith('1.28.2')" >nul 2>nul
if errorlevel 1 (
  echo.
  echo Installing the pinned packages listed in requirements.txt:
  type requirements.txt
  echo.
  echo Packages are downloaded from the Python package index into .venv only.
  "%MARGINALIA_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Marginalia's local environment could not be prepared.
    echo Check the messages above and your internet connection, then run this file again.
    pause
    exit /b 1
  )
)

if /i "%MARGINALIA_SETUP_ONLY%"=="1" exit /b 0
echo Starting Marginalia from auditable source files...
echo Keep this window open while using the application.
echo Press Ctrl+C here to stop Marginalia.
echo.
"%MARGINALIA_PYTHON%" launcher.py
if errorlevel 1 (
  echo.
  echo Marginalia stopped because of an error.
  echo Details are shown above and may also be available in startup_error.log.
  pause
  exit /b 1
)
exit /b 0
