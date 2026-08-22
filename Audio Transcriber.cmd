@echo off
rem ---------------------------------------------------------------------------
rem  Audio Transcriber launcher.
rem
rem  Double-click this file to start the application. The first run creates a
rem  private Python environment in the .venv folder and installs the libraries
rem  the application needs, which takes a minute. Later runs start straight away.
rem ---------------------------------------------------------------------------
setlocal enableextensions
title Audio Transcriber

rem Work in the folder this script lives in, whichever folder it was started from.
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PYW=%VENV_DIR%\Scripts\pythonw.exe"
set "START_LOG=%TEMP%\audio-transcriber-start.log"
rem A copy of requirements.txt taken after the last successful install. While
rem it still matches the real one, nothing needs installing.
set "INSTALLED_STAMP=%VENV_DIR%\requirements.installed"
rem Every library the application imports, for the checks below. Keep this in
rem step with requirements.txt.
set "ALL_LIBRARIES=PySide6, mutagen, av, openai, elevenlabs, assemblyai, deepgram, httpx"

if not exist "%VENV_PY%" call :create_venv
if not exist "%VENV_PY%" goto :no_python

rem Install the libraries only when something has changed, so normal starts are
rem quick. Two checks, because each catches a case the other misses. The stamp
rem file says whether requirements.txt has changed since the last install, which
rem is how a library added to the application reaches a machine that already
rem had the old set. The import check catches a library that is listed and was
rem installed but is broken or half installed, which the stamp cannot see.
set "NEEDS_INSTALL="
if not exist "%INSTALLED_STAMP%" set "NEEDS_INSTALL=1"
if not defined NEEDS_INSTALL (
    fc /b "requirements.txt" "%INSTALLED_STAMP%" >nul 2>&1
    if errorlevel 1 set "NEEDS_INSTALL=1"
)
if not defined NEEDS_INSTALL (
    "%VENV_PY%" -c "import %ALL_LIBRARIES%" >nul 2>&1
    if errorlevel 1 set "NEEDS_INSTALL=1"
)
if defined NEEDS_INSTALL call :install_requirements
if errorlevel 1 goto :setup_failed

rem Check again after installing. An install that reported success but left a
rem library unusable would otherwise be found by the user half way through a
rem run that had already cost money.
"%VENV_PY%" -c "import %ALL_LIBRARIES%" >nul 2>&1
if errorlevel 1 goto :setup_failed

rem Check that the application can be loaded before starting it without a console,
rem so that a broken installation reports itself instead of failing silently.
"%VENV_PY%" -c "import audio_transcriber.app" >"%START_LOG%" 2>&1
if errorlevel 1 goto :start_failed

if exist "%VENV_PYW%" (
    start "" "%VENV_PYW%" -m audio_transcriber
) else (
    start "" "%VENV_PY%" -m audio_transcriber
)
exit /b 0

:create_venv
echo Setting up Audio Transcriber for the first time. Please wait.
py -3 -m venv "%VENV_DIR%" >nul 2>&1
if exist "%VENV_PY%" exit /b 0
python -m venv "%VENV_DIR%" >nul 2>&1
exit /b 0

:install_requirements
echo Installing the libraries Audio Transcriber needs. Please wait.
"%VENV_PY%" -m pip install --disable-pip-version-check --quiet --upgrade pip
"%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 exit /b 1
rem Remember what was installed, so the next start can tell whether the list
rem has changed. Written only after a successful install, so a failed one is
rem tried again next time rather than being taken as done.
copy /y "requirements.txt" "%INSTALLED_STAMP%" >nul
exit /b 0

:no_python
echo.
echo Python was not found on this computer.
echo Install Python 3.11 or newer from https://www.python.org/downloads/windows/
echo and tick "Add python.exe to PATH" during the installation, then run this
echo file again.
echo.
pause
exit /b 1

:setup_failed
echo.
echo The libraries could not be installed. Check your internet connection and
echo run this file again. The messages above explain what went wrong.
echo.
pause
exit /b 1

:start_failed
echo.
echo Audio Transcriber could not start. The reason follows.
echo.
type "%START_LOG%"
echo.
pause
exit /b 1
