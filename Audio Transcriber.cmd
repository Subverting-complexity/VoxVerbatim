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

if not exist "%VENV_PY%" call :create_venv
if not exist "%VENV_PY%" goto :no_python

rem Install the libraries only when something is missing, so normal starts are quick.
"%VENV_PY%" -c "import PySide6, mutagen, av" >nul 2>&1
if errorlevel 1 call :install_requirements
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
exit /b %errorlevel%

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
