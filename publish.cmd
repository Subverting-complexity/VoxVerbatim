@echo off
rem ---------------------------------------------------------------------------
rem  Build the shareable copy of Audio Transcriber.
rem
rem  Double-click this file. It builds the application into the "publish"
rem  folder, as "publish\Audio Transcriber". That folder holds the program and
rem  everything it needs, including Python itself, so it can be copied to
rem  somebody else's Windows computer and run there without anything being
rem  installed first.
rem
rem  The first build takes several minutes and needs an internet connection,
rem  because the libraries the application runs on and the tool that does the
rem  building are all downloaded first. Later builds take about a minute and
rem  a half.
rem ---------------------------------------------------------------------------
setlocal enableextensions
title Publishing Audio Transcriber

rem Work in the folder this script lives in, whichever folder it was started from.
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "PUBLISH_DIR=publish"
set "APP_DIR=%PUBLISH_DIR%\Audio Transcriber"
set "APP_EXE=%APP_DIR%\Audio Transcriber.exe"

if not exist "%VENV_PY%" call :create_venv
if not exist "%VENV_PY%" goto :no_python

echo Checking that the libraries are present and up to date.
"%VENV_PY%" -m pip install --disable-pip-version-check --quiet -r requirements.txt
if errorlevel 1 goto :setup_failed

rem PyInstaller is the tool that turns the application into a program that
rem runs without Python. It belongs to building rather than to running, which
rem is why it is not in requirements.txt.
rem The upper bound is not caution for its own sake. The specification file
rem beside this one is written against PyInstaller 6, and the shape of a
rem specification is the kind of thing a major version is allowed to change.
"%VENV_PY%" -m pip install --disable-pip-version-check --quiet "pyinstaller>=6.16,<7"
if errorlevel 1 goto :setup_failed

rem Remove the previous build. PyInstaller writes over what it produces but
rem does not clear out what it no longer produces, so a file that was needed
rem by an older version of the application would otherwise stay behind for
rem ever and be copied to everybody who was sent the folder.
if exist "%APP_DIR%" rmdir /s /q "%APP_DIR%"
if exist "%APP_DIR%" goto :folder_in_use

echo.
echo Building. This takes a few minutes. Please wait.
echo.
"%VENV_PY%" -m PyInstaller --noconfirm ^
    --distpath "%PUBLISH_DIR%" ^
    --workpath "build\pyinstaller" ^
    "packaging\audio-transcriber.spec"
if errorlevel 1 goto :build_failed
if not exist "%APP_EXE%" goto :build_failed

rem The note for whoever is sent the folder travels with it.
copy /y "packaging\Read me first.txt" "%APP_DIR%\" >nul
if errorlevel 1 goto :note_failed

echo.
echo Done. The application is in:
echo    %CD%\%APP_DIR%
echo.
echo Copy that whole folder to share it. The person you send it to runs
echo "Audio Transcriber.exe" inside it. Nothing needs to be installed on
echo their computer.
echo.
pause
exit /b 0

:create_venv
echo Setting up the build environment for the first time. Please wait.
py -3 -m venv "%VENV_DIR%" >nul 2>&1
if exist "%VENV_PY%" exit /b 0
rem The second attempt is allowed to say what went wrong. A first attempt
rem that fails is ordinary, because "py" is not on every machine, but if the
rem second one fails too then the reason matters, and quietly swallowing it
rem is how somebody ends up installing a second copy of Python they did not
rem need.
python -m venv "%VENV_DIR%"
exit /b 0

:no_python
echo.
echo A Python environment for building could not be created.
echo.
echo Usually that means Python is not installed. Building needs Python 3.11
echo or newer, from https://www.python.org/downloads/windows/ . Tick
echo "Add python.exe to PATH" during the installation, then run this file
echo again.
echo.
echo If Python is installed, any message above says what stopped it. A full
echo disk, a folder that cannot be written to, and the version of Python
echo that comes from the Microsoft Store all fail in this way.
echo.
echo Only building needs Python. The folder that is built does not.
echo.
pause
exit /b 1

:setup_failed
echo.
echo The libraries needed for building could not be installed. The messages
echo above explain what went wrong. The two usual reasons are the internet
echo connection, and this repository sitting in a folder too deep for
echo Windows: some file names inside the ElevenLabs library are long enough
echo to pass the 260-character limit, and installing then stops partway
echo through with a file it cannot find. Moving the repository nearer the
echo top of the drive fixes that one.
echo.
pause
exit /b 1

:folder_in_use
echo.
echo The previous build could not be removed:
echo    %CD%\%APP_DIR%
echo.
echo It is probably still running, or open in a window. Close it and run this
echo file again.
echo.
pause
exit /b 1

:build_failed
echo.
echo The build failed. The messages above explain what went wrong.
echo.
pause
exit /b 1

:note_failed
echo.
echo The application was built, but "Read me first.txt" could not be copied
echo into it from the packaging folder. The folder works; what is missing is
echo the note that explains it to whoever you send it to, including the
echo warning Windows shows the first time they run it. Copy the file in by
echo hand, or run this file again.
echo.
pause
exit /b 1
