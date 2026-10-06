@echo off
rem ---------------------------------------------------------------------------
rem  Brings this copy of the repository up to date with GitHub.
rem
rem  Double-click this file. It fetches every branch and tag from origin,
rem  removes branches that were deleted there, and then fast-forwards the
rem  branch that is checked out. It never makes a merge commit: if your
rem  branch has commits that origin does not have, it stops and says so,
rem  so nothing of yours is changed behind your back.
rem ---------------------------------------------------------------------------
setlocal enableextensions
title Git pull

rem Work in the repository this script lives in, whichever folder it was
rem started from.
cd /d "%~dp0.."

where git >nul 2>nul
if errorlevel 1 (
    echo Git is not installed, or it is not on the PATH.
    goto :failed
)

for /f "delims=" %%r in ('git rev-parse --show-toplevel 2^>nul') do set "REPO=%%r"
if not defined REPO (
    echo This folder is not inside a Git repository: %CD%
    goto :failed
)
cd /d "%REPO%"

for /f "delims=" %%b in ('git branch --show-current') do set "BRANCH=%%b"
if not defined BRANCH set "BRANCH=(no branch checked out)"

echo Repository: %REPO%
echo Branch:     %BRANCH%
echo.

echo Fetching every branch and tag from origin...
git fetch --all --prune --tags
if errorlevel 1 goto :failed
echo.

echo Updating %BRANCH%...
git pull --ff-only
if errorlevel 1 (
    echo.
    echo The branch could not be fast-forwarded. Either you have local
    echo changes that the update would overwrite, or your branch has
    echo commits that origin does not have. Nothing was changed.
    goto :failed
)

echo.
git status --short --branch
echo.
echo Done. The repository is up to date with origin.
echo.
pause
exit /b 0

:failed
echo.
echo The pull did not finish.
echo.
pause
exit /b 1
