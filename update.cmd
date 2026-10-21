@echo off
rem Update the project from its git remote. Only tracked source files change:
rem data\, output\, engine\, runtime\, python_runtime\ and frontend\dist\ are
rem ignored by git and are never touched by this script.
rem
rem Usage: update.cmd [--check] [--rebuild] [--yes]
rem   --check    only report whether an update is available
rem   --rebuild  rebuild the frontend (and the launcher when its sources changed)
rem   --yes      do not ask for confirmation
rem Messages stay in English on purpose: cmd's code page mangles Chinese output.
setlocal
if not exist "%~dp0python_runtime\python.exe" (
  echo Bundled Python is missing. Use a complete portable release, or run install.cmd first.
  pause
  exit /b 1
)
"%~dp0python_runtime\python.exe" "%~dp0scripts\update_project.py" %*
set "LP_EXIT=%ERRORLEVEL%"
if not "%LP_EXIT%"=="0" pause
exit /b %LP_EXIT%
