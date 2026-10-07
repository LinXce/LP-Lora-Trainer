@echo off
setlocal
if not exist "%~dp0python_runtime\python.exe" (
  echo Bundled Python is missing. Use a complete portable release.
  pause
  exit /b 1
)
"%~dp0python_runtime\python.exe" "%~dp0scripts\check_runtime.py"
set "LP_EXIT=%ERRORLEVEL%"
pause
exit /b %LP_EXIT%
