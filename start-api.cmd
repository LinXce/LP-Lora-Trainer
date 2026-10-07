@echo off
setlocal
if not exist "%~dp0python_runtime\python.exe" (
  echo Bundled Python is missing. Use a complete portable release.
  exit /b 1
)
"%~dp0python_runtime\python.exe" -m app.api.server %*
exit /b %ERRORLEVEL%
