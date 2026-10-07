@echo off
setlocal
if not exist "%~dp0python_runtime\python.exe" (
  echo Bundled Python is missing. Use a complete portable release.
  echo Developers: build it with .venv\Scripts\python.exe scripts\build_python_runtime.py
  pause
  exit /b 1
)
if not exist "%~dp0frontend\dist\index.html" (
  echo Frontend build is missing. Developers: npm --prefix frontend run build
  pause
  exit /b 1
)
"%~dp0python_runtime\python.exe" -m desktop.launcher %*
set "LP_EXIT=%ERRORLEVEL%"
if not "%LP_EXIT%"=="0" pause
exit /b %LP_EXIT%
