@echo off
rem Build the three portable deliverables in order. Every step already exists as a
rem script; this only wires them together so a clone can be built with one command.
rem Messages stay in English on purpose: cmd's code page mangles Chinese output.
setlocal
set "LP_ROOT=%~dp0"
set "LP_PY=%LP_ROOT%.venv\Scripts\python.exe"
set "LP_PAUSE=1"
if /I "%~1"=="--no-pause" set "LP_PAUSE=0"

echo [1/4] Building the frontend...
where npm >nul 2>nul
if errorlevel 1 (
  echo   Node.js and npm are required. Install Node 20 or newer from https://nodejs.org/
  goto :failed
)
pushd "%LP_ROOT%frontend"
call npm ci
if errorlevel 1 (
  popd
  goto :failed
)
call npm run build
if errorlevel 1 (
  popd
  goto :failed
)
popd
if not exist "%LP_ROOT%frontend\dist\index.html" (
  echo   frontend\dist\index.html is missing after the build.
  goto :failed
)

echo [2/4] Building the bundled Python runtime...
if not exist "%LP_PY%" (
  rem .venv is a build-time environment only, so recreate it on demand instead of
  rem sending the user away: the runtime that ships is built from it.
  echo   .venv is missing; preparing the build environment with install.cmd...
  call "%LP_ROOT%install.cmd" --no-build --no-pause
  if errorlevel 1 goto :failed
)
if not exist "%LP_PY%" (
  echo   install.cmd did not create .venv. Run it manually and read its output.
  goto :failed
)
rem A low-integrity .venv Python cannot compile the helper class the runtime
rem build loads with Add-Type, so normalize the interpreters we are about to use.
rem Same fix as install.cmd applies; harmless when they are already medium.
rem icacls takes one path per call: passing two returns ERROR_INVALID_PARAMETER.
"%SystemRoot%\System32\icacls.exe" "%LP_PY%" /setintegritylevel M /Q
"%SystemRoot%\System32\icacls.exe" "%LP_ROOT%.venv\Scripts\pythonw.exe" /setintegritylevel M /Q
if exist "%LP_ROOT%python_runtime" (
  rem --replace keeps the current runtime as python_runtime.previous-*
  "%LP_PY%" "%LP_ROOT%scripts\build_python_runtime.py" --replace
) else (
  "%LP_PY%" "%LP_ROOT%scripts\build_python_runtime.py"
)
if errorlevel 1 goto :failed

echo [3/4] Building LP-Lora-Trainer.exe...
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%LP_ROOT%scripts\build_launcher.ps1"
if errorlevel 1 goto :failed

echo [4/4] Verifying the bundled runtime...
"%LP_ROOT%python_runtime\python.exe" "%LP_ROOT%scripts\check_runtime.py"
if errorlevel 1 goto :failed

echo.
echo Build complete.
echo   Run here:      start.cmd  (or LP-Lora-Trainer.exe)
echo   Copy elsewhere: the whole project, including python_runtime\ and frontend\dist\
if "%LP_PAUSE%"=="1" pause
exit /b 0

:failed
echo.
echo Build failed. Fix the messages above and run build.cmd again.
if "%LP_PAUSE%"=="1" pause
exit /b 1
