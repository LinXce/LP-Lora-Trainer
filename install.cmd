@echo off
rem Prepare a development environment without a system Python: fetch uv with
rem PowerShell (which Windows already has), let uv fetch CPython, create .venv,
rem install the application, then hand over to build.cmd.
rem
rem Usage: install.cmd [--no-build] [--no-pause] [--index <package-index-url>]
rem   --no-build  only prepare .venv
rem   --no-pause  do not wait for a key at the end
rem   --index     optional package index (mirror) for pip resolution, e.g.
rem               --index https://pypi.tuna.tsinghua.edu.cn/simple
rem Messages stay in English on purpose: cmd's code page mangles Chinese output.
setlocal
set "LP_ROOT=%~dp0"
set "LP_UV_DIR=%LP_ROOT%runtime\uv"
set "LP_UV=%LP_UV_DIR%\uv.exe"
set "LP_VENV=%LP_ROOT%.venv"
set "LP_PYTHON_VERSION=3.12"
set "LP_BUILD=1"
set "LP_PAUSE=1"
set "LP_INDEX="

:parse
if "%~1"=="" goto :parsed
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--no-build" (
  set "LP_BUILD=0"
  shift
  goto :parse
)
if /I "%~1"=="--no-pause" (
  set "LP_PAUSE=0"
  shift
  goto :parse
)
if /I "%~1"=="--index" (
  if "%~2"=="" goto :missing_index
  if /I "%~2"=="--no-build" goto :missing_index
  if /I "%~2"=="--no-pause" (
    set "LP_PAUSE=0"
    goto :missing_index
  )
  set "LP_INDEX=%~2"
  shift
  shift
  goto :parse
)
echo Unknown option: %~1
goto :usage_error

:parsed
rem Keep uv's cache and interpreters inside the project, exactly like the app does
rem (app/services/installation.py), so nothing leaks into the user profile.
set "UV_CACHE_DIR=%LP_UV_DIR%\cache"
set "UV_PYTHON_INSTALL_DIR=%LP_UV_DIR%\python"
set "LP_UV_TARGET=%LP_UV%"
if not "%LP_INDEX%"=="" set "UV_DEFAULT_INDEX=%LP_INDEX%"

if "%LP_BUILD%"=="0" goto :prerequisites_done
call :check_build_prerequisites
if errorlevel 1 goto :failed
:prerequisites_done

if exist "%LP_UV%" goto :have_uv

echo [1/4] Downloading uv from the official Astral release...
if not exist "%LP_UV_DIR%" mkdir "%LP_UV_DIR%"
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$rel = Invoke-RestMethod -Uri 'https://api.github.com/repos/astral-sh/uv/releases/latest' -Headers @{ 'User-Agent' = 'LP-Lora-Trainer-install' };" ^
  "$asset = $rel.assets | Where-Object { $_.name -eq 'uv-x86_64-pc-windows-msvc.zip' } | Select-Object -First 1;" ^
  "if (-not $asset) { throw 'uv release asset not found' };" ^
  "$stage = Join-Path $env:TEMP ('uv-lp-' + [guid]::NewGuid().ToString('N'));" ^
  "New-Item -ItemType Directory -Path $stage | Out-Null;" ^
  "try {" ^
  "  $zip = Join-Path $stage 'uv.zip';" ^
  "  Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip;" ^
  "  Expand-Archive -LiteralPath $zip -DestinationPath $stage -Force;" ^
  "  Copy-Item -LiteralPath (Join-Path $stage 'uv.exe') -Destination $env:LP_UV_TARGET -Force;" ^
  "} finally { if (Test-Path -LiteralPath $stage) { Remove-Item -Recurse -Force $stage } }"
if errorlevel 1 (
  echo   Automatic download failed.
  echo   Manual fallback: get uv-x86_64-pc-windows-msvc.zip from
  echo     https://github.com/astral-sh/uv/releases
  echo   and extract uv.exe to:
  echo     %LP_UV%
  goto :failed
)
if not exist "%LP_UV%" (
  echo   uv.exe was not created at the expected path.
  goto :failed
)

:have_uv
rem No separate "uv python install": --managed-python makes uv fetch the
rem interpreter on demand, and keeps it inside UV_PYTHON_INSTALL_DIR. Installing
rem it explicitly would also try to write user-level shims and registry entries.
echo [2/4] Creating .venv with CPython %LP_PYTHON_VERSION% ...
rem --allow-existing keeps a .venv from an earlier run, so re-running is safe.
rem Delete .venv yourself when you want a clean environment.
"%LP_UV%" venv --python %LP_PYTHON_VERSION% --managed-python --allow-existing "%LP_VENV%"
if errorlevel 1 (
  echo   If the interpreter download failed, check the network, then retry.
  goto :failed
)
rem A sandbox-marked workspace hands new files a low integrity label, and a
rem low-integrity Python cannot compile the build helper class (Add-Type is
rem denied when the process is lower than the source). The runtime build repairs
rem its own files; this development venv has to be repaired here.
rem icacls takes one path per call: passing two returns ERROR_INVALID_PARAMETER.
"%SystemRoot%\System32\icacls.exe" "%LP_VENV%\Scripts\python.exe" /setintegritylevel M /Q
"%SystemRoot%\System32\icacls.exe" "%LP_VENV%\Scripts\pythonw.exe" /setintegritylevel M /Q

echo [3/4] Installing the application and its dependencies...
pushd "%LP_ROOT%"
if exist "%LP_ROOT%requirements.lock" (
  rem The lock is captured from the environment that produced python_runtime/, so
  rem using it keeps two machines on the same combination.
  echo   Using the pinned requirements.lock.
  "%LP_UV%" pip install --python "%LP_VENV%\Scripts\python.exe" -r "requirements.lock" -e ".[desktop]" packaging
) else (
  echo   No requirements.lock; resolving versions from pyproject.toml.
  "%LP_UV%" pip install --python "%LP_VENV%\Scripts\python.exe" -e ".[desktop]" packaging
)
if errorlevel 1 (
  popd
  echo   If the package index is unreachable, retry with a mirror:
  echo     install.cmd --index https://pypi.tuna.tsinghua.edu.cn/simple
  goto :failed
)
popd

if "%LP_BUILD%"=="0" (
  echo.
  echo Environment ready. Run build.cmd to build the portable deliverables.
  if "%LP_PAUSE%"=="1" pause
  exit /b 0
)

echo [4/4] Building...
call "%LP_ROOT%build.cmd" --no-pause
if errorlevel 1 goto :failed

echo.
echo Done. Environment in .venv, deliverables built. Run start.cmd to use the app.
if "%LP_PAUSE%"=="1" pause
exit /b 0

:check_build_prerequisites
if not "%PROCESSOR_ARCHITECTURE%"=="AMD64" if not "%PROCESSOR_ARCHITEW6432%"=="AMD64" (
  echo This installer supports Windows x64 only.
  exit /b 1
)
where npm >nul 2>nul
if errorlevel 1 (
  echo Node.js 20 or newer with npm is required to build the frontend.
  echo Install Node.js from https://nodejs.org/ and run install.cmd again.
  exit /b 1
)
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
  "$v = (& node --version).Trim() -replace '^v', ''; if ([version]$v -lt [version]'20.0') { exit 1 }"
if errorlevel 1 (
  echo Node.js 20 or newer is required to build the frontend.
  exit /b 1
)
if not exist "%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe" if not exist "%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe" (
  echo .NET Framework 4.x csc.exe is required to build LP-Lora-Trainer.exe.
  exit /b 1
)
exit /b 0

:missing_index
echo --index requires a package index URL.
goto :usage_error

:usage_error
echo Usage: install.cmd [--no-build] [--no-pause] [--index ^<package-index-url^>]
if "%LP_PAUSE%"=="1" pause
exit /b 2

:usage
echo Usage: install.cmd [--no-build] [--no-pause] [--index ^<package-index-url^>]
echo   --no-build  only prepare .venv
echo   --no-pause  do not wait for a key at the end
echo   --index     optional package index URL for Python packages
exit /b 0

:failed
echo.
echo Install failed. Fix the messages above and run install.cmd again.
if "%LP_PAUSE%"=="1" pause
exit /b 1
