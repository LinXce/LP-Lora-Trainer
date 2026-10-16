@echo off
rem Build two delivery folders: one with the engine environments and one without.
rem The source project is only read; nothing here deletes anything.
rem
rem Usage: pack_delivery.cmd [target-directory]
rem   default target (when omitted): <project>\_delivery
rem Output: <target>\LP-Lora-Trainer-core   (program + runtime + frontend, no engines)
rem         <target>\LP-Lora-Trainer-full   (core plus engine/ environments)
rem Author records (data, output, caches, logs) and tool configs (.git, .claude,
rem .commandcode, node_modules) are excluded from both.
rem
rem Copying rules, deliberately: single files are copied with copy, and every
rem robocopy call handles exactly ONE directory and carries RC_EXCLUDES. Passing a
rem file list, or several directories, to robocopy together with /E is what once
rem copied the whole project root (docs, engine, egg-info) into the core folder.
rem Messages stay in English on purpose: cmd's code page mangles Chinese output.
setlocal
set "LP_ROOT=%~dp0.."
pushd "%LP_ROOT%"
set "LP_ROOT=%CD%"
popd
set "LP_TARGET=%~1"
rem %%~f normalises the path, which also removes a trailing backslash that would
rem otherwise break every quoted comparison below.
if defined LP_TARGET for %%A in ("%LP_TARGET%") do set "LP_TARGET=%%~fA"
if not defined LP_TARGET set "LP_TARGET=%LP_ROOT%\_delivery"
set "LP_CORE=%LP_TARGET%\LP-Lora-Trainer-core"
set "LP_FULL=%LP_TARGET%\LP-Lora-Trainer-full"

rem Close the app first: a running backend keeps locks and writes logs.
if exist "%LP_ROOT%runtime\backend.lock" echo WARNING: runtime\backend.lock exists - close LP before packaging.

echo Packaging into: %LP_TARGET%
if exist "%LP_TARGET%" (
  echo   Target already exists. Delete it yourself first to avoid mixing two builds.
  goto :failed
)
rem A target inside the project would be copied into itself, so it is excluded from
rem every source below and refused when it is the project root itself.
if /i "%LP_TARGET%"=="%LP_ROOT%" (
  echo   The target cannot be the project root.
  goto :failed
)
mkdir "%LP_CORE%" || goto :failed

rem Author records, tool configs, the target itself, and engine-internal leftovers
rem (torch compile output, logs) never travel.
set "RC_EXCLUDES=/XD data output _delivery .git .commandcode .claude node_modules __pycache__ .pytest_cache docs engine logs torchinductor_* %LP_TARGET% /XF *.log"
rem /MT copies with several threads: an engine venv is tens of thousands of small
rem files, so per-file overhead (not bytes) dominates the runtime.
set "RC_FLAGS=/E /MT:32 /NFL /NDL /NJH /NJS /NP /R:1 /W:1"

echo [1/7] Core: program files, sources and assets (one directory per robocopy)...
for %%F in (LP-Lora-Trainer.exe README.md README.zh-CN.md LICENSE pyproject.toml requirements.lock build.cmd install.cmd start.cmd start-api.cmd check-runtime.cmd) do (
  if exist "%LP_ROOT%\%%F" copy /y "%LP_ROOT%\%%F" "%LP_CORE%\" >nul
)
for %%D in (app desktop supervisor adapters scripts tests assets) do (
  if exist "%LP_ROOT%\%%D" robocopy "%LP_ROOT%\%%D" "%LP_CORE%\%%D" %RC_FLAGS% %RC_EXCLUDES%
)

echo [2/7] Core: frontend build (no node_modules)...
robocopy "%LP_ROOT%\frontend" "%LP_CORE%\frontend" %RC_FLAGS% %RC_EXCLUDES%

echo [3/7] Core: bundled Python runtime...
robocopy "%LP_ROOT%\python_runtime" "%LP_CORE%\python_runtime" %RC_FLAGS% %RC_EXCLUDES%

echo [4/7] Core: uv tool (its cache and interpreter stock are left behind)...
rem copy never creates directories, and a missing target folder fails silently with
rem output redirected, so both destinations are created first.
if not exist "%LP_CORE%\runtime" mkdir "%LP_CORE%\runtime"
if not exist "%LP_CORE%\runtime\uv" mkdir "%LP_CORE%\runtime\uv"
if exist "%LP_ROOT%\runtime\uv\uv.exe" copy /y "%LP_ROOT%\runtime\uv\uv.exe" "%LP_CORE%\runtime\uv\uv.exe" >nul
if exist "%LP_ROOT%\runtime\windows_integrity.dll" copy /y "%LP_ROOT%\runtime\windows_integrity.dll" "%LP_CORE%\runtime\windows_integrity.dll" >nul

echo [5/7] Full: copy of the core folder...
robocopy "%LP_CORE%" "%LP_FULL%" %RC_FLAGS% %RC_EXCLUDES%

echo [6/7] Full: engines, including venv/.venv and the project-local base Python...
robocopy "%LP_ROOT%\engine" "%LP_FULL%\engine" %RC_FLAGS% %RC_EXCLUDES%

echo [7/7] Delivery notes...
call :notes "%LP_CORE%" "core: program, bundled runtime and built frontend; engines are NOT included"
call :notes "%LP_FULL%" "full: the core package plus engine\ with its environments"

echo.
echo Done. Two folders are ready under %LP_TARGET%:
echo   LP-Lora-Trainer-core  (no engines)
echo   LP-Lora-Trainer-full  (with engines, needs the one-time rebase below)
echo The full package still records %LP_ROOT% inside the engine environments,
echo so its first run needs scripts\repair_engine_venv.py as described in DELIVERY-NOTES.txt.
pause
exit /b 0

:notes
> "%~1\DELIVERY-NOTES.txt" echo LP LoRA Trainer - %~2
>>"%~1\DELIVERY-NOTES.txt" echo.
>>"%~1\DELIVERY-NOTES.txt" echo Removed before packaging: data\, output\, runtime\uv\cache, node_modules,
>>"%~1\DELIVERY-NOTES.txt" echo logs, window state, tool configs (.git, .claude, .commandcode), docs\,
>>"%~1\DELIVERY-NOTES.txt" echo and engine-internal leftovers (torch compile output, engine logs).
>>"%~1\DELIVERY-NOTES.txt" echo.
>>"%~1\DELIVERY-NOTES.txt" echo First run:
>>"%~1\DELIVERY-NOTES.txt" echo   1. Open LP-Lora-Trainer.exe once, or run start.cmd.
>>"%~1\DELIVERY-NOTES.txt" echo   2. Engine page: rescan, then bind an interpreter and diagnose.
>>"%~1\DELIVERY-NOTES.txt" echo      Without engines, press "install engine environment" instead.
>>"%~1\DELIVERY-NOTES.txt" echo   3. Full package only - engine environments store absolute paths, so
>>"%~1\DELIVERY-NOTES.txt" echo      rebase each one once from this folder:
>>"%~1\DELIVERY-NOTES.txt" echo        python_runtime\python.exe scripts\repair_engine_venv.py engine\kohya_ss-26.0.0\venv --old-root %LP_ROOT% --new-root ^<this-folder^>
>>"%~1\DELIVERY-NOTES.txt" echo        python_runtime\python.exe scripts\repair_engine_venv.py engine\musubi-tuner\.venv --old-root %LP_ROOT% --new-root ^<this-folder^>
exit /b 0

:failed
echo.
echo Packaging failed. Fix the message above and run pack_delivery.cmd again.
pause
exit /b 1
