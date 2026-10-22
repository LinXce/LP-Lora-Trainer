@echo off
rem Compatibility wrapper around pack_delivery.ps1.
rem
rem Usage:
rem   scripts\pack_delivery.cmd [target] [-Engines name1,name2] [-ListEngines] [-NoPause]
rem
rem Default target: <project>\_delivery. Default engines: all under engine\.
rem Use -Engines none to build only the core package.
setlocal
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack_delivery.ps1" %*
exit /b %ERRORLEVEL%
