$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path
$source = Join-Path $workspace "desktop\launcher_host.cs"
$output = Join-Path $workspace "LP-Lora-Trainer.exe"
$icon = Join-Path $workspace "assets\logo.ico"
$cscCandidates = @(
    (Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
    (Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe")
)
$csc = $cscCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $csc) { throw "Microsoft .NET Framework csc.exe was not found." }
if (-not (Test-Path -LiteralPath $source)) { throw "Launcher source is missing: $source" }
if (-not (Test-Path -LiteralPath $icon)) { throw "Launcher icon is missing: $icon" }
& $csc /nologo /target:winexe /platform:x64 /optimize+ /out:$output /win32icon:$icon /reference:System.Windows.Forms.dll $source
if ($LASTEXITCODE -ne 0) { throw "Launcher compilation failed with exit code $LASTEXITCODE." }
Write-Output "Built $output"
