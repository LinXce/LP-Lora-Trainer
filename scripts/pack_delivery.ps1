<#
.SYNOPSIS
    Build the release delivery folders for LP LoRA Trainer.

.DESCRIPTION
    Produces LP-Lora-Trainer-core (program, bundled runtime, built frontend,
    application sources) and, when engines are selected, LP-Lora-Trainer-full
    (core plus the selected engine directories and their .lp environments).

    Release mode deliberately leaves author data (.git, data, output, logs and
    caches) out of both folders. The generated README-FIRST.md records the chosen
    engines, requirements and the release update policy.

.PARAMETER Target
    Output directory, created fresh. Defaults to <project>\_delivery.

.PARAMETER Engines
    Engine directory names under engine\ to include in the full package.
    Default: all discovered engines. Use "none" for the core package only.
    Comma-separated values are accepted: -Engines anima_lora,musubi-tuner

.PARAMETER Mode
    Only "release" is supported.

.PARAMETER ListEngines
    Print the engines found under engine\ with their environment status and exit.

.PARAMETER NoPause
    Do not wait for Enter at the end (useful for scripted builds).
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = "",

    [Parameter(Position = 1)]
    [string[]]$Engines = @(),

    [ValidateSet("release")]
    [string]$Mode = "release",

    [switch]$ListEngines,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
$script:ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$script:ExcludeDirs = @(
    "data", "output", "_delivery", ".git", ".commandcode", ".claude",
    "node_modules", "__pycache__", ".pytest_cache", "docs", "engine", "logs",
    "torchinductor_*"
)
$script:ExcludeFiles = @("*.log")

function Get-EngineStatus {
    param([string]$EnginePath)
    if (Test-Path -LiteralPath (Join-Path $EnginePath ".lp\python\python.exe") -PathType Leaf) {
        return "embedded"
    }
    foreach ($name in @(".venv", "venv", "env")) {
        if (Test-Path -LiteralPath (Join-Path $EnginePath "$name\Scripts\python.exe") -PathType Leaf) {
            return "venv"
        }
    }
    return "none"
}

function Get-AvailableEngines {
    $engineRoot = Join-Path $script:ProjectRoot "engine"
    if (-not (Test-Path -LiteralPath $engineRoot -PathType Container)) {
        throw "Engine directory not found: $engineRoot"
    }
    return @(
        Get-ChildItem -LiteralPath $engineRoot -Directory |
            Where-Object { -not $_.Name.StartsWith(".") } |
            Sort-Object -Property Name
    )
}

function Get-AppVersion {
    $initFile = Join-Path $script:ProjectRoot "app\__init__.py"
    try {
        $match = Select-String -LiteralPath $initFile -Pattern 'APP_VERSION\s*=\s*"([^"]+)"' |
            Select-Object -First 1
        if ($match) {
            return $match.Matches[0].Groups[1].Value
        }
    } catch {
        # A missing version must not block packaging.
    }
    return "unknown"
}

function Resolve-TargetPath {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return [System.IO.Path]::GetFullPath((Join-Path $script:ProjectRoot "_delivery"))
    }
    if (-not [System.IO.Path]::IsPathRooted($Value)) {
        $Value = Join-Path $script:ProjectRoot $Value
    }
    return [System.IO.Path]::GetFullPath($Value)
}

function Assert-TargetUsable {
    param([string]$Path)
    if ($Path.TrimEnd('\') -eq $script:ProjectRoot.TrimEnd('\')) {
        throw "The target cannot be the project root."
    }
    foreach ($name in @("app", "adapters", "assets", "desktop", "scripts", "supervisor", "tests", "frontend", "python_runtime", "engine")) {
        $protected = [System.IO.Path]::GetFullPath((Join-Path $script:ProjectRoot $name))
        $prefix = $protected.TrimEnd('\') + '\'
        if ($Path.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "The target must not be inside $name\."
        }
    }
    if (Test-Path -LiteralPath $Path) {
        throw "The target already exists. Delete it yourself first: $Path"
    }
}

function Invoke-CopyTree {
    param(
        [string]$Source,
        [string]$Destination,
        [string[]]$ExcludeDirs = @(),
        [string[]]$ExcludeFiles = @()
    )
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Missing source directory: $Source"
    }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $arguments = @(
        $Source, $Destination, "/E", "/MT:32", "/NFL", "/NDL", "/NJH", "/NJS",
        "/NP", "/R:1", "/W:1"
    )
    if ($ExcludeDirs.Count -gt 0) {
        $arguments += "/XD"
        $arguments += $ExcludeDirs
    }
    if ($ExcludeFiles.Count -gt 0) {
        $arguments += "/XF"
        $arguments += $ExcludeFiles
    }
    & robocopy @arguments | Out-Null
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        throw "robocopy failed with exit code $code`: $Source -> $Destination"
    }
}

function Write-Readme {
    param(
        [string]$Destination,
        [string]$TemplateName,
        [string]$Version,
        [string]$EngineTable = "",
        [int]$EngineCount = 0
    )
    $templatePath = Join-Path $PSScriptRoot $TemplateName
    if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
        throw "Missing README template: $templatePath"
    }
    $text = Get-Content -LiteralPath $templatePath -Raw -Encoding UTF8
    $text = $text.Replace("{{VERSION}}", $Version).Replace("{{DATE}}", (Get-Date -Format "yyyy-MM-dd"))
    if ($TemplateName -eq "delivery_readme_full.md") {
        $text = $text.Replace("{{ENGINES_TABLE}}", $EngineTable).Replace("{{ENGINE_COUNT}}", [string]$EngineCount)
    }
    $text = $text.Replace("`r`n", "`n").Replace("`n", "`r`n")
    [System.IO.File]::WriteAllText($Destination, $text, (New-Object System.Text.UTF8Encoding($false)))
}

function Get-EngineTable {
    param($EngineList)
    $lines = @()
    foreach ($engine in $EngineList) {
        $lines += "| " + $engine.Name + " | " + (Get-EngineStatus $engine.FullName) + " |"
    }
    return ($lines -join "`r`n")
}

function Write-CorePackage {
    param([string]$Destination, [string]$Version)
    Write-Host "[1/5] Core: root files..."
    $rootFiles = @(
        "LP-Lora-Trainer.exe", "README.md", "README.zh-CN.md", "LICENSE",
        "pyproject.toml", "requirements.lock", "build.cmd", "install.cmd",
        "start.cmd", "start-api.cmd", "check-runtime.cmd"
    )
    foreach ($file in $rootFiles) {
        $source = Join-Path $script:ProjectRoot $file
        if (Test-Path -LiteralPath $source -PathType Leaf) {
            Copy-Item -LiteralPath $source -Destination (Join-Path $Destination $file) -Force
        }
    }
    foreach ($name in @("app", "desktop", "supervisor", "adapters", "scripts", "tests", "assets")) {
        Write-Host "      copying $name\"
        Invoke-CopyTree (Join-Path $script:ProjectRoot $name) (Join-Path $Destination $name) $script:ExcludeDirs $script:ExcludeFiles
    }

    Write-Host "[2/5] Core: frontend build..."
    Invoke-CopyTree (Join-Path $script:ProjectRoot "frontend") (Join-Path $Destination "frontend") $script:ExcludeDirs $script:ExcludeFiles

    Write-Host "[3/5] Core: bundled Python runtime..."
    Invoke-CopyTree (Join-Path $script:ProjectRoot "python_runtime") (Join-Path $Destination "python_runtime") $script:ExcludeDirs $script:ExcludeFiles

    Write-Host "[4/5] Core: uv tool..."
    $runtimeTarget = Join-Path $Destination "runtime"
    New-Item -ItemType Directory -Path (Join-Path $runtimeTarget "uv") -Force | Out-Null
    $uvSource = Join-Path $script:ProjectRoot "runtime\uv\uv.exe"
    if (Test-Path -LiteralPath $uvSource -PathType Leaf) {
        Copy-Item -LiteralPath $uvSource -Destination (Join-Path $runtimeTarget "uv\uv.exe") -Force
    }
    $integritySource = Join-Path $script:ProjectRoot "runtime\windows_integrity.dll"
    if (Test-Path -LiteralPath $integritySource -PathType Leaf) {
        Copy-Item -LiteralPath $integritySource -Destination (Join-Path $runtimeTarget "windows_integrity.dll") -Force
    }

    Write-Host "[5/5] Core: README-FIRST.md..."
    Write-Readme -Destination (Join-Path $Destination "README-FIRST.md") `
        -TemplateName "delivery_readme_core.md" -Version $Version
}

function Write-FullPackage {
    param(
        [string]$CorePath,
        [string]$Destination,
        [string]$Version,
        $EngineList
    )
    Write-Host "[Full] copying the core folder..."
    Invoke-CopyTree $CorePath $Destination $script:ExcludeDirs $script:ExcludeFiles
    Write-Host "[Full] copying selected engines..."
    $engineTarget = Join-Path $Destination "engine"
    New-Item -ItemType Directory -Path $engineTarget -Force | Out-Null
    foreach ($engine in $EngineList) {
        Write-Host ("      " + $engine.Name)
        Invoke-CopyTree $engine.FullName (Join-Path $engineTarget $engine.Name) $script:ExcludeDirs $script:ExcludeFiles
    }
    Write-Readme -Destination (Join-Path $Destination "README-FIRST.md") `
        -TemplateName "delivery_readme_full.md" -Version $Version `
        -EngineTable (Get-EngineTable $EngineList) -EngineCount $EngineList.Count
}

function Resolve-Selection {
    param($Available)
    $requested = @()
    foreach ($item in $Engines) {
        foreach ($part in ($item -split ",")) {
            $trimmed = $part.Trim()
            if ($trimmed.Length -gt 0) {
                $requested += $trimmed
            }
        }
    }
    $selected = @()
    $coreOnly = $false
    if ($requested.Count -eq 0 -or ($requested.Count -eq 1 -and $requested[0] -ieq "all")) {
        $selected = @($Available)
    } elseif ($requested.Count -eq 1 -and $requested[0] -ieq "none") {
        $coreOnly = $true
    } else {
        foreach ($name in $requested) {
            if ($name -ieq "none") {
                throw "Use -Engines none only by itself to build the core package."
            }
            $matches = @($Available | Where-Object { $_.Name -ieq $name })
            if ($matches.Count -eq 0) {
                $known = ($Available | ForEach-Object { $_.Name }) -join ", "
                throw "Unknown engine '$name'. Available: $known"
            }
            $already = @($selected | Where-Object { $_.FullName -ieq $matches[0].FullName })
            if ($already.Count -eq 0) {
                $selected += $matches[0]
            }
        }
    }
    return @{ Engines = $selected; CoreOnly = $coreOnly }
}

function Invoke-Main {
    Write-Host "LP LoRA Trainer delivery packager"
    Write-Host "Project: $script:ProjectRoot"

    if ($ListEngines) {
        $available = Get-AvailableEngines
        Write-Host ""
        Write-Host "Engines under engine\ (name / environment):"
        if ($available.Count -eq 0) {
            Write-Host "  (none)"
        } else {
            foreach ($engine in $available) {
                Write-Host ("  {0,-24} {1}" -f $engine.Name, (Get-EngineStatus $engine.FullName))
            }
        }
        return
    }

    if ($Mode -ne "release") {
        throw "Only release mode is supported."
    }

    $targetPath = Resolve-TargetPath $Target
    Assert-TargetUsable $targetPath

    foreach ($required in @("app\__init__.py", "python_runtime\python.exe", "frontend\dist\index.html", "LP-Lora-Trainer.exe")) {
        if (-not (Test-Path -LiteralPath (Join-Path $script:ProjectRoot $required) -PathType Leaf)) {
            throw "Missing required delivery file: $required (build the project first)"
        }
    }

    $version = Get-AppVersion
    $available = @(Get-AvailableEngines)
    $selection = Resolve-Selection $available
    $selected = @($selection.Engines)
    $coreOnly = [bool]$selection.CoreOnly
    if ($selected.Count -eq 0 -and -not $coreOnly) {
        Write-Host "WARNING: no engines found; building the core package only."
        $coreOnly = $true
    }

    Write-Host ""
    Write-Host "Mode: release"
    Write-Host "Target: $targetPath"
    if ($coreOnly) {
        Write-Host "Engines: core package only (no full package)"
    } elseif ($selected.Count -eq $available.Count) {
        Write-Host "Engines: all ($($selected.Count))"
    } else {
        Write-Host "Engines: $(($selected | ForEach-Object { $_.Name }) -join ', ')"
    }

    $corePath = Join-Path $targetPath "LP-Lora-Trainer-core"
    $fullPath = Join-Path $targetPath "LP-Lora-Trainer-full"
    New-Item -ItemType Directory -Path $corePath -Force | Out-Null
    Write-CorePackage -Destination $corePath -Version $version

    if (-not $coreOnly) {
        Write-Host ""
        Write-FullPackage -CorePath $corePath -Destination $fullPath -Version $version -EngineList $selected
    }

    Write-Host ""
    Write-Host "Done."
    Write-Host "  core: $corePath"
    if (-not $coreOnly) {
        Write-Host "  full: $fullPath"
    }
    Write-Host "  README-FIRST.md is generated in each package."
}

try {
    Invoke-Main
    if (-not $NoPause -and -not $ListEngines) {
        [void](Read-Host "Press Enter to exit")
    }
    exit 0
} catch {
    Write-Host ""
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
    if (-not $NoPause) {
        [void](Read-Host "Press Enter to exit")
    }
    exit 1
}
