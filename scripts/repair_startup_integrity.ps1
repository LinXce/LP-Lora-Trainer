param(
    [string]$RuntimeDirectory,
    [switch]$SkipLauncher,
    [switch]$CheckOnly,
    [switch]$Relaunch,
    [string]$LaunchArgumentsBase64 = ''
)
$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$native = Join-Path $workspace 'desktop\windows_integrity.cs'
# Check the source before Add-Type executes any code from it.
$sourceCursor = [System.IO.Path]::GetFullPath($native)
while ($sourceCursor) {
    $entry = Get-Item -LiteralPath $sourceCursor -Force
    if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Permission repair refuses linked paths: $sourceCursor"
    }
    $sourceCursor = [System.IO.Path]::GetDirectoryName($sourceCursor)
}
$icacls = Join-Path $env:SystemRoot 'System32\icacls.exe'
$whoami = Join-Path $env:SystemRoot 'System32\whoami.exe'

# The helper class is how we READ integrity levels. Add-Type's own compilation can
# be refused outright on hardened machines (ERROR_PRIVILEGE_NOT_HELD), but loading
# a prebuilt assembly is not, so compile it once with the framework compiler and
# load that. Verification stays mandatory: -CheckOnly fails when it is unavailable.
$helper = $false
try {
    if (-not ('LPWindowsIntegrity' -as [type])) {
        $dll = Join-Path $workspace 'runtime\windows_integrity.dll'
        if (-not (Test-Path -LiteralPath $dll)) {
            $csc = @(
                (Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'),
                (Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe')
            ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
            if (-not $csc) { throw 'Microsoft .NET Framework csc.exe was not found.' }
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dll) | Out-Null
            & $csc /nologo /target:library /platform:x64 /optimize+ /out:$dll $native
            if ($LASTEXITCODE -ne 0) { throw "Compiling the integrity helper failed with exit code $LASTEXITCODE." }
        }
        Add-Type -Path $dll
    }
    $helper = $true
} catch {
    Write-Warning "Integrity helper unavailable ($($_.Exception.Message))."
}

function Test-ReparsePoint([string]$path) {
    # The helper does the same check in-process; this needs no compilation.
    return ((Get-Item -LiteralPath $path -Force).Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
}

function Test-LowIntegrityProcess {
    # Match the integrity SID, never the localized label text: whoami output is
    # translated on non-English Windows. Medium is S-1-16-8192 and up.
    $groups = & $whoami /groups
    return -not ($groups | Select-String -SimpleMatch 'S-1-16-8192', 'S-1-16-12288', 'S-1-16-16384' -Quiet)
}

if ($helper) {
    [LPWindowsIntegrity]::RejectReparsePath($workspace)
    if ([LPWindowsIntegrity]::ProcessLevel() -lt [LPWindowsIntegrity]::Medium) {
        throw 'Integrity repair must run outside a low-integrity process.'
    }
} else {
    if (Test-ReparsePoint $workspace) { throw "Permission repair refuses linked paths: $workspace" }
    if (Test-LowIntegrityProcess) { throw 'Integrity repair must run outside a low-integrity process.' }
}
if (-not $RuntimeDirectory) { $RuntimeDirectory = Join-Path $workspace 'python_runtime' }
$runtime = [System.IO.Path]::GetFullPath($RuntimeDirectory)
if ([System.IO.Path]::GetDirectoryName($runtime) -ne $workspace -or
    ([System.IO.Path]::GetFileName($runtime) -ne 'python_runtime' -and
     -not [System.IO.Path]::GetFileName($runtime).StartsWith('.python-runtime-build-'))) {
    throw 'Runtime repair target must be a direct application runtime/build directory.'
}
$launcher = Join-Path $workspace 'LP-Lora-Trainer.exe'
$targets = @()
if (-not $SkipLauncher) { $targets += $launcher }
$targets += $runtime
$targets += (Join-Path $runtime 'python.exe')
$targets += (Join-Path $runtime 'pythonw.exe')
# Validate every target before changing any labels. Never grant access, reset ACLs,
# recurse into user engines, or change the user's Downloads/Documents directories.
foreach ($target in $targets) {
    if (-not (Test-Path -LiteralPath $target)) { throw "Missing application file: $target" }
    if ($helper) {
        [LPWindowsIntegrity]::RejectReparsePath($target)
    } elseif (Test-ReparsePoint $target) {
        throw "Permission repair refuses linked paths: $target"
    }
}
if ($Relaunch) {
    if ($SkipLauncher -or $runtime -ne (Join-Path $workspace 'python_runtime')) {
        throw 'Relaunch only supports the installed application runtime.'
    }
    $launchArguments = ''
    if ($LaunchArgumentsBase64) {
        $launchArguments = [System.Text.Encoding]::UTF8.GetString(
            [System.Convert]::FromBase64String($LaunchArgumentsBase64))
    }
}
if ($CheckOnly -and -not $helper) {
    # A release check that cannot read labels must fail, never pass silently.
    throw 'Cannot verify integrity labels: the integrity helper class is unavailable on this machine.'
}
foreach ($target in $targets) {
    $level = if ($helper) { [LPWindowsIntegrity]::FileLevel($target) } else { 0 }
    if (-not $helper -or $level -lt [LPWindowsIntegrity]::Medium) {
        if ($CheckOnly) { throw "Low-integrity release file: $target" }
        & $icacls $target /setintegritylevel M /Q | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Unable to repair application integrity label: $target" }
        if ($helper -and [LPWindowsIntegrity]::FileLevel($target) -lt [LPWindowsIntegrity]::Medium) {
            throw "Application integrity label is still low: $target"
        }
    }
}
if ($Relaunch) {
    # Ask the existing desktop shell to launch as the interactive user rather
    # than carrying the repair helper's elevated token into the application.
    $shell = New-Object -ComObject Shell.Application
    $windows = $null
    $desktop = $null
    $document = $null
    $desktopShell = $null
    try {
        $windows = $shell.Windows()
        $location = 0
        $rootLocation = 0
        $windowHandle = 0
        # SWC_DESKTOP + SWFO_NEEDDISPATCH retrieves the existing Explorer
        # desktop. A new Shell.Application COM object alone can retain elevation.
        $desktop = $windows.FindWindowSW([ref]$location, [ref]$rootLocation, 8, [ref]$windowHandle, 1)
        if (-not $desktop) { throw 'The interactive desktop shell is unavailable.' }
        $document = $desktop.Document
        $desktopShell = $document.Application
        $desktopShell.ShellExecute($launcher, $launchArguments, $workspace, 'open', 1)
    }
    finally {
        foreach ($component in @($desktopShell, $document, $desktop, $windows, $shell)) {
            if ($component) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($component) }
        }
    }
}
