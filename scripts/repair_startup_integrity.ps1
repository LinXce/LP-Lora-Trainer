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
if (-not ('LPWindowsIntegrity' -as [type])) { Add-Type -LiteralPath $native }
[LPWindowsIntegrity]::RejectReparsePath($workspace)
if ([LPWindowsIntegrity]::ProcessLevel() -lt [LPWindowsIntegrity]::Medium) {
    throw 'Integrity repair must run outside a low-integrity process.'
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
    [LPWindowsIntegrity]::RejectReparsePath($target)
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
$icacls = Join-Path $env:SystemRoot 'System32\icacls.exe'
foreach ($target in $targets) {
    if ([LPWindowsIntegrity]::FileLevel($target) -lt [LPWindowsIntegrity]::Medium) {
        if ($CheckOnly) { throw "Low-integrity release file: $target" }
        & $icacls $target /setintegritylevel M /Q | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Unable to repair application integrity label: $target" }
        if ([LPWindowsIntegrity]::FileLevel($target) -lt [LPWindowsIntegrity]::Medium) {
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
