# Launcher for the ChordDetectorTray scheduled task.
# Uses pythonw.exe so no console window flashes at log-on.

$ErrorActionPreference = 'Continue'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

$Pythonw = Join-Path $RepoRoot 'backend\.venv\Scripts\pythonw.exe'
$TrayScript = Join-Path $RepoRoot 'tray\tray_app.py'

if (-not (Test-Path $Pythonw)) {
    # pythonw is in the backend venv alongside python.exe; if it's missing
    # the venv wasn't created with the standard installer.
    exit 1
}

Set-Location $RepoRoot
& $Pythonw $TrayScript
