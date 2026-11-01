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
# PowerShell doesn't wait for GUI-subsystem exes like pythonw, so a bare `&`
# returns immediately and the task ends while the tray keeps running. Task
# Scheduler's MultipleInstances=IgnoreNew then can't see the tray, and the
# next Start-ScheduledTask spawns a duplicate icon. -Wait keeps the task
# alive for the tray's lifetime.
Start-Process -FilePath $Pythonw -ArgumentList "`"$TrayScript`"" -Wait
