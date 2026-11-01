# Remove the Chord Detector scheduled tasks and stop what they launched.
#
# -RepoRoot is the checkout the tasks run from; it defaults to this one.

param(
    [string] $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Continue'
$Tasks = 'ChordDetectorBackend', 'ChordDetectorFrontend', 'ChordDetectorTray'

foreach ($name in $Tasks) {
    Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
}

# Stopping a task only kills its powershell wrapper; the backend, frontend
# and tray processes it started keep running (and keep their ports) until
# killed. service_procs.py finds them by command line and tree-kills them.
$Python = Join-Path $RepoRoot 'backend\.venv\Scripts\python.exe'
$Helper = Join-Path $RepoRoot 'tray\service_procs.py'
if (Test-Path $Python) {
    & $Python $Helper all --root $RepoRoot
} else {
    Write-Host "Backend venv not found at $Python; can't stop leftover processes."
    Write-Host 'Check Task Manager for python/node processes running from' $RepoRoot
}

foreach ($name in $Tasks) {
    try {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction Stop
        Write-Host "Removed: $name"
    } catch {
        Write-Host "Skipped (not registered): $name"
    }
}
