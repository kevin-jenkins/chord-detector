# Launcher for the ChordDetectorBackend scheduled task.
# Keep stdout + stderr in logs\backend.log so the tray's "Open logs" menu works.

$ErrorActionPreference = 'Continue'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$LogDir = Join-Path $RepoRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir 'backend.log'

# Truncate the log if it has grown past ~10 MB so we don't fill the disk.
if ((Test-Path $Log) -and ((Get-Item $Log).Length -gt 10MB)) {
    Clear-Content $Log
}

$Python = Join-Path $RepoRoot 'backend\.venv\Scripts\python.exe'
$Main = Join-Path $RepoRoot 'backend\main.py'

if (-not (Test-Path $Python)) {
    "[$(Get-Date -Format o)] Backend venv not found at $Python" | Add-Content $Log
    exit 1
}

Set-Location (Join-Path $RepoRoot 'backend')

# cmd /c handles the redirection reliably on PS 5.1; running the native exe
# through PowerShell's own 2>&1 wraps every stderr line in an ErrorRecord.
& cmd /c "`"$Python`" `"$Main`" >> `"$Log`" 2>&1"
