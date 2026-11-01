# Launcher for the ChordDetectorFrontend scheduled task.
# Runs the production Next.js server (requires `npm run build` first).

$ErrorActionPreference = 'Continue'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$LogDir = Join-Path $RepoRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir 'frontend.log'

if ((Test-Path $Log) -and ((Get-Item $Log).Length -gt 10MB)) {
    Clear-Content $Log
}

Set-Location (Join-Path $RepoRoot 'frontend')

# Task Scheduler can hand us a leaner PATH than an interactive shell, so
# resolve npm.cmd explicitly if we can.
$Npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
if (-not $Npm) { $Npm = 'npm.cmd' }

if (-not (Test-Path (Join-Path $PWD '.next'))) {
    "[$(Get-Date -Format o)] Frontend not built. Run: cd frontend; npm run build" | Add-Content $Log
    exit 1
}

& cmd /c "`"$Npm`" start >> `"$Log`" 2>&1"
