# Register Task Scheduler jobs so the backend, frontend, and tray icon
# start automatically at log-on. Run this once from a regular (non-admin)
# PowerShell session.
#
# Prereqs (do these first):
#   1. backend venv exists and has `pip install -r backend/requirements.txt`
#   2. backend venv also has `pip install -r tray/requirements.txt`
#   3. frontend has been built: `cd frontend; npm install; npm run build`

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Scripts = Join-Path $RepoRoot 'scripts'

function Register-ChordDetectorTask {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [string] $Script
    )

    # A headless console host, so no window ever appears. `powershell.exe
    # -WindowStyle Hidden` isn't enough on Windows 11: the console gets handed
    # to Windows Terminal, which ignores the hidden flag, so a terminal window
    # opens at log-on and closing it kills the server. The cmd/python/node
    # children inherit this windowless console.
    $argline = "--headless powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Script`""
    $action = New-ScheduledTaskAction `
        -Execute 'conhost.exe' `
        -Argument $argline `
        -WorkingDirectory $RepoRoot

    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

    # ExecutionTimeLimit 0 = no limit. These are meant to run as long as
    # the user is logged in.
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
        -MultipleInstances IgnoreNew `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1)

    $principal = New-ScheduledTaskPrincipal `
        -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType Interactive `
        -RunLevel Limited

    Register-ScheduledTask `
        -TaskName $Name `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Force | Out-Null

    Write-Host "Registered: $Name"
}

Register-ChordDetectorTask -Name 'ChordDetectorBackend'  -Script (Join-Path $Scripts 'run-backend.ps1')
Register-ChordDetectorTask -Name 'ChordDetectorFrontend' -Script (Join-Path $Scripts 'run-frontend.ps1')
Register-ChordDetectorTask -Name 'ChordDetectorTray'     -Script (Join-Path $Scripts 'run-tray.ps1')

Write-Host ''
Write-Host 'Starting all three now...'
Start-ScheduledTask -TaskName 'ChordDetectorBackend'
Start-ScheduledTask -TaskName 'ChordDetectorFrontend'
Start-ScheduledTask -TaskName 'ChordDetectorTray'

Write-Host ''
Write-Host 'Done. Right-click the Chord Detector tray icon to open the app.'
Write-Host 'Logs: ' (Join-Path $RepoRoot 'logs')
