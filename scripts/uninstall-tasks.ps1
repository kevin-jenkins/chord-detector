# Remove the Chord Detector scheduled tasks.

$ErrorActionPreference = 'Continue'

foreach ($name in 'ChordDetectorBackend', 'ChordDetectorFrontend', 'ChordDetectorTray') {
    try {
        Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction Stop
        Write-Host "Removed: $name"
    } catch {
        Write-Host "Skipped (not registered): $name"
    }
}
