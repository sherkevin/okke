$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $scriptDir "run_night_4gpu_watcher.ps1"
$taskName = "Night4GPUWatcherHourly"

Write-Host "Bootstrapping current checklist version..."
powershell -NoProfile -ExecutionPolicy Bypass -File $runner --bootstrap
if ($LASTEXITCODE -ne 0) {
    throw "Bootstrap failed with exit code $LASTEXITCODE."
}

$taskCommand = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$runner`""

Write-Host "Creating or updating scheduled task $taskName ..."
schtasks /Create /F /SC HOURLY /MO 1 /TN $taskName /TR $taskCommand | Out-Null

Write-Host "Scheduled task ready:"
schtasks /Query /TN $taskName /V /FO LIST
