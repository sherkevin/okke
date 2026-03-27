param(
    [Parameter(Mandatory = $true)][string]$Workspace,
    [Parameter(Mandatory = $true)][string]$MainLog,
    [int]$IntervalSec = 90
)
$statusLog = Join-Path $Workspace "veb_workflow_watch.log"
while ($true) {
    $py = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'veb_idea_workflow' -and $_.CommandLine -match [regex]::Escape($Workspace) }
    $tail = if (Test-Path $MainLog) { (Get-Content $MainLog -Tail 1 -ErrorAction SilentlyContinue) } else { "(no main log)" }
    $line = "{0:o} running={1} tail={2}" -f (Get-Date), [bool]$py, $tail
    Add-Content -Path $statusLog -Value $line -Encoding utf8
    if (-not $py) { break }
    Start-Sleep -Seconds $IntervalSec
}
Add-Content -Path $statusLog -Value ("{0:o} watch ended (no matching python)" -f (Get-Date)) -Encoding utf8
