$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$watcher = Join-Path $scriptDir "night_4gpu_watcher.py"

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $watcher @args
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $watcher @args
    exit $LASTEXITCODE
}

throw "Neither 'py' nor 'python' is available in PATH."
