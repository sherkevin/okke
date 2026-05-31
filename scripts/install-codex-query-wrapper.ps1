param(
    [string]$ProfilePath = $PROFILE.CurrentUserAllHosts,
    [string]$QueryScriptPath = (Join-Path $PSScriptRoot 'codex-session-query.ps1')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $QueryScriptPath)) {
    throw "Query script not found: $QueryScriptPath"
}

$resolvedScript = (Resolve-Path -LiteralPath $QueryScriptPath).Path
$escapedScript = $resolvedScript.Replace("'", "''")
$begin = '# >>> codex --query wrapper >>>'
$end = '# <<< codex --query wrapper <<<'

$block = @"
$begin
function codex {
    param([Parameter(ValueFromRemainingArguments = `$true)][string[]]`$CodexArgs)

    if (`$CodexArgs.Count -ge 2 -and `$CodexArgs[0] -eq '--query') {
        `$query = `$CodexArgs[1]
        `$rest = @()
        if (`$CodexArgs.Count -gt 2) {
            `$rest = `$CodexArgs[2..(`$CodexArgs.Count - 1)]
        }
        `$queryParams = @{ Query = `$query }
        for (`$i = 0; `$i -lt `$rest.Count; `$i += 1) {
            `$arg = `$rest[`$i]
            switch (`$arg) {
                { `$_ -in @('--exact', '-Exact') } { `$queryParams.Exact = `$true; break }
                { `$_ -in @('--json', '-Json') } { `$queryParams.Json = `$true; break }
                { `$_ -in @('--first', '-First') } { `$queryParams.First = `$true; break }
                { `$_ -in @('--limit', '-Limit') } {
                    `$i += 1
                    if (`$i -ge `$rest.Count) { throw 'codex --query --limit requires a number.' }
                    `$queryParams.Limit = [int]`$rest[`$i]
                    break
                }
                { `$_ -in @('--codex-home', '-CodexHome') } {
                    `$i += 1
                    if (`$i -ge `$rest.Count) { throw 'codex --query --codex-home requires a path.' }
                    `$queryParams.CodexHome = `$rest[`$i]
                    break
                }
                default {
                    throw "Unsupported codex --query option: `$arg"
                }
            }
        }
        & '$escapedScript' @queryParams
        return
    }

    `$native = @(Get-Command codex -CommandType Application -ErrorAction SilentlyContinue)
    if (`$native.Count -eq 0) {
        `$native = @(Get-Command codex -CommandType ExternalScript -ErrorAction SilentlyContinue)
    }
    if (`$native.Count -eq 0) {
        throw 'Native codex CLI not found on PATH.'
    }

    & `$native[0].Source @CodexArgs
}
$end
"@

$profileDir = Split-Path -Parent $ProfilePath
if ($profileDir -and -not (Test-Path -LiteralPath $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir | Out-Null
}

$existing = ''
if (Test-Path -LiteralPath $ProfilePath) {
    $existing = Get-Content -LiteralPath $ProfilePath -Raw
}

$start = $existing.IndexOf($begin)
$finish = $existing.IndexOf($end)

if ($start -ge 0 -and $finish -gt $start) {
    $finish += $end.Length
    $prefix = $existing.Substring(0, $start).TrimEnd()
    $suffix = $existing.Substring($finish).TrimStart()
    $newContent = @($prefix, $block, $suffix) -ne '' -join "`r`n`r`n"
} else {
    $newContent = @($existing.TrimEnd(), $block) -ne '' -join "`r`n`r`n"
}

Set-Content -LiteralPath $ProfilePath -Value ($newContent + "`r`n") -Encoding UTF8

Write-Output "Installed codex --query wrapper to: $ProfilePath"
Write-Output "Query script: $resolvedScript"
Write-Output "Open a new PowerShell window, or run this once now:"
Write-Output ". '$ProfilePath'"
