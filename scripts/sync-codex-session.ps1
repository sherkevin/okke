param(
    [string]$ThreadId,
    [string]$Query,
    [switch]$Exact,
    [string]$OutDir = (Join-Path (Get-Location) '.codex-session-sync'),
    [string]$CodexHome,
    [switch]$Watch,
    [int]$IntervalSeconds = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-CodexHome {
    param([string]$RequestedHome)

    if ($RequestedHome) {
        return $RequestedHome
    }
    if ($env:CODEX_HOME) {
        return $env:CODEX_HOME
    }
    if (-not $env:USERPROFILE) {
        throw 'Cannot resolve Codex home: CODEX_HOME and USERPROFILE are both unset.'
    }
    return (Join-Path $env:USERPROFILE '.codex')
}

function Normalize-PathForCompare {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }
    $clean = $Path
    if ($clean.StartsWith('\\?\')) {
        $clean = $clean.Substring(4)
    }
    try {
        return ([System.IO.Path]::GetFullPath($clean)).TrimEnd('\').ToLowerInvariant()
    } catch {
        return $clean.TrimEnd('\').ToLowerInvariant()
    }
}

function Read-SessionMeta {
    param([System.IO.FileInfo]$File)

    $empty = [pscustomobject]@{
        id = $null
        timestamp = $null
        cwd = $null
        originator = $null
        cli_version = $null
    }
    if ($null -eq $File -or -not (Test-Path -LiteralPath $File.FullName)) {
        return $empty
    }

    $firstLine = Get-Content -LiteralPath $File.FullName -TotalCount 1
    if (-not $firstLine) {
        return $empty
    }

    try {
        $first = $firstLine | ConvertFrom-Json -ErrorAction Stop
        $payload = $first.payload
        return [pscustomobject]@{
            id = $payload.id
            timestamp = $payload.timestamp
            cwd = $payload.cwd
            originator = $payload.originator
            cli_version = $payload.cli_version
        }
    } catch {
        return $empty
    }
}

function Find-SessionFileById {
    param(
        [string]$ResolvedCodexHome,
        [string]$Id
    )

    $roots = @(
        (Join-Path $ResolvedCodexHome 'sessions'),
        (Join-Path $ResolvedCodexHome 'archived_sessions')
    )
    foreach ($root in $roots) {
        if (-not (Test-Path -LiteralPath $root)) {
            continue
        }
        $match = Get-ChildItem -LiteralPath $root -Recurse -File -Filter "*$Id*.jsonl" |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if ($null -ne $match) {
            return $match
        }
    }
    return $null
}

function Find-LatestSessionForCurrentCwd {
    param([string]$ResolvedCodexHome)

    $sessionsDir = Join-Path $ResolvedCodexHome 'sessions'
    if (-not (Test-Path -LiteralPath $sessionsDir)) {
        throw "Codex sessions directory not found: $sessionsDir"
    }

    $current = Normalize-PathForCompare -Path ((Get-Location).ProviderPath)
    foreach ($file in (Get-ChildItem -LiteralPath $sessionsDir -Recurse -File -Filter '*.jsonl' |
        Sort-Object LastWriteTimeUtc -Descending)) {
        $meta = Read-SessionMeta -File $file
        if ((Normalize-PathForCompare -Path $meta.cwd) -eq $current) {
            return [pscustomobject]@{
                file = $file
                meta = $meta
                thread_name = $null
            }
        }
    }
    throw "No Codex session jsonl found for current cwd: $current"
}

function Resolve-Session {
    param([string]$ResolvedCodexHome)

    if ($ThreadId) {
        $file = Find-SessionFileById -ResolvedCodexHome $ResolvedCodexHome -Id $ThreadId
        if ($null -eq $file) {
            throw "No Codex session file found for thread id: $ThreadId"
        }
        return [pscustomobject]@{
            file = $file
            meta = (Read-SessionMeta -File $file)
            thread_name = $null
        }
    }

    if ($Query) {
        $queryScript = Join-Path $PSScriptRoot 'codex-session-query.ps1'
        if (-not (Test-Path -LiteralPath $queryScript)) {
            throw "Query script not found: $queryScript"
        }
        $args = @('-ExecutionPolicy', 'Bypass', '-File', $queryScript, '-Query', $Query, '-Json', '-First', '-CodexHome', $ResolvedCodexHome)
        if ($Exact) {
            $args += '-Exact'
        }
        $json = & powershell @args
        $record = $json | ConvertFrom-Json -ErrorAction Stop
        if ($record -is [array]) {
            $record = $record[0]
        }
        if (-not $record.session_file) {
            throw "No session_file reported for query: $Query"
        }
        $file = Get-Item -LiteralPath $record.session_file
        return [pscustomobject]@{
            file = $file
            meta = (Read-SessionMeta -File $file)
            thread_name = $record.thread_name
        }
    }

    return Find-LatestSessionForCurrentCwd -ResolvedCodexHome $ResolvedCodexHome
}

function Copy-LiveFile {
    param(
        [string]$SourcePath,
        [string]$DestinationPath
    )

    $destinationDir = Split-Path -Parent $DestinationPath
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
    $tmpPath = "$DestinationPath.tmp"

    $inputStream = [System.IO.File]::Open(
        $SourcePath,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite
    )
    try {
        $outputStream = [System.IO.File]::Open(
            $tmpPath,
            [System.IO.FileMode]::Create,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        try {
            $inputStream.CopyTo($outputStream)
        } finally {
            $outputStream.Dispose()
        }
    } finally {
        $inputStream.Dispose()
    }

    Move-Item -LiteralPath $tmpPath -Destination $DestinationPath -Force
}

function Sync-Once {
    param(
        [object]$Session,
        [string]$ResolvedOutDir
    )

    $threadId = if ($Session.meta.id) { [string]$Session.meta.id } else { [System.IO.Path]::GetFileNameWithoutExtension($Session.file.Name) }
    $threadDir = Join-Path $ResolvedOutDir $threadId
    $targetFile = Join-Path $threadDir $Session.file.Name

    Copy-LiveFile -SourcePath $Session.file.FullName -DestinationPath $targetFile

    $sourceInfo = Get-Item -LiteralPath $Session.file.FullName
    $targetInfo = Get-Item -LiteralPath $targetFile
    $manifest = [pscustomobject]@{
        thread_id = $threadId
        thread_name = $Session.thread_name
        source_file = $Session.file.FullName
        synced_file = $targetInfo.FullName
        source_bytes = $sourceInfo.Length
        synced_bytes = $targetInfo.Length
        source_last_write_utc = $sourceInfo.LastWriteTimeUtc.ToString('o')
        synced_at_utc = ([DateTimeOffset]::UtcNow).ToString('o')
        session_created_at = $Session.meta.timestamp
        cwd = $Session.meta.cwd
        originator = $Session.meta.originator
        mode = 'raw-jsonl'
        warning = 'Raw Codex session logs may contain prompts, tool outputs, file paths, and secrets. Keep this directory ignored unless you intentionally want to commit it.'
    }

    $manifestPath = Join-Path $threadDir 'manifest.json'
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

    Write-Output "Synced Codex session $threadId"
    Write-Output "  source: $($Session.file.FullName)"
    Write-Output "  target: $targetFile"
    Write-Output "  bytes:  $($targetInfo.Length)"
}

if ($IntervalSeconds -lt 1) {
    throw '-IntervalSeconds must be >= 1.'
}

$resolvedCodexHome = Resolve-CodexHome -RequestedHome $CodexHome
$session = Resolve-Session -ResolvedCodexHome $resolvedCodexHome
$resolvedOutDir = if ([System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir
} else {
    Join-Path ((Get-Location).ProviderPath) $OutDir
}

do {
    Sync-Once -Session $session -ResolvedOutDir $resolvedOutDir
    if ($Watch) {
        Start-Sleep -Seconds $IntervalSeconds
    }
} while ($Watch)
