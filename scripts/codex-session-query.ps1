param(
    [Parameter(Position = 0)]
    [string]$Query,

    [switch]$Exact,
    [switch]$Json,
    [switch]$First,
    [int]$Limit = 0,
    [string]$CodexHome
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Stop-WithMessage {
    param(
        [string]$Message,
        [int]$Code = 1
    )

    [Console]::Error.WriteLine($Message)
    exit $Code
}

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

function Read-JsonLines {
    param([string]$Path)

    $lineNo = 0
    foreach ($line in [System.IO.File]::ReadLines($Path)) {
        $lineNo += 1
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        try {
            $obj = $line | ConvertFrom-Json -ErrorAction Stop
            $obj | Add-Member -NotePropertyName '_line' -NotePropertyValue $lineNo -Force
            $obj
        } catch {
            Write-Warning "Skipping malformed JSON in $Path line ${lineNo}: $($_.Exception.Message)"
        }
    }
}

function Get-SortDate {
    param([object]$Value)

    if ($null -eq $Value) {
        return [DateTimeOffset]::MinValue
    }
    try {
        return [DateTimeOffset]::Parse([string]$Value)
    } catch {
        return [DateTimeOffset]::MinValue
    }
}

function Build-SessionFileMap {
    param([string]$SessionsDir)

    $map = @{}
    if (-not (Test-Path -LiteralPath $SessionsDir)) {
        return $map
    }

    $uuidPattern = '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
    foreach ($file in Get-ChildItem -LiteralPath $SessionsDir -Recurse -File -Filter '*.jsonl') {
        if ($file.Name -match $uuidPattern) {
            $id = $Matches[0].ToLowerInvariant()
            if (-not $map.ContainsKey($id) -or $file.LastWriteTimeUtc -gt $map[$id].LastWriteTimeUtc) {
                $map[$id] = $file
            }
        }
    }
    return $map
}

function Read-SessionMeta {
    param([System.IO.FileInfo]$File)

    if ($null -eq $File -or -not (Test-Path -LiteralPath $File.FullName)) {
        return [pscustomobject]@{
            created_at = $null
            cwd = $null
            originator = $null
            cli_version = $null
        }
    }

    $firstLine = Get-Content -LiteralPath $File.FullName -TotalCount 1
    if (-not $firstLine) {
        return [pscustomobject]@{
            created_at = $null
            cwd = $null
            originator = $null
            cli_version = $null
        }
    }

    try {
        $first = $firstLine | ConvertFrom-Json -ErrorAction Stop
        $payload = $first.payload
        return [pscustomobject]@{
            created_at = $payload.timestamp
            cwd = $payload.cwd
            originator = $payload.originator
            cli_version = $payload.cli_version
        }
    } catch {
        return [pscustomobject]@{
            created_at = $null
            cwd = $null
            originator = $null
            cli_version = $null
        }
    }
}

function Test-ThreadNameMatch {
    param(
        [string]$ThreadName,
        [string]$Needle,
        [bool]$UseExact
    )

    if ($null -eq $ThreadName) {
        return $false
    }

    $comparison = [StringComparison]::OrdinalIgnoreCase
    if ($UseExact) {
        return $ThreadName.Equals($Needle, $comparison)
    }
    return $ThreadName.IndexOf($Needle, $comparison) -ge 0
}

function Write-HumanOutput {
    param(
        [string]$Needle,
        [object[]]$Rows,
        [string]$IndexPath
    )

    Write-Output "Codex session query: $Needle"
    Write-Output "Index: $IndexPath"
    Write-Output "Matches: $($Rows.Count)"
    Write-Output ''

    $i = 0
    foreach ($row in $Rows) {
        $i += 1
        Write-Output "[$i] $($row.thread_name)"
        Write-Output "  id: $($row.id)"
        Write-Output "  updated_at: $($row.updated_at)"
        Write-Output "  created_at: $($row.created_at)"
        Write-Output "  cwd: $($row.cwd)"
        Write-Output "  originator: $($row.originator)"
        Write-Output "  session_file: $($row.session_file)"
        Write-Output "  happy_resume: $($row.happy_resume)"
        Write-Output "  codex_resume: $($row.codex_resume)"
        Write-Output ''
    }
}

if ([string]::IsNullOrWhiteSpace($Query)) {
    Stop-WithMessage -Message 'Usage: codex-session-query.ps1 -Query <thread-name> [-Exact] [-Json] [-First] [-Limit N]' -Code 2
}

$resolvedCodexHome = Resolve-CodexHome -RequestedHome $CodexHome
$indexPath = Join-Path $resolvedCodexHome 'session_index.jsonl'
$sessionsDir = Join-Path $resolvedCodexHome 'sessions'

if (-not (Test-Path -LiteralPath $indexPath)) {
    Stop-WithMessage -Message "Codex session index not found: $indexPath" -Code 1
}

$records = @(Read-JsonLines -Path $indexPath | Where-Object {
    $_.PSObject.Properties.Name -contains 'id' -and
    $_.PSObject.Properties.Name -contains 'thread_name' -and
    (Test-ThreadNameMatch -ThreadName $_.thread_name -Needle $Query -UseExact ([bool]$Exact))
})

if ($records.Count -eq 0) {
    Stop-WithMessage -Message "No Codex session found matching '$Query' in $indexPath" -Code 1
}

$records = @($records | Sort-Object @{ Expression = { Get-SortDate $_.updated_at }; Descending = $true })
if ($First) {
    $Limit = 1
}
if ($Limit -gt 0) {
    $records = @($records | Select-Object -First $Limit)
}

$sessionFileById = Build-SessionFileMap -SessionsDir $sessionsDir
$rows = @()
foreach ($record in $records) {
    $id = ([string]$record.id).ToLowerInvariant()
    $file = $null
    if ($sessionFileById.ContainsKey($id)) {
        $file = $sessionFileById[$id]
    }

    $meta = Read-SessionMeta -File $file
    $threadName = [string]$record.thread_name
    $escapedThreadName = $threadName.Replace('"', '\"')
    $sessionPath = if ($null -ne $file) { $file.FullName } else { $null }
    $sessionLastWrite = if ($null -ne $file) { $file.LastWriteTime.ToString('o') } else { $null }

    $rows += [pscustomobject]@{
        id = $id
        thread_name = $threadName
        updated_at = $record.updated_at
        created_at = $meta.created_at
        cwd = $meta.cwd
        originator = $meta.originator
        cli_version = $meta.cli_version
        session_file = $sessionPath
        session_file_last_write = $sessionLastWrite
        index_path = $indexPath
        index_line = $record._line
        happy_resume = "happy codex --resume $id"
        codex_resume = "codex resume `"$escapedThreadName`""
    }
}

if ($Json) {
    $rows | ConvertTo-Json -Depth 6
} else {
    Write-HumanOutput -Needle $Query -Rows $rows -IndexPath $indexPath
}
