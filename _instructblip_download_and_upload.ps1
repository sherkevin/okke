# Download InstructBLIP missing shards locally with curl.exe, SCP to server, delete local
# Requires: VPN active, curl.exe (built-in Windows 10+), OpenSSH

$LocalDir  = "D:\instructblip_tmp"
$RemoteDir = "/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b"
$SshKey    = "C:\Users\shers\.ssh\id_ed25519_autodl"
$SshHost   = "root@connect.westd.seetacloud.com"
$SshPort   = "23427"
$BaseUrl   = "https://huggingface.co/Salesforce/instructblip-vicuna-7b/resolve/main"

$Shards = @(
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors"
)

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
Write-Host "=== InstructBLIP Shard Downloader (curl + scp) ===" -ForegroundColor Cyan
Write-Host "Local temp : $LocalDir"
Write-Host "Server     : ${SshHost}:${RemoteDir}"
Write-Host ""

foreach ($Shard in $Shards) {
    $LocalFile  = Join-Path $LocalDir $Shard
    $RemoteFile = "$RemoteDir/$Shard"

    Write-Host "=== $Shard ===" -ForegroundColor Yellow

    # ── 1. Check if already on server ──────────────────────────────
    Write-Host "  Checking server..."
    $RemoteSize = (ssh -i $SshKey -p $SshPort -o StrictHostKeyChecking=no `
        -o ConnectTimeout=10 $SshHost `
        "test -f '$RemoteFile' && stat -c%s '$RemoteFile' || echo 0" 2>$null).Trim()

    if ([int64]$RemoteSize -gt 4000000000) {
        Write-Host "  [SKIP] Already on server ($([math]::Round([int64]$RemoteSize/1GB,2)) GB)" -ForegroundColor Green
        continue
    }
    Write-Host "  Not on server yet. Downloading locally..."

    # ── 2. Download with curl.exe (supports resume -C -) ───────────
    $Url = "$BaseUrl/$Shard"
    Write-Host "  curl: $Url"
    Write-Host "  Output: $LocalFile"
    Write-Host ""

    # curl.exe with resume (-C -), progress bar, follow redirects
    & curl.exe -L -C - --progress-bar -o $LocalFile $Url
    $curlExit = $LASTEXITCODE

    Write-Host ""
    if ($curlExit -ne 0) {
        Write-Host "  [FAIL] curl exited with code $curlExit" -ForegroundColor Red
        continue
    }

    $LocalItem = Get-Item $LocalFile -ErrorAction SilentlyContinue
    $LocalSize = if ($LocalItem) { $LocalItem.Length } else { 0 }
    if ($LocalSize -lt 4000000000) {
        Write-Host "  [FAIL] File too small: $([math]::Round($LocalSize/1MB,0)) MB" -ForegroundColor Red
        continue
    }
    Write-Host "  [OK] Downloaded: $([math]::Round($LocalSize/1GB,2)) GB" -ForegroundColor Green

    # ── 3. Upload to server via scp ─────────────────────────────────
    Write-Host "  Uploading to server..."
    & scp -i $SshKey -P $SshPort -o StrictHostKeyChecking=no `
        $LocalFile "${SshHost}:${RemoteFile}"
    $scpExit = $LASTEXITCODE

    if ($scpExit -eq 0) {
        Write-Host "  [OK] Uploaded" -ForegroundColor Green
        # Verify
        $VerifySize = (ssh -i $SshKey -p $SshPort -o StrictHostKeyChecking=no `
            $SshHost "stat -c%s '$RemoteFile' 2>/dev/null || echo 0").Trim()
        Write-Host "  [VERIFY] Remote: $([math]::Round([int64]$VerifySize/1GB,2)) GB"
        # Delete local
        Remove-Item $LocalFile -Force
        Write-Host "  [CLEAN] Local copy deleted" -ForegroundColor DarkGray
    } else {
        Write-Host "  [FAIL] scp exited with code $scpExit - local file kept" -ForegroundColor Red
    }
    Write-Host ""
}

# Final server status
Write-Host "=== Server final status ===" -ForegroundColor Cyan
ssh -i $SshKey -p $SshPort -o StrictHostKeyChecking=no $SshHost `
    "ls -lh /root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b/model-*.safetensors 2>/dev/null"

# Clean up local dir if empty
$remaining = Get-ChildItem $LocalDir -ErrorAction SilentlyContinue
if ($remaining.Count -eq 0) {
    Remove-Item $LocalDir -Force
    Write-Host "Temp dir removed." -ForegroundColor DarkGray
} else {
    Write-Host "Remaining local files: $($remaining.Count) in $LocalDir" -ForegroundColor Yellow
}
