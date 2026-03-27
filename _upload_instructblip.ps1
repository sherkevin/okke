# Upload InstructBLIP shards to server, verify, then delete local copies

$SshKey   = "C:\Users\shers\.ssh\id_ed25519_autodl"
$SshHost  = "root@connect.westd.seetacloud.com"
$SshPort  = "23427"
$Remote   = "/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b"
$LocalDir = "D:\chromedown"

$Shards = @(
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors"
)

Write-Host "=== InstructBLIP Upload ===" -ForegroundColor Cyan
Write-Host "$(Get-Date -Format 'HH:mm:ss') Start"
Write-Host ""

foreach ($Shard in $Shards) {
    $Local  = "$LocalDir\$Shard"
    $Remote_ = "${Remote}/$Shard"

    if (-not (Test-Path $Local)) {
        Write-Host "[$Shard] NOT FOUND locally, skipping" -ForegroundColor Red
        continue
    }

    $LocalGB = [math]::Round((Get-Item $Local).Length / 1GB, 2)
    Write-Host "[$Shard] Local size: $LocalGB GB" -ForegroundColor Yellow
    Write-Host "$(Get-Date -Format 'HH:mm:ss') Uploading..."

    # Upload via scp
    & scp -i $SshKey -P $SshPort `
          -o StrictHostKeyChecking=no `
          -o ConnectTimeout=30 `
          $Local "${SshHost}:${Remote_}"
    $exit = $LASTEXITCODE

    if ($exit -ne 0) {
        Write-Host "  [FAIL] scp exited $exit - keeping local file" -ForegroundColor Red
        continue
    }

    # Verify remote size
    $rsize = (ssh -i $SshKey -p $SshPort -o StrictHostKeyChecking=no `
        $SshHost "stat -c%s '$Remote_' 2>/dev/null || echo 0").Trim()
    $rGB = [math]::Round([int64]$rsize / 1GB, 2)

    Write-Host "  Remote: $rGB GB  (local was $LocalGB GB)"

    $lbytes = (Get-Item $Local).Length
    if ([int64]$rsize -ge ($lbytes - 1024)) {
        Write-Host "  [OK] Verified. Deleting local copy..." -ForegroundColor Green
        Remove-Item $Local -Force
        Write-Host "  [DONE] Local deleted" -ForegroundColor DarkGray
    } else {
        Write-Host "  [WARN] Remote size mismatch! Keeping local." -ForegroundColor Yellow
    }
    Write-Host "$(Get-Date -Format 'HH:mm:ss') Done $Shard`n"
}

# Final server check
Write-Host "=== Server final state ===" -ForegroundColor Cyan
ssh -i $SshKey -p $SshPort -o StrictHostKeyChecking=no $SshHost `
    "ls -lh $Remote/model-*.safetensors 2>/dev/null"
Write-Host ""
Write-Host "$(Get-Date -Format 'HH:mm:ss') All done."
