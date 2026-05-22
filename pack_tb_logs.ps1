$RarExe     = "C:\Program Files\WinRAR\Rar.exe"
$ScratchDir = "$PSScriptRoot\src\audio\lab3_results\vits_ruslan_scratch"
$OutRar     = "$PSScriptRoot\vits_ruslan_tb_logs.rar"

if (-not (Test-Path $ScratchDir)) { Write-Error "Not found: $ScratchDir"; exit 1 }

$runs = Get-ChildItem "$ScratchDir\vits_ruslan_scratch-*" -Directory
if (-not $runs) { Write-Host "No run directories found."; exit 0 }

Write-Host "Found $($runs.Count) run(s):"
$runs | ForEach-Object { Write-Host "  $($_.Name)" }

# Pack all run dirs, skip checkpoints and phoneme cache
& $RarExe a -r `
    "-x*.pth" `
    "-x*.npy" `
    $OutRar   `
    "$ScratchDir\vits_ruslan_scratch-*"

if ($LASTEXITCODE -eq 0) {
    $sizeMB = [math]::Round((Get-Item $OutRar).Length / 1MB, 1)
    Write-Host "`nDone: $OutRar ($sizeMB MB)"
} else {
    Write-Error "Rar.exe exited with code $LASTEXITCODE"
}
