param(
    [int]$Target = 500,
    [int]$ChunkSize = 50,
    [string]$Model = "alibaba-nlp_tongyi-deepresearch-30b-a3b",
    [double]$Temperature = 0.1,
    [int]$MaxTokens = 2048,
    [string]$BaseUrl = "http://localhost:1234/v1"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$StateDir = Join-Path $Root ".gemma-research"
$PidFile = Join-Path $StateDir "distillation.pid"
$PauseFile = Join-Path $StateDir "distillation.pause"
$OutLog = Join-Path $StateDir "distillation.out.log"
$ErrLog = Join-Path $StateDir "distillation.err.log"

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
if (Test-Path $PauseFile) {
    Remove-Item -Force $PauseFile
}

if (Test-Path $PidFile) {
    $ExistingPid = [int](Get-Content $PidFile -Raw)
    $ExistingProcess = Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue
    if ($ExistingProcess) {
        Write-Host "Distillation supervisor is already running with PID $ExistingPid"
        exit 0
    }
}

$PythonArgs = @(
    "training\supervise_distillation.py",
    "--target", "$Target",
    "--chunk-size", "$ChunkSize",
    "--model", "$Model",
    "--temperature", "$Temperature",
    "--max-tokens", "$MaxTokens",
    "--base-url", "$BaseUrl"
)

$Process = Start-Process -FilePath "python" -ArgumentList $PythonArgs -WorkingDirectory $Root -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
Set-Content -Path $PidFile -Value $Process.Id
Write-Host "Started distillation supervisor with PID $($Process.Id)"
Write-Host "Logs:"
Write-Host "  $OutLog"
Write-Host "  $ErrLog"
