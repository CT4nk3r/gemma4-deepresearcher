$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$StateDir = Join-Path $Root ".gemma-research"
$PidFile = Join-Path $StateDir "distillation.pid"
$PauseFile = Join-Path $StateDir "distillation.pause"

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
Set-Content -Path $PauseFile -Value "pause requested"

if (-not (Test-Path $PidFile)) {
    Write-Host "No distillation PID file found. Pause file created."
    exit 0
}

$DistillationPid = [int](Get-Content $PidFile -Raw)
$Process = Get-Process -Id $DistillationPid -ErrorAction SilentlyContinue
if (-not $Process) {
    Write-Host "No running process found for PID $DistillationPid. Pause file created."
    Remove-Item -Force $PidFile
    exit 0
}

Stop-Process -Id $DistillationPid
Remove-Item -Force $PidFile
Write-Host "Stopped distillation supervisor PID $DistillationPid"
