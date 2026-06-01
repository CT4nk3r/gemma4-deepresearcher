$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$StateDir = Join-Path $Root ".gemma-research"
$PidFile = Join-Path $StateDir "autonomous_relay.pid"
$PauseFile = Join-Path $StateDir "autonomous_relay.pause"

New-Item -ItemType Directory -Force $StateDir | Out-Null
Set-Content -Path $PauseFile -Value "pause requested at $(Get-Date -Format o)"

if (-not (Test-Path $PidFile)) {
    Write-Host "Pause file created. No relay PID file found."
    exit 0
}

$RelayPid = [int](Get-Content $PidFile -Raw)
$Process = Get-Process -Id $RelayPid -ErrorAction SilentlyContinue
if (-not $Process) {
    Write-Host "No running process found for PID $RelayPid. Pause file created."
    Remove-Item -Force $PidFile
    exit 0
}

Stop-Process -Id $RelayPid
Remove-Item -Force $PidFile
Write-Host "Stopped autonomous relay PID $RelayPid"
