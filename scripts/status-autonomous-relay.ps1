$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$StateDir = Join-Path $Root ".gemma-research"
$PidFile = Join-Path $StateDir "autonomous_relay.pid"
$PauseFile = Join-Path $StateDir "autonomous_relay.pause"
$LogFile = Join-Path $StateDir "autonomous_relay.log"
$WrapperLogFile = Join-Path $StateDir "autonomous_relay.wrapper.log"
$StatusFile = Join-Path $StateDir "autonomous_relay_status.json"

if (Test-Path $PidFile) {
    $RelayPid = [int](Get-Content $PidFile -Raw)
    $Process = Get-Process -Id $RelayPid -ErrorAction SilentlyContinue
    if ($Process) {
        Write-Host "running: true"
        Write-Host "pid: $RelayPid"
    } else {
        Write-Host "running: false"
        Write-Host "stale_pid_cleared: $RelayPid"
        Remove-Item -Force $PidFile
    }
} else {
    Write-Host "running: false"
}

Write-Host "pause_requested: $(Test-Path $PauseFile)"

if (Test-Path $StatusFile) {
    Write-Host "`nstatus:"
    Get-Content $StatusFile
}

if (Test-Path $LogFile) {
    Write-Host "`nlog tail:"
    Get-Content $LogFile -Tail 40
}

if (Test-Path $WrapperLogFile) {
    Write-Host "`nwrapper log tail:"
    Get-Content $WrapperLogFile -Tail 20
}
