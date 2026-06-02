param(
    [double]$Hours = 18,
    [int]$TeacherTarget = 1000,
    [int]$TeacherCycleSize = 100,
    [int]$TeacherChunkSize = 25,
    [int]$TrainSteps = 60,
    [int]$TrainMaxLength = 512,
    [string]$VenvPython = ".venv-rocm\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$StateDir = Join-Path $Root ".gemma-research"
New-Item -ItemType Directory -Force $StateDir | Out-Null

$PidFile = Join-Path $StateDir "autonomous_relay.pid"
$PauseFile = Join-Path $StateDir "autonomous_relay.pause"
$DistillationPauseFile = Join-Path $StateDir "distillation.pause"
$LogFile = Join-Path $StateDir "autonomous_relay.log"
$WrapperLogFile = Join-Path $StateDir "autonomous_relay.wrapper.log"
$StatusFile = Join-Path $StateDir "autonomous_relay_status.json"

if (Test-Path $PidFile) {
    $ExistingPid = [int](Get-Content $PidFile -Raw)
    $ExistingProcess = Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue
    if ($ExistingProcess) {
        Write-Host "Autonomous relay is already running with PID $ExistingPid"
        exit 0
    }
    Remove-Item -Force $PidFile
}

Remove-Item -Force $PauseFile -ErrorAction SilentlyContinue
Remove-Item -Force $DistillationPauseFile -ErrorAction SilentlyContinue

$PythonPath = $VenvPython
if (-not [System.IO.Path]::IsPathRooted($PythonPath)) {
    $PythonPath = Join-Path $Root $PythonPath
}
if (-not (Test-Path $PythonPath -PathType Leaf)) {
    throw "Python not found: $PythonPath"
}

$Command = @(
    "Set-Location -LiteralPath '$Root';",
    "'$PythonPath' training\autonomous_relay.py",
    "--hours $Hours",
    "--teacher-target $TeacherTarget",
    "--teacher-cycle-size $TeacherCycleSize",
    "--teacher-chunk-size $TeacherChunkSize",
    "--train-steps $TrainSteps",
    "--train-max-length $TrainMaxLength",
    "--status-output '$StatusFile'",
    "--log-output '$LogFile'",
    "--pause-file '$PauseFile'",
    "--distillation-pause-file '$DistillationPauseFile'"
) -join " "

$Process = Start-Process -FilePath "powershell" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "$Command *> '$WrapperLogFile'") `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $PidFile -Value $Process.Id
Write-Host "Started autonomous relay PID $($Process.Id)"
Write-Host "Log: $LogFile"
Write-Host "Wrapper log: $WrapperLogFile"
Write-Host "Status: $StatusFile"
