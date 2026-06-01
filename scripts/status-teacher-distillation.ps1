$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$StateDir = Join-Path $Root ".gemma-research"
$PidFile = Join-Path $StateDir "distillation.pid"
$PauseFile = Join-Path $StateDir "distillation.pause"
$StatusFile = Join-Path $StateDir "distillation_status.json"
$OutLog = Join-Path $StateDir "distillation.out.log"
$ErrLog = Join-Path $StateDir "distillation.err.log"

Set-Location $Root

if (Test-Path $PidFile) {
    $DistillationPid = [int](Get-Content $PidFile -Raw)
    $Process = Get-Process -Id $DistillationPid -ErrorAction SilentlyContinue
    if ($Process) {
        Write-Host "running: true"
        Write-Host "pid: $DistillationPid"
    } else {
        Write-Host "running: false"
        Write-Host "stale_pid_cleared: $DistillationPid"
        Remove-Item -Force $PidFile
    }
} else {
    Write-Host "running: false"
}
Write-Host "pause_requested: $(Test-Path $PauseFile)"

function Count-JsonlExamples {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return 0
    }
    return (Get-Content $Path | Where-Object { $_.Trim() } | Measure-Object).Count
}

$RawPath = "data\teacher_distilled_starter_sft.jsonl"
$CleanPath = "data\teacher_distilled_clean_sft.jsonl"
Write-Host "$RawPath`: $(Count-JsonlExamples $RawPath) examples"
Write-Host "$CleanPath`: $(Count-JsonlExamples $CleanPath) examples"

if (Test-Path $StatusFile) {
    Write-Host ""
    Write-Host "status:"
    Get-Content $StatusFile
}

if (Test-Path $OutLog) {
    Write-Host ""
    Write-Host "last stdout:"
    Get-Content $OutLog -Tail 20
}

if (Test-Path $ErrLog) {
    $ErrTail = Get-Content $ErrLog -Tail 20
    if ($ErrTail) {
        Write-Host ""
        Write-Host "last stderr:"
        $ErrTail
    }
}
