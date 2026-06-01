param(
    [string]$Input = "data\uncertainty_repair_seed_sft.jsonl",
    [string]$RawOutput = "data\uncertainty_repair_distilled_sft.jsonl",
    [string]$CleanOutput = "data\uncertainty_repair_clean_sft.jsonl",
    [string]$StatsOutput = "data\uncertainty_repair_clean_sft.stats.json",
    [string]$TeacherModel = "alibaba-nlp_tongyi-deepresearch-30b-a3b",
    [string]$BaseUrl = "http://localhost:1234/v1",
    [int]$Target = 60,
    [int]$MaxTokens = 1536,
    [double]$Temperature = 0.1,
    [switch]$WaitForTraining,
    [switch]$PauseRelayAfterTraining
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$StateDir = Join-Path $Root ".gemma-research"
New-Item -ItemType Directory -Force $StateDir | Out-Null

$PauseFile = Join-Path $StateDir "autonomous_relay.pause"
$PidFile = Join-Path $StateDir "autonomous_relay.pid"
$LogFile = Join-Path $StateDir "uncertainty_repair_distillation.log"
$DistillPauseFile = Join-Path $StateDir "uncertainty_repair_distillation.pause"

function Write-Log {
    param([string]$Message)
    $Line = "[$((Get-Date).ToString('o'))] $Message"
    Add-Content -Path $LogFile -Value $Line
    Write-Host $Line
}

function Get-TrainLoraProcess {
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like '*training*train_lora.py*' -or $_.CommandLine -like '*train_lora.py*' } |
        Select-Object -First 1
}

if (-not (Test-Path $Input -PathType Leaf)) {
    throw "Seed input not found: $Input"
}

if ($WaitForTraining) {
    Write-Log "Waiting for active train_lora.py process to finish"
    while ($true) {
        $TrainProcess = Get-TrainLoraProcess
        if (-not $TrainProcess) {
            break
        }
        Write-Log "Still training with PID $($TrainProcess.ProcessId)"
        Start-Sleep -Seconds 30
    }
}

if ($PauseRelayAfterTraining) {
    Write-Log "Requesting relay pause at safe point"
    Set-Content -Path $PauseFile -Value "pause after current training for uncertainty repair distillation"
    if (Test-Path $PidFile) {
        $RelayPid = [int](Get-Content $PidFile -Raw)
        while (Get-Process -Id $RelayPid -ErrorAction SilentlyContinue) {
            Write-Log "Waiting for relay PID $RelayPid to exit after pause request"
            Start-Sleep -Seconds 10
        }
    }
}

Write-Log "Starting LM Studio server"
& lms server start *>> $LogFile

Write-Log "Loading teacher model $TeacherModel"
& lms load $TeacherModel --identifier $TeacherModel --gpu max --context-length 4096 -y *>> $LogFile

try {
    Write-Log "Distilling uncertainty repair examples"
    & python "training\distill_with_lmstudio.py" `
        --input $Input `
        --output $RawOutput `
        --base-url $BaseUrl `
        --model $TeacherModel `
        --max-examples $Target `
        --temperature $Temperature `
        --max-tokens $MaxTokens `
        --timeout 600 `
        --sleep 0.1 `
        --retries 3 `
        --retry-sleep 10 `
        --resume `
        --on-error skip *>> $LogFile
    if ($LASTEXITCODE -ne 0) {
        throw "distill_with_lmstudio.py exited with code $LASTEXITCODE"
    }

    Write-Log "Cleaning uncertainty repair examples"
    & python "training\clean_sft_dataset.py" --input $RawOutput --output $CleanOutput *>> $LogFile
    if ($LASTEXITCODE -ne 0) {
        throw "clean_sft_dataset.py exited with code $LASTEXITCODE"
    }

    Write-Log "Validating clean uncertainty repair examples"
    Remove-Item -Force $StatsOutput -ErrorAction SilentlyContinue
    & python "training\validate_sft_dataset.py" $CleanOutput --json *>> $StatsOutput
    if ($LASTEXITCODE -ne 0) {
        throw "validate_sft_dataset.py exited with code $LASTEXITCODE"
    }
} finally {
    Write-Log "Unloading teacher model $TeacherModel"
    & lms unload $TeacherModel *>> $LogFile
}

Write-Log "Uncertainty repair distillation complete: $CleanOutput"
