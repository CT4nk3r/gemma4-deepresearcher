param(
    [string]$Adapter = "runs\gemma4-e4b-deepresearch-lora-latest",
    [string]$BaseModel = "google/gemma-4-e4b-it",
    [string]$VenvPython = ".venv-rocm\Scripts\python.exe",
    [string]$EvalSet = "data\eval_set.jsonl",
    [string]$OutputDir = "eval\results",
    [int]$PollSeconds = 30,
    [int]$StableSeconds = 15,
    [int]$TimeoutMinutes = 0,
    [int]$MaxNewTokens = 512,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Resolve-RepoPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path $Root $PathValue)
}

function Test-AdapterReady {
    param([string]$AdapterDir)
    return (
        (Test-Path $AdapterDir -PathType Container) -and
        (Test-Path (Join-Path $AdapterDir "adapter_config.json") -PathType Leaf) -and
        (Test-Path (Join-Path $AdapterDir "adapter_model.safetensors") -PathType Leaf)
    )
}

$AdapterPath = Resolve-RepoPath $Adapter
$PythonPath = Resolve-RepoPath $VenvPython

if (-not (Test-Path $PythonPath -PathType Leaf)) {
    throw "Python not found: $PythonPath"
}

$StartTime = Get-Date
while (-not (Test-AdapterReady $AdapterPath)) {
    if ($TimeoutMinutes -gt 0 -and ((Get-Date) - $StartTime).TotalMinutes -ge $TimeoutMinutes) {
        throw "Timed out waiting for adapter: $AdapterPath"
    }
    Write-Host "Waiting for adapter files in $AdapterPath"
    Start-Sleep -Seconds $PollSeconds
}

if ($StableSeconds -gt 0) {
    Write-Host "Adapter files found; waiting $StableSeconds seconds for copy to settle"
    Start-Sleep -Seconds $StableSeconds
    if (-not (Test-AdapterReady $AdapterPath)) {
        throw "Adapter files disappeared while waiting for a stable copy: $AdapterPath"
    }
}

if (-not $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL) {
    $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"
}

& $PythonPath "training\eval_adapter.py" `
    --adapter $Adapter `
    --base-model $BaseModel `
    --eval-set $EvalSet `
    --output-dir $OutputDir `
    --max-new-tokens $MaxNewTokens `
    @ExtraArgs

exit $LASTEXITCODE
