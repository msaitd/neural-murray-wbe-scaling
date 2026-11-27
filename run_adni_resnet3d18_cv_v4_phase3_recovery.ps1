[CmdletBinding()]
param(
    [string]$MatlabPath = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$workRoot = Join-Path $projectRoot "adni_work"
$logRoot = Join-Path $workRoot "pipeline_logs"

if ([string]::IsNullOrWhiteSpace($MatlabPath)) {
    $matlabCommand = Get-Command matlab -ErrorAction SilentlyContinue
    if ($null -eq $matlabCommand) {
        throw "MATLAB was not found on PATH; pass -MatlabPath explicitly."
    }
    $MatlabPath = $matlabCommand.Source
}

if (-not (Test-Path -LiteralPath $MatlabPath -PathType Leaf)) {
    throw "MATLAB executable not found: $MatlabPath"
}
if (-not (Test-Path -LiteralPath $workRoot -PathType Container)) {
    throw "ADNI work directory not found: $workRoot"
}
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Invoke-MatlabStage {
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter(Mandatory)]
        [string]$ScriptName
    )

    $scriptPath = Join-Path (Join-Path $projectRoot "matlab") $ScriptName
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
        throw "MATLAB stage script not found: $scriptPath"
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdoutPath = Join-Path $logRoot "${timestamp}_${Name}_stdout.log"
    $stderrPath = Join-Path $logRoot "${timestamp}_${Name}_stderr.log"
    $matlabProjectRoot = $projectRoot.Replace("\", "/").Replace("'", "''")
    $matlabScriptPath = $scriptPath.Replace("\", "/").Replace("'", "''")
    $batchCommand = "cd('$matlabProjectRoot'); run('$matlabScriptPath');"

    Write-Host "Starting MATLAB stage '$Name'..."
    Write-Host "stdout: $stdoutPath"
    Write-Host "stderr: $stderrPath"
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $MatlabPath -batch $batchCommand `
            1> $stdoutPath 2> $stderrPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0) {
        Write-Host "MATLAB stage '$Name' failed with exit code $exitCode."
        if (Test-Path -LiteralPath $stderrPath) {
            Get-Content -LiteralPath $stderrPath -Tail 80
        }
        if (Test-Path -LiteralPath $stdoutPath) {
            Get-Content -LiteralPath $stdoutPath -Tail 80
        }
        throw "Recovery pipeline stopped at MATLAB stage '$Name'."
    }

    Write-Host "MATLAB stage '$Name' completed."
    Get-Content -LiteralPath $stdoutPath -Tail 30
}

Invoke-MatlabStage `
    -Name "cv4_phase3_recovery_preflight" `
    -ScriptName "check_adni_resnet3d18_cv_v4_phase3_recovery_prelaunch.m"
try {
    foreach ($fold in 1..5) {
        foreach ($seed in 20260820..20260824) {
            $env:ADNI_V4_CELL_FOLD = [string]$fold
            $env:ADNI_V4_CELL_SEED = [string]$seed
            Invoke-MatlabStage `
                -Name "cv4_audit_recovery_fold${fold}_seed${seed}" `
                -ScriptName "run_adni_resnet3d18_cv_v4_validation_audit_cell_recovery.m"
        }
    }
    Invoke-MatlabStage `
        -Name "cv4_validation_pretest_audit_recovery" `
        -ScriptName "validate_adni_resnet3d18_cv_v4_validation_training_recovery.m"
    Invoke-MatlabStage `
        -Name "cv4_phase3_recovery_authorization" `
        -ScriptName "authorize_adni_resnet3d18_cv_v4_phase3_recovery.m"
}
finally {
    Remove-Item Env:ADNI_V4_CELL_FOLD -ErrorAction SilentlyContinue
    Remove-Item Env:ADNI_V4_CELL_SEED -ErrorAction SilentlyContinue
}
Write-Host "Phase-3 recovery audit completed successfully; outer test remains sealed."
