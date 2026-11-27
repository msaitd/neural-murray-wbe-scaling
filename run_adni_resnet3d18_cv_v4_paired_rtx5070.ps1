[CmdletBinding()]
param(
    [ValidateSet("preflight", "cells", "aggregate", "all")]
    [string]$Stage = "preflight",

    [string]$MatlabPath = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$workRoot = Join-Path $projectRoot "adni_work"
$logRoot = Join-Path $workRoot "pipeline_logs"
$benchmarkCellRoot = Join-Path $workRoot "adni_3d_cv_v4\paired_rtx5070_cells"
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
        Get-Content -LiteralPath $stderrPath -Tail 120 -ErrorAction SilentlyContinue
        Get-Content -LiteralPath $stdoutPath -Tail 120 -ErrorAction SilentlyContinue
        throw "Paired RTX 5070 benchmark stopped at MATLAB stage '$Name'."
    }
    Write-Host "MATLAB stage '$Name' completed."
    Get-Content -LiteralPath $stdoutPath -Tail 30
}

function Invoke-BenchmarkCells {
    $previousFold = $env:ADNI_V4_BENCHMARK_FOLD
    $previousSeed = $env:ADNI_V4_BENCHMARK_SEED
    try {
        foreach ($fold in 1..5) {
            foreach ($seed in 20260820..20260824) {
                $cellOutput = Join-Path $benchmarkCellRoot `
                    "fold_$fold\seed_$seed\benchmark_cell_result.mat"
                if (Test-Path -LiteralPath $cellOutput -PathType Leaf) {
                    Write-Host "Skipping completed benchmark fold $fold seed $seed; aggregate validation will verify it."
                    continue
                }
                $env:ADNI_V4_BENCHMARK_FOLD = [string]$fold
                $env:ADNI_V4_BENCHMARK_SEED = [string]$seed
                Invoke-MatlabStage `
                    -Name "cv4_rtx5070_fold_${fold}_seed_${seed}" `
                    -ScriptName "run_adni_resnet3d18_cv_v4_benchmark_cell.m"
            }
        }
    }
    finally {
        if ($null -eq $previousFold) {
            Remove-Item Env:ADNI_V4_BENCHMARK_FOLD -ErrorAction SilentlyContinue
        }
        else {
            $env:ADNI_V4_BENCHMARK_FOLD = $previousFold
        }
        if ($null -eq $previousSeed) {
            Remove-Item Env:ADNI_V4_BENCHMARK_SEED -ErrorAction SilentlyContinue
        }
        else {
            $env:ADNI_V4_BENCHMARK_SEED = $previousSeed
        }
    }
}

switch ($Stage) {
    "preflight" {
        Invoke-MatlabStage `
            -Name "cv4_paired_rtx5070_preflight" `
            -ScriptName "check_adni_resnet3d18_cv_v4_paired_rtx5070_prelaunch.m"
    }
    "cells" {
        Invoke-BenchmarkCells
    }
    "aggregate" {
        Invoke-MatlabStage `
            -Name "cv4_paired_rtx5070_aggregate" `
            -ScriptName "aggregate_adni_resnet3d18_cv_v4_paired_rtx5070.m"
    }
    "all" {
        Invoke-MatlabStage `
            -Name "cv4_paired_rtx5070_preflight" `
            -ScriptName "check_adni_resnet3d18_cv_v4_paired_rtx5070_prelaunch.m"
        Invoke-BenchmarkCells
        Invoke-MatlabStage `
            -Name "cv4_paired_rtx5070_aggregate" `
            -ScriptName "aggregate_adni_resnet3d18_cv_v4_paired_rtx5070.m"
    }
}
Write-Host "Requested paired RTX 5070 benchmark stage '$Stage' completed successfully."
