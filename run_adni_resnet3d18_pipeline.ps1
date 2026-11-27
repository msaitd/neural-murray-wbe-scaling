[CmdletBinding()]
param(
    [ValidateSet(
        "cv4internalpreflight",
        "cv4smokepreflight",
        "cv4smoke",
        "cv4fullpreflight",
        "cv4full",
        "cv4fullqa"
    )]
    [string]$Stage = "cv4internalpreflight",

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
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $MatlabPath -batch $batchCommand 1> $stdoutPath 2> $stderrPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0) {
        Get-Content -LiteralPath $stderrPath -Tail 120 -ErrorAction SilentlyContinue
        Get-Content -LiteralPath $stdoutPath -Tail 120 -ErrorAction SilentlyContinue
        throw "Pipeline stopped at MATLAB stage '$Name'."
    }
    Write-Host "MATLAB stage '$Name' completed."
    Get-Content -LiteralPath $stdoutPath -Tail 30
}

if ($Stage -eq "cv4full") {
    try {
        foreach ($fold in 1..5) {
            foreach ($seed in 20260820..20260824) {
                $env:ADNI_V4_CELL_FOLD = [string]$fold
                $env:ADNI_V4_CELL_SEED = [string]$seed
                $env:ADNI_V4_CELL_ACTION = "train-artifacts"
                Invoke-MatlabStage `
                    -Name "cv4_full_fold${fold}_seed${seed}_training" `
                    -ScriptName "run_adni_resnet3d18_cv_v4_validation_cell_stage.m"
                $env:ADNI_V4_CELL_ACTION = "finalize-validation"
                Invoke-MatlabStage `
                    -Name "cv4_full_fold${fold}_seed${seed}_validation" `
                    -ScriptName "run_adni_resnet3d18_cv_v4_validation_cell_stage.m"
            }
        }
        Invoke-MatlabStage `
            -Name "cv4_full_validation_training_aggregate" `
            -ScriptName "run_adni_resnet3d18_cv_v4_validation_training.m"
    }
    finally {
        Remove-Item Env:ADNI_V4_CELL_FOLD -ErrorAction SilentlyContinue
        Remove-Item Env:ADNI_V4_CELL_SEED -ErrorAction SilentlyContinue
        Remove-Item Env:ADNI_V4_CELL_ACTION -ErrorAction SilentlyContinue
    }
    return
}

if ($Stage -eq "cv4fullqa") {
    try {
        foreach ($fold in 1..5) {
            foreach ($seed in 20260820..20260824) {
                $env:ADNI_V4_CELL_FOLD = [string]$fold
                $env:ADNI_V4_CELL_SEED = [string]$seed
                Invoke-MatlabStage `
                    -Name "cv4_audit_fold${fold}_seed${seed}" `
                    -ScriptName "run_adni_resnet3d18_cv_v4_validation_audit_cell.m"
            }
        }
        Invoke-MatlabStage `
            -Name "cv4_validation_pretest_audit" `
            -ScriptName "validate_adni_resnet3d18_cv_v4_validation_training.m"
    }
    finally {
        Remove-Item Env:ADNI_V4_CELL_FOLD -ErrorAction SilentlyContinue
        Remove-Item Env:ADNI_V4_CELL_SEED -ErrorAction SilentlyContinue
    }
    return
}

$stageDefinitions = switch ($Stage) {
    "cv4internalpreflight" {
        @(@{
            Name = "cv4_locked_internal_validation_preflight"
            Script = "check_adni_resnet3d18_cv_v4_locked_internal_validation_prelaunch.m"
        })
    }
    "cv4smokepreflight" {
        @(@{
            Name = "cv4_engineering_smoke_preflight"
            Script = "check_adni_resnet3d18_cv_v4_engineering_smoke_prelaunch.m"
        })
    }
    "cv4smoke" {
        @(
            @{
                Name = "cv4_engineering_smoke"
                Script = "run_adni_resnet3d18_cv_v4_engineering_smoke.m"
            },
            @{
                Name = "cv4_engineering_smoke_validation"
                Script = "validate_adni_resnet3d18_cv_v4_engineering_smoke.m"
            }
        )
    }
    "cv4fullpreflight" {
        @(@{
            Name = "cv4_full_validation_preflight"
            Script = "check_adni_resnet3d18_cv_v4_full_validation_prelaunch.m"
        })
    }
}

foreach ($definition in $stageDefinitions) {
    Invoke-MatlabStage -Name $definition.Name -ScriptName $definition.Script
}
