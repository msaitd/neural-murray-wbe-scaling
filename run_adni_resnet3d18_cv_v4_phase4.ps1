[CmdletBinding()]
param(
    [ValidateSet("preflight", "outer", "validate", "analyze", "all")]
    [string]$Stage = "preflight",

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
        Get-Content -LiteralPath $stderrPath -Tail 120 -ErrorAction SilentlyContinue
        Get-Content -LiteralPath $stdoutPath -Tail 120 -ErrorAction SilentlyContinue
        throw "Phase-4 pipeline stopped at MATLAB stage '$Name'."
    }
    Write-Host "MATLAB stage '$Name' completed."
    Get-Content -LiteralPath $stdoutPath -Tail 50
}

$stages = switch ($Stage) {
    "preflight" {
        @([pscustomobject]@{
            Name = "cv4_phase4_atomic_preflight"
            Script = "check_adni_resnet3d18_cv_v4_phase4_atomic_prelaunch.m"
        })
    }
    "outer" {
        @([pscustomobject]@{
            Name = "cv4_atomic_outer_test"
            Script = "run_adni_resnet3d18_cv_v4_atomic_outer_test.m"
        })
    }
    "validate" {
        @([pscustomobject]@{
            Name = "cv4_atomic_outer_test_validation"
            Script = "validate_adni_resnet3d18_cv_v4_atomic_outer_test.m"
        })
    }
    "analyze" {
        @([pscustomobject]@{
            Name = "cv4_outer_test_analysis"
            Script = "analyze_adni_resnet3d18_cv_v4_outer_test.m"
        })
    }
    "all" {
        @(
            [pscustomobject]@{
                Name = "cv4_phase4_atomic_preflight"
                Script = "check_adni_resnet3d18_cv_v4_phase4_atomic_prelaunch.m"
            }
            [pscustomobject]@{
                Name = "cv4_atomic_outer_test"
                Script = "run_adni_resnet3d18_cv_v4_atomic_outer_test.m"
            }
            [pscustomobject]@{
                Name = "cv4_atomic_outer_test_validation"
                Script = "validate_adni_resnet3d18_cv_v4_atomic_outer_test.m"
            }
            [pscustomobject]@{
                Name = "cv4_outer_test_analysis"
                Script = "analyze_adni_resnet3d18_cv_v4_outer_test.m"
            }
        )
    }
}
foreach ($entry in $stages) {
    Invoke-MatlabStage -Name $entry.Name -ScriptName $entry.Script
}
Write-Host "Requested Phase-4 stage '$Stage' completed successfully."
