<#
.SYNOPSIS
    Launcher script for distributed regression hyperparameter tuning.
.DESCRIPTION
    Supports dynamic resource allocation and checkpointed execution on Windows.
.PARAMETER Profile
    Preset resource profile:
    - 'default': 15 threads, 16GB driver heap
    - 'eco':     8 threads, 10GB driver heap
    - 'turbo':   16 threads, 22GB driver heap
.PARAMETER Threads
    Explicit thread count (overrides profile).
.PARAMETER DriverMemory
    Explicit driver heap size (overrides profile).
.EXAMPLE
    .\scripts\run_regression_tuning.ps1
    .\scripts\run_regression_tuning.ps1 -Profile eco
    .\scripts\run_regression_tuning.ps1 -Profile turbo
#>
param(
    [ValidateSet("default", "eco", "turbo")]
    [string]$Profile = "default",
    [int]$Threads = 0,
    [string]$DriverMemory = ""
)

$ErrorActionPreference = "Continue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Determine resource settings from profile or explicit parameters
if ($Threads -eq 0 -or [string]::IsNullOrWhiteSpace($DriverMemory)) {
    switch ($Profile) {
        "eco" {
            if ($Threads -eq 0) { $Threads = 8 }
            if ([string]::IsNullOrWhiteSpace($DriverMemory)) { $DriverMemory = "10g" }
        }
        "turbo" {
            if ($Threads -eq 0) { $Threads = 16 }
            if ([string]::IsNullOrWhiteSpace($DriverMemory)) { $DriverMemory = "22g" }
        }
        Default {
            if ($Threads -eq 0) { $Threads = 15 }
            if ([string]::IsNullOrWhiteSpace($DriverMemory)) { $DriverMemory = "16g" }
        }
    }
}

# Set portable Java 17, Hadoop, and Python paths
$env:JAVA_HOME = Join-Path $projectRoot "temp\jdk-17\jdk-17.0.20.1+1"
$env:HADOOP_HOME = Join-Path $projectRoot "temp\hadoop"
$venvScripts = Join-Path $projectRoot ".venv\Scripts"
$env:SPARK_HOME = Join-Path $projectRoot ".venv\Lib\site-packages\pyspark"
$env:PATH = "$venvScripts;$($env:JAVA_HOME)\bin;$($env:HADOOP_HOME)\bin;$($env:PATH)"
$env:PYTHONUNBUFFERED = "1"
$env:PYSPARK_PYTHON = "python"
$env:PYSPARK_DRIVER_PYTHON = "python"

# Pass Spark resource settings to Python SparkSession builder
$env:SPARK_THREADS = "$Threads"
$env:SPARK_DRIVER_MEMORY = "$DriverMemory"

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$targetScript = Join-Path $projectRoot "sfdws_model_spark_regression.py"
$logsDir = Join-Path $projectRoot "logs"
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir -Force | Out-Null }
$logFile = Join-Path $logsDir "regression_training.log"
$checkpointFile = Join-Path $projectRoot "models\regression_grid_combined_checkpoint.json"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Spark Regression Tuning Launcher" -ForegroundColor Green
Write-Host "Profile:        $Profile"
Write-Host "Active Threads: $Threads"
Write-Host "Driver Memory:  $DriverMemory"
Write-Host "Java Home:      $($env:JAVA_HOME)"
Write-Host "Checkpoint:     $checkpointFile"
Write-Host "Log Output:     $logFile"
Write-Host "==========================================================" -ForegroundColor Cyan

$startTime = Get-Date

try {
    & $pythonExe -u $targetScript 2>&1 | Tee-Object -FilePath $logFile
    if ($LASTEXITCODE -ne 0) {
        throw "Python script exited with non-zero exit code $LASTEXITCODE"
    }
    $elapsed = (Get-Date) - $startTime
    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host "Execution finished successfully in $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
}
catch {
    Write-Host "`n==========================================================" -ForegroundColor Red
    Write-Host "Execution stopped or encountered an error: $_" -ForegroundColor Red
    Write-Host "==========================================================" -ForegroundColor Red
    exit 1
}
