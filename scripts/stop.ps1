# scripts/stop.ps1
# ============================================================
# One-command teardown for the Disease Prediction MLOps stack.
# Run from repository root:
#   .\scripts\stop.ps1               # stops active services, preserves named volumes
#   .\scripts\stop.ps1 -Volumes      # stops and PURGES named volumes (wipes data)
#   .\scripts\stop.ps1 -Rmi          # also removes built Docker images
# ============================================================

param(
    [switch]$Volumes,
    [switch]$Rmi
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Yellow
Write-Host "  Disease Prediction MLOps Stack" -ForegroundColor Yellow
Write-Host "  Stopping services..." -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow
Write-Host ""

# 1. Verify we are at the repo root
if (-not (Test-Path "docker-compose.yml")) {
    Write-Error "ERROR: docker-compose.yml not found. Run this script from the repository root."
    exit 1
}

# 2. Build the docker compose down command
$downArgs = @("compose", "down")

if ($Volumes) {
    $downArgs += "-v"
    Write-Host "  [!] -Volumes flag passed: ALL named volumes will be REMOVED." -ForegroundColor Red
    Write-Host "      This purges MLflow sqlite db, feedback.csv, prod logs, and Grafana state." -ForegroundColor Red
    Write-Host ""
}

if ($Rmi) {
    $downArgs += "--rmi", "all"
    Write-Host "  [!] -Rmi flag passed: Built container images will be REMOVED." -ForegroundColor Red
    Write-Host ""
}

# 3. Stop services
Write-Host "[1/2] Stopping containers..." -ForegroundColor Yellow
& docker @downArgs

# 4. Summary
Write-Host ""
Write-Host "[2/2] Post-stop verification:" -ForegroundColor Yellow
docker compose ps 2>$null

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Stack stopped." -ForegroundColor Green
if (-not $Volumes) {
    Write-Host "  Named volumes retained (MLflow, feedback, and logs preserved)." -ForegroundColor Green
    Write-Host "  To wipe volumes next time: .\scripts\stop.ps1 -Volumes" -ForegroundColor DarkGray
} else {
    Write-Host "  Named volumes removed (clean slate)." -ForegroundColor Yellow
}
Write-Host "  To restart: .\scripts\start.ps1" -ForegroundColor DarkGray
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
