# scripts/stop.ps1
# ============================================================
# One-command teardown for the Disease Prediction MLOps stack.
# Run from the repository root:
#   .\scripts\stop.ps1
#
# Options (pass as arguments):
#   -Volumes   Also remove named volumes (wipes all persistent data)
#   -Rmi       Also remove built Docker images
# ============================================================

param(
    [switch]$Volumes,
    [switch]$Rmi
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Yellow
Write-Host "  Disease Prediction MLOps Stack" -ForegroundColor Yellow
Write-Host "  Stopping all services..." -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow
Write-Host ""

# ── 1. Verify we are at the repo root ────────────────────────
if (-not (Test-Path "docker-compose.yml")) {
    Write-Error "ERROR: docker-compose.yml not found. Run this script from the repository root."
    exit 1
}

# ── 2. Build the docker compose down command ─────────────────
$downArgs = @("compose", "down")

if ($Volumes) {
    $downArgs += "-v"
    Write-Host "  [!] --Volumes flag set: named volumes will be DELETED." -ForegroundColor Red
    Write-Host "      This removes MLflow data, feedback, prod logs, Grafana state." -ForegroundColor Red
    Write-Host ""
}

if ($Rmi) {
    $downArgs += "--rmi", "all"
    Write-Host "  [!] --Rmi flag set: built Docker images will be REMOVED." -ForegroundColor Red
    Write-Host ""
}

# ── 3. Stop services ─────────────────────────────────────────
Write-Host "[1/2] Stopping and removing containers..." -ForegroundColor Yellow
& docker @downArgs

# ── 4. Summary ───────────────────────────────────────────────
Write-Host ""
Write-Host "[2/2] Remaining Docker resources:" -ForegroundColor Yellow
docker compose ps 2>$null
docker volume ls --filter "name=diseaseprediction" 2>$null

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Stack stopped." -ForegroundColor Green
if (-not $Volumes) {
    Write-Host "  Named volumes retained (data preserved)." -ForegroundColor Green
    Write-Host "  To also remove volumes: .\scripts\stop.ps1 -Volumes" -ForegroundColor DarkGray
}
Write-Host "  To restart: .\scripts\start.ps1" -ForegroundColor DarkGray
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
