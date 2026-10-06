# scripts/start.ps1
# ============================================================
# One-command startup for the Disease Prediction MLOps stack.
# Run from the repository root:
#   .\scripts\start.ps1
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Disease Prediction MLOps Stack" -ForegroundColor Cyan
Write-Host "  Starting all services..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Verify we are at the repo root ────────────────────────
if (-not (Test-Path "docker-compose.yml")) {
    Write-Error "ERROR: docker-compose.yml not found. Run this script from the repository root."
    exit 1
}

# ── 2. Verify .env exists ────────────────────────────────────
if (-not (Test-Path ".env")) {
    Write-Warning ".env not found — copying from .env.example. Edit .env before running in production."
    Copy-Item ".env.example" ".env"
}

# ── 3. Build and start services ──────────────────────────────
Write-Host "[1/3] Building Docker images..." -ForegroundColor Yellow
docker compose build

Write-Host "[2/3] Starting services in background..." -ForegroundColor Yellow
docker compose up -d

# ── 4. Wait for healthchecks ─────────────────────────────────
Write-Host "[3/3] Waiting for services to become healthy (up to 60s)..." -ForegroundColor Yellow
$timeout = 60
$elapsed = 0
$interval = 5

while ($elapsed -lt $timeout) {
    Start-Sleep -Seconds $interval
    $elapsed += $interval
    $ps = docker compose ps --format json 2>$null | ConvertFrom-Json
    $unhealthy = $ps | Where-Object { $_.Health -eq "starting" -or $_.Health -eq "unhealthy" }
    if ($unhealthy.Count -eq 0) { break }
    Write-Host "  ...still waiting ($elapsed s)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Services Status" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
docker compose ps

Write-Host ""
Write-Host "  FastAPI      → http://localhost:8000" -ForegroundColor Cyan
Write-Host "  Swagger UI   → http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  MLflow UI    → http://localhost:5000" -ForegroundColor Cyan
Write-Host "  Prometheus   → http://localhost:9090" -ForegroundColor Cyan
Write-Host "  Grafana      → http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Quick test:" -ForegroundColor DarkGray
Write-Host "  Invoke-RestMethod http://localhost:8000/health" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  To stop: .\scripts\stop.ps1" -ForegroundColor DarkGray
Write-Host ""
