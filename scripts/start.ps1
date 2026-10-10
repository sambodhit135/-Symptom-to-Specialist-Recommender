# scripts/start.ps1
# ============================================================
# One-command startup for the Disease Prediction MLOps stack.
# Run from repository root:
#   .\scripts\start.ps1               # starts "full" profile (all 5 services)
#   .\scripts\start.ps1 core          # starts "core" profile (frontend + api + mlflow)
#   .\scripts\start.ps1 monitoring    # starts "monitoring" profile (prometheus + grafana)
#   .\scripts\start.ps1 -Build        # force build before startup
# ============================================================

param(
    [Parameter(Position = 0)]
    [ValidateSet("core", "monitoring", "full")]
    [string]$Profile = "full",

    [switch]$Build
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Disease Prediction MLOps Stack" -ForegroundColor Cyan
Write-Host "  Profile: $Profile" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verify we are at the repo root
if (-not (Test-Path "docker-compose.yml")) {
    Write-Error "ERROR: docker-compose.yml not found. Run this script from the repository root."
    exit 1
}

# 2. Verify .env exists
if (-not (Test-Path ".env")) {
    Write-Warning ".env not found - copying from .env.example. Edit .env if needed."
    Copy-Item ".env.example" ".env"
}

# 3. Build images if requested
if ($Build) {
    Write-Host "[1/3] Building Docker images for profile: $Profile..." -ForegroundColor Yellow
    docker compose --profile $Profile build
} else {
    Write-Host "[1/3] Skipping build (pass -Build to rebuild)..." -ForegroundColor DarkGray
}

# 4. Start services
Write-Host "[2/3] Starting services in background (profile: $Profile)..." -ForegroundColor Yellow
docker compose --profile $Profile up -d

# 5. Wait for all started services to become healthy
Write-Host "[3/3] Waiting for services to become healthy (up to 90s)..." -ForegroundColor Yellow
$timeout = 90
$elapsed = 0
$interval = 5

while ($elapsed -lt $timeout) {
    Start-Sleep -Seconds $interval
    $elapsed += $interval
    
    $raw = docker compose --profile $Profile ps --format json 2>$null
    if ($raw) {
        $containers = $raw | ConvertFrom-Json
        $pending = $containers | Where-Object {
            $_.Health -eq "starting" -or ($_.State -eq "running" -and $_.Health -eq "unhealthy")
        }
        if (-not $pending -or $pending.Count -eq 0) {
            $running = $containers | Where-Object { $_.State -eq "running" }
            if ($running.Count -gt 0) { break }
        }
    }
    Write-Host "  ...waiting for health probes ($elapsed s / $timeout s)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Stack Status ($Profile)" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
docker compose --profile $Profile ps

Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Cyan
if ($Profile -eq "core" -or $Profile -eq "full") {
    Write-Host "  Frontend Web UI   -> http://localhost:8080" -ForegroundColor Cyan
    Write-Host "  FastAPI Backend   -> http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  FastAPI Docs      -> http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "  MLflow UI         -> http://localhost:5000" -ForegroundColor Cyan
}
if ($Profile -eq "monitoring" -or $Profile -eq "full") {
    Write-Host "  Prometheus Scraper-> http://localhost:9090" -ForegroundColor Cyan
    Write-Host "  Grafana Dashboard -> http://localhost:3000" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Verifying HTTP responses..." -ForegroundColor Yellow
try {
    if ($Profile -eq "core" -or $Profile -eq "full") {
        $ui = (Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing -TimeoutSec 5).StatusCode
        $api = (Invoke-RestMethod -Uri "http://localhost:8080/api/health" -TimeoutSec 5).status
        Write-Host "  Frontend (port 8080): HTTP $ui (Health: $api)" -ForegroundColor Green
    }
    if ($Profile -eq "monitoring" -or $Profile -eq "full") {
        $prom = (Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -UseBasicParsing -TimeoutSec 5).StatusCode
        $graf = (Invoke-RestMethod -Uri "http://localhost:3000/api/health" -TimeoutSec 5).database
        Write-Host "  Prometheus (port 9090): HTTP $prom | Grafana (port 3000): $graf" -ForegroundColor Green
    }
} catch {
    Write-Warning "Initial HTTP probe caught: $_"
}

Write-Host ""
Write-Host "To stop the stack:" -ForegroundColor DarkGray
Write-Host "  .\scripts\stop.ps1" -ForegroundColor DarkGray
Write-Host ""
