#!/usr/bin/env bash
# scripts/start.sh
# ============================================================
# One-command startup for the Disease Prediction MLOps stack.
# Run from repository root:
#   ./scripts/start.sh
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo " Starting Disease Prediction MLOps Platform"
echo "============================================================"

# 1. Prerequisite checks
if ! command -v docker &> /dev/null; then
    echo "ERROR: 'docker' command not found. Please install Docker." >&2
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "ERROR: 'docker compose' plugin not found. Please install Docker Compose v2." >&2
    exit 1
fi

# 2. Environment file check
if [ ! -f .env ]; then
    echo "[!] .env not found. Creating from .env.example..."
    cp .env.example .env
fi

# 3. Create required runtime directories if missing
mkdir -p data/prod_logs data/retraining reports/drift

# 4. Build and start services
echo "[*] Building and starting Docker Compose services in background..."
docker compose up -d --build

# 5. Wait for API health check
echo "[*] Waiting for API health check at http://localhost:8000/health..."
MAX_RETRIES=30
RETRY_COUNT=0
HEALTHY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        HEALTHY=true
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep 2
done

if [ "$HEALTHY" = true ]; then
    echo "[+] API is healthy!"
else
    echo "[!] API did not report healthy within 60s, but containers are starting up."
fi

# 6. Service dashboard summary
echo ""
echo "============================================================"
echo " MLOps Services Active"
echo "============================================================"
echo " - FastAPI API Docs:    http://localhost:8000/docs"
echo " - FastAPI Health:      http://localhost:8000/health"
echo " - Prometheus Metrics:  http://localhost:9090"
echo " - Grafana Dashboard:   http://localhost:3000 (admin / see .env)"
echo " - Airflow Webserver:   http://localhost:8080 (admin / see .env)"
echo "============================================================"
echo "To view live logs:   docker compose logs -f api"
echo "To stop stack:       ./scripts/stop.sh"
echo "============================================================"
