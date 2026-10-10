# Local Docker Deployment Guide

> **Branch:** `feature/phase-12-docker-deploy`  
> **Target Environment:** Local Developer Machine / Laptop (Windows, macOS, or Linux with Docker Desktop)  
> **Course Requirement:** Local Compose deployment only (No cloud services required).

---

## 1. System Requirements

| Specification | Minimum (Core Profile) | Recommended (Full Profile) |
|---|---|---|
| **Host RAM** | 6 GB (≥ 3 GB allocated to Docker) | 8 GB (≥ 4.5 GB allocated to Docker) |
| **Available Disk** | 5 GB free disk space | 8 GB free disk space |
| **CPU** | 2 cores | 4 cores |
| **OS** | Windows 10/11 with WSL 2, macOS, Linux | Windows 11 with WSL 2 |
| **Docker Engine** | Docker Desktop 4.25+ / Docker Engine 24+ | Docker Compose v2.20+ |

---

## 2. Service Catalog & Real Resource Consumption

Measured using `docker stats --no-stream` under active local execution:

| Service | Container Name | Host Port | Internal Port | Compose Profiles | Purpose | Real Memory Usage | Healthcheck Probe |
|---|---|---|---|---|---|---|---|
| **Frontend UI** | `disease-frontend` | `8080` | `80` | `core`, `full` | Nginx web server & `/api/` reverse proxy | **~10.4 MB** | `wget -q --spider http://127.0.0.1/` |
| **FastAPI Backend** | `disease-api` | `8000` | `8000` | `core`, `full` | Clinical decision support prediction service | **~187 MB** | `curl -f http://localhost:8000/health` |
| **MLflow Server** | `mlflow-server` | `5000` | `5000` | `core`, `full` | Model tracking server & SQLite registry | **~2.42 GB** | `curl -f http://localhost:5000/health` |
| **Prometheus** | `prometheus-monitoring` | `9090` | `9090` | `monitoring`, `full` | Time-series scraper for API metrics | **~86.9 MB** | `wget -q --spider http://localhost:9090/-/healthy` |
| **Grafana** | `grafana-dashboards` | `3000` | `3000` | `monitoring`, `full` | Observability dashboards & SLO panels | **~147.7 MB** | `wget -q --spider http://localhost:3000/api/health` |

### RAM Headroom & Compose Profiles
- **Core Profile (`core`):** `frontend` + `api` + `mlflow` consumes **~2.61 GB** RAM. Recommended for developer workstations with 8 GB total physical RAM where host OS requires 4 GB.
- **Monitoring Profile (`monitoring`):** `prometheus` + `grafana` consumes **~235 MB** RAM.
- **Full Stack (`full`):** All 5 containers consume **~2.85 GB** RAM.

---

## 3. Data Persistence Architecture (Named Volumes)

All persistent state is stored in Docker named volumes (`driver: local`). Data completely survives `docker compose down` and container restarts, and is only purged upon explicit `docker compose down -v`:

| Volume Name | Mounted Path Inside Container | What It Persists |
|---|---|---|
| `mlflow_data` | `/mlflow_data` (in `mlflow-server`) | SQLite database (`mlflow.db`) tracking runs, experiments, and model artifacts |
| `feedback_data` | `/app/data/feedback` (in `disease-api`) | Real clinician feedback rows (`feedback.csv`) submitted via the UI |
| `prod_logs_data` | `/app/data/prod_logs` (in `disease-api`) | Production prediction audit logs (`requests.csv`) |
| `prometheus_data` | `/prometheus` (in `prometheus-monitoring`) | Prometheus TSDB historical metrics and scrapings |
| `grafana_data` | `/var/lib/grafana` (in `grafana-dashboards`) | Grafana user settings, annotations, and dashboard states |

---

## 4. Secrets & Environment Configuration

All environment configuration is managed via `.env` (strictly ignored by `.git` via `.gitignore`).  
A complete template is tracked as [`.env.example`](file:///d:/Disease%20Prediction%20Using%20Machine%20Learning/.env.example).

### Key Variables:
- `COMPOSE_PROFILES`: Default profile to activate (`full` or `core`).
- `FRONTEND_PORT`: Port for the Web UI (default `8080`).
- `API_PORT`: Port for FastAPI backend (default `8000`).
- `ADMIN_API_KEY`: Secret key required to access administrative routes like `POST /admin/reload-model`.
- `GRAFANA_ADMIN_PASSWORD`: Administrator password for Grafana UI.
- `AIRFLOW_PORT`: Port reserved for offline Airflow DAG orchestration (set to `8088` to avoid port conflict with `8080`).

---

## 5. Step-by-Step Operations

### Option A: Standard Full Stack (Recommended)
```powershell
# 1. Start all 5 services with automatic health checks
.\scripts\start.ps1 full

# 2. Open Web UI in browser
Start-Process "http://localhost:8080"
```

### Option B: Core Lightweight Stack (For 8 GB Laptops)
```powershell
# Starts only Frontend, API, and MLflow (~2.6 GB RAM)
.\scripts\start.ps1 core
```

### Option C: Stopping the Stack
```powershell
# Normal stop (preserves all feedback, logs, and MLflow data)
.\scripts\stop.ps1

# Complete teardown (PURGES all persistent volumes and resets to clean slate)
.\scripts\stop.ps1 -Volumes
```

---

## 6. Service URLs & Quick Verification

| Service | URL | Quick Verification Command (PowerShell) | Expected Status |
|---|---|---|---|
| **Frontend UI** | [http://localhost:8080](http://localhost:8080) | `(Invoke-WebRequest http://localhost:8080 -UseBasicParsing).StatusCode` | `200` |
| **API Health** | [http://localhost:8080/api/health](http://localhost:8080/api/health) | `(Invoke-RestMethod http://localhost:8080/api/health).status` | `"healthy"` |
| **FastAPI Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | `(Invoke-WebRequest http://localhost:8000/docs -UseBasicParsing).StatusCode` | `200` |
| **MLflow UI** | [http://localhost:5000](http://localhost:5000) | `(Invoke-WebRequest http://localhost:5000 -UseBasicParsing).StatusCode` | `200` |
| **Prometheus** | [http://localhost:9090/-/healthy](http://localhost:9090/-/healthy) | `(Invoke-WebRequest http://localhost:9090/-/healthy -UseBasicParsing).StatusCode` | `200` |
| **Grafana** | [http://localhost:3000/api/health](http://localhost:3000/api/health) | `(Invoke-RestMethod http://localhost:3000/api/health).database` | `"ok"` |

---

## 7. Troubleshooting

| Issue / Symptom | Root Cause | Verified Resolution |
|---|---|---|
| `Port already allocated: 8080` | Another process or old container is holding port 8080. | Stop the conflicting process or change `FRONTEND_PORT=8082` in `.env`. |
| `disease-api` remains unhealthy | MLflow not ready or models directory missing in container. | Verify `models/production/model.joblib` exists before build. Check `docker compose logs api`. |
| Frontend shows "Cannot reach the API" | Reverse proxy cannot reach `http://api:8000`. | Run `docker compose ps` to ensure `disease-api` is healthy. |
| High RAM usage / system slowdown | Running all 5 services on limited RAM laptop. | Run `.\scripts\stop.ps1`, then start with lightweight core profile: `.\scripts\start.ps1 core`. |
| Grafana shows "Datasource Prometheus down" | Prometheus container still initializing. | Wait 15 seconds after startup. Verify Prometheus status at `http://localhost:9090/-/healthy`. |
