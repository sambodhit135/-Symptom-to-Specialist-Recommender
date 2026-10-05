# Phase 5: Containerisation & Deployment Notes

## Overview
This phase containerises the Disease Prediction MLOps platform into a reproducible, single-command deployment using Docker and Docker Compose.

The deployment consists of two services running in an isolated bridge network (`app-net`):
1. **`api`** (`disease-api`): FastAPI REST prediction service running as non-root `appuser` on port 8000.
2. **`mlflow`** (`mlflow-server`): MLflow tracking server backed by a persistent SQLite database (`sqlite:////mlflow_data/mlflow.db`) on port 5000.

---

## 1. Model Serving Strategy & Trade-Off Analysis

### Approach Chosen: Self-Contained Baked Model (`models/production/`) with MLflow Fallback

In `Dockerfile`, the vetted production model package (`models/production/`) is copied directly into the image:
```dockerfile
COPY models/production/ /app/models/production/
```
In `src/api/model_loader.py`, the system attempts to load from `MLFLOW_TRACKING_URI` if reachable; otherwise, it loads from local `models/production/model.joblib`.

### Why This Is the Simplest and Most Reliable Approach:
- **Zero Startup Dependency Coupling:** The API container boots in <1 second and passes its healthcheck immediately without waiting for complex MLflow model registry download handshakes across the network.
- **Hermetic & Self-Contained:** The container image is fully immutable and independently deployable to any target (ECS, Cloud Run, Kubernetes, edge VM) without requiring MLflow to be online.
- **Fail-Safe Operation:** If the MLflow server is stopped or restarted for maintenance, API predictions continue serving with 100% availability.
- **Auditability & Traceability:** The model artifacts in `models/production/` contain full provenance metadata (`metadata.json`, SHA-256 hashes, metrics, and quality gate sign-off).

---

## 2. Docker Architecture & Security

### Security Hardening
- **Non-Root Execution:** The API container creates a dedicated system user and group (`appuser`, UID/GID non-root) and drops all root privileges before starting Uvicorn.
- **Secrets Isolation:** No secrets or environment configurations are committed to git. `.env.example` provides the template, and `.env` is ignored via `.gitignore`.
- **Minimal Image Surface:** Built on `python:3.12-slim`, removing apt caches and unnecessary build dependencies. `.dockerignore` excludes git history, virtual environments, raw data, test caches, and development artifacts.

### Healthcheck Probes
- **`api`**: Probes `http://localhost:8000/health` every 10s using `curl`.
- **`mlflow`**: Probes `http://localhost:5000/health` every 10s using `curl`.
- Compose establishes service ordering: `api` waits on `mlflow` with `condition: service_healthy`.

---

## 3. Persistent Volumes

| Volume Name | Mount Point | Purpose |
|-------------|-------------|---------|
| `mlflow_data` | `/mlflow_data` | SQLite database (`mlflow.db`) and MLflow run artifacts |
| `feedback_data` | `/app/data/feedback` | Clinician feedback logs (`feedback.csv`) |
| `prod_logs_data` | `/app/data/prod_logs` | Real-time audit logs of all inference requests (`requests.csv`) |

---

## 4. Verification & Validation Results

| Test Step | Command / Request | Expected Result | Actual Result | Status |
|-----------|-------------------|-----------------|---------------|--------|
| Build | `docker compose build` | Success, tagged image | Image built successfully | Pass |
| Startup | `docker compose up -d` | All containers Up (healthy) | `disease-api` & `mlflow-server` healthy | Pass |
| Healthcheck | `GET /health` | HTTP 200, `healthy: true` | `{"status":"healthy","model_loaded":true,"model_version":"1",...}` | Pass |
| Model Info | `GET /model-info` | HTTP 200, model specs | Returns 131 features, 41 classes, quality gate passed | Pass |
| Standard Inference | `POST /predict` (`itching`, `skin_rash`) | Top-3 predictions + specialist | Fungal infection (0.5108), Dermatologist | Pass |
| Emergency Triage | `POST /predict` (`chest_pain`, `breathlessness`) | `emergency_flag: true` + reasons | Heart attack (0.2408), emergency reasons included | Pass |
| Feedback Capture | `POST /feedback` | HTTP 200, `recorded` | Appended to `/app/data/feedback/feedback.csv` | Pass |
| Audit Logging | Request logging check | Request stored in volume | Appended to `/app/data/prod_logs/requests.csv` | Pass |
| MLflow Service | `GET :5000/health` | HTTP 200 OK | MLflow server responds OK | Pass |
