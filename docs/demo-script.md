# Demo Script — Disease Prediction MLOps Live Demonstration

> **Run this script top-to-bottom before the demo** to verify every step.
> Expected total time: ~15–20 minutes with all services running.

---

## Pre-Demo Checklist

- [ ] Docker Desktop is running
- [ ] `venv` is activated: `.\venv\Scripts\Activate.ps1`
- [ ] `.env` file exists (copied from `.env.example`)
- [ ] Browser tabs open for: MLflow (:5000), Grafana (:3000), Swagger (:8000/docs)
- [ ] Terminal at repo root

---

## Step 1 · Show the GitHub Repository

**Action:** Open repo in browser / VS Code.

**Talking points:**
- 10 phases, each on its own feature branch, merged to `main` via PR.
- `README.md` covers setup, API reference, ports, troubleshooting.
- All config externalised — no hardcoded credentials.

**Expected:** Tidy git log, all branches visible.

```powershell
git log --oneline -12
```

**Expected output:**
```
<hash>  Merge feature/phase-10-docs into main - Documentation (Phase 10)
<hash>  Merge feature/phase-9-airflow into main - Automated retraining (Phase 9)
<hash>  Merge feature/phase-8-drift into main - Drift Detection (Phase 8)
...
```

**PASS ✅ / FAIL ❌**

---

## Step 2 · Training Pipeline

**Action:** Run the training pipeline from a clean state.

```powershell
# Show raw data
Get-ChildItem data\raw\

# Preprocess
python src/preprocess.py

# Train (3 models logged to MLflow)
python src/train.py
```

**Expected output (train.py):**
```
[INFO] Training logistic_regression ...
[INFO]   val_top3_accuracy: 1.0000  val_macro_f1: 1.0000
[INFO] Training random_forest ...
[INFO] Training xgboost ...
[INFO] Best model: logistic_regression (run_id=...)
```

**PASS ✅ / FAIL ❌**

---

## Step 3 · MLflow Tracking & Registry

**Action:** Open `http://localhost:5000` (or start MLflow locally).

```powershell
# Start MLflow locally (if not in Docker)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

**Demonstrate:**
1. **Experiments** tab → `symptom-disease` experiment → compare 3 runs by `val_top3_accuracy`.
2. **Models** tab → `symptom-disease-classifier` → Version 1 with alias `production`.
3. Click a run → show logged parameters (solver, C), metrics, and artifacts (model.pkl).

**Expected:** Three runs visible; production alias set on best model.

**PASS ✅ / FAIL ❌**

---

## Step 4 · Quality Gate & Model Registration

```powershell
python src/register_best.py
```

**Expected output:**
```
[INFO] Quality gate PASSED for run <run_id>
[INFO] Copying model artefacts to models/production/
[INFO] Registered model version 1 with alias 'production'
```

**Show:** `models/production/metadata.json`

```powershell
Get-Content models\production\metadata.json
```

**Expected:** quality_gate_passed: true, val_top3_accuracy: 1.0.

**PASS ✅ / FAIL ❌**

---

## Step 5 · pytest Test Suite

```powershell
.\venv\Scripts\pytest tests\ -v
```

**Expected output:**
```
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_predict_valid_symptoms PASSED
...
==================== 42 passed, 14 warnings in ~30s ====================
```

**Talking points:**
- 42 tests across 5 modules: data, model, API, drift, retraining.
- Tests run without Docker or MLflow server (use committed `models/production/`).
- pytest configured in `pyproject.toml`.

**PASS ✅ / FAIL ❌**

---

## Step 6 · CI/CD Pipeline (GitHub Actions)

**Action:** Open `.github/workflows/ci.yml` and `.github/workflows/cd.yml`.

**Talking points:**
- **CI** runs on every PR: ruff lint → pytest → docker build (no push).
- **CD** runs on push to `main`: build → push to `ghcr.io/<owner>/disease-prediction-api`.
- No secrets hardcoded — uses `GITHUB_TOKEN` (built-in) for GHCR.
- Show a completed workflow run in GitHub Actions tab.

**Show files:**
```powershell
Get-Content .github\workflows\ci.yml
```

**PASS ✅ / FAIL ❌**

---

## Step 7 · Docker Stack

```powershell
# Build and start all services
docker compose up --build -d

# Wait ~30 seconds for health checks
Start-Sleep -Seconds 30

# Verify all containers healthy
docker compose ps
```

**Expected output:**
```
NAME                    STATUS
disease-api             running (healthy)
mlflow-server           running (healthy)
prometheus-monitoring   running
grafana-dashboards      running
```

**PASS ✅ / FAIL ❌**

---

## Step 8 · FastAPI — Live Prediction

**Option A: Swagger UI**  
Open `http://localhost:8000/docs` → `POST /predict` → Try it out.

**Option B: curl / PowerShell**

```powershell
# Health check
Invoke-RestMethod http://localhost:8000/health

# List symptoms
(Invoke-RestMethod http://localhost:8000/symptoms).count

# Predict — cardiac emergency example
$body = '{"symptoms": ["chest_pain", "breathlessness", "sweating", "vomiting"]}'
Invoke-RestMethod -Method POST `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 5
```

**Expected response (abbreviated):**
```json
{
  "predictions": [
    {"disease": "Heart attack", "confidence": 0.87, "specialist": "Cardiologist"}
  ],
  "emergency_flag": true,
  "emergency_reasons": ["Potential cardiac event detected"],
  "disclaimer": "Suggestion tool, not a medical diagnosis."
}
```

**Also demonstrate feedback endpoint:**
```powershell
$fb = '{"request_id": "demo-001", "helpful": true}'
Invoke-RestMethod -Method POST `
  -Uri http://localhost:8000/feedback `
  -ContentType "application/json" `
  -Body $fb
```

**PASS ✅ / FAIL ❌**

---

## Step 9 · Prometheus & Grafana Monitoring

**Generate traffic first:**
```powershell
python scripts\generate_traffic.py --n 50 --url http://localhost:8000
```

**Prometheus:**
1. Open `http://localhost:9090`
2. Query: `api_requests_total` → shows request counts by endpoint and status.
3. Query: `histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m]))` → p95 latency.
4. Status → Targets → `disease-api` should show state **UP**.

**Grafana:**
1. Open `http://localhost:3000` (admin / password from `.env`).
2. Navigate to **Disease Prediction API** dashboard.
3. Show panels: Request Rate, p95 Latency, Error Rate, Predictions by Specialist, Emergency Flags, Confidence Histogram, Feedback Count.

**PASS ✅ / FAIL ❌**

---

## Step 10 · Drift Detection

```powershell
# Simulate shifted traffic (drifted symptom distribution)
python scripts\simulate_drift.py --mode shifted --n 150

# Run Evidently drift analysis
python src\monitoring\drift.py
```

**Expected output:**
```
[INFO] Loaded reference: 3984 rows, 132 features
[INFO] Loaded current:   150 rows
[INFO] Drift detected: True
[INFO] Drifted features: 27/132 (20.5%)
[INFO] HTML report saved: reports/drift/drift_report_latest.html
[INFO] JSON summary saved: reports/drift/drift_summary.json
```

**Show HTML report:**
```powershell
Start-Process reports\drift\drift_report_latest.html
```

**Show JSON summary:**
```powershell
Get-Content reports\drift\drift_summary.json | ConvertFrom-Json | Format-List
```

**PASS ✅ / FAIL ❌**

---

## Step 11 · Automated Retraining (Airflow DAG)

**Action:** Show the DAG structure and walk through the code.

```powershell
# Show DAG file
Get-Content dags\retrain_symptom_model.py | Select-Object -First 30

# Show the task graph (text summary)
python -c "
import sys, os
sys.path.insert(0, '.')
from dags.retrain_symptom_model import dag
print('DAG ID:', dag.dag_id)
print('Tasks:', [t.task_id for t in dag.tasks])
"
```

**Expected output:**
```
DAG ID: retrain_symptom_model
Tasks: ['check_drift', 'branch_on_retrain_needed', 'build_retraining_dataset',
        'retrain', 'compare_with_production', 'branch_on_promotion',
        'promote', 'reload_api', 'log_summary']
```

**Talking points:**
- DAG runs daily; triggers if drift detected OR feedback count ≥ threshold.
- Candidate model must beat production accuracy by 1% minimum to be promoted.
- `POST /admin/reload-model` called automatically after promotion — zero-downtime hot-reload.
- Airflow requires Linux/WSL2 for actual scheduling; DAG is fully testable on Windows via mocks.

**PASS ✅ / FAIL ❌**

---

## Demo Summary Card

| Step | Component | Status |
|------|-----------|--------|
| 1 | GitHub repo / git log | ✅ / ❌ |
| 2 | Training pipeline | ✅ / ❌ |
| 3 | MLflow tracking & registry | ✅ / ❌ |
| 4 | Quality gate & registration | ✅ / ❌ |
| 5 | pytest (42 tests) | ✅ / ❌ |
| 6 | CI/CD workflows | ✅ / ❌ |
| 7 | Docker stack (4 services healthy) | ✅ / ❌ |
| 8 | FastAPI /predict + /feedback | ✅ / ❌ |
| 9 | Prometheus targets UP + Grafana panels | ✅ / ❌ |
| 10 | Evidently drift report + JSON | ✅ / ❌ |
| 11 | Airflow DAG structure | ✅ / ❌ |
