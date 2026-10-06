# MLOps Workflow — Disease Prediction Project

This document describes the operational workflow: how a change moves from raw data
through to a deployed, monitored model, and how that model is automatically retrained.

---

## 1 · Data Ingestion

| Step | Command | Output |
|------|---------|--------|
| Audit raw data | `python src/data_audit.py` | `docs/data_audit.md`, `docs/data_hashes.txt` |
| Preprocess | `python src/preprocess.py` | `data/processed/` (train/val/test CSVs + symptom/disease JSON) |

**Raw data rules:**
- Files in `data/raw/` are **read-only** — never modified by any script.
- SHA-256 hashes stored in `docs/data_hashes.txt` for integrity verification.

---

## 2 · Training & Experiment Tracking

```powershell
python src/train.py
```

- Trains three models: **Logistic Regression**, **Random Forest**, **XGBoost**.
- Logs each run to **MLflow** (`sqlite:///mlflow.db` locally, or a remote URI in Docker).
- Logged per run: hyperparameters, validation accuracy, macro F1, test accuracy, confusion matrix.
- MLflow UI: `http://localhost:5000`

### Quality Gate (`src/register_best.py`)

```powershell
python src/register_best.py
```

A model run is promoted only if it meets **all** thresholds in `config/model_config.yaml`:

| Metric | Minimum |
|--------|---------|
| `val_top3_accuracy` | ≥ 0.90 |
| `val_macro_f1` | ≥ 0.85 |
| `test_top3_accuracy` | ≥ 0.90 |
| `train_val_gap` | ≤ 0.15 |

On pass: model files copied to `models/production/`, `metadata.json` updated,
model registered in MLflow Model Registry with alias `production`.

---

## 3 · Testing

```powershell
.\venv\Scripts\pytest tests\ -q
```

| Test file | What it tests |
|-----------|--------------|
| `test_data.py` | No missing values, label consistency, schema match |
| `test_model.py` | Model loads, 41-class output, quality gate metrics, metadata |
| `test_api.py` | All endpoints: health, symptoms, predict, feedback, metrics, reload |
| `test_drift.py` | Evidently pipeline: insufficient data guard, drift detection, JSON keys |
| `test_retrain.py` | Retraining functions: dataset build, compare, promote, DAG import |

---

## 4 · CI/CD (GitHub Actions)

### CI — every Pull Request

File: `.github/workflows/ci.yml`

Steps:
1. Python 3.10 setup + pip cache
2. `pip install -r requirements.txt -r requirements-dev.txt`
3. `ruff check src tests dags scripts`
4. `pytest tests/ -q`
5. `docker build . --no-push`

### CD — push to `main`

File: `.github/workflows/cd.yml`

Steps:
1. Login to GitHub Container Registry (GHCR) using `GITHUB_TOKEN`
2. Build Docker image tagged `latest` + Git SHA
3. Push to `ghcr.io/<owner>/disease-prediction-api`

**Repository settings required:**
- Actions → Workflow permissions → Read and write (for GHCR push)
- Enable "Allow GitHub Actions to create and approve pull requests" if needed

---

## 5 · Docker Stack

```powershell
# Start all services
docker compose up --build -d

# View logs
docker compose logs -f api

# Check health
docker compose ps
```

Services started:

| Container | Image | Port |
|-----------|-------|------|
| `disease-api` | Dockerfile (built) | 8000 |
| `mlflow-server` | same image | 5000 |
| `prometheus-monitoring` | `prom/prometheus:v2.51.0` | 9090 |
| `grafana-dashboards` | `grafana/grafana:10.4.0` | 3000 |

---

## 6 · Monitoring

### Prometheus metrics (exposed at `GET /metrics`)

| Metric | Type | Labels |
|--------|------|--------|
| `api_requests_total` | Counter | `endpoint`, `status_code` |
| `api_request_latency_seconds` | Histogram | `endpoint` |
| `api_predictions_by_disease_total` | Counter | `disease` |
| `api_predictions_by_specialist_total` | Counter | `specialist` |
| `api_emergency_flags_total` | Counter | — |
| `api_top1_confidence` | Histogram | — |
| `api_feedback_total` | Counter | `helpful` |
| `api_model_info` | Info | `model_name`, `version`, `type`, `alias` |

### Grafana dashboard

Pre-provisioned at `monitoring/grafana/dashboards/disease_prediction_api.json`.  
Panels: request rate, p95 latency, error rate, specialist split, emergency flags,
average confidence, feedback count, disease distribution.

---

## 7 · Drift Detection

```powershell
# Run drift analysis against current prod_logs
python src/monitoring/drift.py

# Simulate normal traffic then check drift
python scripts/simulate_drift.py --mode normal --n 200
python src/monitoring/drift.py

# Simulate shifted traffic then check drift
python scripts/simulate_drift.py --mode shifted --n 200
python src/monitoring/drift.py
```

Output files:
- `reports/drift/drift_report_latest.html` — visual Evidently report
- `reports/drift/drift_summary.json` — machine-readable summary with keys:
  `drift_detected`, `share_drifted_features`, `top_drifted_features`,
  `prediction_distribution_drift`, `n_rows_current`, `feedback_wrong_share`

Thresholds live in `config/drift_config.yaml`:
```yaml
drift_share_threshold: 0.20      # flag if >20% features drifted
min_sample_size: 30              # skip if fewer rows
feedback_wrong_threshold: 0.30   # concept drift proxy
```

---

## 8 · Automated Retraining (Airflow DAG)

DAG: `dags/retrain_symptom_model.py`  
ID: `retrain_symptom_model`  
Schedule: `@daily` (configurable)

### Task chain

```
check_drift → branch_on_retrain_needed
                  ├── [skip] → log_summary
                  └── build_retraining_dataset
                        └── retrain
                              └── compare_with_production
                                    ├── [candidate loses] → log_summary
                                    └── promote
                                          └── reload_api → log_summary
```

### Trigger conditions

The DAG retrains when **either** is true:
- `drift_detected = true` in the latest `drift_summary.json`
- Feedback rows with validated disease label ≥ `FEEDBACK_THRESHOLD` (default 50)

### Model promotion gate

Candidate beats production on **test accuracy** by at least `MIN_IMPROVEMENT` (default 0.01).
Only then is `models/production/` overwritten and `/admin/reload-model` called.

---

## 9 · Branching Strategy

See [`docs/git-workflow.md`](git-workflow.md) for the full branching guide.

Summary:
- `main` — deployable at all times; protected
- `feature/*` — one branch per phase/feature
- PRs required to merge to main
- Commit message style: `type(scope): description` (Conventional Commits)
