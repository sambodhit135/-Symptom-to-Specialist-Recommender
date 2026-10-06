# Architecture — Disease Prediction MLOps System

## Overview

The system is designed around the **MLOps maturity model**: every stage of the
machine-learning lifecycle is automated, versioned, observable, and reproducible.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                         │
│  data/raw/           → immutable source CSVs (Kaggle)                       │
│  data/processed/     → train / val / test splits + symptoms.json            │
│  data/prod_logs/     → per-request binary feature vectors (drift input)     │
│  data/feedback/      → clinician helpful/unhelpful + true_disease labels    │
│  data/retraining/    → timestamped merged datasets for re-training runs     │
└──────────────────┬──────────────────────────────────────────────────────────┘
                   │ src/preprocess.py
                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING LAYER                                      │
│  src/train.py          → trains LR / RF / XGBoost; logs to MLflow          │
│  src/register_best.py  → quality gate (val top-3 acc ≥ 0.90, F1 ≥ 0.85)   │
│                           promotes to models/production/                    │
│  config/model_config.yaml → hyperparameters + quality-gate thresholds      │
└──────────────────┬──────────────────────────────────────────────────────────┘
                   │ models/production/
                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SERVING LAYER (FastAPI :8000)                      │
│  POST /predict         → top-3 disease predictions + specialist + emergency │
│  POST /feedback        → clinician feedback log                             │
│  GET  /health          → liveness + model readiness                         │
│  GET  /model-info      → version, type, quality-gate status                 │
│  GET  /symptoms        → 131 valid symptom names for client validation      │
│  GET  /metrics         → Prometheus exposition format                       │
│  POST /admin/reload-model → hot-reload model (authenticated)               │
│                                                                             │
│  src/api/emergency.py  → rule-based red-flag evaluator (config JSON)       │
│  src/api/model_loader.py → ModelManager: joblib load + predict_top3        │
└──────────────────┬──────────────────────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐    ┌────────────────────────────────────────────┐
│  MLflow :5000 │    │        MONITORING LAYER                     │
│  experiment   │    │  Prometheus :9090 — scrapes GET /metrics    │
│  tracking,    │    │  Grafana :3000    — pre-provisioned         │
│  model        │    │  dashboard: request rate, p95 latency,      │
│  registry     │    │  error rate, predictions by specialist,     │
└───────────────┘    │  emergency flags, confidence histogram,     │
                     │  feedback count                             │
                     └──────────────┬─────────────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────────────────┐
                     │       DRIFT DETECTION LAYER              │
                     │  src/monitoring/drift.py                 │
                     │  Reference: data/processed/train.csv     │
                     │  Current:   data/prod_logs/requests.csv  │
                     │  Outputs:   reports/drift/               │
                     │    ├── drift_report_*.html (Evidently)   │
                     │    └── drift_summary.json                │
                     │  Thresholds: config/drift_config.yaml    │
                     └──────────────┬───────────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────────────────┐
                     │       ORCHESTRATION LAYER (Airflow)      │
                     │  dags/retrain_symptom_model.py           │
                     │                                          │
                     │  check_drift                             │
                     │    └── branch_on_retrain_needed          │
                     │          ├── build_retraining_dataset    │
                     │          │     └── retrain               │
                     │          │           └── compare_with_   │
                     │          │                 production     │
                     │          │                   └── branch_ │
                     │          │                       promote  │
                     │          │                         ├── promote         │
                     │          │                         └── reload_api      │
                     │          └── log_summary           │
                     └──────────────────────────────────────────┘
```

---

## CI/CD Layer (GitHub Actions)

```
On PR → .github/workflows/ci.yml
  1. Checkout code
  2. Set up Python 3.10
  3. Cache pip dependencies
  4. pip install requirements.txt requirements-dev.txt
  5. ruff check src tests dags scripts
  6. pytest tests/ -q
  7. docker build . --no-push (smoke test)

On push to main → .github/workflows/cd.yml
  1. Build Docker image
  2. Push to ghcr.io/<owner>/disease-prediction-api:latest
  3. Tag with Git SHA for rollback
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Logistic Regression as production model | Highest quality-gate score; interpretable |
| `models/production/` committed to repo | CI can run tests without MLflow server |
| Rule-based emergency evaluator | Independent of model — never affected by drift |
| Binary feature vector logging | Matches exact training feature space; enables Evidently drift |
| Airflow mock classes in DAG | Allows `import dags.retrain_symptom_model` on Windows CI without Airflow installed |
| `prometheus_client` multi-process mode disabled | Single-worker Uvicorn; no multiproc dir needed |
| SQLite for MLflow backend | Zero-dependency local tracking; swap URI for PostgreSQL in production |

---

## Data Flow for a Single Prediction

```
Client  →  POST /predict {symptoms: [...]}
           │
           ├─ validate symptom names against symptoms.json
           ├─ emergency_evaluator.evaluate(symptoms)   ← config/emergency_rules.json
           ├─ model_manager.predict_top3(symptoms)     ← models/production/model.joblib
           │     └── build binary feature vector
           │         → model.predict_proba()
           │         → sort top-3 confidences
           │         → map disease → specialist (disease_specialist.csv)
           ├─ Prometheus counters/histograms updated
           ├─ log_prediction_request() → data/prod_logs/requests.csv
           └─ return PredictResponse (JSON)
```
