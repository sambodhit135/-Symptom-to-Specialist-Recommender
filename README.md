# Disease Prediction – Symptom-to-Disease-to-Specialist Recommender

## Description
A full-lifecycle MLOps project built for the *MLOps and AI Development* college course.
Given a set of binary symptom inputs, the system predicts the **top-3 most likely diseases**
with confidence scores, maps each disease to the appropriate **medical specialist**, raises an
**emergency flag** when needed, and always shows the disclaimer:
*"Suggestion tool, not a medical diagnosis."*
The project follows a complete MLOps lifecycle: data audit → training → evaluation →
MLflow experiment tracking & model registry → pytest → GitHub Actions CI/CD →
Docker/Compose → FastAPI → Prometheus + Grafana monitoring → Evidently drift detection →
Apache Airflow automated retraining → redeploy.

## Dataset
Kaggle – *Disease Prediction Using Machine Learning* by kaushil268.
132 binary symptom columns, target column `prognosis` (41 diseases).
Files placed in `data/raw/` (never modified directly).

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd "Disease Prediction Using Machine Learning"

# 2. Create and activate virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run data audit
python src/data_audit.py
# Output written to docs/data_audit.md
```

## Project Structure
```
├── config/          # YAML config files
├── data/
│   ├── raw/         # Original CSVs (read-only)
│   ├── processed/   # Cleaned / feature-engineered data
│   ├── feedback/    # User feedback logs
│   └── prod_logs/   # Production prediction logs
├── docs/            # Phase notes, audit reports, hashes
├── models/          # Saved model artefacts
├── notebooks/       # Exploratory analysis
├── scripts/         # Utility / one-off scripts
├── src/             # Source code
└── tests/           # pytest test suite
```

## Phases
| # | Phase | Branch |
|---|-------|--------|
| 1 | Project setup & data audit | `feature/phase-1-setup` |
| 2 | Model training | `feature/phase-2-training` |
| 3 | Evaluation & MLflow | `feature/phase-3-mlflow` |
| 4 | pytest & CI/CD | `feature/phase-4-cicd` |
| 5 | FastAPI service | `feature/phase-5-api` |
| 6 | Docker / Compose | `feature/phase-6-docker` |
| 7 | Monitoring (Prometheus + Grafana) | `feature/phase-7-monitoring` |
| 8 | Drift detection (Evidently) | `feature/phase-8-drift` |
| 9 | Retraining pipeline (Airflow) | `feature/phase-9-airflow` |
