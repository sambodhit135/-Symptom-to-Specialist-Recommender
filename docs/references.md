# References — Disease Prediction MLOps Project

> **Instructions for the team:** Replace every `[PLACEHOLDER]` with the actual detail
> before submitting. Entries marked **[TEAM: ADD]** require you to supply information
> that only you have access to (URLs, names, tutorial links you followed, etc.).

---

## Dataset

| Field | Detail |
|-------|--------|
| **Name** | Disease Prediction Using Machine Learning |
| **Source** | Kaggle |
| **Author** | kaushil268 |
| **URL** | https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning |
| **Licence** | [TEAM: ADD — check the Kaggle dataset licence] |
| **Files used** | `Training.csv`, `Testing.csv` |
| **Description** | 4,920 rows, 133 columns (132 binary symptom features + `prognosis`), 41 disease classes |
| **Accessed** | [TEAM: ADD — date you downloaded the dataset] |

---

## Core Libraries & Frameworks

| Library | Version | Purpose | URL |
|---------|---------|---------|-----|
| Python | 3.10 | Runtime | https://www.python.org |
| FastAPI | 0.115.x | REST API framework | https://fastapi.tiangolo.com |
| Uvicorn | 0.34.x | ASGI server | https://www.uvicorn.org |
| Pydantic | 2.x | Data validation | https://docs.pydantic.dev |
| scikit-learn | 1.6.x | ML models + preprocessing | https://scikit-learn.org |
| XGBoost | 2.x | Gradient boosted trees | https://xgboost.readthedocs.io |
| MLflow | 2.x | Experiment tracking + model registry | https://mlflow.org |
| Evidently | 0.5.x | Data drift detection | https://docs.evidentlyai.com |
| Apache Airflow | 2.x | Workflow orchestration | https://airflow.apache.org |
| prometheus-client | 0.26.x | Prometheus metrics exposition | https://github.com/prometheus/client_python |
| pandas | 2.x | Data manipulation | https://pandas.pydata.org |
| numpy | 1.x / 2.x | Numerical computing | https://numpy.org |
| joblib | 1.x | Model serialisation | https://joblib.readthedocs.io |
| pytest | 8.x | Testing framework | https://docs.pytest.org |
| ruff | 0.x | Python linter & formatter | https://docs.astral.sh/ruff |
| httpx | 0.28.x | HTTP client (TestClient) | https://www.python-httpx.org |

---

## Infrastructure & DevOps

| Tool | Version | Purpose | URL |
|------|---------|---------|-----|
| Docker | 4.28+ | Containerisation | https://www.docker.com |
| Docker Compose | v2.24+ | Multi-container orchestration | https://docs.docker.com/compose |
| GitHub Actions | — | CI/CD automation | https://docs.github.com/en/actions |
| GitHub Container Registry | — | Docker image registry | https://ghcr.io |
| Prometheus | v2.51.0 | Metrics time-series database | https://prometheus.io |
| Grafana | 10.4.0 | Metrics visualisation | https://grafana.com |

---

## Course & Academic References

| Reference | Detail |
|-----------|--------|
| **Course** | [TEAM: ADD — full course name, code, and institution] |
| **Lecturer / Supervisor** | [TEAM: ADD — name(s)] |
| **Assignment brief** | [TEAM: ADD — title and date] |

---

## External Tutorials & Documentation Used

> [TEAM: ADD] — List any YouTube videos, blog posts, StackOverflow answers, or tutorials
> you followed. Include title, URL, and what you used it for. This is important for
> academic integrity.

| Title | URL | Used for |
|-------|-----|---------|
| [TEAM: ADD] | [TEAM: ADD] | [TEAM: ADD] |

---

## AI-Assisted Development

> [TEAM: ADD] — Declare any use of AI coding assistants (e.g., GitHub Copilot,
> ChatGPT, Gemini) per your institution's academic integrity policy. Describe
> what was generated and how it was reviewed/modified.

[TEAM: ADD — e.g., "Gemini was used to scaffold the initial Airflow DAG structure,
which was then reviewed, adapted, and tested by the team."]

---

## Code Reuse & Attribution

Any code adapted from open-source projects or documentation examples is listed below.

| Component | Source | Licence | Adaptation |
|-----------|--------|---------|------------|
| FastAPI CORS middleware pattern | FastAPI official docs | MIT | Used as-is |
| Prometheus histogram_quantile query | Prometheus docs | Apache 2.0 | Adapted for API |
| [TEAM: ADD] | [TEAM: ADD] | [TEAM: ADD] | [TEAM: ADD] |
