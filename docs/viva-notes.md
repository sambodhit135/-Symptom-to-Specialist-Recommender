# Viva Notes — Disease Prediction MLOps Project

> For each tool: **purpose**, **how configured in this project**, and **3 likely viva questions with answers**.

---

## 1 · Git

### Purpose
Distributed version control — tracks every change to code, config, and documentation,
enables collaboration, and underpins the CI/CD pipeline.

### How configured in this project
- 10 feature branches (`feature/phase-N-*`), each merged to `main` via `--no-ff` merge commits.
- `.gitignore` excludes `venv/`, `__pycache__/`, `mlruns/`, `mlartifacts/`, `.env`.
- `docs/data_hashes.txt` stores SHA-256 hashes of raw data files for integrity checking.
- PR template in `.github/pull_request_template.md` enforces checklist on every merge.
- Commit messages follow **Conventional Commits**: `feat(api): add /predict endpoint`.

### Viva Q&A

**Q1: Why do we use `--no-ff` when merging feature branches?**  
`--no-ff` forces a merge commit even if a fast-forward is possible. This preserves the
history that work happened on a separate branch, making it easy to `git revert` an entire
feature or see when it landed in `main`. Fast-forward merges make the history look linear
and lose that context.

**Q2: What is in `.gitignore` and why?**  
`venv/` (Python packages are in `requirements.txt`, not source-controlled),
`__pycache__/` and `*.pyc` (compiled bytecode, auto-generated),
`mlruns/` and `mlartifacts/` (large binary artefacts managed by MLflow),
`.env` (contains secrets — only `.env.example` is committed),
`*.db` except `mlflow.db` (local SQLite state).

**Q3: What is the difference between `git merge` and `git rebase`?**  
`merge` creates a merge commit that joins two branch histories — it is non-destructive
and preserves exactly what happened. `rebase` replays commits on top of another branch,
producing a linear history but rewriting SHAs. In team workflows `rebase` onto `main`
before a PR keeps the log clean; `merge --no-ff` into `main` preserves the branch record.

---

## 2 · MLflow

### Purpose
Open-source platform for managing the ML lifecycle: experiment tracking, model registry,
and model serving (optional). Enables reproducibility and comparison across training runs.

### How configured in this project
- **Tracking URI:** `sqlite:///mlflow.db` locally; `http://mlflow:5000` inside Docker.
- **Experiment:** `symptom-disease` — one experiment, multiple runs.
- **Logged per run:** hyperparameters (`solver`, `C`, `n_estimators`), metrics
  (`val_top3_accuracy`, `val_macro_f1`, `test_accuracy`), confusion matrix PNG, model artifact.
- **Registry:** `symptom-disease-classifier` registered model; Version 1 has aliases
  `staging` and `production` set by `src/register_best.py`.
- **Production copy:** model files copied to `models/production/` so CI can run tests
  without a live MLflow server.

### Viva Q&A

**Q1: What is the difference between an MLflow Run and a registered Model Version?**  
A **Run** is a single execution of training code — it logs parameters, metrics, and
artifacts to the tracking server. Runs are immutable once complete. A **registered Model
Version** is created by calling `mlflow.register_model()` on a run's artifact URI — it
gives the model a name and version number in the Model Registry, which is separate from
the tracking store. You can then assign aliases (e.g., `production`) to a version for
serving.

**Q2: How does the quality gate work in this project?**  
`src/register_best.py` queries all runs in the `symptom-disease` experiment, finds the
one with the highest `val_top3_accuracy`, then checks four thresholds from
`config/model_config.yaml` (val top-3 acc ≥ 0.90, val macro F1 ≥ 0.85, test top-3 acc ≥ 0.90,
train-val gap ≤ 0.15). If all pass, the model is registered and promoted to the
`production` alias; if not, the script exits non-zero and CI fails.

**Q3: Why is the MLflow backend SQLite rather than PostgreSQL here?**  
SQLite is zero-configuration — no extra service required — which is fine for a
single-developer or single-node project. In a real production environment you would use
PostgreSQL (for concurrent writers and durability) and an S3-compatible object store
for artifacts, so that multiple workers can access the same registry.

---

## 3 · pytest

### Purpose
Python testing framework — runs unit, integration, and end-to-end tests, reports
pass/fail, and integrates with CI pipelines.

### How configured in this project
- `pyproject.toml` → `[tool.pytest.ini_options]`: `testpaths = ["tests"]`,
  `addopts = "-q --tb=short"`.
- `conftest.py` provides a shared `client` fixture (FastAPI `TestClient`) used by all API tests.
- 42 tests across 5 files; none require a running Docker container or MLflow server.
- `requirements-dev.txt` pins `pytest` and `pytest-cov`.
- CI runs `pytest tests/ -q` as a required check before merge.

### Viva Q&A

**Q1: What is a pytest fixture and why is `client` defined in `conftest.py`?**  
A fixture is a function decorated with `@pytest.fixture` that provides reusable test
setup/teardown. `conftest.py` is auto-discovered by pytest, so fixtures defined there are
available to all test files without explicit imports. The `client` fixture creates a
`TestClient` wrapping the FastAPI app — an in-process HTTP client that doesn't need a
running server — so all API tests share the same setup without repeating it.

**Q2: How do the drift tests avoid writing to production directories?**  
The drift tests use `tmp_path` (a built-in pytest fixture that provides a temporary
directory unique to each test) and `monkeypatch` to redirect the report output path.
This ensures tests never touch `reports/drift/` or `data/prod_logs/` and are safe to run
in parallel.

**Q3: What does `--tb=short` do?**  
It controls the traceback format on failure. `short` shows only the assert line and the
exception message — not the full stack — keeping output readable in CI logs. Other options
are `long` (full stack), `line` (one line per failure), `no` (suppress tracebacks), and
`auto` (full for first failure, short for subsequent).

---

## 4 · GitHub Actions (CI/CD)

### Purpose
Cloud-based automation platform integrated with GitHub — runs workflows on events such as
push and pull_request. Used here for lint, test, Docker build, and image publishing.

### How configured in this project
- **CI** (`.github/workflows/ci.yml`): triggered on `push` and `pull_request` to any
  branch. Steps: checkout → Python 3.10 → pip cache → install deps → ruff → pytest →
  docker build (no push).
- **CD** (`.github/workflows/cd.yml`): triggered on `push` to `main` only.
  Steps: login to GHCR with `GITHUB_TOKEN` → build → push `latest` + SHA tag.
- No secrets hardcoded — `GITHUB_TOKEN` is provided automatically by Actions.
- Repository setting required: Actions → Workflow permissions → "Read and write".

### Viva Q&A

**Q1: How does the CD workflow push to GHCR without hardcoding a password?**  
GitHub automatically injects a short-lived `GITHUB_TOKEN` into every workflow run.
The `docker/login-action` uses `secrets.GITHUB_TOKEN` (referenced as `${{ secrets.GITHUB_TOKEN }}`)
to authenticate to `ghcr.io`. No manual secret rotation is needed; the token expires when
the workflow job ends.

**Q2: Why does CI build the Docker image but not push it?**  
Building confirms that the `Dockerfile` is correct and that the image can be constructed
from the current code — a "smoke test". Pushing is only done from `main` (CD) because only
code that has passed all tests and been approved via PR should enter the registry.

**Q3: What is pip caching in the CI workflow and why does it matter?**  
The `actions/cache` step saves the pip download cache (`~/.cache/pip`) between workflow
runs, keyed on a hash of `requirements.txt`. On a cache hit, pip finds packages locally
and skips downloading, reducing a typical install from ~90 s to ~10 s and saving GitHub
Actions minutes.

---

## 5 · Docker

### Purpose
Packages an application and all its dependencies into a portable, reproducible container
image that runs identically on any machine with Docker installed.

### How configured in this project
- **Dockerfile:** multi-stage build — `python:3.10-slim` base, `pip install` from
  `requirements.txt`, copy source and model artefacts, expose port 8000, entrypoint
  `uvicorn src.api.main:app`.
- `.dockerignore` excludes `venv/`, `mlruns/`, `.env`, `*.pyc`.
- The same image (`diseasepredictionusingmachinelearning-api`) is reused by both the
  `api` and `mlflow` services in docker-compose to avoid building twice.

### Viva Q&A

**Q1: What is the purpose of `.dockerignore`?**  
Like `.gitignore`, it tells the Docker build context which files to exclude when sending
files to the Docker daemon. Excluding `venv/` (hundreds of MB), `mlruns/` (large binary
artefacts), and `.env` (secrets) keeps the build context small, speeds up the build, and
prevents accidental secret leakage into the image.

**Q2: Why use `python:3.10-slim` rather than the full `python:3.10` image?**  
`slim` variants omit development tools, documentation, and many OS utilities, reducing the
image from ~900 MB to ~130 MB. Smaller images pull faster, use less registry storage,
and reduce the attack surface. A production hardening step would also run as a non-root
user and add a security scanner (e.g., Trivy).

**Q3: What is the difference between `CMD` and `ENTRYPOINT` in a Dockerfile?**  
`ENTRYPOINT` defines the executable that always runs — it cannot be overridden by
`docker run` arguments (only by `--entrypoint`). `CMD` provides default arguments to the
entrypoint, which can be replaced at runtime. Using `ENTRYPOINT ["uvicorn"]` with
`CMD ["src.api.main:app", "--host", "0.0.0.0"]` lets you override just the arguments
(e.g., change the port) without changing the entrypoint.

---

## 6 · Docker Compose

### Purpose
Defines and runs multi-container Docker applications. A single `docker-compose.yml`
file declares all services, their dependencies, networks, volumes, and health checks.

### How configured in this project
- 4 services: `api`, `mlflow`, `prometheus`, `grafana`.
- All on a shared bridge network `app-net` — services communicate by container name.
- Named volumes for persistent state: `mlflow_data`, `feedback_data`, `prod_logs_data`,
  `prometheus_data`, `grafana_data`.
- Environment variables sourced from `.env` — no hardcoded values.
- `depends_on` with `condition: service_healthy` ensures the API only starts after MLflow
  passes its healthcheck.

### Viva Q&A

**Q1: What is the difference between `depends_on` and a healthcheck?**  
`depends_on` alone only waits for a container to *start* — not for the service inside to
be *ready*. Adding `condition: service_healthy` makes Compose also wait until the
container's healthcheck returns success. This prevents the API from crashing on startup
because MLflow isn't ready yet.

**Q2: What is a named volume in Docker Compose and why is it used?**  
A named volume (`mlflow_data:`, `grafana_data:`, etc.) is managed by Docker and persists
independently of the container lifecycle. When you `docker compose down` (without `-v`),
the volumes survive — so MLflow experiment history, Grafana dashboards, and feedback data
are not lost. Anonymous volumes and bind mounts are the alternatives; named volumes are
preferred for production data.

**Q3: How would you scale the API service horizontally with Compose?**  
`docker compose up --scale api=3` starts 3 replicas of the API container. You would also
need a load balancer (e.g., nginx or Traefik) defined as a service to distribute traffic,
and the port mapping on the `api` service would move to the load balancer. The API is
stateless (model is loaded from `models/production/`), so horizontal scaling works without
shared session state.

---

## 7 · FastAPI

### Purpose
Modern, high-performance Python web framework for building APIs with automatic OpenAPI/
Swagger documentation, type validation via Pydantic, and async support via Starlette.

### How configured in this project
- App defined in `src/api/main.py`; served by Uvicorn.
- Endpoints: `GET /health`, `GET /symptoms`, `GET /model-info`, `GET /metrics`,
  `POST /predict`, `POST /feedback`, `POST /admin/reload-model`.
- Pydantic schemas in `src/api/schemas.py` validate request/response bodies automatically.
- CORS middleware allows all origins (`*`) — should be restricted in production.
- JSON-structured middleware logs every request (timestamp, path, status, latency, request_id).
- `X-Request-ID` header echoed back to client for distributed tracing.

### Viva Q&A

**Q1: How does FastAPI validate incoming request bodies?**  
Request bodies are declared as Pydantic `BaseModel` subclasses (e.g., `PredictRequest`).
FastAPI automatically deserialises the JSON body and validates each field against the
Pydantic type annotations. If validation fails, FastAPI returns a 422 Unprocessable Entity
response with a detailed error message — no manual validation code is needed.

**Q2: What is the role of the `ModelManager` class?**  
`ModelManager` (`src/api/model_loader.py`) encapsulates all model I/O: loading
`model.joblib`, `symptoms.json`, `diseases.json`, and `disease_specialist.csv` from
`models/production/` at startup; exposing `predict_top3(symptoms)` which builds the binary
feature vector, calls `model.predict_proba()`, sorts the top-3 results, and maps diseases
to specialists. Using a class allows hot-reload (`load()` called by `/admin/reload-model`)
without restarting the process.

**Q3: Why is the `/metrics` endpoint excluded from Prometheus instrumentation in the middleware?**  
If Prometheus scraped `/metrics` and that scrape itself was counted and added to the
metrics output, it would inflate request counts with scraper traffic, making dashboards
misleading. The `_TRACKED_ENDPOINTS` set in `main.py` excludes `/metrics` so only real
business traffic is counted.

---

## 8 · Prometheus

### Purpose
Open-source time-series monitoring system that scrapes metric endpoints at regular
intervals, stores the data, and exposes a query language (PromQL) for alerts and dashboards.

### How configured in this project
- Config at `monitoring/prometheus/prometheus.yml`: scrapes `api:8000/metrics` every 15 s.
- `prometheus_client` Python library defines metrics in `src/api/metrics.py`.
- `GET /metrics` returns metrics in Prometheus text exposition format (returned by
  `generate_latest()`).
- Counters: `api_requests_total`, `api_predictions_by_disease_total`,
  `api_predictions_by_specialist_total`, `api_emergency_flags_total`, `api_feedback_total`.
- Histograms: `api_request_latency_seconds`, `api_top1_confidence`.
- Info: `api_model_info` (model name, version, type, alias).

### Viva Q&A

**Q1: What is the difference between a Counter, Gauge, and Histogram in Prometheus?**  
A **Counter** only increases (or resets on restart) — used for request counts, error
counts, etc. A **Gauge** can go up and down — used for current queue depth, memory usage,
active connections. A **Histogram** samples observations into configurable buckets and
exposes count, sum, and per-bucket counts — used for latency and response size, enabling
percentile queries. We use Counters and Histograms; we don't need Gauges because we have
no fluctuating inventory metric.

**Q2: How would you query the 95th-percentile latency for the /predict endpoint?**  
```promql
histogram_quantile(
  0.95,
  rate(api_request_latency_seconds_bucket{endpoint="/predict"}[5m])
)
```
This uses a 5-minute rolling rate of histogram bucket observations to compute p95.

**Q3: Why do we avoid high-cardinality labels?**  
Prometheus stores one time series per unique label combination. If we used
`request_id` or a raw symptom list as a label, there would be millions of unique series —
consuming huge amounts of memory and storage (the "cardinality explosion" problem).
Labels should only take a small, bounded number of values (e.g., endpoint names,
status codes, specialist types).

---

## 9 · Grafana

### Purpose
Open-source analytics and visualisation platform that queries data sources (including
Prometheus) and renders dashboards with graphs, gauges, tables, and alerts.

### How configured in this project
- Provisioned automatically on startup — no manual setup required.
- **Datasource:** `monitoring/grafana/provisioning/datasources/prometheus.yml` — points
  to `http://prometheus:9090`.
- **Dashboard:** `monitoring/grafana/provisioning/dashboards/dashboards.yml` scans
  `monitoring/grafana/dashboards/` for JSON files.
- Dashboard JSON at `monitoring/grafana/dashboards/disease_prediction_api.json` with
  panels: request rate, p95 latency, error rate, specialist split, emergency flags,
  confidence histogram, feedback count, disease distribution.
- Admin password from `.env` via `GF_SECURITY_ADMIN_PASSWORD`.

### Viva Q&A

**Q1: What is Grafana provisioning and why is it useful?**  
Provisioning allows Grafana to automatically configure datasources and dashboards from
YAML/JSON files at startup, rather than requiring a user to click through the UI.
This makes the monitoring setup reproducible and version-controllable — anyone who runs
`docker compose up` gets the same dashboard without manual steps.

**Q2: How would you set up a Grafana alert for high error rate?**  
In the dashboard panel for error rate (e.g., `rate(api_requests_total{status_code=~"5.."}[1m])`),
create an Alert Rule with a threshold condition (e.g., > 0.05 requests/s), configure
an evaluation interval, and attach a Notification Policy pointing to a contact point
(email, Slack, PagerDuty). The alert fires when the condition is true for the configured
pending period.

**Q3: What does the Grafana dashboard show for "p95 latency" and how is it computed?**  
p95 latency is the 95th percentile of response times — 95% of requests are faster than
this value. It is computed from the Prometheus histogram `api_request_latency_seconds`
using: `histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m]))`.
A spike in p95 latency indicates tail latency issues, even if the average looks healthy.

---

## 10 · Evidently

### Purpose
Open-source ML monitoring library that generates data and model quality reports by
comparing a reference dataset (training data) to a current dataset (production requests).

### How configured in this project
- `src/monitoring/drift.py` runs the full drift pipeline.
- **Reference:** `data/processed/train.csv` (4 k rows, 132 binary features).
- **Current:** `data/prod_logs/requests.csv` (accumulated production requests).
- **Report:** `DataDriftPreset` generates an HTML visual report saved to `reports/drift/`.
- **JSON summary** contains: `drift_detected`, `share_drifted_features`,
  `top_drifted_features`, `n_rows_current`, `feedback_wrong_share` (concept drift proxy).
- Thresholds in `config/drift_config.yaml`:
  `drift_share_threshold: 0.20`, `min_sample_size: 30`, `feedback_wrong_threshold: 0.30`.
- `scripts/simulate_drift.py` generates normal or shifted traffic for testing.

### Viva Q&A

**Q1: What statistical test does Evidently use to detect drift in binary features?**  
For binary (categorical) features, Evidently uses the **chi-squared test** by default.
For continuous numerical features it uses the **Wasserstein distance** or the
**Kolmogorov-Smirnov test**, depending on the number of unique values and the preset.
The test compares the distribution of each feature in the reference vs. current dataset
and reports a p-value; features below the significance threshold are flagged as drifted.

**Q2: What is the difference between data drift and concept drift?**  
**Data drift** (covariate shift): the input distribution P(X) changes — users are
presenting different symptom combinations than seen in training. **Concept drift**:
the relationship P(Y|X) changes — the same symptoms now predict different outcomes,
perhaps because a new disease variant has emerged. We proxy concept drift via the
feedback signal: a high share of "unhelpful" feedback may indicate the model's
predictions are no longer accurate even for familiar symptom patterns.

**Q3: Why is a minimum sample size check important before running drift analysis?**  
Statistical tests are unreliable on very small samples — they may falsely flag drift
(high variance from small N) or fail to detect real drift (insufficient power).
`config/drift_config.yaml` sets `min_sample_size: 30`; if `requests.csv` has fewer
rows, `drift.py` returns `{"status": "insufficient_data"}` and the Airflow DAG skips
retraining rather than making a decision on noisy statistics.

---

## 11 · Apache Airflow

### Purpose
Open-source workflow orchestration platform — defines, schedules, and monitors
data pipelines as DAGs (Directed Acyclic Graphs) of tasks.

### How configured in this project
- DAG: `dags/retrain_symptom_model.py`, ID `retrain_symptom_model`, schedule `@daily`.
- **Tasks:** `check_drift → branch_on_retrain_needed → build_retraining_dataset →
  retrain → compare_with_production → branch_on_promotion → promote → reload_api → log_summary`.
- Pipeline functions in `src/retrain_pipeline.py`; DAG imports them via `PythonOperator`.
- Airflow mock classes included so the DAG can be imported and tested on Windows CI
  without native Airflow installed.
- In production: deployed via Docker Compose (separate `docker-compose.airflow.yml`),
  with `dags/` and `src/` mounted as volumes.
- `FEEDBACK_THRESHOLD` and `MIN_IMPROVEMENT` set via Airflow Variables or `.env`.

### Viva Q&A

**Q1: What is a DAG in Airflow and how are task dependencies defined?**  
A **DAG** (Directed Acyclic Graph) is a Python object that groups tasks and defines the
order in which they run. Dependencies are set using the `>>` (right-shift) operator:
`task_a >> task_b` means `task_b` runs only after `task_a` succeeds. The graph must be
acyclic — no circular dependencies — which Airflow enforces. Tasks are instances of
Operators (e.g., `PythonOperator`, `BranchPythonOperator`).

**Q2: What does `BranchPythonOperator` do and when is it used?**  
`BranchPythonOperator` calls a Python function that returns the `task_id` (or a list of
`task_id`s) of the next task(s) to run. All other downstream tasks are skipped. In this
project it is used twice: first to branch between "retrain" and "skip" based on drift/
feedback, and second to branch between "promote" and "skip" based on whether the
candidate beats the production model.

**Q3: How does the DAG avoid overwriting raw training data?**  
`build_retraining_dataset()` in `src/retrain_pipeline.py` writes only to
`data/retraining/` (timestamped files) and explicitly checks that it never touches
`data/raw/`. The function merges the original `data/processed/train.csv` with validated
feedback rows and writes a new CSV — it never modifies any existing file. The test
`test_build_retraining_dataset_never_touches_raw` verifies this constraint.
