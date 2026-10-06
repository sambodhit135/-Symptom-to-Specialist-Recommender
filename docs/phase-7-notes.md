# Phase 7: Operational Monitoring & Grafana Dashboard

## Overview

Phase 7 implements live operational observability for the Disease Prediction REST API using **Prometheus** for metric collection and **Grafana** for real-time visualization and alerting dashboards.

All metrics are exposed at `GET /metrics` in standard Prometheus exposition format and collected every 15 seconds by the Prometheus server.

---

## Architecture & Data Flow

```
   ┌────────────────────────────────────────────────────────┐
   │                  FastAPI Prediction API                │
   │  • Middleware records request latency & status codes   │
   │  • Predict endpoint tracks top diseases & specialists  │
   │  • Emergency evaluator records red flag triggers       │
   │  • Feedback endpoint records user ratings              │
   └──────────────────────────┬─────────────────────────────┘
                              │
                    GET /metrics (every 15s)
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │                   Prometheus Server                    │
   │  • Scrapes api:8000/metrics                            │
   │  • Stores time-series data in TSDB                     │
   │  • Evaluates PromQL rate/quantile queries              │
   └──────────────────────────┬─────────────────────────────┘
                              │
                    PromQL API queries
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │                   Grafana Dashboard                    │
   │  • Auto-provisioned Prometheus datasource              │
   │  • Auto-loaded dashboard: "Disease Prediction API"     │
   │  • Port 3000 (admin / $GRAFANA_ADMIN_PASSWORD)         │
   └────────────────────────────────────────────────────────┘
```

---

## Metrics Exposed

All metrics are defined in `src/api/metrics.py` and conform to Prometheus naming conventions and label cardinality guidelines (no unbounded IDs or symptom lists).

| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `api_requests_total` | Counter | `endpoint`, `status_code` | Total incoming HTTP requests per endpoint and status code (excludes `/metrics` to avoid scrape recursion). |
| `api_request_latency_seconds` | Histogram | `endpoint` | Request latency in seconds with buckets `[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]`. |
| `api_predictions_by_disease_total` | Counter | `disease` | Frequency of top-1 predicted diseases (bounded to 41 classes). |
| `api_predictions_by_specialist_total` | Counter | `specialist` | Frequency of top-1 recommended medical specialists. |
| `api_emergency_flags_total` | Counter | _none_ | Cumulative count of requests triggering the emergency safety red flag. |
| `api_prediction_confidence` | Histogram | _none_ | Histogram of top-1 prediction confidence probabilities with buckets `0.1` to `1.0`. |
| `api_feedback_total` | Counter | `helpful` (`"True"` / `"False"`) | Submissions from the clinician/user feedback endpoint. |
| `api_model_info` | Info / Gauge | `model_name`, `model_version`, `model_type`, `model_alias` | Static gauge detailing serving model metadata. |

---

## How to Access Services

### Starting the Stack

```powershell
# Copy environment template if not already done
cp .env.example .env

# Build and start all 4 services (api, mlflow, prometheus, grafana)
docker compose up -d

# Verify all containers are running and healthy
docker compose ps
```

### URLs & Credentials

| Service | URL | Credentials / Notes |
| :--- | :--- | :--- |
| **API** | `http://localhost:8000` | Swagger UI at `http://localhost:8000/docs` |
| **Metrics Endpoint** | `http://localhost:8000/metrics` | Raw Prometheus text format |
| **MLflow Server** | `http://localhost:5000` | Model registry & experiment tracking |
| **Prometheus Web UI** | `http://localhost:9090` | Targets status: `http://localhost:9090/targets` |
| **Grafana Dashboard** | `http://localhost:3000` | User: `admin`, Password: `${GRAFANA_ADMIN_PASSWORD}` (default: `admin`) |

---

## How to Run the Synthetic Traffic Generator

To populate the Prometheus time-series database and generate live activity in Grafana:

```powershell
# Activate local virtual environment
.\venv\Scripts\Activate.ps1

# Send 100 realistic requests (default: http://localhost:8000)
python scripts/generate_traffic.py --n 100

# Custom parameters:
python scripts/generate_traffic.py --n 250 --url http://localhost:8000 --delay 0.02 --seed 123
```

The script includes 30 curated clinical symptom clusters representing genuine disease profiles (e.g., Fungal infection, Dengue, Pneumonia, Myocardial Infarction red-flag combos).

---

## Dashboard Panels & Screenshot Instructions

The pre-provisioned dashboard **"Disease Prediction API"** (`monitoring/grafana/dashboards/disease_prediction_api.json`) is organized into 4 functional rows containing 12 panels:

### Row 1: Request Traffic
1. **Request Rate by Endpoint (req/s)**: Time-series showing traffic distribution across `/predict`, `/feedback`, `/health`, `/symptoms`, and `/model-info`.
   - PromQL: `sum(rate(api_requests_total[2m])) by (endpoint)`
2. **p95 Request Latency by Endpoint**: 95th percentile latency per endpoint.
   - PromQL: `histogram_quantile(0.95, sum(rate(api_request_latency_seconds_bucket[5m])) by (le, endpoint))`
3. **Error Rate (4xx + 5xx)**: Fraction of requests resulting in client or server errors.
   - PromQL: `sum(rate(api_requests_total{status_code=~"4..|5.."}[5m])) / sum(rate(api_requests_total[5m]))`
4. **Predict Latency (p50 / p95)**: Stat card displaying median and p95 inference latency for the `/predict` endpoint.

### Row 2: Predictions & Diseases
5. **Predictions by Specialist (rate, top-10)**: Time-series of top recommended specialists (e.g., Dermatologist, Cardiologist, Pulmonologist).
   - PromQL: `topk(10, sum(rate(api_predictions_by_specialist_total[5m])) by (specialist))`
6. **Predictions by Disease (rate, top-10)**: Time-series of top predicted primary diseases.
   - PromQL: `topk(10, sum(rate(api_predictions_by_disease_total[5m])) by (disease))`

### Row 3: Safety & Confidence
7. **Emergency Flag Rate**: Real-time rate of emergency triage red-flag activations.
   - PromQL: `rate(api_emergency_flags_total[5m])`
8. **Average Top-1 Prediction Confidence**: Rolling average of model confidence on top predictions.
   - PromQL: `sum(rate(api_prediction_confidence_sum[5m])) / sum(rate(api_prediction_confidence_count[5m]))`

### Row 4: Feedback & Totals
9. **Feedback Rate (helpful / not helpful)**: Submissions breakdown by user rating.
   - PromQL: `sum(rate(api_feedback_total[5m])) by (helpful)`
10. **Total Predictions**: Cumulative counter stat panel.
11. **Total Emergency Flags**: Cumulative emergency count stat panel.
12. **Total Feedback Submissions**: Cumulative feedback submissions stat panel.

---

### Screenshot Instructions for Review / Submission

When documenting or showcasing Phase 7, take screenshots of the following 3 views:

1. **Screenshot A – Full Dashboard Overview**:
   - In Grafana (`http://localhost:3000`), open **Dashboards** > **Disease Prediction** > **Disease Prediction API**.
   - Set time range to **Last 15 minutes** and auto-refresh to **5s**.
   - Run `python scripts/generate_traffic.py --n 100` and capture the entire dashboard showing active request rates, p95 latency, and summary stats.
2. **Screenshot B – Predictions by Specialist & Top Diseases**:
   - Focus / zoom in on Panels 5 & 6 (**Predictions by Specialist** and **Predictions by Disease**) showing the distribution across specialties.
3. **Screenshot C – Prometheus Targets Health**:
   - Navigate to `http://localhost:9090/targets` in Prometheus and capture the **UP** status for the `disease-prediction-api` endpoint.
