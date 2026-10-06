# Phase 8: Data & Concept Drift Detection

## Overview

Phase 8 implements automated continuous monitoring for **feature data drift**, **prediction distribution drift**, and **feedback-driven concept drift**. Using **Evidently** (`v0.7.23`), production request logs are statistically compared against the reference training dataset (`data/processed/train.csv`).

---

## Architecture & Detection Pipeline

```
 ┌────────────────────────────────────────────────────────┐
 │            Production Request Logs & Feedback          │
 │       • data/prod_logs/requests.csv (current data)     │
 │       • data/feedback/feedback.csv  (concept proxy)    │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │              src/monitoring/drift.py                   │
 │   1. Sample size guard: requests < min_sample_size?    │
 │      → Returns "insufficient_data" (no guessing)       │
 │   2. Feature drift: Evidently DataDriftPreset (131)    │
 │   3. Prediction drift: ValueDrift on top_disease       │
 │   4. Concept drift: Unhelpful feedback ratio           │
 └──────────────────────────┬─────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
 ┌──────────────────────┐       ┌───────────────────────┐
 │ reports/drift/       │       │ Prometheus & Grafana  │
 │ • drift_report.html  │       │ • api_drift_detected  │
 │ • drift_summary.json │       │ • api_drift_share     │
 └──────────────────────┘       └───────────────────────┘
```

---

## Configuration & Thresholds (`config/drift_config.yaml`)

All statistical thresholds and sample size guardrails are configured externally:

```yaml
min_sample_size: 50                # Minimum production requests before evaluating drift
feature_drift_p_value: 0.05        # Statistical test p-value threshold per symptom feature
drift_share_threshold: 0.20        # >20% of 131 symptoms drifting triggers overall dataset drift
prediction_drift_p_value: 0.05     # Statistical p-value threshold for top-1 disease shift
feedback_unhelpful_threshold: 0.25 # >25% unhelpful feedback triggers concept drift alert
min_feedback_sample_size: 10       # Minimum feedback rows before calculating ratio
```

---

## Verification Scenarios & Real Results

### 1. Normal Scenario (Identical / Empirically Sampled Traffic)
- **Command**:
  ```powershell
  python scripts/simulate_drift.py --mode normal --n 100 --dest log
  python src/monitoring/drift.py
  ```
- **Real Summary Output** (`reports/drift/drift_summary_normal.json`):
  ```json
  {
    "status": "completed",
    "drift_detected": false,
    "share_of_drifted_features": 0.0,
    "drift_share_threshold": 0.2,
    "drifted_features_count": 0,
    "total_features": 131,
    "top_drifted_features": [],
    "prediction_distribution_drift": {
      "drift_detected": false,
      "p_value": 0.18401,
      "threshold": 0.05
    },
    "rows_compared": {
      "reference_rows": 243,
      "current_rows": 100
    },
    "feedback_concept_drift": {
      "status": "insufficient_data",
      "message": "Sample size 3 below minimum requirement (10).",
      "concept_drift_warning": false
    }
  }
  ```

### 2. Shifted Scenario (Over-represented Symptom Clusters)
- **Command**:
  ```powershell
  python scripts/simulate_drift.py --mode shifted --n 100 --dest log
  python src/monitoring/drift.py
  ```
- **Real Summary Output** (`reports/drift/drift_summary_shifted.json`):
  ```json
  {
    "status": "completed",
    "drift_detected": true,
    "share_of_drifted_features": 0.3359,
    "drift_share_threshold": 0.2,
    "drifted_features_count": 44,
    "total_features": 131,
    "top_drifted_features": [
      { "feature": "vomiting", "p_value": 0.0, "threshold": 0.05 },
      { "feature": "burning_micturition", "p_value": 0.0, "threshold": 0.05 },
      { "feature": "spotting_urination", "p_value": 0.0, "threshold": 0.05 },
      { "feature": "fatigue", "p_value": 0.0, "threshold": 0.05 },
      { "feature": "high_fever", "p_value": 0.0, "threshold": 0.05 },
      { "feature": "loss_of_appetite", "p_value": 0.0, "threshold": 0.05 },
      { "feature": "bladder_discomfort", "p_value": 0.0, "threshold": 0.05 },
      { "feature": "foul_smell_of_urine", "p_value": 0.0, "threshold": 0.05 },
      { "feature": "continuous_feel_of_urine", "p_value": 0.0, "threshold": 0.05 },
      { "feature": "internal_itching", "p_value": 0.0, "threshold": 0.05 },
      { "feature": "toxic_look_(typhos)", "p_value": 0.0, "threshold": 0.05 },
      { "feature": "altered_sensorium", "p_value": 0.0, "threshold": 0.05 }
    ],
    "prediction_distribution_drift": {
      "drift_detected": true,
      "p_value": 0.0,
      "threshold": 0.05
    },
    "rows_compared": {
      "reference_rows": 243,
      "current_rows": 100
    }
  }
  ```

---

## Sample Size Guard & Small-Sample Caveats

### Guardrail Behavior
When production request logs contain fewer rows than `min_sample_size` (50):
- `run_drift_analysis` returns `status: "insufficient_data"`.
- It explicitly logs `drift_detected: false`, `share_of_drifted_features: 0.0`, and does not run hypothesis tests on undersized sample groups.

### Honest Statistical Caveats for Small Samples (N < 200)
1. **High Variance in Discrete Sparse Features**: Many symptoms in the dataset have low baseline prevalence (< 5% of training samples). In small production samples (e.g. 50–100 rows), observing 2 or 3 instances can yield low p-values due to Poisson noise rather than true population shift.
2. **False Discovery Rate**: Across 131 simultaneous statistical tests, an uncorrected alpha of 0.05 may theoretically produce ~6 false positive feature drifts by random chance. The `drift_share_threshold: 0.20` (>26 drifted features) acts as a robust multi-test guard against false alarms.
3. **Feedback Proxy Lag**: Clinical feedback arrives asynchronously and sparsely compared to inference traffic. A sample size guard of 10 responses is enforced to avoid premature concept drift alerts based on 1 or 2 isolated negative ratings.

---

## HTML Report Artifacts

The generated reports are standalone HTML dashboards with interactive Plotly visualisations:
- `reports/drift/drift_report_latest.html`: Current report.
- `reports/drift/drift_report_normal.html`: Normal baseline report.
- `reports/drift/drift_report_shifted.html`: Shifted alert report.

Open in any browser:
```powershell
Start-Process "reports\drift\drift_report_latest.html"
```
