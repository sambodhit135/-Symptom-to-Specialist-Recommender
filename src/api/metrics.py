"""
src/api/metrics.py
==================
Prometheus metrics definitions for the Disease Prediction API.

All metrics are defined here and imported by main.py.
Avoids high-cardinality labels (no request_id, no raw symptom lists).
"""

from prometheus_client import Counter, Histogram, Info

# ── Request-level counters ────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total HTTP requests received by endpoint and status.",
    ["endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ── Prediction-level metrics ──────────────────────────────────────────────────

PREDICTIONS_BY_DISEASE = Counter(
    "api_predictions_by_disease_total",
    "Total predictions by top-1 disease name (low-cardinality: 41 classes).",
    ["disease"],
)

PREDICTIONS_BY_SPECIALIST = Counter(
    "api_predictions_by_specialist_total",
    "Total predictions by top-1 recommended specialist.",
    ["specialist"],
)

EMERGENCY_FLAG_COUNT = Counter(
    "api_emergency_flags_total",
    "Total prediction requests that triggered the emergency flag.",
)

CONFIDENCE_HISTOGRAM = Histogram(
    "api_prediction_confidence",
    "Histogram of top-1 prediction confidence scores.",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Feedback metrics ──────────────────────────────────────────────────────────

FEEDBACK_COUNT = Counter(
    "api_feedback_total",
    "Total feedback submissions received.",
    ["helpful"],
)

# ── Model info gauge ──────────────────────────────────────────────────────────

MODEL_INFO = Info(
    "api_model",
    "Static information about the currently serving model.",
)
