"""
src/api/main.py
===============
FastAPI Prediction Service for Symptom-to-Disease-to-Specialist Recommendation.

Features:
  - GET /health: Service liveness & model readiness.
  - GET /symptoms: Ordered feature list for UI and client validation.
  - GET /model-info: Model metadata, version, load time, and quality gate status.
  - GET /metrics: Prometheus exposition format for scraping.
  - POST /predict: Top-3 disease predictions with probabilities, specialists,
                   independent emergency triage rules, and request logging.
  - POST /feedback: Clinician/user feedback logging with disease validation.
  - JSON-structured middleware logging for observability.
"""

import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.api.emergency import EmergencyEvaluator
from src.api.metrics import (
    CONFIDENCE_HISTOGRAM,
    EMERGENCY_FLAG_COUNT,
    FEEDBACK_COUNT,
    MODEL_INFO,
    PREDICTIONS_BY_DISEASE,
    PREDICTIONS_BY_SPECIALIST,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from src.api.model_loader import ModelManager
from src.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionItem,
    PredictRequest,
    PredictResponse,
)

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION (JSON-Structured)
# -----------------------------------------------------------------------------
logger = logging.getLogger("symptom_api")
logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter("%(message)s"))
logger.handlers = [log_handler]

# -----------------------------------------------------------------------------
# APPLICATION LIFECYCLE & GLOBAL MANAGERS
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Symptom-to-Disease-to-Specialist API",
    description="Clinical Decision Support System predicting differential diagnoses and emergency triage recommendations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()
emergency_evaluator = EmergencyEvaluator()

# Seed the model info gauge once at startup
MODEL_INFO.info(
    {
        "model_name": "symptom-disease-classifier",
        "model_version": model_manager.model_version,
        "model_type": model_manager.model_type,
        "model_alias": model_manager.model_alias,
    }
)

# Thread locks for safe CSV appends
feedback_lock = threading.Lock()
request_log_lock = threading.Lock()

# Endpoints to track - /metrics excluded to avoid scraper noise
_TRACKED_ENDPOINTS = {
    "/health",
    "/symptoms",
    "/model-info",
    "/predict",
    "/feedback",
    "/admin/reload-model",
}


# -----------------------------------------------------------------------------
# MIDDLEWARE: JSON STRUCTURED REQUEST LOGGING + PROMETHEUS INSTRUMENTATION
# -----------------------------------------------------------------------------
@app.middleware("http")
async def json_structured_logging_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    request.state.request_id = req_id
    start_time = time.time()

    response = await call_next(request)
    latency_s = time.time() - start_time
    latency_ms = round(latency_s * 1000, 2)

    endpoint = request.url.path
    status_code = str(response.status_code)

    # Instrument Prometheus — exclude /metrics scrape endpoint itself
    if endpoint in _TRACKED_ENDPOINTS:
        REQUEST_COUNT.labels(endpoint=endpoint, status_code=status_code).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_s)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": req_id,
        "method": request.method,
        "path": endpoint,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "client_ip": request.client.host if request.client else "unknown",
    }
    logger.info(json.dumps(log_entry))
    response.headers["X-Request-ID"] = req_id
    return response


# -----------------------------------------------------------------------------
# HELPERS FOR FILE LOGGING
# -----------------------------------------------------------------------------
def log_prediction_request(
    request_id: str,
    symptoms: List[str],
    predictions: List[Dict[str, Any]],
    emergency_flag: bool,
):
    """Log production request symptom vector to data/prod_logs/requests.csv for drift detection."""
    log_dir = Path(os.environ.get("PROD_LOGS_DIR", "data/prod_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    requests_csv = log_dir / "requests.csv"

    # Create binary feature map matching model_manager.symptoms
    presented_set = {s.strip().lower() for s in symptoms}
    vector = [1 if sym in presented_set else 0 for sym in model_manager.symptoms]

    top_disease = predictions[0]["disease"] if predictions else "none"
    top_confidence = predictions[0]["confidence"] if predictions else 0.0

    with request_log_lock:
        file_exists = requests_csv.exists() and requests_csv.stat().st_size > 0
        with open(requests_csv, "a", encoding="utf-8") as f:
            if not file_exists:
                header = ["timestamp", "request_id", "top_disease", "top_confidence", "emergency_flag"] + model_manager.symptoms
                f.write(",".join(header) + "\n")
            row = [
                datetime.now(timezone.utc).isoformat(),
                request_id,
                f'"{top_disease}"',
                str(top_confidence),
                str(emergency_flag),
            ] + [str(v) for v in vector]
            f.write(",".join(row) + "\n")


def log_feedback_entry(request_id: str, helpful: bool, true_disease: str = None):
    """Log user/clinician feedback to data/feedback/feedback.csv."""
    feedback_dir = Path(os.environ.get("FEEDBACK_DIR", "data/feedback"))
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_csv = feedback_dir / "feedback.csv"

    with feedback_lock:
        file_exists = feedback_csv.exists() and feedback_csv.stat().st_size > 0
        with open(feedback_csv, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("timestamp,request_id,helpful,true_disease\n")
            row = [
                datetime.now(timezone.utc).isoformat(),
                request_id,
                str(helpful),
                f'"{true_disease}"' if true_disease else "",
            ]
            f.write(",".join(row) + "\n")


# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    """Healthcheck endpoint for liveness probes and container orchestrators."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.model is not None,
        model_version=model_manager.model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metrics")
def metrics():
    """Prometheus metrics exposition endpoint (scraped by Prometheus)."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/symptoms")
def get_symptoms():
    """Return the ordered list of all 131 valid symptom feature names."""
    return {
        "count": len(model_manager.symptoms),
        "symptoms": model_manager.symptoms,
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Return metadata about the currently serving production model."""
    return ModelInfoResponse(
        model_name="symptom-disease-classifier",
        model_version=model_manager.model_version,
        model_alias=model_manager.model_alias,
        model_type=model_manager.model_type,
        loaded_from=model_manager.loaded_from,
        load_time_seconds=round(model_manager.load_time_seconds, 4),
        num_features=len(model_manager.symptoms),
        num_classes=len(model_manager.diseases),
        quality_gate_passed=model_manager.quality_gate_passed,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, req: Request):
    """
    Predict top-3 diseases with confidence scores, recommended specialist,
    and emergency red-flag evaluation.
    """
    # 1. Validate empty symptoms
    if not request.symptoms:
        raise HTTPException(
            status_code=422,
            detail="At least one symptom must be provided in 'symptoms'.",
        )

    # 2. Validate unknown symptoms
    unknown = [s for s in request.symptoms if s.strip().lower() not in model_manager.symptom_set]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown symptom(s): {unknown}. Use GET /symptoms to inspect the 131 valid symptom names.",
        )

    # 3. Evaluate emergency red-flag rules (independent of model)
    emergency_flag, emergency_reasons = emergency_evaluator.evaluate(request.symptoms)

    # 4. Predict Top-3 diseases from ML model
    predictions_raw = model_manager.predict_top3(request.symptoms)
    predictions = [PredictionItem(**item) for item in predictions_raw]

    # 5. Extract request_id
    request_id = getattr(req.state, "request_id", uuid.uuid4().hex)

    # 6. Update Prometheus metrics
    top = predictions_raw[0]
    PREDICTIONS_BY_DISEASE.labels(disease=top["disease"]).inc()
    PREDICTIONS_BY_SPECIALIST.labels(specialist=top["specialist"]).inc()
    CONFIDENCE_HISTOGRAM.observe(top["confidence"])
    if emergency_flag:
        EMERGENCY_FLAG_COUNT.inc()

    # 7. Log production request for drift detection
    try:
        log_prediction_request(
            request_id=request_id,
            symptoms=request.symptoms,
            predictions=predictions_raw,
            emergency_flag=emergency_flag,
        )
    except Exception as e:
        logger.error(f"Failed to log production request: {e}")

    return PredictResponse(
        request_id=request_id,
        model_version=model_manager.model_version,
        predictions=predictions,
        emergency_flag=emergency_flag,
        emergency_reasons=emergency_reasons,
        disclaimer="Suggestion tool, not a medical diagnosis.",
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    """
    Record user or clinician feedback on prediction usefulness,
    optionally logging confirmed diagnosis.
    """
    if request.true_disease:
        disease_clean = request.true_disease.strip()
        if disease_clean not in model_manager.disease_set:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown true_disease: '{disease_clean}'. Must match one of the 41 canonical disease names.",
            )

    try:
        log_feedback_entry(
            request_id=request.request_id,
            helpful=request.helpful,
            true_disease=request.true_disease,
        )
        # Update Prometheus feedback counter
        FEEDBACK_COUNT.labels(helpful=str(request.helpful)).inc()
    except Exception as e:
        logger.error(f"Failed to append feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback.")

    return FeedbackResponse(status="recorded", request_id=request.request_id)

@app.post("/admin/reload-model")
def reload_model(x_api_key: str = Header(None, alias="X-API-Key")):
    """
    Safely reload the active production model and schemas into memory.
    Protected by ADMIN_API_KEY environment variable.
    """
    expected_key = os.environ.get("ADMIN_API_KEY", "default-admin-secret")
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Invalid or missing administrative API key in X-API-Key header.",
        )

    try:
        model_manager.load()
        MODEL_INFO.info(
            {
                "model_name": "symptom-disease-classifier",
                "model_version": model_manager.model_version,
                "model_type": model_manager.model_type,
                "model_alias": model_manager.model_alias,
            }
        )
        return {
            "status": "reloaded",
            "model_version": model_manager.model_version,
            "model_type": model_manager.model_type,
            "loaded_from": model_manager.loaded_from,
            "load_time_seconds": round(model_manager.load_time_seconds, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


