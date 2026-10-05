# Phase 4 Notes: FastAPI Prediction Service, Safety Triage & Structured Logging

## 1. Overview and Objectives
Phase 4 delivers an enterprise-grade, high-performance REST API built with FastAPI. It serves the trained production disease classification model, performs independent emergency triage checks, captures clinician/user feedback, logs request symptom vectors for drift monitoring, and outputs structured JSON logs.

---

## 2. API Endpoints Specification

| Method | Endpoint | Description | Request Payload | Response / Status |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service liveness & model readiness | None | `200 OK`: `{"status": "healthy", "model_loaded": true, "model_version": "1", ...}` |
| `GET` | `/symptoms` | Ordered feature names | None | `200 OK`: `{"count": 131, "symptoms": [...]}` |
| `GET` | `/model-info` | Production model metadata & latency | None | `200 OK`: `{"model_name": "symptom-disease-classifier", "model_alias": "production", ...}` |
| `POST` | `/predict` | Top-3 differential predictions + triage | `{"symptoms": ["itching", "skin_rash"]}` | `200 OK`: Top-3 diseases, confidences, specialists, emergency flag, disclaimer |
| `POST` | `/feedback` | Record prediction utility & true diagnosis | `{"request_id": "...", "helpful": true, "true_disease": "Fungal infection"}` | `200 OK`: `{"status": "recorded", "request_id": "..."}` |

---

## 3. Safety Rules & Independent Emergency Triage
- **Config**: [`config/emergency_rules.json`](file:///d:/Disease%20Prediction%20Using%20Machine%20Learning/config/emergency_rules.json)
- **Status**: `"needs review by the team"` (explicitly marked for clinical validation).
- **Architecture**: Evaluated via [`EmergencyEvaluator`](file:///d:/Disease%20Prediction%20Using%20Machine%20Learning/src/api/emergency.py) independently of the machine learning model.
- **Rule Categories**:
  1. **Individual Red-Flags**: Critical indicators requiring immediate emergency intervention (`chest_pain`, `breathlessness`, `altered_sensorium`, `coma`, `weakness_of_one_body_side`, `acute_liver_failure`, `stomach_bleeding`, `blood_in_sputum`).
  2. **High-Risk Combinations**: Combinations indicating acute syndromes (e.g. `chest_pain + sweating` for Acute Myocardial Infarction; `high_fever + altered_sensorium` for Meningitis/Sepsis; `dizziness + loss_of_balance + slurred_speech` for Acute Stroke).
  - All symptoms strictly verified to exist in `symptoms.json`.

---

## 4. Input Validation & Error Handling
- **Empty Symptom List**: Rejected with `HTTP 422 Unprocessable Entity`:
  `{"detail": "At least one symptom must be provided in 'symptoms'."}`
- **Unknown Symptom Names**: Rejected with `HTTP 422 Unprocessable Entity`:
  `{"detail": "Unknown symptom(s): ['nonexistent_symptom']. Use GET /symptoms to inspect the 131 valid symptom names."}`
- **Feedback True Disease Validation**: Validated against `diseases.json`. Unknown disease names rejected with `HTTP 422`.

---

## 5. Structured Observability & Data Capture
1. **JSON-Structured Application Logs**:
   Every request intercepted by ASGI middleware, logging:
   ```json
   {"timestamp": "2026-10-05T16:23:43.562423+00:00", "request_id": "ca7e8bab9ec24abc9b5a18dcb29fc388", "method": "POST", "path": "/predict", "status_code": 200, "latency_ms": 38.06, "client_ip": "127.0.0.1"}
   ```
2. **Production Drift Logging**:
   Symptom vectors logged to [`data/prod_logs/requests.csv`](file:///d:/Disease%20Prediction%20Using%20Machine%20Learning/data/prod_logs/requests.csv) with columns: `timestamp,request_id,top_disease,top_confidence,emergency_flag,<131 symptom binary columns>`.
3. **Feedback Storage**:
   Logged to [`data/feedback/feedback.csv`](file:///d:/Disease%20Prediction%20Using%20Machine%20Learning/data/feedback/feedback.csv) with thread-safe file appends.

---

## 6. Test Suite Results (`pytest -q`)
- **Total Tests**: 25 passed.
- **Phase 2 Data Tests**: 9 passed.
- **Phase 3 Model Tests**: 6 passed.
- **Phase 4 API Tests**: 10 passed (`tests/test_api.py` covering health, predict, 422 validation, emergency flags, feedback temp logging, and schema consistency).
