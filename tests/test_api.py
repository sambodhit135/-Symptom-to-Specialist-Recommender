"""
tests/test_api.py
=================
Integration tests for the FastAPI prediction service using TestClient.
Zero dependency on external running servers or active network connections.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, model_manager


@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance for testing the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /health returns 200 and reports model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] != ""
    assert "timestamp" in data


def test_symptoms_endpoint(client):
    """Verify /symptoms returns the 131 standard symptom features."""
    response = client.get("/symptoms")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 131
    assert len(data["symptoms"]) == 131
    assert "itching" in data["symptoms"]
    assert "chest_pain" in data["symptoms"]


def test_model_info_endpoint(client):
    """Verify /model-info returns complete production model metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "symptom-disease-classifier"
    assert data["model_alias"] == "production"
    assert data["num_features"] == 131
    assert data["num_classes"] == 41
    assert data["quality_gate_passed"] is True


def test_valid_predict_top3(client):
    """Verify /predict returns 3 sorted predictions with valid specialists and disclaimer."""
    payload = {"symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "request_id" in data
    assert data["model_version"] != ""
    assert data["disclaimer"] == "Suggestion tool, not a medical diagnosis."
    assert data["emergency_flag"] is False
    assert data["emergency_reasons"] == []

    # Verify exactly 3 predictions
    predictions = data["predictions"]
    assert len(predictions) == 3

    # Verify sorted descending by confidence
    confidences = [p["confidence"] for p in predictions]
    assert confidences == sorted(confidences, reverse=True)

    # Verify top prediction is Fungal infection with a Dermatologist
    assert predictions[0]["disease"] == "Fungal infection"
    assert predictions[0]["specialist"] == "Dermatologist"
    assert predictions[0]["confidence"] > 0.5

    # Verify each prediction has a valid specialist
    for p in predictions:
        assert p["disease"] in model_manager.disease_set
        assert p["specialist"] != ""


def test_predict_empty_symptoms_returns_422(client):
    """Verify empty symptom list is rejected with HTTP 422."""
    response = client.post("/predict", json={"symptoms": []})
    assert response.status_code == 422
    data = response.json()
    assert "At least one symptom must be provided" in data["detail"]


def test_predict_unknown_symptom_returns_422(client):
    """Verify unknown symptom name is rejected with HTTP 422 and helpful message."""
    response = client.post("/predict", json={"symptoms": ["magic_cure_symptom", "itching"]})
    assert response.status_code == 422
    data = response.json()
    assert "Unknown symptom(s)" in data["detail"]
    assert "magic_cure_symptom" in data["detail"]


def test_emergency_flag_triggers_on_red_flag_symptom(client):
    """Verify emergency flag triggers when a critical individual symptom is presented."""
    response = client.post("/predict", json={"symptoms": ["chest_pain"]})
    assert response.status_code == 200
    data = response.json()
    assert data["emergency_flag"] is True
    assert len(data["emergency_reasons"]) >= 1
    assert any("chest_pain" in r or "coronary" in r.lower() for r in data["emergency_reasons"])


def test_emergency_flag_triggers_on_combination(client):
    """Verify emergency flag triggers on a critical symptom combination."""
    response = client.post("/predict", json={"symptoms": ["chest_pain", "sweating"]})
    assert response.status_code == 200
    data = response.json()
    assert data["emergency_flag"] is True
    assert len(data["emergency_reasons"]) >= 2  # individual chest_pain + combo
    assert any("myocardial infarction" in r.lower() for r in data["emergency_reasons"])


def test_feedback_saved_to_temp_path(client, monkeypatch, tmp_path):
    """Verify feedback is recorded and appended safely to CSV."""
    monkeypatch.setenv("FEEDBACK_DIR", str(tmp_path))

    feedback_payload = {
        "request_id": "req-uuid-test-999",
        "helpful": True,
        "true_disease": "Fungal infection",
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "recorded"
    assert data["request_id"] == "req-uuid-test-999"

    feedback_csv = tmp_path / "feedback.csv"
    assert feedback_csv.exists()
    content = feedback_csv.read_text(encoding="utf-8")
    assert "req-uuid-test-999" in content
    assert "Fungal infection" in content
    assert "True" in content


def test_feedback_unknown_disease_returns_422(client):
    """Verify submitting feedback with an invalid disease name returns 422."""
    feedback_payload = {
        "request_id": "req-uuid-test-888",
        "helpful": False,
        "true_disease": "InventedAlienDisease",
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 422
    data = response.json()
    assert "Unknown true_disease" in data["detail"]
