"""
tests/test_model.py
===================
Automated test suite for the trained production model.

Validates:
  1. The model loads reliably from models/production/.
  2. The model outputs probabilities for all 41 diseases.
  3. The predicted probabilities sum to ~1.0 for each input sample.
  4. Hold-out test set top-3 accuracy meets the configured quality gate (>= 0.90).
  5. The model classes match the canonical diseases.json list.
  6. Production metadata is consistent and documents quality gate passage.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture(scope="module")
def model_and_metadata():
    """Load model from models/production/ and associated artifacts."""
    prod_dir = os.path.join("models", "production")
    model_path = os.path.join(prod_dir, "model.joblib")
    meta_path = os.path.join(prod_dir, "metadata.json")
    symptoms_path = os.path.join(prod_dir, "symptoms.json")
    diseases_path = os.path.join(prod_dir, "diseases.json")

    assert os.path.exists(model_path), f"Model binary missing at {model_path}"
    assert os.path.exists(meta_path), f"Metadata missing at {meta_path}"

    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    with open(symptoms_path, "r") as f:
        symptoms = json.load(f)
    with open(diseases_path, "r") as f:
        diseases = json.load(f)

    return {
        "model": model,
        "metadata": metadata,
        "symptoms": symptoms,
        "diseases": diseases,
    }


@pytest.fixture(scope="module")
def test_data():
    """Load hold-out test set."""
    test_path = os.path.join("data", "processed", "test.csv")
    assert os.path.exists(test_path), f"Test set missing at {test_path}"
    df = pd.read_csv(test_path)
    X = df.drop(columns=["prognosis"])
    y = df["prognosis"]
    return X, y


@pytest.fixture(scope="module")
def quality_gate_config():
    """Load quality gate thresholds from model_config.yaml."""
    cfg_path = os.path.join("config", "model_config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("quality_gate", {})


def test_model_loads_from_production(model_and_metadata):
    """Verify that model object has predict and predict_proba methods."""
    model = model_and_metadata["model"]
    assert hasattr(model, "predict"), "Model must implement predict()"
    assert hasattr(model, "predict_proba"), "Model must implement predict_proba()"


def test_model_outputs_41_disease_probabilities(model_and_metadata, test_data):
    """Verify model outputs 41 probability scores per prediction."""
    model = model_and_metadata["model"]
    X, _ = test_data
    probs = model.predict_proba(X)

    assert probs.ndim == 2, f"Expected 2D probability array, got shape {probs.shape}"
    assert probs.shape[0] == len(X), f"Expected {len(X)} probability rows, got {probs.shape[0]}"
    assert probs.shape[1] == 41, f"Expected exactly 41 class probabilities, got {probs.shape[1]}"


def test_probabilities_sum_to_one(model_and_metadata, test_data):
    """Verify that output probabilities sum to 1.0 within floating point tolerance."""
    model = model_and_metadata["model"]
    X, _ = test_data
    probs = model.predict_proba(X)
    row_sums = probs.sum(axis=1)

    assert np.allclose(row_sums, 1.0, atol=1e-3), (
        f"Some probability rows do not sum to 1.0. Min sum: {row_sums.min():.5f}, Max sum: {row_sums.max():.5f}"
    )


def test_holdout_accuracy_meets_quality_gate(model_and_metadata, test_data, quality_gate_config):
    """Verify hold-out test set top-3 accuracy satisfies the minimum quality gate."""
    model = model_and_metadata["model"]
    X, y = test_data
    probs = model.predict_proba(X)
    classes = np.array(model.classes_)

    top3_indices = np.argsort(probs, axis=1)[:, -3:]
    hits = 0
    y_arr = y.to_numpy()
    for i, true_label in enumerate(y_arr):
        top3_diseases = classes[top3_indices[i]]
        if true_label in top3_diseases:
            hits += 1

    test_top3_acc = hits / len(y_arr)
    min_top3 = quality_gate_config.get("min_test_top3_accuracy", 0.90)

    assert test_top3_acc >= min_top3, (
        f"Hold-out top-3 accuracy {test_top3_acc:.4f} is below minimum quality gate {min_top3:.4f}"
    )


def test_model_classes_match_diseases_json(model_and_metadata):
    """Verify model.classes_ contains the exact 41 canonical disease names."""
    model = model_and_metadata["model"]
    expected_diseases = sorted(model_and_metadata["diseases"])
    actual_classes = sorted(list(model.classes_))

    assert actual_classes == expected_diseases, (
        f"Model classes mismatch diseases.json. Diff: {set(actual_classes) ^ set(expected_diseases)}"
    )


def test_production_metadata_consistency(model_and_metadata, quality_gate_config):
    """Verify metadata.json records passing the quality gate and valid metrics."""
    meta = model_and_metadata["metadata"]
    assert meta["quality_gate_passed"] is True, "Production model metadata must show quality gate passed."
    assert "production" in meta["aliases"], "Production model must have 'production' alias in metadata."
    assert "val_top3_accuracy" in meta["metrics"], "val_top3_accuracy must be in metadata metrics."
    assert meta["metrics"]["val_top3_accuracy"] >= quality_gate_config.get("min_val_top3_accuracy", 0.90)
