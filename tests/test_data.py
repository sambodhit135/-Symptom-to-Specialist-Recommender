"""
tests/test_data.py
==================
Automated data quality tests for the processed dataset.

Run with:
    pytest -q tests/test_data.py

All tests load from data/processed/ and data/disease_specialist.csv.
No network calls; no model loading. Tests are deterministic.
"""

import json
import os

import pandas as pd
import pytest

# ------------------------------------------------------------------
# FIXTURES: load once, reuse across tests
# ------------------------------------------------------------------
PROCESSED_DIR = os.path.join("data", "processed")
SPECIALIST_CSV = os.path.join("data", "disease_specialist.csv")
LABEL_COL = "prognosis"


@pytest.fixture(scope="session")
def train_df():
    """Load the processed training split."""
    return pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))


@pytest.fixture(scope="session")
def val_df():
    """Load the processed validation split."""
    return pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))


@pytest.fixture(scope="session")
def test_df():
    """Load the processed test (hold-out) split."""
    return pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))


@pytest.fixture(scope="session")
def specialist_df():
    """Load the disease-to-specialist mapping."""
    return pd.read_csv(SPECIALIST_CSV)


@pytest.fixture(scope="session")
def symptoms():
    """Load the ordered symptom column list from symptoms.json."""
    with open(os.path.join(PROCESSED_DIR, "symptoms.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def diseases():
    """Load the canonical disease list from diseases.json."""
    with open(os.path.join(PROCESSED_DIR, "diseases.json"), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# TEST 1: Every disease has exactly one non-empty specialist
# ------------------------------------------------------------------
def test_every_disease_has_one_specialist(specialist_df):
    """
    Each row in disease_specialist.csv must have:
      - a non-empty specialist name
      - exactly one mapping per disease (no duplicates)
    """
    # No empty specialist values
    empty = specialist_df[specialist_df["specialist"].isna() |
                          (specialist_df["specialist"].str.strip() == "")]
    assert len(empty) == 0, (
        f"These diseases have an empty specialist: {empty['disease'].tolist()}"
    )

    # No duplicate disease entries
    duplicates = specialist_df[specialist_df.duplicated(subset=["disease"])]
    assert len(duplicates) == 0, (
        f"Duplicate disease entries in specialist CSV: {duplicates['disease'].tolist()}"
    )


# ------------------------------------------------------------------
# TEST 2: Specialist CSV contains no unknown disease names
# ------------------------------------------------------------------
def test_specialist_mapping_no_unknown_diseases(specialist_df, diseases):
    """
    Every disease in disease_specialist.csv must be in diseases.json
    (i.e., only canonical cleaned names are used).
    """
    csv_diseases = set(specialist_df["disease"].tolist())
    json_diseases = set(diseases)
    unknown = csv_diseases - json_diseases
    assert unknown == set(), (
        f"Diseases in specialist CSV not in diseases.json: {unknown}"
    )


# ------------------------------------------------------------------
# TEST 3: All specialist CSV diseases are covered in diseases.json
# ------------------------------------------------------------------
def test_all_diseases_have_specialist(specialist_df, diseases):
    """
    Every disease in diseases.json must appear in disease_specialist.csv.
    Ensures no disease is left without a specialist mapping.
    """
    csv_diseases = set(specialist_df["disease"].tolist())
    json_diseases = set(diseases)
    missing = json_diseases - csv_diseases
    assert missing == set(), (
        f"Diseases in diseases.json with no specialist mapping: {missing}"
    )


# ------------------------------------------------------------------
# TEST 4: train / val / test share the same ordered columns
# ------------------------------------------------------------------
def test_column_consistency(train_df, val_df, test_df):
    """
    All three splits must have identical column names in identical order.
    This ensures the model sees the same feature layout regardless of split.
    """
    assert list(train_df.columns) == list(val_df.columns), (
        "train and val have different columns"
    )
    assert list(train_df.columns) == list(test_df.columns), (
        "train and test have different columns"
    )


# ------------------------------------------------------------------
# TEST 5: All symptom values are strictly 0 or 1
# ------------------------------------------------------------------
def test_symptom_values_are_binary(train_df, val_df, test_df, symptoms):
    """
    Symptom columns must contain only the values 0 and 1.
    Any other value (e.g., NaN, 2, -1) would corrupt model training.
    """
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sym_data = df[symptoms]
        unique_vals = set(sym_data.values.flatten().tolist())
        # Allow only {0, 1} (Python ints or numpy int64 both compare fine)
        assert unique_vals <= {0, 1}, (
            f"{name}: symptom columns contain unexpected values: "
            f"{unique_vals - {0, 1}}"
        )


# ------------------------------------------------------------------
# TEST 6: No missing values in any split
# ------------------------------------------------------------------
def test_no_missing_values(train_df, val_df, test_df):
    """
    All three splits must be free of NaN / null values.
    Missing values would break sklearn and FastAPI request handling.
    """
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        null_counts = df.isnull().sum()
        bad_cols = null_counts[null_counts > 0]
        assert len(bad_cols) == 0, (
            f"{name} has missing values:\n{bad_cols}"
        )


# ------------------------------------------------------------------
# TEST 7: Test labels are a subset of train labels
# ------------------------------------------------------------------
def test_test_labels_subset_of_train(train_df, test_df):
    """
    Every disease class in the test set must also appear in the training set.
    If a test label is unknown to the model, prediction is impossible.
    """
    train_labels = set(train_df[LABEL_COL].unique())
    test_labels  = set(test_df[LABEL_COL].unique())
    unseen = test_labels - train_labels
    assert unseen == set(), (
        f"Test set contains disease classes not in training: {unseen}"
    )


# ------------------------------------------------------------------
# TEST 8: No duplicate rows between train and val
# ------------------------------------------------------------------
def test_no_overlap_between_train_and_val(train_df, val_df):
    """
    No row should appear in both the training and validation splits.
    Overlap would cause optimistic validation metrics (data leakage).
    We use ALL columns (symptoms + label) for the duplicate check.
    """
    merged = pd.merge(train_df, val_df, how="inner")
    assert len(merged) == 0, (
        f"{len(merged)} identical rows appear in both train and val "
        f"(data leakage!)"
    )


# ------------------------------------------------------------------
# TEST 9: symptoms.json matches the symptom columns in train
# ------------------------------------------------------------------
def test_symptoms_json_matches_train_columns(train_df, symptoms):
    """
    The ordered list in symptoms.json must exactly match the symptom columns
    (all columns except 'prognosis') of train.csv, in the same order.
    The API uses symptoms.json to build the feature vector; any mismatch
    would cause silent column misalignment at inference time.
    """
    train_symptom_cols = [c for c in train_df.columns if c != LABEL_COL]
    assert train_symptom_cols == symptoms, (
        "symptoms.json order does not match train.csv column order.\n"
        f"First mismatch: "
        f"{next((a, b) for a, b in zip(train_symptom_cols, symptoms) if a != b)}"
    )
