"""
tests/test_drift.py
===================
Automated tests for Evidently Data & Concept Drift Monitoring.

Verifies:
  - Insufficient data guard when production requests < min_sample_size.
  - No drift detected on identical/same-distribution data.
  - Drift correctly detected on perturbed/shifted distributions.
  - Required JSON summary structure and fields.
  - HTML report file generation.
  - Concept drift proxy calculation from feedback logs.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.monitoring.drift import (
    evaluate_feedback_signal,
    run_drift_analysis,
)


@pytest.fixture(scope="module")
def sample_training_data():
    """Load baseline training data."""
    train_path = Path("data/processed/train.csv")
    assert train_path.exists(), "Processed train.csv must exist."
    return pd.read_csv(train_path)


def test_insufficient_data_guard(tmp_path, sample_training_data):
    """Verify system reports 'insufficient_data' instead of guessing on tiny samples."""
    # Create empty/small current data (only 5 rows, min required is 50)
    cur_df = sample_training_data.iloc[:5].copy()
    cur_df["top_disease"] = cur_df["prognosis"]

    summary = run_drift_analysis(
        override_reference_df=sample_training_data,
        override_current_df=cur_df,
        report_dir=str(tmp_path),
    )

    assert summary["status"] == "insufficient_data"
    assert summary["drift_detected"] is False
    assert summary["rows_compared"]["current_rows"] == 5
    assert "insufficient data" in summary["message"].lower()

    # Check that summary file was written
    summary_file = tmp_path / "drift_summary.json"
    assert summary_file.exists()


def test_no_drift_on_same_distribution(tmp_path, sample_training_data):
    """Verify that same-distribution data does not trigger false drift alarms."""
    ref_df = sample_training_data.copy()

    # Sample 80 rows with replacement to simulate identical distribution
    cur_df = sample_training_data.sample(n=80, replace=True, random_state=42).copy()
    cur_df["top_disease"] = cur_df["prognosis"]

    summary = run_drift_analysis(
        override_reference_df=ref_df,
        override_current_df=cur_df,
        report_dir=str(tmp_path),
    )

    assert summary["status"] == "completed"
    assert summary["drift_detected"] is False
    assert summary["share_of_drifted_features"] <= summary["drift_share_threshold"]
    assert summary["rows_compared"]["current_rows"] == 80


def test_drift_detected_on_shifted_distribution(tmp_path, sample_training_data):
    """Verify that heavily shifted symptom patterns reliably trigger drift alerts."""
    ref_df = sample_training_data.copy()
    symptom_cols = [c for c in ref_df.columns if c != "prognosis"]

    # Generate 80 shifted records with heavy urine/toxic cluster symptoms
    shifted_cluster = [
        "foul_smell_of_urine",
        "continuous_feel_of_urine",
        "bladder_discomfort",
        "burning_micturition",
        "internal_itching",
        "spotting_urination",
        "toxic_look_(typhos)",
        "altered_sensorium",
    ]

    shifted_rows = []
    for _ in range(80):
        row = {s: 0 for s in symptom_cols}
        for s in shifted_cluster:
            row[s] = 1
        row["top_disease"] = "Urinary tract infection"
        shifted_rows.append(row)

    cur_df = pd.DataFrame(shifted_rows)

    summary = run_drift_analysis(
        override_reference_df=ref_df,
        override_current_df=cur_df,
        report_dir=str(tmp_path),
    )

    assert summary["status"] == "completed"
    assert summary["drift_detected"] is True
    assert summary["share_of_drifted_features"] > summary["drift_share_threshold"]
    assert len(summary["top_drifted_features"]) > 0

    # Ensure shifted symptoms appear in top drifted features
    drifted_names = {item["feature"] for item in summary["top_drifted_features"]}
    assert any(s in drifted_names for s in shifted_cluster)


def test_drift_summary_json_keys(tmp_path, sample_training_data):
    """Verify all mandatory summary keys are present in the JSON output."""
    cur_df = sample_training_data.sample(n=60, replace=True, random_state=123).copy()
    cur_df["top_disease"] = cur_df["prognosis"]

    summary = run_drift_analysis(
        override_reference_df=sample_training_data,
        override_current_df=cur_df,
        report_dir=str(tmp_path),
    )

    expected_keys = [
        "status",
        "timestamp",
        "drift_detected",
        "share_of_drifted_features",
        "drift_share_threshold",
        "drifted_features_count",
        "total_features",
        "top_drifted_features",
        "prediction_distribution_drift",
        "rows_compared",
        "feedback_concept_drift",
        "html_report_path",
    ]
    for key in expected_keys:
        assert key in summary, f"Expected key '{key}' missing from summary dictionary."


def test_html_report_file_exists_and_non_empty(tmp_path, sample_training_data):
    """Verify Evidently HTML report is created and contains HTML markup."""
    cur_df = sample_training_data.sample(n=60, replace=True, random_state=99).copy()
    cur_df["top_disease"] = cur_df["prognosis"]

    summary = run_drift_analysis(
        override_reference_df=sample_training_data,
        override_current_df=cur_df,
        report_dir=str(tmp_path),
    )

    report_path = Path(summary["html_report_path"])
    assert report_path.exists()
    assert report_path.stat().st_size > 50000  # Report is rich standalone HTML

    content = report_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content or "<html" in content


def test_feedback_concept_drift_calculation(tmp_path):
    """Verify concept drift calculation from user feedback records."""
    fb_file = tmp_path / "feedback.csv"

    # 1. Test insufficient feedback data (< 10)
    fb_df_small = pd.DataFrame({
        "timestamp": ["2026-10-06T10:00:00"] * 4,
        "request_id": ["req-1", "req-2", "req-3", "req-4"],
        "helpful": [True, False, True, True],
        "true_disease": ["Dengue", "Malaria", "Typhoid", "Dengue"],
    })
    fb_df_small.to_csv(fb_file, index=False)

    sig_small = evaluate_feedback_signal(fb_file, min_sample_size=10, unhelpful_threshold=0.25)
    assert sig_small["status"] == "insufficient_data"
    assert sig_small["concept_drift_warning"] is False

    # 2. Test high unhelpful feedback triggering concept drift warning (40% unhelpful > 25% threshold)
    fb_df_large = pd.DataFrame({
        "timestamp": ["2026-10-06T10:00:00"] * 20,
        "request_id": [f"req-{i}" for i in range(20)],
        "helpful": [True] * 12 + [False] * 8,  # 8/20 = 40% unhelpful
        "true_disease": ["Dengue"] * 20,
    })
    fb_df_large.to_csv(fb_file, index=False)

    sig_large = evaluate_feedback_signal(fb_file, min_sample_size=10, unhelpful_threshold=0.25)
    assert sig_large["status"] == "evaluated"
    assert sig_large["unhelpful_count"] == 8
    assert sig_large["unhelpful_share"] == 0.40
    assert sig_large["concept_drift_warning"] is True
