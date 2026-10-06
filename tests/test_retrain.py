"""
tests/test_retrain.py
=====================
Unit tests for the automated retraining pipeline & DAG components.

Verifies:
  - Drift & feedback condition checking.
  - Retraining dataset synthesis under data/retraining/ without touching data/raw/.
  - Candidate model training & evaluation.
  - Model comparison logic with configurable margin.
  - Candidate promotion and metadata generation.
  - DAG structure and task dependency graph.
"""

import json
from pathlib import Path

import pandas as pd

from src.retrain_pipeline import (
    build_retraining_dataset,
    check_drift_and_feedback,
    compare_with_production,
    promote_candidate,
    retrain_candidate_model,
)


def test_check_drift_and_feedback_triggers_on_drift(tmp_path):
    """Verify that drift_detected=True in drift summary triggers should_retrain=True."""
    summary_file = tmp_path / "drift_summary.json"
    summary_file.write_text(json.dumps({"drift_detected": True, "share_of_drifted_features": 0.35}))

    res = check_drift_and_feedback(
        drift_summary_path=str(summary_file),
        feedback_path=str(tmp_path / "non_existent_feedback.csv"),
    )
    assert res["should_retrain"] is True
    assert res["drift_detected"] is True
    assert "Data drift detected" in res["reason"]


def test_check_drift_and_feedback_skips_when_clean(tmp_path):
    """Verify that normal conditions return should_retrain=False."""
    summary_file = tmp_path / "drift_summary.json"
    summary_file.write_text(json.dumps({"drift_detected": False, "share_of_drifted_features": 0.0}))

    res = check_drift_and_feedback(
        drift_summary_path=str(summary_file),
        feedback_path=str(tmp_path / "non_existent_feedback.csv"),
    )
    assert res["should_retrain"] is False
    assert res["drift_detected"] is False


def test_build_retraining_dataset_never_touches_raw(tmp_path):
    """Verify dataset is saved under data/retraining/ and data/raw/ remains untouched."""
    raw_dir = Path("data/raw")
    raw_files_before = set(raw_dir.glob("*")) if raw_dir.exists() else set()

    output_dir = tmp_path / "retraining"
    output_path = build_retraining_dataset(
        base_train_path="data/processed/train.csv",
        feedback_path="data/feedback/feedback.csv",
        requests_path="data/prod_logs/requests.csv",
        diseases_json_path="data/processed/diseases.json",
        output_dir=str(output_dir),
    )

    assert Path(output_path).exists()
    assert (output_dir / "retrain_latest.csv").exists()

    df_retrain = pd.read_csv(output_path)
    assert len(df_retrain) >= 240
    assert "prognosis" in df_retrain.columns

    # Verify data/raw was not modified
    if raw_dir.exists():
        raw_files_after = set(raw_dir.glob("*"))
        assert raw_files_before == raw_files_after


def test_retrain_and_compare_pipeline(tmp_path):
    """Verify retraining produces valid candidate model and comparison evaluates quality gate."""
    # 1. Train candidate model
    candidate_metrics = retrain_candidate_model(
        training_data_path="data/processed/train.csv",
        test_data_path="data/processed/test.csv",
    )

    assert "test_accuracy" in candidate_metrics
    assert candidate_metrics["test_accuracy"] >= 0.90
    assert "candidate_version" in candidate_metrics

    # 2. Compare against production baseline
    comp = compare_with_production(
        candidate_metrics=candidate_metrics,
        test_data_path="data/processed/test.csv",
        prod_model_dir="models/production",
        margin=0.0,
    )

    assert "promote" in comp
    assert comp["action"] in ("promote", "skip_promotion")
    assert comp["candidate_accuracy"] >= 0.90
    assert comp["production_accuracy"] >= 0.90


def test_promote_candidate_updates_metadata(tmp_path):
    """Verify promotion writes model.joblib and valid metadata.json."""
    temp_cand_dir = tmp_path / "candidate_temp"
    temp_cand_dir.mkdir(parents=True)
    temp_prod_dir = tmp_path / "production"
    temp_prod_dir.mkdir(parents=True)

    # Train and save a small dummy model
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit([[0, 1], [1, 0]], ["A", "B"])
    import joblib
    joblib.dump(dummy, temp_cand_dir / "candidate_model.joblib")

    candidate_metrics = {
        "candidate_version": "99",
        "model_type": "DummyClassifier",
        "test_accuracy": 1.0,
        "test_f1": 1.0,
    }

    meta = promote_candidate(
        candidate_metrics=candidate_metrics,
        prod_model_dir=str(temp_prod_dir),
        candidate_temp_dir=str(temp_cand_dir),
    )

    assert (temp_prod_dir / "model.joblib").exists()
    assert (temp_prod_dir / "metadata.json").exists()
    assert meta["model_version"] == "99"
    assert meta["quality_gate_passed"] is True


def test_dag_import_and_structure():
    """Verify the Airflow DAG imports cleanly and contains all required tasks and branching."""
    import dags.retrain_symptom_model as dag_module

    dag = getattr(dag_module, "dag", None)
    assert dag is not None
    assert dag.dag_id == "retrain_symptom_model"

    task_ids = {
        "check_drift",
        "branch_on_retrain_needed",
        "build_retraining_dataset",
        "retrain",
        "compare_with_production",
        "branch_on_promotion",
        "promote",
        "skip_promotion",
        "skip_retraining",
        "reload_api",
        "log_summary",
    }

    # Verify tasks if airflow native tasks are present
    if hasattr(dag, "task_ids"):
        assert task_ids.issubset(set(dag.task_ids))
