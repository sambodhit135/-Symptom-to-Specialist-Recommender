"""
src/retrain_pipeline.py
=======================
Automated Retraining and Model Promotion Engine.

Executes:
  1. Drift & Feedback inspection: determines if retraining condition is met.
  2. Dataset synthesis: merges verified feedback into baseline training data
     under data/retraining/ (never overwrites data/raw).
  3. Candidate training: trains updated model and logs to MLflow.
  4. Production comparison: evaluates candidate vs current production model
     on hold-out test set with quality gate margin.
  5. Promotion: updates MLflow 'production' alias and refreshes models/production/.
  6. API hot-reload: triggers POST /admin/reload-model.
"""

import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath("."))

logger = logging.getLogger("retrain_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def check_drift_and_feedback(
    drift_summary_path: str = "reports/drift/drift_summary.json",
    feedback_path: str = "data/feedback/feedback.csv",
    feedback_threshold: int = 10,
) -> Dict[str, Any]:
    """
    Check if retraining is required due to data/concept drift OR accumulated verified feedback.
    """
    drift_detected = False
    drift_share = 0.0
    p_drift_summary = Path(drift_summary_path)

    if p_drift_summary.exists():
        try:
            with open(p_drift_summary, "r", encoding="utf-8") as f:
                d = json.load(f)
            drift_detected = bool(d.get("drift_detected", False))
            drift_share = float(d.get("share_of_drifted_features", 0.0))
        except Exception as e:
            logger.warning(f"Error reading drift summary: {e}")

    # Count validated unhelpful feedback with labeled ground truth
    feedback_count = 0
    p_fb = Path(feedback_path)
    if p_fb.exists() and p_fb.stat().st_size > 0:
        try:
            df_fb = pd.read_csv(p_fb)
            if "helpful" in df_fb.columns and "true_disease" in df_fb.columns:
                # Count feedback with unhelpful label and provided ground truth disease
                valid_fb = df_fb[
                    df_fb["helpful"].isin([False, "False", "false", 0]) &
                    df_fb["true_disease"].notna() &
                    (df_fb["true_disease"].astype(str).str.strip() != "")
                ]
                feedback_count = len(valid_fb)
        except Exception as e:
            logger.warning(f"Error reading feedback file: {e}")

    feedback_condition = feedback_count >= feedback_threshold
    should_retrain = bool(drift_detected or feedback_condition)

    reasons: List[str] = []
    if drift_detected:
        reasons.append(f"Data drift detected (share={drift_share:.2%})")
    if feedback_condition:
        reasons.append(f"Feedback threshold reached ({feedback_count}/{feedback_threshold} verified corrections)")

    reason_str = "; ".join(reasons) if reasons else "No drift detected and feedback below threshold."

    logger.info(f"Retrain check: should_retrain={should_retrain} ({reason_str})")
    return {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "feedback_count": feedback_count,
        "feedback_threshold": feedback_threshold,
        "feedback_condition_met": feedback_condition,
        "should_retrain": should_retrain,
        "reason": reason_str,
    }


def build_retraining_dataset(
    base_train_path: str = "data/processed/train.csv",
    feedback_path: str = "data/feedback/feedback.csv",
    requests_path: str = "data/prod_logs/requests.csv",
    diseases_json_path: str = "data/processed/diseases.json",
    output_dir: str = "data/retraining",
) -> str:
    """
    Synthesize an updated training dataset by merging original processed data with validated feedback.
    Never modifies or overwrites data/raw/.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_p = Path(base_train_path)
    if not base_p.exists():
        raise FileNotFoundError(f"Base training dataset not found at {base_train_path}")

    base_df = pd.read_csv(base_p)
    logger.info(f"Loaded base training data: {len(base_df)} rows")

    # Load canonical diseases
    with open(diseases_json_path, "r", encoding="utf-8") as f:
        valid_diseases = set(json.load(f))

    symptom_cols = [c for c in base_df.columns if c != "prognosis"]
    new_rows: List[Dict[str, Any]] = []

    # Check for feedback corrections
    fb_p = Path(feedback_path)
    req_p = Path(requests_path)

    if fb_p.exists() and fb_p.stat().st_size > 0 and req_p.exists() and req_p.stat().st_size > 0:
        try:
            df_fb = pd.read_csv(fb_p)
            df_req = pd.read_csv(req_p)

            # Filter valid corrections
            valid_fb = df_fb[
                df_fb["helpful"].isin([False, "False", "false", 0]) &
                df_fb["true_disease"].notna()
            ]

            # Merge with request symptom vectors
            for _, fb_row in valid_fb.iterrows():
                true_dis = str(fb_row["true_disease"]).strip()
                if true_dis in valid_diseases:
                    req_id = fb_row["request_id"]
                    req_match = df_req[df_req["request_id"] == req_id]
                    if not req_match.empty:
                        matched_row = req_match.iloc[0]
                        row_dict = {col: int(matched_row.get(col, 0)) for col in symptom_cols}
                        row_dict["prognosis"] = true_dis
                        new_rows.append(row_dict)
        except Exception as e:
            logger.warning(f"Failed to extract feedback rows: {e}")

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        combined_df = pd.concat([base_df, df_new], ignore_index=True)
        # Deduplicate
        combined_df = combined_df.drop_duplicates(subset=symptom_cols + ["prognosis"])
        logger.info(f"Merged {len(new_rows)} validated feedback rows into retraining set. Total: {len(combined_df)}")
    else:
        combined_df = base_df.copy()
        logger.info("No validated feedback rows available; using base training distribution.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamped_file = out_dir / f"retrain_{timestamp}.csv"
    latest_file = out_dir / "retrain_latest.csv"

    combined_df.to_csv(timestamped_file, index=False)
    combined_df.to_csv(latest_file, index=False)

    logger.info(f"Retraining dataset written to {timestamped_file} (and {latest_file})")
    return str(latest_file.resolve())


def retrain_candidate_model(
    training_data_path: str = "data/retraining/retrain_latest.csv",
    test_data_path: str = "data/processed/test.csv",
    experiment_name: str = "symptom-disease",
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrain a candidate RandomForestClassifier model on the updated retraining dataset.
    Logs run and registers model in MLflow.
    """
    train_df = pd.read_csv(training_data_path)
    test_df = pd.read_csv(test_data_path)

    symptom_cols = [c for c in train_df.columns if c != "prognosis"]

    X_train = train_df[symptom_cols]
    y_train = train_df["prognosis"]

    X_test = test_df[symptom_cols]
    y_test = test_df["prognosis"]

    # Train model
    candidate_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
    )
    candidate_model.fit(X_train, y_train)

    # Evaluate on Hold-out Test set
    y_pred = candidate_model.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))
    test_f1 = float(f1_score(y_test, y_pred, average="macro"))

    logger.info(f"Candidate model trained: Holdout Accuracy={test_acc:.4f}, Macro-F1={test_f1:.4f}")

    # Set up MLflow
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    run_id = "local_candidate_run"
    model_version = "1"

    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="retrained_candidate") as run:
            run_id = run.info.run_id
            mlflow.log_params({
                "n_estimators": 100,
                "model_type": "RandomForestClassifier",
                "training_samples": len(train_df),
                "retraining_trigger": "drift_or_feedback",
            })
            mlflow.log_metrics({
                "test_accuracy": test_acc,
                "test_macro_f1": test_f1,
            })

            # Save model artifact
            mlflow.sklearn.log_model(
                sk_model=candidate_model,
                artifact_path="model",
                registered_model_name="symptom-disease-classifier",
            )
            # Find registered version
            client = mlflow.tracking.MlflowClient()
            reg_versions = client.search_model_versions("name='symptom-disease-classifier'")
            if reg_versions:
                model_version = str(reg_versions[0].version)
    except Exception as e:
        logger.warning(f"MLflow tracking/registration had a warning (proceeding with local metadata): {e}")

    # Save candidate binary locally in temp cache for promotion
    temp_dir = Path("models/candidate_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate_model, temp_dir / "candidate_model.joblib")

    return {
        "candidate_run_id": run_id,
        "candidate_version": model_version,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "model_type": "RandomForestClassifier",
        "training_samples": len(train_df),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def compare_with_production(
    candidate_metrics: Dict[str, Any],
    test_data_path: str = "data/processed/test.csv",
    prod_model_dir: str = "models/production",
    margin: float = 0.0,
) -> Dict[str, Any]:
    """
    Compare candidate model against the active production model on the hold-out test set.
    Candidate must beat or match production accuracy (cand_acc >= prod_acc - margin).
    """
    test_df = pd.read_csv(test_data_path)
    symptom_cols = [c for c in test_df.columns if c != "prognosis"]
    X_test = test_df[symptom_cols]
    y_test = test_df["prognosis"]

    prod_dir = Path(prod_model_dir)
    prod_model_file = prod_dir / "model.joblib"

    if prod_model_file.exists():
        prod_model = joblib.load(prod_model_file)
        y_prod_pred = prod_model.predict(X_test)
        prod_acc = float(accuracy_score(y_test, y_prod_pred))
        prod_f1 = float(f1_score(y_test, y_prod_pred, average="macro"))
    else:
        prod_acc = 0.95
        prod_f1 = 0.95

    cand_acc = float(candidate_metrics["test_accuracy"])
    cand_f1 = float(candidate_metrics["test_f1"])

    # Comparison decision: candidate must satisfy accuracy quality gate
    promote_decision = bool(cand_acc >= (prod_acc - margin))
    action = "promote" if promote_decision else "skip_promotion"

    diff = round(cand_acc - prod_acc, 4)

    logger.info(
        f"Model Comparison: Candidate Acc={cand_acc:.4f} vs Production Acc={prod_acc:.4f} "
        f"(diff={diff:+.4f}, margin={margin}) -> Action: {action.upper()}"
    )

    return {
        "promote": promote_decision,
        "action": action,
        "candidate_accuracy": cand_acc,
        "candidate_f1": cand_f1,
        "production_accuracy": prod_acc,
        "production_f1": prod_f1,
        "accuracy_diff": diff,
        "margin": margin,
    }


def promote_candidate(
    candidate_metrics: Dict[str, Any],
    prod_model_dir: str = "models/production",
    candidate_temp_dir: str = "models/candidate_temp",
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Promote candidate model: updates MLflow 'production' alias and refreshes models/production/ on disk.
    """
    prod_dir = Path(prod_model_dir)
    prod_dir.mkdir(parents=True, exist_ok=True)

    cand_file = Path(candidate_temp_dir) / "candidate_model.joblib"
    if not cand_file.exists():
        raise FileNotFoundError(f"Candidate model artifact not found at {cand_file}")

    # 1. Update MLflow Alias
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    alias_updated = False
    cand_version = candidate_metrics.get("candidate_version", "2")

    try:
        import mlflow
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)
        client.set_registered_model_alias(
            name="symptom-disease-classifier",
            alias="production",
            version=cand_version,
        )
        alias_updated = True
        logger.info(f"Updated MLflow model alias 'production' to version {cand_version}")
    except Exception as e:
        logger.warning(f"Could not update MLflow alias (continuing with local promotion): {e}")

    # 2. Copy binary to models/production/
    target_joblib = prod_dir / "model.joblib"
    shutil.copyfile(cand_file, target_joblib)

    # 3. Compute SHA256
    h = hashlib.sha256()
    with open(target_joblib, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    model_sha256 = h.hexdigest()

    # 4. Update metadata.json
    metadata = {
        "model_name": "symptom-disease-classifier",
        "model_type": candidate_metrics.get("model_type", "RandomForestClassifier"),
        "model_version": cand_version,
        "model_alias": "production",
        "sha256": model_sha256,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "holdout_accuracy": candidate_metrics.get("test_accuracy", 1.0),
        "holdout_f1": candidate_metrics.get("test_f1", 1.0),
        "quality_gate_passed": True,
        "mlflow_alias_set": alias_updated,
    }
    with open(prod_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Promoted new model version {cand_version} to {prod_dir}")
    return metadata


def reload_api_model(
    api_url: str = "http://localhost:8000",
    admin_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call POST /admin/reload-model on the API service to hot-reload the newly promoted model.
    """
    key = admin_api_key or os.environ.get("ADMIN_API_KEY", "admin-secret-reload-key")
    headers = {"X-API-Key": key}
    url = f"{api_url}/admin/reload-model"

    try:
        resp = requests.post(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"API hot-reload succeeded: version={data.get('model_version')}")
            return {"status": "success", "response": data}
        else:
            logger.warning(f"API reload returned HTTP {resp.status_code}: {resp.text}")
            return {"status": "failed", "status_code": resp.status_code, "detail": resp.text}
    except Exception as e:
        logger.warning(f"Could not connect to API for reload at {url}: {e}")
        return {"status": "unreachable", "error": str(e)}
