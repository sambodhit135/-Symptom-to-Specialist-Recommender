"""
dags/retrain_symptom_model.py
=============================
Airflow DAG for Automated Symptom Disease Model Retraining & Continuous Delivery.

Workflow:
  1. check_drift: Inspects latest data drift report and feedback database.
  2. branch_on_retrain_needed: Branches to build_retraining_dataset if drift or feedback threshold met,
     otherwise skips directly to log_summary.
  3. build_retraining_dataset: Synthesizes new dataset with validated feedback under data/retraining/.
  4. retrain: Trains candidate model on updated data and logs run to MLflow.
  5. compare_with_production: Evaluates candidate vs active production model on hold-out set.
  6. branch_on_promotion: Promotes if candidate meets quality gate, else skips promotion.
  7. promote: Moves MLflow 'production' alias and refreshes models/production/.
  8. reload_api: Calls POST /admin/reload-model on the FastAPI service.
  9. log_summary: Emits execution summary and metric comparison.
"""

import logging
import os
import sys
from datetime import datetime, timedelta

# Ensure repository root is on path for custom pipeline imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ruff: noqa: E402 – imports must come after sys.path manipulation
from src.retrain_pipeline import (  # noqa: E402, I001
    build_retraining_dataset as _build_dataset,
    check_drift_and_feedback as _check_drift,
    compare_with_production as _compare_models,
    promote_candidate as _promote_candidate,
    reload_api_model as _reload_api,
    retrain_candidate_model as _retrain_model,
)

# Safe imports for Airflow with local fallback
try:
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator, PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    # Mock classes for local Windows unit testing without native airflow package
    class DAG:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.dag_id = args[0] if args else kwargs.get("dag_id", "mock_dag")
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class PythonOperator:  # type: ignore
        def __init__(self, task_id, python_callable, **kwargs):
            self.task_id = task_id
            self.python_callable = python_callable
        def __rshift__(self, other): return other

    class BranchPythonOperator(PythonOperator):  # type: ignore
        pass

    class EmptyOperator(PythonOperator):  # type: ignore
        def __init__(self, task_id, **kwargs):
            super().__init__(task_id, lambda: None, **kwargs)



logger = logging.getLogger("airflow.retrain_dag")

# DAG Default Arguments
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# ── Python Task Callables ───────────────────────────────────────────────────

def task_check_drift(**context):
    """Check latest drift report and feedback count."""
    res = _check_drift(
        drift_summary_path=os.path.join(BASE_DIR, "reports/drift/drift_summary.json"),
        feedback_path=os.path.join(BASE_DIR, "data/feedback/feedback.csv"),
        feedback_threshold=int(os.environ.get("FEEDBACK_THRESHOLD", 10)),
    )
    if "ti" in context:
        context["ti"].xcom_push(key="drift_check_result", value=res)
    return res


def task_branch_on_retrain_needed(**context):
    """Determine whether retraining is necessary."""
    ti = context.get("ti")
    if ti:
        drift_res = ti.xcom_pull(task_ids="check_drift", key="drift_check_result")
    else:
        drift_res = _check_drift()

    if drift_res and drift_res.get("should_retrain", False):
        logger.info(f"Retraining triggered: {drift_res.get('reason')}")
        return "build_retraining_dataset"
    else:
        logger.info("No retraining necessary. Skipping to summary.")
        return "skip_retraining"


def task_build_dataset(**context):
    """Synthesize updated retraining dataset."""
    output_path = _build_dataset(
        base_train_path=os.path.join(BASE_DIR, "data/processed/train.csv"),
        feedback_path=os.path.join(BASE_DIR, "data/feedback/feedback.csv"),
        requests_path=os.path.join(BASE_DIR, "data/prod_logs/requests.csv"),
        diseases_json_path=os.path.join(BASE_DIR, "data/processed/diseases.json"),
        output_dir=os.path.join(BASE_DIR, "data/retraining"),
    )
    if "ti" in context:
        context["ti"].xcom_push(key="retraining_data_path", value=output_path)
    return output_path


def task_retrain(**context):
    """Train new candidate model on synthesized dataset."""
    ti = context.get("ti")
    data_path = ti.xcom_pull(task_ids="build_retraining_dataset", key="retraining_data_path") if ti else None
    if not data_path:
        data_path = os.path.join(BASE_DIR, "data/retraining/retrain_latest.csv")

    candidate_res = _retrain_model(
        training_data_path=data_path,
        test_data_path=os.path.join(BASE_DIR, "data/processed/test.csv"),
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    )
    if ti:
        ti.xcom_push(key="candidate_metrics", value=candidate_res)
    return candidate_res


def task_compare(**context):
    """Compare candidate model against production baseline."""
    ti = context.get("ti")
    candidate_metrics = ti.xcom_pull(task_ids="retrain", key="candidate_metrics") if ti else None
    if not candidate_metrics:
        raise ValueError("Missing candidate_metrics from retrain task.")

    margin = float(os.environ.get("PROMOTION_MARGIN", 0.0))
    comp_res = _compare_models(
        candidate_metrics=candidate_metrics,
        test_data_path=os.path.join(BASE_DIR, "data/processed/test.csv"),
        prod_model_dir=os.path.join(BASE_DIR, "models/production"),
        margin=margin,
    )
    if ti:
        ti.xcom_push(key="comparison_result", value=comp_res)
    return comp_res


def task_branch_on_promotion(**context):
    """Branch to promote or skip_promotion based on metric comparison."""
    ti = context.get("ti")
    comp_res = ti.xcom_pull(task_ids="compare_with_production", key="comparison_result") if ti else None
    if comp_res and comp_res.get("promote", False):
        return "promote"
    return "skip_promotion"


def task_promote(**context):
    """Promote candidate model into production."""
    ti = context.get("ti")
    candidate_metrics = ti.xcom_pull(task_ids="retrain", key="candidate_metrics") if ti else None
    if not candidate_metrics:
        raise ValueError("Missing candidate metrics for promotion.")

    meta = _promote_candidate(
        candidate_metrics=candidate_metrics,
        prod_model_dir=os.path.join(BASE_DIR, "models/production"),
        candidate_temp_dir=os.path.join(BASE_DIR, "models/candidate_temp"),
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    )
    if ti:
        ti.xcom_push(key="promoted_metadata", value=meta)
    return meta


def task_reload_api(**context):
    """Trigger API hot-reload."""
    api_url = os.environ.get("API_INTERNAL_URL", os.environ.get("API_URL", "http://api:8000"))
    admin_key = os.environ.get("ADMIN_API_KEY", "admin-secret-reload-key")
    res = _reload_api(api_url=api_url, admin_api_key=admin_key)
    if "ti" in context:
        context["ti"].xcom_push(key="api_reload_result", value=res)
    return res


def task_log_summary(**context):
    """Log complete execution summary."""
    ti = context.get("ti")
    drift_res = ti.xcom_pull(task_ids="check_drift", key="drift_check_result") if ti else {}
    comp_res = ti.xcom_pull(task_ids="compare_with_production", key="comparison_result") if ti else {}
    prom_meta = ti.xcom_pull(task_ids="promote", key="promoted_metadata") if ti else {}
    reload_res = ti.xcom_pull(task_ids="reload_api", key="api_reload_result") if ti else {}

    summary = {
        "execution_time": datetime.now().isoformat(),
        "drift_detected": drift_res.get("drift_detected", False),
        "feedback_count": drift_res.get("feedback_count", 0),
        "should_retrain": drift_res.get("should_retrain", False),
        "candidate_accuracy": comp_res.get("candidate_accuracy"),
        "production_accuracy": comp_res.get("production_accuracy"),
        "promote_decision": comp_res.get("action", "skipped"),
        "promoted_version": prom_meta.get("model_version"),
        "api_reload_status": reload_res.get("status"),
    }
    logger.info("=== RETRAINING PIPELINE EXECUTION SUMMARY ===")
    logger.info(summary)
    return summary


# ── DAG Definition ──────────────────────────────────────────────────────────

with DAG(
    dag_id="retrain_symptom_model",
    default_args=default_args,
    description="Automated symptom model retraining on drift or validated feedback.",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["retraining", "mlops", "symptom-disease"],
) as dag:

    check_drift_step = PythonOperator(
        task_id="check_drift",
        python_callable=task_check_drift,
    )

    branch_retrain_step = BranchPythonOperator(
        task_id="branch_on_retrain_needed",
        python_callable=task_branch_on_retrain_needed,
    )

    build_dataset_step = PythonOperator(
        task_id="build_retraining_dataset",
        python_callable=task_build_dataset,
    )

    retrain_step = PythonOperator(
        task_id="retrain",
        python_callable=task_retrain,
    )

    compare_step = PythonOperator(
        task_id="compare_with_production",
        python_callable=task_compare,
    )

    branch_promo_step = BranchPythonOperator(
        task_id="branch_on_promotion",
        python_callable=task_branch_on_promotion,
    )

    promote_step = PythonOperator(
        task_id="promote",
        python_callable=task_promote,
    )

    skip_promo_step = EmptyOperator(
        task_id="skip_promotion",
    )

    skip_retrain_step = EmptyOperator(
        task_id="skip_retraining",
    )

    reload_api_step = PythonOperator(
        task_id="reload_api",
        python_callable=task_reload_api,
    )

    log_summary_step = PythonOperator(
        task_id="log_summary",
        python_callable=task_log_summary,
        trigger_rule="none_failed_min_one_success",
    )

    # Define Graph Dependencies
    check_drift_step >> branch_retrain_step
    branch_retrain_step >> [build_dataset_step, skip_retrain_step]
    build_dataset_step >> retrain_step >> compare_step >> branch_promo_step
    branch_promo_step >> [promote_step, skip_promo_step]
    promote_step >> reload_api_step >> log_summary_step
    skip_promo_step >> log_summary_step
    skip_retrain_step >> log_summary_step
