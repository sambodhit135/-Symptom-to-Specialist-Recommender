"""
src/monitoring/drift.py
=======================
Data and Concept Drift Detection using Evidently.

Compares production request logs against training baseline:
  - Feature drift: assesses shift across all 131 binary symptom features.
  - Prediction drift: assesses shift in top-1 predicted disease distribution.
  - Concept drift proxy: assesses clinician/user feedback unhelpful rating rate.
  - Insufficient data guard: refuses to guess when sample size is below threshold.

Outputs:
  - reports/drift/drift_report_latest.html
  - reports/drift/drift_summary.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath("."))

from evidently import Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset

try:
    from src.api.metrics import (
        DRIFT_CONCEPT_DETECTED,
        DRIFT_DETECTED,
        DRIFT_PREDICTION_DETECTED,
        DRIFT_SHARE,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger("drift_monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_CONFIG: Dict[str, Any] = {
    "min_sample_size": 50,
    "feature_drift_p_value": 0.05,
    "drift_share_threshold": 0.20,
    "prediction_drift_p_value": 0.05,
    "feedback_unhelpful_threshold": 0.25,
    "min_feedback_sample_size": 10,
}


def load_drift_config(config_path: str = "config/drift_config.yaml") -> Dict[str, Any]:
    """Load drift thresholds and sample size settings from YAML file."""
    p = Path(config_path)
    if not p.exists():
        logger.warning(f"Config file {config_path} not found. Using default parameters.")
        return DEFAULT_CONFIG.copy()

    try:
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        merged = DEFAULT_CONFIG.copy()
        merged.update(cfg)
        return merged
    except Exception as e:
        logger.warning(f"Failed to parse {config_path}: {e}. Falling back to defaults.")
        return DEFAULT_CONFIG.copy()


def evaluate_feedback_signal(
    feedback_path: Path,
    min_sample_size: int,
    unhelpful_threshold: float,
) -> Dict[str, Any]:
    """Calculate feedback-based concept drift proxy signal."""
    if not feedback_path.exists() or feedback_path.stat().st_size == 0:
        return {
            "status": "insufficient_data",
            "message": "No feedback log file found.",
            "total_feedback": 0,
            "unhelpful_count": 0,
            "unhelpful_share": None,
            "threshold": unhelpful_threshold,
            "concept_drift_warning": False,
        }

    try:
        df_fb = pd.read_csv(feedback_path)
        if df_fb.empty or len(df_fb) < min_sample_size:
            return {
                "status": "insufficient_data",
                "message": f"Sample size {len(df_fb)} below minimum requirement ({min_sample_size}).",
                "total_feedback": len(df_fb),
                "unhelpful_count": int(df_fb["helpful"].isin([False, "False", "false", 0]).sum()) if "helpful" in df_fb.columns else 0,
                "unhelpful_share": None,
                "threshold": unhelpful_threshold,
                "concept_drift_warning": False,
            }

        unhelpful_mask = df_fb["helpful"].isin([False, "False", "false", 0])
        unhelpful_count = int(unhelpful_mask.sum())
        total_count = len(df_fb)
        unhelpful_share = round(unhelpful_count / total_count, 4)
        warning = bool(unhelpful_share > unhelpful_threshold)

        return {
            "status": "evaluated",
            "message": "Feedback concept drift proxy evaluated successfully.",
            "total_feedback": total_count,
            "unhelpful_count": unhelpful_count,
            "unhelpful_share": unhelpful_share,
            "threshold": unhelpful_threshold,
            "concept_drift_warning": warning,
        }
    except Exception as e:
        logger.error(f"Error parsing feedback log: {e}")
        return {
            "status": "error",
            "message": str(e),
            "total_feedback": 0,
            "unhelpful_count": 0,
            "unhelpful_share": None,
            "threshold": unhelpful_threshold,
            "concept_drift_warning": False,
        }


def run_drift_analysis(
    reference_path: str = "data/processed/train.csv",
    current_path: str = "data/prod_logs/requests.csv",
    feedback_path: str = "data/feedback/feedback.csv",
    config_path: str = "config/drift_config.yaml",
    report_dir: str = "reports/drift",
    update_metrics: bool = False,
    override_reference_df: Optional[pd.DataFrame] = None,
    override_current_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Execute Evidently data & prediction drift analysis comparing reference to current data.
    """
    cfg = load_drift_config(config_path)
    min_sample = int(cfg["min_sample_size"])
    feat_pval_thresh = float(cfg["feature_drift_p_value"])
    drift_share_thresh = float(cfg["drift_share_threshold"])
    pred_pval_thresh = float(cfg["prediction_drift_p_value"])
    fb_thresh = float(cfg["feedback_unhelpful_threshold"])
    min_fb_sample = int(cfg.get("min_feedback_sample_size", 10))

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "drift_summary.json"

    # 1. Load Reference Data
    if override_reference_df is not None:
        ref_df = override_reference_df.copy()
    else:
        ref_file = Path(reference_path)
        if not ref_file.exists():
            raise FileNotFoundError(f"Reference training dataset not found at {reference_path}")
        ref_df = pd.read_csv(ref_file)

    # 2. Load Current Production Data
    if override_current_df is not None:
        cur_df = override_current_df.copy()
    else:
        cur_file = Path(current_path)
        if not cur_file.exists() or cur_file.stat().st_size == 0:
            cur_df = pd.DataFrame()
        else:
            cur_df = pd.read_csv(cur_file)

    current_rows = len(cur_df)
    reference_rows = len(ref_df)

    # Concept drift proxy from feedback
    fb_signal = evaluate_feedback_signal(
        Path(feedback_path),
        min_sample_size=min_fb_sample,
        unhelpful_threshold=fb_thresh,
    )

    # 3. Insufficient Data Check
    if current_rows < min_sample:
        msg = f"Insufficient data: found {current_rows} rows, but minimum {min_sample} required."
        logger.info(msg)
        summary = {
            "status": "insufficient_data",
            "message": msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drift_detected": False,
            "share_of_drifted_features": 0.0,
            "top_drifted_features": [],
            "prediction_distribution_drift": {
                "drift_detected": False,
                "p_value": None,
                "threshold": pred_pval_thresh,
            },
            "rows_compared": {
                "reference_rows": reference_rows,
                "current_rows": current_rows,
            },
            "feedback_concept_drift": fb_signal,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    # 4. Prepare feature columns (131 symptoms)
    symptom_cols = [c for c in ref_df.columns if c not in ("prognosis", "top_disease", "request_id", "timestamp", "top_confidence", "emergency_flag")]

    # Ensure current data has all symptom columns (fill missing with 0)
    for col in symptom_cols:
        if col not in cur_df.columns:
            cur_df[col] = 0

    eval_ref = ref_df[symptom_cols].copy()
    eval_cur = cur_df[symptom_cols].copy()

    # Align prediction column if available
    eval_prediction_drift = False
    ref_pred_col = "prognosis" if "prognosis" in ref_df.columns else ("top_disease" if "top_disease" in ref_df.columns else None)
    cur_pred_col = "top_disease" if "top_disease" in cur_df.columns else ("prognosis" if "prognosis" in cur_df.columns else None)

    if ref_pred_col and cur_pred_col and not cur_df[cur_pred_col].dropna().empty:
        eval_ref["top_disease"] = ref_df[ref_pred_col].astype(str)
        eval_cur["top_disease"] = cur_df[cur_pred_col].astype(str)
        eval_prediction_drift = True

    # 5. Build Evidently Report
    metrics: List[Any] = [
        DataDriftPreset(
            columns=symptom_cols,
            drift_share=drift_share_thresh,
            cat_threshold=feat_pval_thresh,
        )
    ]
    if eval_prediction_drift:
        metrics.append(ValueDrift(column="top_disease", threshold=pred_pval_thresh))

    report = Report(metrics=metrics)
    snapshot = report.run(reference_data=eval_ref, current_data=eval_cur)
    snap_dict = snapshot.dict()

    # 6. Extract Feature Drift Metrics
    metrics_data = snap_dict.get("metrics", [])
    drifted_count = 0.0
    drift_share = 0.0
    top_drifted: List[Dict[str, Any]] = []

    # First metric is DriftedColumnsCount
    if metrics_data:
        val = metrics_data[0].get("value", {})
        if isinstance(val, dict):
            drifted_count = float(val.get("count", 0.0))
            drift_share = float(val.get("share", 0.0))

    # Inspect per-column ValueDrift metrics
    for m in metrics_data[1:]:
        cfg_m = m.get("config", {})
        col_name = cfg_m.get("column")
        if col_name in symptom_cols:
            raw_val = m.get("value")
            try:
                pval = float(raw_val)
            except (TypeError, ValueError):
                continue
            thresh = float(cfg_m.get("threshold", feat_pval_thresh))
            if pval < thresh:
                top_drifted.append({
                    "feature": col_name,
                    "p_value": round(pval, 6),
                    "threshold": thresh,
                })

    top_drifted.sort(key=lambda x: x["p_value"])

    # 7. Extract Prediction Distribution Drift
    pred_drift_detected = False
    pred_p_value = None
    if eval_prediction_drift:
        for m in metrics_data:
            cfg_m = m.get("config", {})
            if cfg_m.get("column") == "top_disease":
                raw_val = m.get("value")
                try:
                    pred_p_value = round(float(raw_val), 6)
                    pred_drift_detected = bool(pred_p_value < pred_pval_thresh)
                except (TypeError, ValueError):
                    pass
                break

    # 8. Overall Drift Decision
    overall_drift_detected = bool(drift_share > drift_share_thresh)

    # 9. Save Evidently HTML Report
    html_path = out_dir / "drift_report_latest.html"
    snapshot.save_html(str(html_path))

    # 10. Assemble JSON Summary
    summary = {
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_detected": overall_drift_detected,
        "share_of_drifted_features": round(drift_share, 4),
        "drift_share_threshold": drift_share_thresh,
        "drifted_features_count": int(drifted_count),
        "total_features": len(symptom_cols),
        "top_drifted_features": top_drifted[:15],
        "prediction_distribution_drift": {
            "drift_detected": pred_drift_detected,
            "p_value": pred_p_value,
            "threshold": pred_pval_thresh,
        },
        "rows_compared": {
            "reference_rows": reference_rows,
            "current_rows": current_rows,
        },
        "feedback_concept_drift": fb_signal,
        "html_report_path": str(html_path.resolve()),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Drift Analysis Complete: drift_detected={overall_drift_detected}, "
        f"share={drift_share:.2%} ({drifted_count}/{len(symptom_cols)} features), "
        f"pred_drift={pred_drift_detected}"
    )

    # 11. Optionally update Prometheus gauges
    if update_metrics and METRICS_AVAILABLE:
        try:
            DRIFT_DETECTED.set(1 if overall_drift_detected else 0)
            DRIFT_SHARE.set(drift_share)
            DRIFT_PREDICTION_DETECTED.set(1 if pred_drift_detected else 0)
            if fb_signal.get("concept_drift_warning") is not None:
                DRIFT_CONCEPT_DETECTED.set(1 if fb_signal["concept_drift_warning"] else 0)
        except Exception as e:
            logger.warning(f"Could not update Prometheus drift gauges: {e}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Data and Concept Drift using Evidently.")
    parser.add_argument("--reference", default="data/processed/train.csv", help="Path to reference CSV")
    parser.add_argument("--current", default="data/prod_logs/requests.csv", help="Path to production requests CSV")
    parser.add_argument("--feedback", default="data/feedback/feedback.csv", help="Path to feedback CSV")
    parser.add_argument("--config", default="config/drift_config.yaml", help="Path to drift configuration YAML")
    parser.add_argument("--output-dir", default="reports/drift", help="Output directory for reports")
    parser.add_argument("--update-metrics", action="store_true", help="Update Prometheus metrics")
    args = parser.parse_args()

    summary = run_drift_analysis(
        reference_path=args.reference,
        current_path=args.current,
        feedback_path=args.feedback,
        config_path=args.config,
        report_dir=args.output_dir,
        update_metrics=args.update_metrics,
    )
    print("\n" + "=" * 60)
    print("EVIDENTLY DRIFT ANALYSIS SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
