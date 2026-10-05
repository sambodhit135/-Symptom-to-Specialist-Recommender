"""
src/register_best.py
====================
Selects the best trained model using validation metrics, registers it in the
MLflow Model Registry, assigns model aliases ('staging' and 'production'),
and exports a standalone copy to models/production/.

Selection criteria:
  1. Primary: validation top-3 accuracy (descending)
  2. Secondary: validation macro F1 (descending)
  (Never evaluated or chosen using the hold-out test set!)

Quality Gate:
  Configured in config/model_config.yaml (min_val_top3_accuracy >= 0.90).
  Only runs passing the quality gate are assigned the 'production' alias.
"""

import os
import sys
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
sys.path.insert(0, os.path.abspath("."))
import shutil
import json
import yaml
import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.sklearn import SERIALIZATION_FORMAT_CLOUDPICKLE

from src.train import XGBoostDiseaseClassifier


def load_config(config_path: str = "config/model_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def register_best_model(config_path: str = "config/model_config.yaml"):
    cfg = load_config(config_path)

    tracking_uri = cfg["experiment"]["tracking_uri"]
    exp_name = cfg["experiment"]["name"]
    model_name = cfg["experiment"]["registered_model_name"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{exp_name}' not found. Run src/train.py first.")

    # Search runs ordered by val_top3_accuracy DESC, then val_macro_f1 DESC
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.val_top3_accuracy DESC", "metrics.val_macro_f1 DESC"],
    )

    if not runs:
        raise RuntimeError(f"No runs found in experiment '{exp_name}'.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_model_name = best_run.data.tags.get("model_type", best_run.data.tags.get("mlflow.runName", "unknown"))

    val_top3 = best_run.data.metrics.get("val_top3_accuracy", 0.0)
    val_f1 = best_run.data.metrics.get("val_macro_f1", 0.0)
    val_acc = best_run.data.metrics.get("val_accuracy", 0.0)
    test_top3 = best_run.data.metrics.get("test_top3_accuracy", 0.0)
    test_f1 = best_run.data.metrics.get("test_macro_f1", 0.0)
    test_acc = best_run.data.metrics.get("test_accuracy", 0.0)

    print("\n" + "="*80)
    print(f"BEST RUN IDENTIFIED (Validation-based selection):")
    print(f"  Model Type:        {best_model_name}")
    print(f"  Run ID:            {best_run_id}")
    print(f"  Val Top-3 Acc:     {val_top3:.4f}")
    print(f"  Val Macro F1:      {val_f1:.4f}")
    print(f"  Val Accuracy:      {val_acc:.4f}")
    print(f"  (Test Top-3 Acc):  {test_top3:.4f} [Holdout, unexamined during selection]")
    print("="*80)

    # Register model in MLflow Model Registry
    model_uri = f"runs:/{best_run_id}/model"
    reg_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = reg_model.version

    print(f"\nRegistered model '{model_name}' version {version}")

    # Set alias 'staging'
    client.set_registered_model_alias(name=model_name, alias="staging", version=version)
    print(f"  -> Assigned alias: 'staging' to version {version}")

    # Quality Gate Check
    gate_cfg = cfg.get("quality_gate", {})
    min_top3 = gate_cfg.get("min_val_top3_accuracy", 0.90)
    min_f1 = gate_cfg.get("min_val_macro_f1", 0.85)

    passes_gate = (val_top3 >= min_top3) and (val_f1 >= min_f1)

    if passes_gate:
        client.set_registered_model_alias(name=model_name, alias="production", version=version)
        print(f"  -> QUALITY GATE PASSED (val_top3={val_top3:.4f} >= {min_top3}, val_f1={val_f1:.4f} >= {min_f1})")
        print(f"  -> Assigned alias: 'production' to version {version}")
    else:
        print(f"  -> QUALITY GATE FAILED! (val_top3={val_top3:.4f} < {min_top3} or val_f1={val_f1:.4f} < {min_f1})")
        print("  -> Production alias NOT assigned.")

    # Export a copy of the production model to models/production/
    prod_dir = Path("models/production")
    if prod_dir.exists():
        shutil.rmtree(prod_dir)
    prod_dir.mkdir(parents=True, exist_ok=True)

    # Load from MLflow run and save standalone
    loaded_model = mlflow.sklearn.load_model(model_uri)
    
    # Save standard MLflow model bundle
    mlflow.sklearn.save_model(
        loaded_model,
        str(prod_dir / "mlflow_model"),
        serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    
    # Also save native joblib binary for ultra-lightweight standalone loading
    joblib.dump(loaded_model, prod_dir / "model.joblib")

    # Copy metadata
    shutil.copy(cfg["data"]["symptoms_path"], prod_dir / "symptoms.json")
    shutil.copy(cfg["data"]["diseases_path"], prod_dir / "diseases.json")
    if os.path.exists("data/disease_specialist.csv"):
        shutil.copy("data/disease_specialist.csv", prod_dir / "disease_specialist.csv")

    metadata = {
        "model_name": model_name,
        "model_version": version,
        "model_type": best_model_name,
        "run_id": best_run_id,
        "aliases": ["staging", "production"] if passes_gate else ["staging"],
        "metrics": {
            "val_top3_accuracy": val_top3,
            "val_macro_f1": val_f1,
            "val_accuracy": val_acc,
            "test_top3_accuracy": test_top3,
            "test_macro_f1": test_f1,
            "test_accuracy": test_acc,
        },
        "quality_gate_passed": passes_gate,
    }
    with open(prod_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nStandalone production model exported to: {prod_dir.resolve()}")
    print(f"  Files created in {prod_dir}:")
    for f in prod_dir.iterdir():
        print(f"    - {f.name}")
    print("="*80 + "\n")

    return version, metadata


if __name__ == "__main__":
    register_best_model()
