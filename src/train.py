"""
src/train.py
============
Model training and MLflow experiment tracking pipeline.

Trains:
  1. Logistic Regression
  2. Random Forest
  3. XGBoost

Evaluates each on Train, Validation, and Hold-out Test sets:
  - Accuracy
  - Macro F1
  - Top-3 Accuracy
  - Train vs Validation Gap

Tracks in MLflow:
  - Backend: sqlite:///mlflow.db
  - Artifacts: ./mlartifacts
  - Experiment: 'symptom-disease'
  - Params, metrics, model signatures, input examples, file hashes, git commit,
    symptoms.json and diseases.json.
"""

import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import json
import hashlib
import subprocess
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.sklearn import SERIALIZATION_FORMAT_CLOUDPICKLE

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


class XGBoostDiseaseClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper around XGBClassifier for string disease labels.
    Encodes string targets for multi:softprob, exposes standard classes_ and predict_proba.
    """
    def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="mlogloss", random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.encoder_ = LabelEncoder()
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        y_enc = self.encoder_.fit_transform(y)
        self.classes_ = self.encoder_.classes_
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
        )
        self.model_.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc = self.model_.predict(X)
        return self.encoder_.inverse_transform(y_enc)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def compute_sha256(filepath: str) -> str:
    """Return the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Return the current git commit HEAD hexsha."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit
    except Exception:
        return "unknown"


def compute_top3_accuracy(model, X: pd.DataFrame, y: pd.Series) -> float:
    """Compute Top-3 Accuracy: fraction of samples where true disease is in top-3 probabilities."""
    probs = model.predict_proba(X)
    classes = np.array(model.classes_)
    # argsort along axis 1 gives ascending index order; take the last 3 for top-3
    top3_indices = np.argsort(probs, axis=1)[:, -3:]
    hits = 0
    y_arr = y.to_numpy()
    for i, true_label in enumerate(y_arr):
        top3_diseases = classes[top3_indices[i]]
        if true_label in top3_diseases:
            hits += 1
    return hits / len(y_arr)


def evaluate_split(model, X: pd.DataFrame, y: pd.Series, prefix: str) -> dict:
    """Evaluate accuracy, macro F1, and top-3 accuracy on a given dataset split."""
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro", zero_division=0)
    top3_acc = compute_top3_accuracy(model, X, y)
    return {
        f"{prefix}_accuracy": float(acc),
        f"{prefix}_macro_f1": float(macro_f1),
        f"{prefix}_top3_accuracy": float(top3_acc),
    }


def load_config(config_path: str = "config/model_config.yaml") -> dict:
    """Load YAML model and experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_training(config_path: str = "config/model_config.yaml"):
    """Execute training for Logistic Regression, Random Forest, and XGBoost with MLflow logging."""
    cfg = load_config(config_path)

    # 1. Setup MLflow
    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    exp_name = cfg["experiment"]["name"]
    artifact_loc = str(Path(cfg["experiment"]["artifact_location"]).resolve().as_uri())

    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_name, artifact_location=artifact_loc)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)

    # 2. Load data
    train_df = pd.read_csv(cfg["data"]["train_path"])
    val_df = pd.read_csv(cfg["data"]["val_path"])
    test_df = pd.read_csv(cfg["data"]["test_path"])

    target_col = cfg["data"]["target_col"]
    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_val, y_val = val_df.drop(columns=[target_col]), val_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    # 3. File hashes and git metadata
    file_hashes = {
        "hash_train_data": compute_sha256(cfg["data"]["train_path"]),
        "hash_val_data": compute_sha256(cfg["data"]["val_path"]),
        "hash_test_data": compute_sha256(cfg["data"]["test_path"]),
        "hash_symptoms": compute_sha256(cfg["data"]["symptoms_path"]),
        "hash_diseases": compute_sha256(cfg["data"]["diseases_path"]),
    }
    git_commit = get_git_commit()

    # 4. Instantiate model candidate dict
    seed = cfg["training"]["random_seed"]
    lr_params = cfg["models"]["logistic_regression"]
    rf_params = cfg["models"]["random_forest"]
    xgb_params = cfg["models"]["xgboost"]

    candidates = {
        "logistic_regression": {
            "model": LogisticRegression(
                solver=lr_params.get("solver", "lbfgs"),
                max_iter=lr_params.get("max_iter", 1000),
                C=lr_params.get("C", 1.0),
                random_state=seed,
            ),
            "params": {**lr_params, "random_state": seed},
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=rf_params.get("n_estimators", 100),
                max_depth=rf_params.get("max_depth", None),
                random_state=seed,
            ),
            "params": {**rf_params, "random_state": seed},
        },
        "xgboost": {
            "model": XGBoostDiseaseClassifier(
                n_estimators=xgb_params.get("n_estimators", 100),
                max_depth=xgb_params.get("max_depth", 4),
                learning_rate=xgb_params.get("learning_rate", 0.1),
                eval_metric=xgb_params.get("eval_metric", "mlogloss"),
                random_state=seed,
            ),
            "params": {**xgb_params, "random_state": seed},
        },
    }

    results = []

    print(f"\n{'='*80}")
    print(f"Starting MLflow Experiment: '{exp_name}' (ID: {exp_id})")
    print(f"Tracking URI: {cfg['experiment']['tracking_uri']}")
    print(f"Git Commit: {git_commit}")
    print(f"{'='*80}\n")

    for model_name, info in candidates.items():
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            print(f"---> Training: {model_name} (Run ID: {run_id})")

            model = info["model"]
            params = info["params"]

            # Log tags
            mlflow.set_tags({
                "model_type": model_name,
                "git_commit": git_commit,
                **file_hashes,
            })

            # Log parameters
            mlflow.log_params(params)

            # Fit model
            model.fit(X_train, y_train)

            # Evaluate on train, val, and test
            m_train = evaluate_split(model, X_train, y_train, prefix="train")
            m_val = evaluate_split(model, X_val, y_val, prefix="val")
            m_test = evaluate_split(model, X_test, y_test, prefix="test")

            # Compute train-val gaps
            gap_acc = m_train["train_accuracy"] - m_val["val_accuracy"]
            gap_top3 = m_train["train_top3_accuracy"] - m_val["val_top3_accuracy"]
            gap_f1 = m_train["train_macro_f1"] - m_val["val_macro_f1"]

            metrics = {
                **m_train,
                **m_val,
                **m_test,
                "train_val_gap_accuracy": float(gap_acc),
                "train_val_gap_top3": float(gap_top3),
                "train_val_gap_macro_f1": float(gap_f1),
            }

            mlflow.log_metrics(metrics)

            # Signature and input example
            input_example = X_train.iloc[:2]
            signature = infer_signature(X_train, model.predict(X_train))

            # Log model artifact
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=input_example,
                serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            # Log metadata artifacts
            mlflow.log_artifact(cfg["data"]["symptoms_path"], artifact_path="metadata")
            mlflow.log_artifact(cfg["data"]["diseases_path"], artifact_path="metadata")
            if os.path.exists("data/disease_specialist.csv"):
                mlflow.log_artifact("data/disease_specialist.csv", artifact_path="metadata")

            record = {
                "model_name": model_name,
                "run_id": run_id,
                **metrics,
            }
            results.append(record)

    results_df = pd.DataFrame(results)
    print("\n" + "="*95)
    print("TRAINING RUN RESULTS SUMMARY")
    print("="*95)
    print(results_df[[
        "model_name", "val_top3_accuracy", "val_macro_f1", "val_accuracy",
        "test_top3_accuracy", "test_macro_f1", "test_accuracy", "train_val_gap_accuracy"
    ]].to_string(index=False))
    print("="*95 + "\n")

    return results_df


if __name__ == "__main__":
    run_training()
