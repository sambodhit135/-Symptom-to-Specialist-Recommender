"""
src/api/model_loader.py
=======================
Model loading and lifecycle management.
Loads the production model either from MLflow (if MLFLOW_TRACKING_URI is set)
or falls back gracefully to standalone models/production/.
Loads symptoms.json and disease_specialist.csv at startup.
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path so model wrappers can deserialize
sys.path.insert(0, os.path.abspath("."))
from src.train import XGBoostDiseaseClassifier  # noqa: F401


class ModelManager:
    """Manages model loading, feature schemas, specialist lookup, and top-3 inference."""

    def __init__(self, prod_dir: str = "models/production", specialist_csv: str = "data/disease_specialist.csv"):
        self.prod_dir = Path(prod_dir)
        self.specialist_csv = Path(specialist_csv)

        self.model: Any = None
        self.model_version: str = "unknown"
        self.model_alias: str = "production"
        self.model_type: str = "unknown"
        self.loaded_from: str = "unknown"
        self.load_time_seconds: float = 0.0
        self.quality_gate_passed: bool = False

        self.symptoms: List[str] = []
        self.symptom_set: set = set()
        self.diseases: List[str] = []
        self.disease_set: set = set()
        self.specialist_map: Dict[str, str] = {}

        self.load()

    def load(self):
        """Load schemas, specialist map, and trained model binary."""
        t0 = time.time()

        # 1. Load schemas
        symptoms_path = self.prod_dir / "symptoms.json"
        if not symptoms_path.exists():
            symptoms_path = Path("data/processed/symptoms.json")
        with open(symptoms_path, "r", encoding="utf-8") as f:
            self.symptoms = json.load(f)
        self.symptom_set = set(self.symptoms)

        diseases_path = self.prod_dir / "diseases.json"
        if not diseases_path.exists():
            diseases_path = Path("data/processed/diseases.json")
        with open(diseases_path, "r", encoding="utf-8") as f:
            self.diseases = json.load(f)
        self.disease_set = set(self.diseases)

        # 2. Load specialist table
        spec_path = self.specialist_csv
        if not spec_path.exists():
            spec_path = self.prod_dir / "disease_specialist.csv"
        df_spec = pd.read_csv(spec_path)
        self.specialist_map = dict(zip(df_spec["disease"].str.strip(), df_spec["specialist"].str.strip()))

        # 3. Load metadata if available
        meta_path = self.prod_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.model_version = str(meta.get("model_version", "1"))
                self.model_type = meta.get("model_type", "unknown")
                self.quality_gate_passed = meta.get("quality_gate_passed", True)

        # 4. Attempt MLflow model load if MLFLOW_TRACKING_URI is provided
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        loaded = False
        if tracking_uri:
            try:
                import mlflow
                import mlflow.sklearn
                mlflow.set_tracking_uri(tracking_uri)
                model_uri = "models:/symptom-disease-classifier@production"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.loaded_from = f"mlflow ({tracking_uri})"
                loaded = True
            except Exception as e:
                # Log warning and fall back to local disk
                self.loaded_from = f"fallback (MLflow load failed: {str(e)})"

        # 5. Fallback: load directly from local disk (zero MLflow server required)
        if not loaded:
            joblib_path = self.prod_dir / "model.joblib"
            if joblib_path.exists():
                self.model = joblib.load(joblib_path)
                self.loaded_from = str(joblib_path)
            else:
                import mlflow.sklearn
                self.model = mlflow.sklearn.load_model(str(self.prod_dir / "mlflow_model"))
                self.loaded_from = str(self.prod_dir / "mlflow_model")

        self.load_time_seconds = time.time() - t0

    def predict_top3(self, presented_symptoms: List[str]) -> List[Dict[str, Any]]:
        """
        Convert presented symptom list into full 131-element binary feature vector,
        run inference, and return the top-3 predictions with probabilities and specialists.
        """
        # Create binary feature vector matching self.symptoms order
        presented_set = {s.strip().lower() for s in presented_symptoms}
        feature_vector = [1 if s in presented_set else 0 for s in self.symptoms]

        df_input = pd.DataFrame([feature_vector], columns=self.symptoms)
        probs = self.model.predict_proba(df_input)[0]
        classes = np.array(self.model.classes_)

        # Top-3 indices sorted by confidence descending
        top3_indices = np.argsort(probs)[-3:][::-1]

        predictions = []
        for idx in top3_indices:
            disease = str(classes[idx])
            conf = float(probs[idx])
            spec = self.specialist_map.get(disease, "General Physician")
            predictions.append({
                "disease": disease,
                "confidence": round(conf, 4),
                "specialist": spec,
            })

        return predictions
