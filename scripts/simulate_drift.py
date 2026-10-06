"""
scripts/simulate_drift.py
=========================
Generates synthetic traffic for drift evaluation:
  (a) Normal scenario: symptoms sampled directly from the training data distribution.
  (b) Shifted scenario: heavily perturbed distribution with certain symptoms strongly over-represented.

Traffic can either be written directly to `data/prod_logs/requests.csv` (instant offline simulation)
or sent via HTTP requests to a running API instance (`POST /predict`).

Usage:
    python scripts/simulate_drift.py --mode normal --n 100 --dest log
    python scripts/simulate_drift.py --mode shifted --n 100 --dest log
    python scripts/simulate_drift.py --mode shifted --n 100 --dest api --api-url http://localhost:8000
"""

import argparse
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.api.model_loader import ModelManager

# Prominent shifted symptoms cluster for the shifted scenario
SHIFTED_CLUSTER = [
    "foul_smell_of_urine",
    "continuous_feel_of_urine",
    "bladder_discomfort",
    "burning_micturition",
    "internal_itching",
    "spotting_urination",
    "toxic_look_(typhos)",
    "altered_sensorium",
]


def generate_normal_sample(train_df: pd.DataFrame, symptom_cols: List[str]) -> List[str]:
    """Sample symptoms following real empirical training cases."""
    row = train_df.sample(n=1).iloc[0]
    active = [col for col in symptom_cols if row[col] == 1]
    # Optionally drop 1 symptom or add 1 noise symptom
    if len(active) > 2 and random.random() < 0.2:
        active.pop(random.randint(0, len(active) - 1))
    return active if active else ["itching", "fatigue"]


def generate_shifted_sample(symptom_cols: List[str]) -> List[str]:
    """Generate sample with heavily shifted symptom distributions."""
    # 85% of cases contain several shifted cluster symptoms
    k = random.randint(3, len(SHIFTED_CLUSTER))
    active = random.sample(SHIFTED_CLUSTER, k)

    # Add 1-2 random other symptoms
    noise = random.sample(symptom_cols, random.randint(1, 2))
    result = list(dict.fromkeys(active + noise))
    return result


def main():
    parser = argparse.ArgumentParser(description="Simulate normal or shifted traffic for drift detection.")
    parser.add_argument("--mode", choices=["normal", "shifted"], default="normal", help="Simulation mode")
    parser.add_argument("--n", type=int, default=100, help="Number of records to generate (default: 100)")
    parser.add_argument("--dest", choices=["log", "api"], default="log", help="Output destination (log file or live API)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL (if dest=api)")
    parser.add_argument("--output-file", default="data/prod_logs/requests.csv", help="Log destination CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--append", action="store_true", help="Append to log file rather than overwrite")
    args = parser.parse_args()

    random.seed(args.seed)

    train_path = Path("data/processed/train.csv")
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_df = pd.read_csv(train_path)
    symptom_cols = [c for c in train_df.columns if c != "prognosis"]

    print(f"--- Simulating Drift: Mode={args.mode.upper()}, Count={args.n}, Destination={args.dest} ---")

    if args.dest == "api":
        session = requests.Session()
        success = 0
        failed = 0
        for i in range(1, args.n + 1):
            if args.mode == "normal":
                symptoms = generate_normal_sample(train_df, symptom_cols)
            else:
                symptoms = generate_shifted_sample(symptom_cols)

            try:
                resp = session.post(f"{args.api_url}/predict", json={"symptoms": symptoms}, timeout=5)
                if resp.status_code == 200:
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                if i == 1:
                    print(f"Connection failed: {e}")

            if i % 10 == 0 or i == args.n:
                print(f"  Sent {i}/{args.n} requests (success={success}, failed={failed})")

        print(f"Completed API simulation: {success} succeeded, {failed} failed.")

    else:
        # Destination: direct to log file
        log_file = Path(args.output_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        model_mgr = ModelManager()
        rows = []

        for _ in range(args.n):
            if args.mode == "normal":
                symptoms = generate_normal_sample(train_df, symptom_cols)
            else:
                symptoms = generate_shifted_sample(symptom_cols)

            preds = model_mgr.predict_top3(symptoms)
            top_disease = preds[0]["disease"] if preds else "Unknown"
            top_conf = preds[0]["confidence"] if preds else 0.0

            # Emergency flag evaluation
            emergency = "chest_pain" in symptoms or "fast_heart_rate" in symptoms

            # Binary vector
            symptom_set = set(symptoms)
            vector = [1 if s in symptom_set else 0 for s in symptom_cols]

            row_dict = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": uuid.uuid4().hex,
                "top_disease": top_disease,
                "top_confidence": top_conf,
                "emergency_flag": emergency,
            }
            for col, val in zip(symptom_cols, vector):
                row_dict[col] = val

            rows.append(row_dict)

        df_out = pd.DataFrame(rows)

        if args.append and log_file.exists() and log_file.stat().st_size > 0:
            df_out.to_csv(log_file, mode="a", header=False, index=False)
            print(f"Appended {args.n} records to {log_file} (total rows: {len(pd.read_csv(log_file))}).")
        else:
            df_out.to_csv(log_file, index=False)
            print(f"Wrote {args.n} records to {log_file}.")


if __name__ == "__main__":
    main()
