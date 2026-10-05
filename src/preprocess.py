"""
src/preprocess.py
=================
Preprocessing pipeline for the Disease Prediction dataset.

What it does (in order):
  1. Load data/raw/Training.csv and Testing.csv  (never modified).
  2. Drop the phantom 'Unnamed: 133' column (CSV artefact in Training only).
  3. Resolve the duplicated fluid_overload column:
       - 'fluid_overload'   -> always 0 in Training -> drop it.
       - 'fluid_overload.1' -> the real data        -> rename to 'fluid_overload'.
  4. Normalise symptom column names: strip spaces, lowercase, replace inner
     spaces with underscores, collapse duplicate underscores.
  5. Strip whitespace from the prognosis column.
  6. Apply DISEASE_NAME_MAP to fix typos and inconsistent capitalisation.
  7. Drop exact-duplicate rows from the training data (documented below).
  8. Stratified 80/20 train/validation split (fixed seed=42).
  9. Write:
       data/processed/train.csv
       data/processed/val.csv
       data/processed/test.csv   (Testing.csv, cleaned but never split)
       data/processed/symptoms.json  (ordered feature list)
       data/processed/diseases.json  (sorted list of clean disease names)

Run as a script:
    python src/preprocess.py

Import as a module:
    from src.preprocess import run_preprocessing
    paths = run_preprocessing()
"""

import os
import json
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
RANDOM_SEED   = 42        # fixed seed for reproducibility
VAL_SIZE      = 0.20      # 20 % of deduplicated training rows go to validation
LABEL_COL     = "prognosis"

# -- Disease name corrections --------------------------------------
# KEY   = exact string as it appears in the raw CSVs (both files).
# VALUE = corrected, canonical name used everywhere downstream.
# Source: Phase 1 audit (docs/data_audit.md).
DISEASE_NAME_MAP = {
    # Typo: "Paroymsal" should be "Paroxysmal"; also has a double space
    "(vertigo) Paroymsal  Positional Vertigo": "Paroxysmal Positional Vertigo",
    # Trailing whitespace
    "Diabetes ":      "Diabetes",
    "Hypertension ":  "Hypertension",
    # Wrong capitalisation (lowercase 'h')
    "hepatitis A":    "Hepatitis A",
    # Misspelling: "Arthristis" -> "Arthritis"
    "Osteoarthristis": "Osteoarthritis",
    # Misspelling: "hemmorhoids" -> "Hemorrhoids"; add space before "(piles)"
    "Dimorphic hemmorhoids(piles)": "Dimorphic Hemorrhoids (Piles)",
    # Misspelling: "diseae" -> "Disease"; consistent title case
    "Peptic ulcer diseae": "Peptic Ulcer Disease",
}

# -- Symptom column name corrections (documented) ------------------
# Known cases found in audit:
#   'spotting_ urination' -> 'spotting_urination'
#   'foul_smell_of urine' -> 'foul_smell_of_urine'
#   'dischromic _patches' -> 'dischromic_patches'
# These are handled by the general normalise_symptom_name() function below.


def normalise_symptom_name(name: str) -> str:
    """
    Convert a raw symptom column name to clean lowercase snake_case.
    Steps:
      - strip leading/trailing whitespace
      - lowercase
      - replace spaces with underscores
      - collapse multiple consecutive underscores to one
    """
    name = name.strip().lower()
    name = name.replace(" ", "_")
    # collapse runs like '__' -> '_'
    while "__" in name:
        name = name.replace("__", "_")
    return name


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def file_sha256(path: str) -> str:
    """Return hex SHA-256 of a file (used to verify reproducibility)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def clean_columns(df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
    """
    Remove artefact columns and normalise symptom names.

    For Training.csv only:
      - Drop 'Unnamed: 133'          (all-NaN CSV artefact)
      - Drop 'fluid_overload'        (always 0; phantom duplicate)
      - Rename 'fluid_overload.1' -> 'fluid_overload'  (the real data column)
    Then normalise ALL non-label column names to snake_case.
    """
    df = df.copy()

    if is_training:
        # Drop the phantom NaN column created by a trailing comma in the CSV
        if "Unnamed: 133" in df.columns:
            df = df.drop(columns=["Unnamed: 133"])
            print("  [drop] Unnamed: 133  (all-NaN artefact)")

        # 'fluid_overload' is always 0 in training (verified in audit).
        # pandas auto-renamed the real second occurrence to 'fluid_overload.1'.
        # Correct action: drop the zero-only column, rename the real one back.
        if "fluid_overload" in df.columns and "fluid_overload.1" in df.columns:
            assert df["fluid_overload"].sum() == 0, \
                "fluid_overload unexpectedly has non-zero values"
            df = df.drop(columns=["fluid_overload"])
            df = df.rename(columns={"fluid_overload.1": "fluid_overload"})
            print("  [fix]  fluid_overload (dropped zero column; renamed .1 back)")

    # Normalise all symptom column names (skip the label column)
    rename_map = {
        col: normalise_symptom_name(col)
        for col in df.columns
        if col != LABEL_COL and normalise_symptom_name(col) != col
    }
    if rename_map:
        df = df.rename(columns=rename_map)
        for old, new in rename_map.items():
            print(f"  [rename] '{old}' -> '{new}'")

    return df


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from prognosis values and apply DISEASE_NAME_MAP.
    Both steps operate on the raw string values and are idempotent.
    """
    df = df.copy()
    # Step 1: strip (catches 'Diabetes ' and 'Hypertension ')
    df[LABEL_COL] = df[LABEL_COL].str.strip()
    # Step 2: apply explicit correction map
    df[LABEL_COL] = df[LABEL_COL].replace(DISEASE_NAME_MAP)
    return df


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------

def run_preprocessing() -> dict:
    """
    Execute the full preprocessing pipeline.

    Returns a dict with file paths and key statistics so callers
    (e.g. notebooks or the API) can inspect the results.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # -- 1. Load raw data ------------------------------------------
    train_raw_path = os.path.join(RAW_DIR, "Training.csv")
    test_raw_path  = os.path.join(RAW_DIR, "Testing.csv")

    print(f"Loading {train_raw_path} ...")
    train_raw = pd.read_csv(train_raw_path)
    print(f"  raw shape: {train_raw.shape}")

    print(f"Loading {test_raw_path} ...")
    test_raw  = pd.read_csv(test_raw_path)
    print(f"  raw shape: {test_raw.shape}")

    # -- 2. Clean columns ------------------------------------------
    print("\nCleaning Training columns ...")
    train = clean_columns(train_raw, is_training=True)

    print("Cleaning Testing columns ...")
    test  = clean_columns(test_raw, is_training=False)

    # -- 3. Clean labels -------------------------------------------
    print("\nCleaning disease labels ...")
    train = clean_labels(train)
    test  = clean_labels(test)

    # Verify: all disease names in test are also in train
    train_diseases = set(train[LABEL_COL].unique())
    test_diseases  = set(test[LABEL_COL].unique())
    only_test = test_diseases - train_diseases
    if only_test:
        raise ValueError(f"Test set has unknown diseases not in train: {only_test}")

    # -- 4. Align columns ------------------------------------------
    # Use the train symptom column order as the canonical order.
    symptom_cols = [c for c in train.columns if c != LABEL_COL]

    test_symptom_cols = [c for c in test.columns if c != LABEL_COL]
    missing_in_test   = set(symptom_cols) - set(test_symptom_cols)
    extra_in_test     = set(test_symptom_cols) - set(symptom_cols)

    if missing_in_test:
        print(f"  [warn] Columns in train not in test (filling with 0): {missing_in_test}")
        for col in missing_in_test:
            test[col] = 0

    if extra_in_test:
        print(f"  [warn] Extra columns in test (dropping): {extra_in_test}")
        test = test.drop(columns=list(extra_in_test))

    # Reorder test columns to match train exactly
    test = test[symptom_cols + [LABEL_COL]]

    # -- 5. Remove exact duplicates from training data -------------
    # WHY: Training.csv has 4,616 duplicate rows (94 % of 4,920 rows).
    # The dataset is synthetic; each disease class has 120 rows built from
    # a small number of unique symptom combos repeated many times.
    # Keeping duplicates would cause data leakage: identical rows would appear
    # in both the training and validation splits, inflating validation accuracy.
    # Decision: deduplicate BEFORE splitting.
    n_before = len(train)
    train = train.drop_duplicates()
    n_after  = len(train)
    n_removed = n_before - n_after
    print(f"\nDuplicate rows removed from training: {n_removed} "
          f"({n_before} -> {n_after})")

    # -- 6. Stratified train / validation split --------------------
    # stratify=y ensures every disease class is proportionally represented.
    X = train[symptom_cols]
    y = train[LABEL_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[LABEL_COL] = y_train.values

    val_df = X_val.copy()
    val_df[LABEL_COL] = y_val.values

    print(f"\nSplit sizes:")
    print(f"  train : {len(train_df)} rows")
    print(f"  val   : {len(val_df)} rows")
    print(f"  test  : {len(test)} rows (hold-out, never split)")

    # -- 7. Save processed CSV files -------------------------------
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path   = os.path.join(PROCESSED_DIR, "val.csv")
    test_path  = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test.to_csv(test_path,      index=False)

    print(f"\nSaved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")

    # -- 8. Save feature/label metadata ---------------------------
    # symptoms.json: ORDERED list of symptom column names.
    # This exact order MUST be used by the API when building feature vectors.
    symptoms_path = os.path.join(PROCESSED_DIR, "symptoms.json")
    diseases_path = os.path.join(PROCESSED_DIR, "diseases.json")

    with open(symptoms_path, "w", encoding="utf-8") as f:
        json.dump(symptom_cols, f, indent=2)

    all_diseases = sorted(train_df[LABEL_COL].unique().tolist())
    with open(diseases_path, "w", encoding="utf-8") as f:
        json.dump(all_diseases, f, indent=2)

    print(f"Saved: {symptoms_path}  ({len(symptom_cols)} features)")
    print(f"Saved: {diseases_path}  ({len(all_diseases)} diseases)")

    # -- 9. Verify raw file hashes haven't changed -----------------
    expected_hashes = {
        "Training.csv":
            "4EC0D80A829B1588B5B3290D29051017783992BEC8331C93255A41660AA43F58",
        "Testing.csv":
            "F4F8E2EEFAD5401948AA355D082D9F51C7F0546F9F01FD836E5BBD964D7BAD21",
    }
    print("\nRaw file integrity check:")
    for fname, expected_hash in expected_hashes.items():
        actual = file_sha256(os.path.join(RAW_DIR, fname))
        status = "OK" if actual == expected_hash else "MISMATCH!"
        print(f"  {fname}: {status}")
        if status != "OK":
            raise RuntimeError(
                f"Raw file {fname} has been modified! "
                f"Expected {expected_hash}, got {actual}"
            )

    return {
        "train_path":    train_path,
        "val_path":      val_path,
        "test_path":     test_path,
        "symptoms_path": symptoms_path,
        "diseases_path": diseases_path,
        "n_train":       len(train_df),
        "n_val":         len(val_df),
        "n_test":        len(test),
        "n_removed":     n_removed,
        "n_symptoms":    len(symptom_cols),
        "n_diseases":    len(all_diseases),
    }


# ------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Disease Prediction - Preprocessing Pipeline")
    print("=" * 60)
    result = run_preprocessing()
    print("\n[DONE] Summary:")
    for k, v in result.items():
        print(f"  {k}: {v}")
