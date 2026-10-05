# Phase 2: Data Preprocessing, Specialist Mapping, and Data Tests

## 1. Overview and Objectives
Phase 2 transforms the raw Kaggle "Disease Prediction Using Machine Learning" dataset into clean, reproducible, and tested artifacts ready for ML modeling.

Objectives completed:
1. **Sanitize columns**: Drop empty/unnamed columns, deduplicate symptom columns, and normalize symptom names to clean `snake_case`.
2. **Canonicalize disease labels**: Fix typos, inconsistent capitalization, and extraneous whitespace using an explicit mapping table.
3. **De-duplicate training data**: Remove exact row duplicates from training data to prevent train-validation data leakage while keeping `Testing.csv` as an untouched holdout set.
4. **Stratified Split**: Partition deduplicated training data into train (80%) and validation (20%) sets with a fixed random seed.
5. **Specialist Mapping**: Construct a comprehensive medical specialty mapping (`disease_specialist.csv`) for all 41 diseases across 15 distinct medical specialties.
6. **Automated Quality Tests**: Implement 9 automated tests via `pytest` to guarantee data integrity, schema consistency, and zero leakage.

---

## 2. Preprocessing Decisions and Pipeline (`src/preprocess.py`)

### 2.1 Column Cleaning
- **`Unnamed: 133`**: Dropped. This column in `Training.csv` was a CSV trailing comma artifact containing 100% NaN values.
- **`fluid_overload` resolution**: In `Training.csv`, pandas read the duplicate header as `fluid_overload` and `fluid_overload.1`. The Phase 1 audit revealed that `fluid_overload` was all zeros, while `fluid_overload.1` contained the active feature values. The pipeline drops the redundant zero-column and standardizes the true feature to `fluid_overload`.
- **Symptom Name Normalization**: Handled by `normalise_symptom_name()`:
  - Strips leading and trailing whitespace.
  - Converts names to lowercase.
  - Replaces spaces with underscores (e.g., `'spotting_ urination'` -> `'spotting_urination'`, `'foul_smell_of urine'` -> `'foul_smell_of_urine'`, `'dischromic _patches'` -> `'dischromic_patches'`).
  - Collapses consecutive underscores (`'__'` -> `'_'`).
  - Total symptom features after cleaning: **131 binary features**.

### 2.2 Disease Label Normalization
Raw disease labels contained misspellings and formatting defects. The pipeline standardizes them via `DISEASE_NAME_MAP`:
- `"(vertigo) Paroymsal  Positional Vertigo"` -> `"Paroxysmal Positional Vertigo"`
- `"Diabetes "` -> `"Diabetes"`
- `"Hypertension "` -> `"Hypertension"`
- `"hepatitis A"` -> `"Hepatitis A"`
- `"Osteoarthristis"` -> `"Osteoarthritis"`
- `"Dimorphic hemmorhoids(piles)"` -> `"Dimorphic Hemorrhoids (Piles)"`
- `"Peptic ulcer diseae"` -> `"Peptic Ulcer Disease"`

Exactly **41 clean diseases** are preserved.

### 2.3 Exact Duplicate Row Removal & Train/Val Leakage Prevention
- **Raw Training Rows**: 4,920
- **Unique Rows**: 304
- **Duplicates Removed**: 4,616
- **Rationale**: The Kaggle training dataset was synthetically expanded by repeating identical symptom patterns across rows. If random train/test splitting is performed on raw data, identical feature-label rows end up in both the training and validation sets. This produces severe **data leakage** and artificially inflates validation accuracy to ~100%, masking true generalization error. Deduplicating prior to splitting ensures the validation set tests model generalization on unseen feature combinations.
- **Hold-out Test Set (`Testing.csv`)**: Cleaned (columns and labels standardized) but preserved as an untouched external benchmark (42 rows).

---

## 3. Dataset Splits & Summary

| Split / File | Rows | Features (Symptoms) | Target (`prognosis`) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `data/processed/train.csv` | 243 | 131 | 1 | 80% stratified split of unique training rows |
| `data/processed/val.csv` | 61 | 131 | 1 | 20% stratified split of unique training rows |
| `data/processed/test.csv` | 42 | 131 | 1 | Holdout evaluation benchmark from `Testing.csv` |
| `data/processed/symptoms.json` | 131 | - | - | Ordered feature list for inference/API |
| `data/processed/diseases.json` | 41 | - | - | Sorted list of all 41 target disease classes |
| `data/disease_specialist.csv` | 41 | - | - | Mapping of 41 diseases to 15 specialist categories |

---

## 4. Specialist Lookup Table (`data/disease_specialist.csv`)

Maps each of the 41 canonical diseases to a medical specialist and triage category across 15 distinct specialties:
- **Allergist/Immunologist**: AIDS, Allergy
- **Cardiologist**: Heart attack, Hypertension
- **Dermatologist**: Acne, Chicken pox, Fungal infection, Psoriasis, Impetigo
- **Endocrinologist**: Diabetes, Hypoglycemia, Hypothyroidism, Hyperthyroidism
- **ENT Specialist**: (Vertigo) Paroxysmal Positional Vertigo
- **Gastroenterologist**: Chronic cholestasis, GERD, Peptic Ulcer Disease, Jaundice
- **General Physician**: Common Cold, Dengue, Malaria, Typhoid, Dimorphic Hemorrhoids (Piles), Alcohol-related Hepatitis
- **Gynecologist**: Urinary tract infection
- **Hepatologist**: Hepatitis A, Hepatitis B, Hepatitis C, Hepatitis D, Hepatitis E
- **Infectious Disease Specialist**: Tuberculosis
- **Neurologist**: Migraine, Paralysis (brain hemorrhage)
- **Orthopedist**: Arthritis, Cervical spondylosis, Osteoarthritis
- **Physiotherapist**: Bronchial Asthma
- **Pulmonologist**: Bronchial Asthma, Pneumonia
- **Rheumatologist**: Varicose veins

---

## 5. Automated Data Quality Tests (`tests/test_data.py`)

Run via `pytest -v tests/test_data.py`. All 9 tests pass:
1. `test_every_disease_has_one_specialist`: Every disease appears exactly once in the specialist table.
2. `test_specialist_mapping_no_unknown_diseases`: No extraneous or mismatched disease labels in the specialist table.
3. `test_all_diseases_have_specialist`: Every disease present in training, validation, and test datasets has an associated specialist.
4. `test_column_consistency`: Feature and target columns match precisely across train, validation, and test splits.
5. `test_symptom_values_are_binary`: All 131 symptom features contain strictly binary values `{0, 1}`.
6. `test_no_missing_values`: Zero NaN or missing values across all processed splits.
7. `test_test_labels_subset_of_train`: All target labels in validation and test are a valid subset of training labels.
8. `test_no_overlap_between_train_and_val`: Exact zero row overlap between train and validation sets (data leakage prevention).
9. `test_symptoms_json_matches_train_columns`: `symptoms.json` matches the exact feature columns in `train.csv` in identical order.

---

## 6. Reproducibility & Idempotence
The preprocessing script (`src/preprocess.py`) was verified for strict reproducibility. Re-running the pipeline generates identical SHA-256 hashes across repeated executions:
- `train.csv`, `val.csv`, `test.csv`, `symptoms.json`, `diseases.json` are byte-for-byte deterministic.
- Raw CSV files in `data/raw/` were verified against their baseline SHA-256 hashes and remain untouched.
