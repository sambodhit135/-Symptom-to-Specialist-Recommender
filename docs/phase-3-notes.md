# Phase 3 Notes: Model Training, MLflow Tracking, and Quality Gate Evaluation

## 1. Overview and Objectives
Phase 3 trains three classification algorithms on the preprocessed disease data, tracks all experiments in MLflow with a SQLite database backend and local artifact storage, evaluates models against a rigorous quality gate, and registers the best model into production.

---

## 2. Model Performance Summary

| Model | Val Top-3 Acc | Val Macro F1 | Val Accuracy | Test Top-3 Acc | Test Macro F1 | Test Accuracy | Train-Val Accuracy Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0000** |
| **Random Forest** | 1.0000 | 0.9707 | 0.9836 | 0.9762 | 0.9837 | 0.9762 | 0.0164 |
| **XGBoost** | 0.9672 | 0.8927 | 0.9344 | 1.0000 | 1.0000 | 1.0000 | 0.0656 |

### Selected Best Model
- **Winning Architecture**: `LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)`
- **Selection Metric**: Primary = `val_top3_accuracy` (1.0000), Secondary = `val_macro_f1` (1.0000).
- **Hold-out Test Set**: Held strictly untouched during model selection; evaluated post-selection for final benchmark reporting.

---

## 3. High Accuracy & Synthetic Data Analysis (Quality Audit)

### 3.1 Data Leakage Audit
A comprehensive audit was conducted to verify that high accuracy is not caused by technical errors or data leakage:
1. **Train/Validation Partitioning**: In Phase 2, 4,616 exact duplicate rows were stripped prior to stratified train/validation splitting. Automated test `test_no_overlap_between_train_and_val` proved 0 identical rows between `train.csv` and `val.csv`.
2. **Feature Preprocessing**: All features are binary indicators `{0, 1}`. No data-dependent scalers, target encoders, or imputers were fitted across splits.
3. **Target Integrity**: The target label (`prognosis`) was strictly isolated from input feature matrices.

### 3.2 Root Cause: Synthetic Dataset Characteristics
The near-perfect accuracy (100% on validation and hold-out test sets) is a direct consequence of the dataset's synthetic origin:
1. **Deterministic Symptom Rules**: The Kaggle dataset was artificially generated from standardized clinical archetype templates rather than noisy electronic health records (EHRs).
2. **Disjoint Symptom Clusters**: Many diseases possess non-overlapping symptom combinations (e.g. `itching + skin_rash + nodal_skin_eruptions` uniquely identifying Fungal Infection). Linear hyperplanes effortlessly separate these clusters.
3. **Absence of Clinical Noise**: The dataset contains zero missing values, zero false-positive symptom reports, no comorbidities (patients presenting multiple conditions simultaneously), and no ambiguous borderline cases.

### 3.3 Real-World Performance Caveat
> [!IMPORTANT]
> **Clinical Reality**: In actual clinical practice, patients present incomplete, vague, and subjective symptoms with widespread overlap (e.g. headache, fever, fatigue). Real-world diagnostic accuracy would be significantly lower than 100%. The model must always serve strictly as an advisory triage suggestion tool rather than an autonomous diagnostic system.

---

## 4. MLflow Experiment Tracking Architecture
- **Backend Store URI**: `sqlite:///mlflow.db`
- **Artifact Location**: `./mlartifacts`
- **Experiment Name**: `symptom-disease`
- **Tracked Metadata per Run**:
  - Model hyperparameters (`C`, `max_iter`, `n_estimators`, `max_depth`, `learning_rate`, `seed`)
  - Full metrics suite (train/val/test accuracy, macro F1, top-3 accuracy, and train-val gaps)
  - Model signatures and input examples
  - File SHA-256 hashes (`train.csv`, `val.csv`, `test.csv`, `symptoms.json`, `diseases.json`)
  - Git commit HEAD SHA
  - Artifacts: `symptoms.json`, `diseases.json`, and `disease_specialist.csv`

---

## 5. Model Registry & Quality Gate
- **Registered Model**: `symptom-disease-classifier`
- **Version**: `1`
- **Aliases**:
  - `staging`: assigned upon registration.
  - `production`: assigned after passing quality gate (`val_top3_accuracy >= 0.90` and `val_macro_f1 >= 0.85`).
- **Standalone Export Directory**: `models/production/`
  - `mlflow_model/` (portable MLflow model bundle with MLmodel specification)
  - `model.joblib` (lightweight serialized binary for zero-server inference)
  - `metadata.json` (version, metrics, run ID, and quality gate record)
  - `symptoms.json`, `diseases.json`, `disease_specialist.csv` (complete feature and specialist schemas)
