# Phase 1 Notes – Project Setup and Data Audit

## What Was Built

### Folder Skeleton
We created a standard MLOps project layout inside the existing workspace folder:

```
data/raw/          – holds the original, read-only CSVs
data/processed/    – will hold cleaned data in later phases
data/feedback/     – will hold user feedback at runtime
data/prod_logs/    – will hold production prediction logs
src/               – Python source code
tests/             – pytest test files
models/            – saved model files (.pkl, .joblib)
notebooks/         – Jupyter exploration notebooks
docs/              – reports, notes, hashes
config/            – YAML config files
scripts/           – one-off utility scripts
```

Empty directories have a `.gitkeep` placeholder so Git tracks them even though they have no files yet.

### Why This Layout?
Separating raw data from processed data means we can always re-run preprocessing from scratch without losing the original files. The `src/` package approach (rather than a flat folder of scripts) keeps code organised and importable.

### Data Hashes (`docs/data_hashes.txt`)
We recorded the SHA-256 fingerprint of each raw CSV the moment they were placed in `data/raw/`. If anyone accidentally edits those files, re-running `Get-FileHash` will show a mismatch. This is a basic data integrity check.

### Virtual Environment
We created a Python 3.12.1 venv and installed:
- **pandas 3.0.6** – data loading and manipulation
- **scikit-learn 1.9.1** – machine learning (used from Phase 2 onwards)

All installed packages (including transitive dependencies like numpy, scipy, joblib) are pinned in `requirements.txt` so any team member can reproduce the exact environment with `pip install -r requirements.txt`.

### Data Audit (`src/data_audit.py` → `docs/data_audit.md`)
The script reads both CSVs and records the following facts (all numbers are real, from actual runs):

| Finding | Detail |
|---------|--------|
| Training shape | 4920 rows × 134 columns |
| Testing shape | 42 rows × 133 columns |
| Disease classes | 41 |
| Symptom columns | 132 real + 1 ghost (`Unnamed: 133`) |
| Missing values in Training | 4920 (entire `Unnamed: 133` column is NaN) |
| Missing values in Testing | 0 |
| Duplicate rows in Training | 4616 out of 4920 |
| Duplicate rows in Testing | 0 |
| Shared rows (exact match) | 3048 |
| Symptom never active (train) | `fluid_overload` (always 0) |
| Disease name whitespace bugs | `'Hypertension '` and `'Diabetes '` have trailing spaces |
| Case inconsistencies | None |

### Key Issues to Fix in Phase 2 (preprocessing)
1. **`Unnamed: 133`** – drop this column (all NaN, artefact of extra comma in CSV).
2. **`fluid_overload` vs `fluid_overload.1`** – `fluid_overload` is never 1 in training; pandas renamed the second occurrence to `fluid_overload.1`. Need to investigate and deduplicate.
3. **Trailing spaces** in `Hypertension ` and `Diabetes ` – strip whitespace from the `prognosis` column.
4. **4616 duplicate rows** – this is intentional in this dataset (each disease class has 120 copies of carefully constructed symptom combos). We will keep them for training but document the decision.
5. **3048 shared rows between train and test** – the test set is a strict subset of training rows. This is common in student Kaggle datasets. We will note it but not modify raw files.

### Git Workflow
- One initial commit on `main` with the project skeleton.
- All Phase 1 work on branch `feature/phase-1-setup`.
- We never pushed to GitHub (no remote configured yet).
