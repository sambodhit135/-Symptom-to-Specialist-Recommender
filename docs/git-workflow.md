# Git Workflow & CI/CD Strategy

This document outlines the branching strategy, commit conventions, pull request process, and CI/CD pipelines for the **Disease Prediction Recommender** MLOps project.

---

## 1. Branching Strategy

The repository follows a structured Gitflow-inspired workflow designed for continuous integration and automated deployment:

```
feature/phase-x-name ────┐
                          ▼ (PR + CI pass)
                       develop ────────────────┐
                                                ▼ (Release PR + CI pass)
                                               main ──► CD (GHCR Docker Push)
hotfix/fix-name ────────────────────────────────┘
```

### Core Branches
- **`main`**:
  - Production-ready branch.
  - Protected branch (requires passing CI checks and PR review).
  - Pushes or merges to `main` trigger the **CD Pipeline** (`cd.yml`), which builds and publishes the production Docker image to GitHub Container Registry (`ghcr.io`).
- **`develop`**:
  - Integration branch for ongoing development and testing across phases.
  - Feature branches merge into `develop` via pull requests.

### Supporting Branches
- **`feature/<name>`** (e.g., `feature/phase-6-cicd`):
  - Created from `main` (or `develop`) for discrete features or MLOps phases.
  - Tested continuously by the **CI Pipeline** on every push and pull request.
- **`fix/<name>`** or **`hotfix/<name>`**:
  - Created to address critical bugs or regression issues.
  - Merged back with corresponding tests.

---

## 2. Commit Message Style (Conventional Commits)

All commit messages must adhere to the [Conventional Commits v1.0.0](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short imperative description>

[optional body explaining rationale]

[optional footer(s)]
```

### Allowed Types
| Type | Description | Example |
|---|---|---|
| `feat` | A new feature, pipeline component, or endpoint | `feat(api): add emergency triage rules to predict endpoint` |
| `fix` | A bug fix | `fix(data): handle leading whitespace in disease labels` |
| `docs` | Documentation only changes | `docs(git): add branching strategy and CI/CD guide` |
| `test` | Adding or correcting automated tests | `test(model): add hold-out test set accuracy verification` |
| `ci` | Changes to CI/CD workflows and scripts | `ci(github): add Ruff linting and pytest automated workflow` |
| `refactor` | Code change that neither fixes a bug nor adds a feature | `refactor(train): extract XGBoost classifier into class wrapper` |
| `chore` | Maintenance tasks, dependency updates, configs | `chore(deps): pin ruff in requirements-dev.txt` |

---

## 3. Pull Request (PR) Lifecycle

1. **Create Branch**: Branch off `main` or `develop` using `feature/<feature-name>`.
2. **Local Pre-Flight Checks**:
   ```bash
   # 1. Run linter
   ruff check .

   # 2. Run automated tests
   pytest

   # 3. Test Docker build
   docker build -t disease-api:local .
   ```
3. **Open PR**: Fill out the PR template (`.github/pull_request_template.md`).
4. **Automated CI Validation**:
   - GitHub Actions executes `.github/workflows/ci.yml`.
   - Runs `ruff check .` across all Python code.
   - Executes all 25 unit/integration/data tests via `pytest`.
   - Performs a test build of the Docker container.
5. **Review & Merge**:
   - Merge only when CI status is **green** and checks have passed.
   - Use `Squash and merge` or `Create a merge commit` depending on project policy.

---

## 4. CI/CD Pipelines

### 4.1 Continuous Integration (`.github/workflows/ci.yml`)
- **Triggers**:
  - `push` to `main`, `develop`, `feature/**`, `fix/**`
  - `pull_request` targeting `main`, `develop`
- **Steps**:
  1. Checks out repository.
  2. Sets up Python 3.12 with pip caching.
  3. Installs pinned dependencies (`requirements-dev.txt`).
  4. Runs `ruff check .`.
  5. Executes full test suite with `pytest`.
     - *Hermetic execution*: Tests utilize `models/production/` and `data/processed/` with zero requirement for an external running MLflow server.
  6. Dry-run builds the Docker image using Docker Buildx (no push).

### 4.2 Continuous Deployment (`.github/workflows/cd.yml`)
- **Triggers**:
  - `push` to `main` (merges from pull requests).
- **Steps**:
  1. Checks out repository.
  2. Sets up Docker Buildx.
  3. Logs into GitHub Container Registry (`ghcr.io`) using the built-in `${{ secrets.GITHUB_TOKEN }}`.
  4. Generates OCI metadata tags (`latest`, `sha-<commit>`).
  5. Builds and pushes the multi-stage production Docker image to:
     `ghcr.io/<owner>/<repo>:latest` and `ghcr.io/<owner>/<repo>:sha-<short-sha>`

---

## 5. Required GitHub Repository Settings

To ensure the CD pipeline runs smoothly without permission errors, verify the following in your GitHub repository:

1. **Workflow Permissions**:
   - Navigate to: **Settings** → **Actions** → **General** → **Workflow permissions**.
   - Select: **Read and write permissions** (or ensure workflow explicitly declares `permissions: { contents: read, packages: write }`).
   - Check: **Allow GitHub Actions to create and approve pull requests** (optional, recommended).
2. **GitHub Container Registry (GHCR) Package Settings**:
   - By default, images pushed by private repositories are set to private visibility.
   - To make the container image publicly pullable:
     - Navigate to your profile/organization **Packages** tab.
     - Click `disease-prediction-api` (or repo package name).
     - Go to **Package settings** → **Danger Zone** → **Change package visibility** → **Public**.
3. **Branch Protection Rules (Recommended)**:
   - Navigate to: **Settings** → **Branches** → **Add branch protection rule**.
   - Branch name pattern: `main`.
   - Enable: **Require a pull request before merging**.
   - Enable: **Require status checks to pass before merging** (check `Lint & Pytest Suite` and `Build Docker Image (Dry-Run)`).

---

## 6. Model & Data Availability Strategy for CI

### Strategy Chosen: Version-Controlled Artifacts in Repo

The model binary (`models/production/model.joblib`), metadata (`models/production/metadata.json`), schemas (`symptoms.json`, `diseases.json`), and processed hold-out data (`data/processed/`) are tracked directly in Git.

### Why:
1. **Lightweight (< 100 KB total):** The entire production model package is ~47 KB, well within standard Git file size best practices.
2. **Hermetic & Deterministic:** CI does not retrain models or re-scrape data on every push; it tests the exact, immutable artifact that will run in production.
3. **Fast CI Execution (< 2 minutes):** Eliminates heavy training loops, external storage download latency, or S3/GCS credential setup in CI runners.
