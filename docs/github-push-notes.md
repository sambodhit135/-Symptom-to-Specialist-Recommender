# GitHub Push & Branching Notes

## Branching Strategy

This repository follows a strict **Trunk-Based / Feature-Branch Flow**:
- `main` serves as the primary, production-ready branch.
- Each development phase was isolated into a dedicated feature branch (`feature/phase-*`).
- Feature branches are developed and committed independently, preserving granular step-by-step audit trails.
- Merges into `main` preserve complete commit history and merge points for course auditability and reproducibility.
- Continuous Integration (`.github/workflows/ci.yml`) and Continuous Delivery (`.github/workflows/cd.yml`) trigger on pushes and pull requests to `main`.

## Branch Inventory & Commit Counts

| Branch | Commit Count | Status vs `main` | Purpose |
|---|---|---|---|
| `main` | 22 commits | Base branch | Production branch containing all integrated phases up to Phase 11b |
| `feature/phase-1-setup` | 3 commits | Merged into `main` | Repository skeleton, dependency definitions, data audit |
| `feature/phase-2-preprocessing` | 4 commits | Merged into `main` | Data cleaning, encoding, train/val/test splits, specialist mapping |
| `feature/phase-3-training` | 5 commits | Merged into `main` | Model training, cross-validation, MLflow tracking, candidate promotion |
| `feature/phase-4-api` | 6 commits | Merged into `main` | FastAPI prediction service, emergency triage rules, feedback capture |
| `feature/phase-4b-frontend` | 18 commits | Merged into `main` | Early web UI prototyping & component testing |
| `feature/phase-5-docker` | 7 commits | Merged into `main` | Docker containerisation for API and MLflow tracking server |
| `feature/phase-6-cicd` | 9 commits | Merged into `main` | Automated GitHub Actions CI test suite and CD container build workflow |
| `feature/phase-7-monitoring` | 11 commits | Merged into `main` | Prometheus metrics export, Grafana dashboard provisioning, traffic simulator |
| `feature/phase-8-drift` | 13 commits | Merged into `main` | Evidently AI data & concept drift detection, drift reporting, unit tests |
| `feature/phase-9-airflow` | 15 commits | Merged into `main` | Apache Airflow automated retraining DAG & model redeploy pipeline |
| `feature/phase-10-docs` | 17 commits | Merged into `main` | Comprehensive documentation, demo scripts, viva notes, repo audit |
| `feature/phase-11-frontend` | 19 commits | Merged into `main` | Responsive vanilla JS frontend UI & Nginx reverse proxy |
| `feature/phase-11b-polish` | 21 commits | Merged into `main` | UI polish, low-confidence disclaimer, typo map, feedback toast auto-dismiss |
| `feature/phase-12-docker-deploy` | 23 commits | **Unmerged (PR pending)** | Production multi-service Docker Compose deployment, orchestrator scripts, deployment guide |

## Pull Requests

### Pending Pull Request
- **Branch**: `feature/phase-12-docker-deploy`
- **Compare URL**: [https://github.com/sambodhit135/-Symptom-to-Specialist-Recommender/compare/main...feature/phase-12-docker-deploy](https://github.com/sambodhit135/-Symptom-to-Specialist-Recommender/compare/main...feature/phase-12-docker-deploy)
