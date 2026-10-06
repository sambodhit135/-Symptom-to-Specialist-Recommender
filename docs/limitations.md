# Limitations — Disease Prediction MLOps Project

> This document is required reading before any demonstration or deployment.

---

## 1 · Synthetic / Simplified Dataset

| Limitation | Detail |
|------------|--------|
| **Source** | Kaggle — *"Disease Prediction Using Machine Learning"* by kaushil268 |
| **Nature** | Artificially generated; symptom-disease combinations are not derived from real patient records |
| **Binary features** | Each of 132 symptoms is either present (1) or absent (0) — no severity, duration, or frequency encoding |
| **41 disease classes** | Covers only 41 diseases; thousands of real conditions are absent |
| **Perfect separability** | The dataset is nearly perfectly separable, yielding unrealistically high accuracy (val top-3 ≈ 100%) |
| **No patient demographics** | Age, sex, comorbidities, medications, and family history are absent |
| **No temporal data** | Symptom onset order, duration, and progression are not captured |

**Implication:** The model cannot generalise to real clinical populations and must not be used for any actual triage or diagnostic decision.

---

## 2 · Not Medical Advice

> **IMPORTANT — Do not use this system for clinical decisions under any circumstances.**

- The system outputs a ranked list of possible diseases for educational purposes only.
- Every API response includes the mandatory disclaimer: *"Suggestion tool, not a medical diagnosis."*
- The disclaimer is hardcoded and cannot be disabled via the API.
- Real symptom checkers are developed with licensed clinicians, regulated medical device approval processes (e.g., FDA 510(k), CE mark), and extensive clinical validation.

---

## 3 · Emergency Rules Require Clinical Review

The emergency red-flag rules in `config/emergency_rules.json` were assembled by the project
team for demonstration purposes. They have **not** been reviewed by a medical professional.

Known gaps:
- Rules are based on a small set of symptoms — many genuine emergencies would not trigger the flag.
- The rule thresholds (number of matching symptoms required) are arbitrary.
- Paediatric, obstetric, and geriatric emergencies are not covered.
- Symptom overlap (e.g., "chest_pain" appearing in both cardiac and non-cardiac conditions) is not disambiguated.

**Action required before any real-world use:** All rules must be reviewed and approved by a licensed clinician or medical informaticist.

---

## 4 · Specialist Mapping Requires Review

The disease-to-specialist mapping in `data/disease_specialist.csv` was manually created
by the project team for this course.

Known limitations:
- Some diseases could reasonably map to multiple specialties; only one is recorded.
- Specialty names follow common UK/US conventions — they may not match local healthcare system terminology.
- Mapping does not account for specialist availability, referral pathways, or healthcare system context.

**Action required:** All mappings should be reviewed against local clinical guidelines before use in any real system.

---

## 5 · Model & MLOps Limitations

| Area | Limitation |
|------|------------|
| **Model type** | Logistic Regression — interpretable but may underperform DL models on complex symptom interactions |
| **No uncertainty calibration** | Confidence scores are raw softmax outputs; not calibrated probabilities |
| **Single-label prediction** | Model predicts one disease at a time; differential diagnosis often involves concurrent conditions |
| **Drift baseline** | Drift reference is the training split — any shift in real patient populations would require re-baselining |
| **Airflow on Windows** | Apache Airflow does not support native Windows execution; the DAG is testable via mocks but requires Linux/Mac or WSL2/Docker for production scheduling |
| **SQLite MLflow backend** | Suitable for development; production would require a PostgreSQL backend and S3-compatible artifact store |
| **Single-node deployment** | docker-compose runs all services on one machine; no horizontal scaling, load balancing, or high-availability |
| **No authentication on API** | Except `/admin/reload-model`, all endpoints are unauthenticated; add API gateway or OAuth2 before any internet-facing deployment |
| **No HTTPS** | Docker stack serves HTTP; TLS termination must be added (e.g., via nginx or Caddy) for production |

---

## 6 · What This Project Does Demonstrate

Despite the above limitations, the project successfully demonstrates:

- ✅ End-to-end MLOps lifecycle on a real dataset
- ✅ Reproducible training with MLflow experiment tracking and model registry
- ✅ Quality-gated model promotion (no silent degradation)
- ✅ Containerised, observable microservice deployment
- ✅ Automated testing at every layer (data, model, API, drift, retraining)
- ✅ Production monitoring with Prometheus and Grafana
- ✅ Evidently-based data drift detection with configurable thresholds
- ✅ Airflow DAG for automated, conditional retraining and safe model promotion
- ✅ CI/CD pipeline with GitHub Actions (lint + test + build + push)
- ✅ Git branching strategy and PR workflow
