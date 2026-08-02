# MolOps

**MLOps-Driven Cheminformatics Pipeline for Data-Driven Drug Bioactivity Prediction**

[![CI](https://github.com/LYHAMSEA/MolOps/actions/workflows/ci.yml/badge.svg)](https://github.com/LYHAMSEA/MolOps/actions/workflows/ci.yml)
[![Docker Hub](https://img.shields.io/docker/pulls/lyhamsea/molops)](https://hub.docker.com/r/lyhamsea/molops)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

MolOps is an end-to-end, reproducible cheminformatics pipeline that treats drug discovery ML the way production software is treated — version-controlled data, tracked experiments, tested code, a deployed inference API, and a live observability stack. The target is EGFR kinase (CHEMBL203), a clinically relevant anti-cancer target with abundant public bioactivity data in ChEMBL.

---

## Why MolOps exists

Most published ML models for drug discovery cannot be reproduced six months after the paper was written. There is no versioned data, no experiment tracking, no tested featurisation code, and no way to query the model without rerunning a Jupyter notebook.

MolOps answers that problem by applying MLOps discipline to the full cheminformatics workflow:

- Data is downloaded programmatically and validated with tests
- Every training run is logged to MLflow with parameters, metrics, and model artifacts
- The trained model is served through a FastAPI inference API with a model evaluation gate in CI
- Every prediction includes an applicability domain verdict — the API tells you whether to trust the number
- The entire stack is instrumented with Prometheus metrics and visualised in Grafana

---

## Architecture

```
ChEMBL REST API
      |
      v
 ingestion.py          Download -> curate -> validate -> save CSV
      |
      v
featurisation.py       SMILES -> Morgan fingerprints (ECFP4, 2048-bit)
                               + physicochemical descriptors (MW, LogP, HBD, HBA, TPSA)
      |
      v
  training.py          Random Forest + XGBoost -> MLflow experiment tracking
      |
      v
 evaluation.py         Tanimoto-based applicability domain check
      |
      v
    api.py             FastAPI inference server
      |
      +---> /metrics ---> Prometheus (scrapes every 10s) ---> Grafana dashboards
      +---> MLflow    ---> Experiment tracking UI (all training runs + predictions)
```

---

## Project structure

```
MolOps/
├── molops/
│   ├── pipeline/
│   │   ├── ingestion.py        ChEMBL download and IC50 curation
│   │   ├── featurisation.py    RDKit fingerprints and descriptors
│   │   ├── training.py         RF + XGBoost with MLflow tracking
│   │   └── evaluation.py       Tanimoto applicability domain
│   ├── api.py                  FastAPI inference server
│   ├── cli.py                  Command-line interface
│   └── config.py               Pydantic settings from .env
│
├── tests/
│   ├── unit/
│   │   ├── test_ingestion.py      7 curation and pIC50 tests
│   │   ├── test_featurisation.py  12 RDKit and fingerprint tests
│   │   └── test_evaluation.py     7 applicability domain tests
│   └── integration/
│       └── test_pipeline.py       4 end-to-end pipeline tests
│
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml      Scrape config (API + self-monitoring)
│   │   └── alerts.yml          5 alert rules
│   └── grafana/
│       ├── provisioning/       Auto-wired datasource and dashboard provider
│       └── dashboards/         9-panel dashboard JSON (infrastructure as code)
│
├── k8s/base/                   Kubernetes manifests (Kustomize)
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret-template.yaml
│   ├── api-deployment.yaml     Deployment + Service (2 replicas)
│   ├── mlflow-deployment.yaml  MLflow server + PVC
│   ├── hpa.yaml                HorizontalPodAutoscaler (2-6 replicas, 70% CPU)
│   └── kustomization.yaml
│
├── linting/
│   └── molops_linter.py        Custom AST-based pipeline linter (ML001-ML005)
├── scripts/
│   └── dev.sh                  All developer commands
├── .github/workflows/ci.yml    4-job CI/CD pipeline
├── Dockerfile                  Two-stage build via Miniconda for RDKit
├── docker-compose.yml          API + MLflow + Prometheus + Grafana
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## Getting started

### Prerequisites

- Ubuntu 20.04+ or macOS
- Miniconda (https://docs.conda.io/en/latest/miniconda.html) -- required for RDKit
- Docker and Docker Compose
- Git

### 1. Clone and set up the environment

```bash
git clone https://github.com/LYHAMSEA/MolOps.git
cd MolOps

./scripts/dev.sh setup
conda activate molops

cp .env.example .env
```

### 2. Verify the full stack

```bash
./scripts/dev.sh verify
```

This checks RDKit, ChEMBL connectivity, scikit-learn, XGBoost, MLflow, FastAPI, and all lint tools. Every check must be green before proceeding.

### 3. Run the full pipeline

```bash
./scripts/dev.sh ingest    # download and curate ChEMBL data (~5 min)
./scripts/dev.sh train     # train RF + XGBoost, log to MLflow (~10 min)
./scripts/dev.sh run       # start inference API on port 8001
```

### 4. Make your first prediction

```bash
# Gefitinib -- approved EGFR drug (expect pIC50 ~8, within AD)
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"}' \
  | python3 -m json.tool

# Aspirin -- not an EGFR inhibitor (expect pIC50 ~5.5, outside AD)
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}' \
  | python3 -m json.tool
```

---

## The pipeline in detail

### Stage 1 -- Data ingestion

Connects to the ChEMBL public REST API and downloads all IC50 bioactivity records for EGFR kinase (target ID: CHEMBL203). Applies a standardised curation protocol:

| Step | What it removes |
|---|---|
| Null SMILES or IC50 | Records that cannot be featurised or labelled |
| Zero or negative IC50 | Physically impossible values |
| Non-numeric IC50 | String values from the API that cannot be parsed |
| pIC50 outside 3-12 | Physically unrealistic (below 1 pM or above 1 mM) |
| Duplicate ChEMBL IDs | Same compound measured in multiple assays |

```bash
./scripts/dev.sh ingest
# Saves to data/processed/egfr_bioactivity.csv
# Typical output: ~10,000-11,000 curated compounds
```

**Why pIC50?** IC50 values span seven orders of magnitude (1 nM to 10,000,000 nM). Converting to pIC50 = -log10(IC50_M) compresses this to a 0-12 scale where higher means more potent. pIC50 > 8 is very potent (IC50 < 10 nM); pIC50 < 5 is inactive (IC50 > 10 uM).

---

### Stage 2 -- Featurisation

Converts SMILES strings into numerical feature matrices that ML models can learn from.

**Morgan fingerprints (ECFP4)**

Each molecule is encoded as a 2048-bit binary vector. Each bit represents the presence or absence of a circular substructural feature (atom + neighbours up to radius 2). Two molecules with similar fingerprints are structurally similar.

```python
MORGAN_RADIUS = 2     # ECFP4 -- extended connectivity, diameter 4
MORGAN_NBITS  = 2048  # standard size in cheminformatics
```

**Physicochemical descriptors**

Nine descriptors computed via RDKit:

| Descriptor | Drug-likeness relevance |
|---|---|
| MW | < 500 Da for Lipinski Rule of Five |
| LogP | < 5 for oral absorption |
| HBD | < 5 H-bond donors (Lipinski) |
| HBA | < 10 H-bond acceptors (Lipinski) |
| TPSA | < 140 Angstrom squared for membrane permeability |
| RotBonds | Molecular flexibility |
| RingCount | Structural complexity |
| AromaticRings | Aromatic system count |
| HeavyAtoms | Molecular size |

---

### Stage 3 -- Model training

Two models are trained and compared. Every run is logged to MLflow.

| Model | Hyperparameters | Features |
|---|---|---|
| Random Forest | n_estimators=500, max_depth=10, min_samples_leaf=2 | Morgan ECFP4 (2048-bit) |
| XGBoost | n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8 | Morgan ECFP4 (2048-bit) |

**Performance gates** -- CI will reject a Docker build if these thresholds are not met:

```python
MIN_R2   = 0.50   # model must explain at least 50% of pIC50 variance
MAX_RMSE = 1.20   # prediction error must be below 1.2 pIC50 units
```

**Typical results on EGFR ChEMBL data:**

| Model | RMSE | R2 | Interpretation |
|---|---|---|---|
| Random Forest | ~0.65-0.80 | ~0.60-0.72 | RMSE of 0.7 = ~5-fold IC50 error |
| XGBoost | ~0.60-0.75 | ~0.65-0.75 | Usually matches or beats RF |

```bash
./scripts/dev.sh train
# Open http://localhost:5000 to compare runs in MLflow UI
```

---

### Stage 4 -- Applicability domain

The applicability domain (AD) answers the question every ML model avoids: *should I trust this prediction?*

A model trained on EGFR kinase inhibitors knows the chemistry of kinase inhibitors. If you ask it to predict bioactivity for a steroid or a carbohydrate -- compounds structurally unlike anything in its training data -- it will return a number. That number is not a prediction; it is extrapolation presented as fact.

**Implementation: Tanimoto nearest-neighbour AD**

```
1. Pre-compute Morgan fingerprints for all training set molecules (at API startup)
2. For each inference request:
   a. Compute Morgan fingerprint of the query molecule
   b. Compute Tanimoto similarity to every training fingerprint
   c. Take the maximum similarity (nearest neighbour)
   d. If max_tanimoto >= 0.4: within AD -> prediction is reliable
      If max_tanimoto <  0.4: outside AD -> prediction flagged as unreliable
3. Return the max_tanimoto value and AD verdict in the API response
4. Increment the molops_outside_applicability_domain_total Prometheus counter
```

**Real prediction results:**

| Molecule | pIC50 predicted | Max Tanimoto | AD verdict | Scientific interpretation |
|---|---|---|---|---|
| Gefitinib (EGFR drug) | ~8.2 | ~0.74 | within | Correct -- potent EGFR inhibitor |
| Erlotinib (EGFR drug) | ~7.8 | ~0.68 | within | Correct -- potent EGFR inhibitor |
| Aspirin | ~5.5 | ~0.36 | outside | Correct -- COX inhibitor, not EGFR |
| Ibuprofen | ~5.6 | ~0.33 | outside | Correct -- COX inhibitor, not EGFR |

---

### Stage 5 -- Inference API

FastAPI server serving predictions from SMILES input.

```
GET  /healthz    ->  liveness probe (Docker HEALTHCHECK, Kubernetes liveness)
GET  /readyz     ->  readiness probe (model loaded check, Kubernetes readiness)
POST /predict    ->  main prediction endpoint
GET  /metrics    ->  Prometheus scrape endpoint
GET  /docs       ->  auto-generated Swagger UI
```

**Prediction request:**

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "model_name": "random_forest"}'
```

**Prediction response:**

```json
{
  "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
  "pIC50_predicted": 8.214,
  "drug_likeness": {
    "MW": 446.9,
    "LogP": 3.74,
    "HBD": 1,
    "HBA": 7,
    "TPSA": 74.9,
    "RotBonds": 6,
    "lipinski_pass": true
  },
  "applicability_domain": "within",
  "max_tanimoto": 0.741,
  "confidence": 0.741,
  "model_used": "random_forest"
}
```

**pIC50 interpretation:**

| pIC50 | IC50 | Activity class |
|---|---|---|
| > 8.0 | < 10 nM | Very potent -- excellent lead candidate |
| 7.0 - 8.0 | 10-100 nM | Potent -- strong hit |
| 6.0 - 7.0 | 100 nM - 1 uM | Moderate -- weak hit |
| 5.0 - 6.0 | 1-10 uM | Low activity -- borderline |
| < 5.0 | > 10 uM | Inactive |

---

## Monitoring

### Start the full monitoring stack

```bash
# Docker Compose -- all services containerised
docker compose up --build -d

# Or direct systemd installation on Ubuntu
sudo systemctl start prometheus grafana-server
uvicorn molops.api:app --host 0.0.0.0 --port 8001 --workers 1
```

| Service | URL | Default credentials |
|---|---|---|
| MolOps API | http://localhost:8001 | -- |
| API docs | http://localhost:8001/docs | -- |
| MLflow | http://localhost:5000 | -- |
| Prometheus | http://localhost:9090 | -- |
| Grafana | http://localhost:3000 | admin / molops_dev |

### Prometheus metrics

| Metric | Type | Description |
|---|---|---|
| `molops_predictions_total` | Counter | Total predictions by model and outcome |
| `molops_prediction_latency_seconds` | Histogram | End-to-end prediction time |
| `molops_last_confidence_score` | Gauge | Tanimoto score of the most recent prediction |
| `molops_last_max_tanimoto` | Gauge | Max Tanimoto to training set |
| `molops_outside_applicability_domain_total` | Counter | Cumulative out-of-domain predictions |
| `molops_predicted_pic50` | Summary | Distribution of predicted pIC50 values |

### Alert rules

| Alert | Severity | Condition | Duration |
|---|---|---|---|
| `MolOpsAPIDown` | critical | API unreachable | 1 min |
| `HighPredictionErrorRate` | warning | > 10% errors | 3 min |
| `HighApplicabilityDomainViolations` | warning | > 50% outside AD | 5 min |
| `SlowPredictionLatency` | warning | P95 > 2s | 5 min |
| `LowModelConfidence` | warning | avg Tanimoto < 0.3 | 10 min |

---

## CI/CD pipeline

```
Lint -> Test (3.11 + 3.12) -> Docker build + push -> GitHub Release
```

| Job | What runs |
|---|---|
| Lint | ruff, mypy, bandit, MolOps pipeline linter, secrets check |
| Test | pytest on Python 3.11 and 3.12, coverage report, integration tests |
| Build | multi-arch Docker build, push to Docker Hub, Trivy scan |
| Release | GitHub Release with changelog on version tags |

**Required GitHub Secrets:** `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`

---

## Custom pipeline linter

`linting/molops_linter.py` enforces domain-specific rules that ruff and mypy cannot know:

| Rule | Severity | What it checks |
|---|---|---|
| ML001 | WARNING | Module missing a docstring |
| ML002 | ERROR | AD threshold outside 0.0-1.0 |
| ML003 | ERROR | pH constant outside 0-14 |
| ML004 | WARNING | Function missing a docstring |
| ML005 | ERROR | Bare except clause in pipeline code |

```bash
python linting/molops_linter.py molops/pipeline/ --strict
```

---

## Kubernetes deployment

```bash
kubectl apply -f k8s/base/namespace.yaml

kubectl create secret generic molops-secrets \
  --namespace=molops \
  --from-literal=MOLOPS_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" \
  --from-literal=GF_SECURITY_ADMIN_USER="admin" \
  --from-literal=GF_SECURITY_ADMIN_PASSWORD="your_password"

kubectl apply -k k8s/base

kubectl get pods -n molops -w
```

**Resources deployed:**

| Resource | Description |
|---|---|
| `Deployment/molops-api` | 2 replicas, liveness + readiness probes |
| `Deployment/mlflow` | MLflow server with PVC-backed storage |
| `HorizontalPodAutoscaler` | 2-6 replicas at 70% CPU |
| `ConfigMap/molops-config` | Non-secret environment variables |

---

## Testing

```bash
./scripts/dev.sh test        # unit tests with coverage
./scripts/dev.sh test-all    # unit + integration tests
```

| File | Tests | Coverage |
|---|---|---|
| `test_ingestion.py` | 7 | curation logic, pIC50 conversion, boundary values |
| `test_featurisation.py` | 12 | SMILES parsing, fingerprints, Lipinski, Tanimoto |
| `test_evaluation.py` | 7 | AD threshold, within/outside classification |
| `test_pipeline.py` | 4 | full curation -> featurisation -> AD chain |

---

## Developer commands

```
./scripts/dev.sh setup        create conda env and install dependencies
./scripts/dev.sh verify       check every component of the stack
./scripts/dev.sh ingest       download and curate ChEMBL data
./scripts/dev.sh train        train RF + XGBoost, log to MLflow
./scripts/dev.sh predict      predict from CLI without starting the API
./scripts/dev.sh pipeline     ingest + train + predict in sequence
./scripts/dev.sh lint         ruff + mypy + bandit + MolOps linter
./scripts/dev.sh test         unit tests with coverage report
./scripts/dev.sh test-all     unit + integration tests
./scripts/dev.sh run          start API in dev mode (auto-reload)
./scripts/dev.sh mlflow       start MLflow UI on port 5000
./scripts/dev.sh jupyter      start JupyterLab for interactive exploration
./scripts/dev.sh docker-up    start full stack with Docker Compose
./scripts/dev.sh docker-down  stop all containers
./scripts/dev.sh docker-logs  tail API container logs
```

---

## Scientific background

**EGFR as a drug discovery target**

Epidermal Growth Factor Receptor (EGFR, CHEMBL203) is a receptor tyrosine kinase overexpressed in several cancers including non-small cell lung cancer, colorectal cancer, and head and neck squamous cell carcinoma. Approved drugs targeting it include Gefitinib, Erlotinib, Osimertinib, and Lapatinib, making it a well-characterised benchmark target with thousands of publicly measured inhibitors in ChEMBL.

Because EGFR has approved drugs with known IC50 values, predictions can be biologically validated: Gefitinib and Erlotinib should score high (pIC50 > 7), while unrelated drugs like Aspirin and Ibuprofen should score low and be flagged as outside the applicability domain.

**Applicability domain in the literature**

Applicability domain is an active research area in cheminformatics. The principle that a model should express uncertainty when predicting outside its training distribution is recognised as critical for trustworthy ML in drug discovery -- see the published work of the Bender group (Cambridge), Svensson et al. (AstraZeneca), and the broader QSAR literature. MolOps operationalises this concept as a production monitoring metric rather than a paper result.

---

## Roadmap

- [ ] Graph Neural Network (PyTorch Geometric MPNN) as a third model
- [ ] ChemBERTa fine-tuning for SMILES-based prediction
- [ ] Multi-target prediction (extend beyond EGFR)
- [ ] ADMET prediction endpoints (solubility, BBB permeability, hERG toxicity)
- [ ] Database persistence for prediction history
- [ ] Alertmanager Slack integration
- [ ] Streamlit or Gradio frontend for non-API users

---

## Author

**Ismail Abiodun Onile**
BSc Industrial Chemistry (OAU) · HND Pharmaceutical Technology (MAPOLY, Best Graduating Student)
Former QC Analyst, GlaxoSmithKline Consumer Nigeria

- GitHub: https://github.com/LYHAMSEA
- Docker Hub: https://hub.docker.com/u/lyhamsea
- Email: lyhamseaelino99@gmail.com

---

## License

MIT