#!/usr/bin/env bash
set -euo pipefail

CMD=${1:-help}

case "$CMD" in

  setup)
    echo "Setting up MolOps conda environment..."
    conda create -n molops -c conda-forge python=3.11 rdkit -y
    conda run -n molops pip install -e ".[dev,lint]"
    conda run -n molops python -m ipykernel install --user \
      --name molops --display-name "MolOps (Python 3.11)"
    echo ""
    echo "Done. Activate with: conda activate molops"
    echo "Then run:            ./scripts/dev.sh verify"
    ;;

  verify)
    echo "Verifying full stack..."
    python - << 'PYEOF'
checks = {}

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    assert mol is not None
    assert round(Descriptors.MolWt(mol), 0) == 180.0
    checks["RDKit"] = "OK  (aspirin MW=180)"
except Exception as e:
    checks["RDKit"] = f"FAIL -- {e}"

try:
    from chembl_webresource_client.new_client import new_client
    t = new_client.target
    name = list(t.filter(chembl_id="CHEMBL203").only(["pref_name"]))[0]["pref_name"]
    checks["ChEMBL"] = f"OK  (EGFR = {name[:40]})"
except Exception as e:
    checks["ChEMBL"] = f"FAIL -- {e}"

try:
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    import numpy as np
    X, y = np.random.rand(20, 100), np.random.rand(20)
    RandomForestRegressor(n_estimators=5).fit(X, y)
    XGBRegressor(n_estimators=5, verbosity=0).fit(X, y)
    checks["sklearn + XGBoost"] = "OK"
except Exception as e:
    checks["sklearn + XGBoost"] = f"FAIL -- {e}"

try:
    import mlflow
    checks["MLflow"] = f"OK  (v{mlflow.__version__})"
except Exception as e:
    checks["MLflow"] = f"FAIL -- {e}"

try:
    import fastapi, prometheus_client, pydantic
    checks["FastAPI stack"] = f"OK  (FastAPI {fastapi.__version__})"
except Exception as e:
    checks["FastAPI stack"] = f"FAIL -- {e}"

try:
    import subprocess
    r = subprocess.run(["ruff", "--version"], capture_output=True, text=True)
    checks["Ruff"] = f"OK  ({r.stdout.strip()})"
except Exception as e:
    checks["Ruff"] = f"FAIL -- {e}"

print("")
print("=" * 55)
print("  MolOps Stack Verification")
print("=" * 55)
for name, status in checks.items():
    icon = "✓" if status.startswith("OK") else "✗"
    print(f"  {icon}  {name:<22} {status}")
print("=" * 55)
failures = [k for k, v in checks.items() if not v.startswith("OK")]
if failures:
    print(f"\n  FAILED: {', '.join(failures)}")
    raise SystemExit(1)
else:
    print("\n  All checks passed -- ready to build MolOps")
PYEOF
    ;;

  ingest)
    echo "Downloading and curating ChEMBL data for EGFR..."
    python -m molops.cli ingest
    ;;

  train)
    echo "Training Random Forest and XGBoost models..."
    python -m molops.cli train
    echo ""
    echo "Open MLflow UI to compare runs:"
    echo "  mlflow ui --host 0.0.0.0 --port 5000"
    ;;

  predict)
    SMILES="${2:-CC(=O)Oc1ccccc1C(=O)O}"
    echo "Predicting bioactivity for: $SMILES"
    python -m molops.cli predict "$SMILES"
    ;;

  lint)
    echo "Running ruff..."
    ruff check molops/ tests/ --fix
    ruff format molops/ tests/
    echo "Running mypy..."
    mypy molops/
    echo "Running bandit..."
    bandit -r molops/ -c pyproject.toml
    echo "Running MolOps pipeline linter..."
    python linting/molops_linter.py molops/pipeline/ --strict
    echo ""
    echo "All lint checks passed"
    ;;

  test)
    echo "Running unit tests..."
    pytest tests/unit/ -v --cov=molops --cov-report=term-missing -m "not integration"
    ;;

  test-all)
    echo "Running all tests (unit + integration)..."
    pytest tests/ -v --cov=molops --cov-report=term-missing --cov-report=html
    echo ""
    echo "Coverage report: htmlcov/index.html"
    ;;

  run)
    echo "Starting MolOps API in dev mode (auto-reload)..."
    echo "API docs: http://localhost:8001/docs"
    uvicorn molops.api:app --host 0.0.0.0 --port 8001 --reload --log-level debug
    ;;

  mlflow)
    echo "Starting MLflow UI..."
    echo "Open: http://localhost:5000"
    mlflow ui --host 0.0.0.0 --port 5000
    ;;

  jupyter)
    echo "Starting JupyterLab..."
    echo "Open the URL printed below in your browser"
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
    ;;

  docker-up)
    echo "Starting full stack: API + MLflow + Prometheus + Grafana..."
    docker compose up --build -d
    echo ""
    echo "Services running:"
    echo "  API:        http://localhost:8001"
    echo "  API docs:   http://localhost:8001/docs"
    echo "  MLflow:     http://localhost:5000"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana:    http://localhost:3000  (admin / molops_dev)"
    ;;

  docker-down)
    docker compose down
    echo "All containers stopped"
    ;;

  docker-logs)
    docker compose logs -f molops-api
    ;;

  pipeline)
    echo "Running full pipeline: ingest -> train -> verify"
    ./scripts/dev.sh ingest
    ./scripts/dev.sh train
    echo ""
    echo "Pipeline complete. Predict aspirin:"
    ./scripts/dev.sh predict "CC(=O)Oc1ccccc1C(=O)O"
    ;;

  help|*)
    echo ""
    echo "MolOps Dev Scripts"
    echo ""
    echo "Usage: ./scripts/dev.sh <command>"
    echo ""
    echo "Environment:"
    echo "  setup          Create conda env and install all dependencies"
    echo "  verify         Check every component of the stack is working"
    echo ""
    echo "Pipeline:"
    echo "  ingest         Download and curate ChEMBL EGFR bioactivity data"
    echo "  train          Featurise molecules and train RF + XGBoost models"
    echo "  predict        Predict bioactivity for a SMILES string"
    echo "  pipeline       Run ingest + train + predict in sequence"
    echo ""
    echo "Development:"
    echo "  lint           Run ruff, mypy, bandit, and MolOps linter"
    echo "  test           Run unit tests with coverage"
    echo "  test-all       Run unit + integration tests"
    echo "  run            Start API server with auto-reload"
    echo "  mlflow         Start MLflow experiment tracking UI"
    echo "  jupyter        Start JupyterLab for interactive exploration"
    echo ""
    echo "Docker:"
    echo "  docker-up      Start full stack (API + MLflow + Prometheus + Grafana)"
    echo "  docker-down    Stop all containers"
    echo "  docker-logs    Tail API container logs"
    ;;
esac