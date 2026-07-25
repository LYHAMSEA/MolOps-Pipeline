"""
MolOps command-line interface.

Usage:
  python -m molops.cli ingest
  python -m molops.cli train
  python -m molops.cli predict "CC(=O)Oc1ccccc1C(=O)O"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_ingest() -> None:
    """Download and curate ChEMBL data."""
    from molops.pipeline.ingestion import run_ingestion
    df = run_ingestion()
    print(f"\nIngestion complete: {len(df)} compounds saved to data/processed/egfr_bioactivity.csv")
    print(f"pIC50 range: {df['pIC50'].min():.2f} -- {df['pIC50'].max():.2f}")


def cmd_train() -> None:
    """Featurise data and train both models."""
    import numpy as np
    import pandas as pd
    from molops.pipeline.featurisation import featurise_dataset
    from molops.pipeline.training import (
        meets_minimum_performance,
        train_random_forest,
        train_xgboost,
    )
    import joblib, os

    df = pd.read_csv("data/processed/egfr_bioactivity.csv")
    print(f"Loaded {len(df)} compounds")

    print("Featurising molecules...")
    fp_matrix, desc_df, valid_idx = featurise_dataset(df)
    y = df.iloc[valid_idx]["pIC50"].values
    print(f"Feature matrix: {fp_matrix.shape}")

    print("\nTraining Random Forest...")
    rf_run, rf_rmse, rf_r2 = train_random_forest(fp_matrix, y)
    print(f"  RMSE={rf_rmse:.3f}  R2={rf_r2:.3f}")

    print("\nTraining XGBoost...")
    xgb_run, xgb_rmse, xgb_r2 = train_xgboost(fp_matrix, y)
    print(f"  RMSE={xgb_rmse:.3f}  R2={xgb_r2:.3f}")

    # Save the better model for the API
    os.makedirs("models", exist_ok=True)
    best_rmse = min(rf_rmse, xgb_rmse)
    if not meets_minimum_performance(best_rmse, max(rf_r2, xgb_r2)):
        print(f"\nWARNING: Best RMSE={best_rmse:.3f} does not meet minimum performance gate")

    # Save Random Forest (simpler, more interpretable) as default
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    rf_model_uri = f"runs:/{rf_run}/random_forest"
    import mlflow.sklearn
    model = mlflow.sklearn.load_model(rf_model_uri)
    joblib.dump(model, "models/random_forest.joblib")

    # Save training SMILES for applicability domain
    valid_smiles = df.iloc[valid_idx]["canonical_smiles"].tolist()
    train_n = int(len(valid_smiles) * 0.8)
    with open("models/training_smiles.txt", "w") as f:
        f.write("\n".join(valid_smiles[:train_n]))

    print(f"\nModels saved to models/")
    print(f"Run: mlflow ui --port 5000  to compare experiments")


def cmd_predict(smiles: str) -> None:
    """Quick CLI prediction without starting the API server."""
    import joblib
    import numpy as np
    from molops.pipeline.featurisation import (
        lipinski_pass, morgan_fingerprint,
        physicochemical_descriptors, smiles_to_mol,
    )

    mol = smiles_to_mol(smiles)
    if mol is None:
        print(f"ERROR: Invalid SMILES: {smiles}")
        sys.exit(1)

    model = joblib.load("models/random_forest.joblib")
    fp = morgan_fingerprint(mol).reshape(1, -1)
    pic50 = float(model.predict(fp)[0])
    desc = physicochemical_descriptors(mol)

    result = {
        "smiles": smiles,
        "pIC50_predicted": round(pic50, 3),
        "drug_likeness": desc,
        "lipinski_pass": lipinski_pass(desc),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="molops", description="MolOps CLI")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("ingest", help="Download and curate ChEMBL data")
    sub.add_parser("train",  help="Featurise and train ML models")
    pred_p = sub.add_parser("predict", help="Predict bioactivity for a SMILES")
    pred_p.add_argument("smiles", help="SMILES string of the molecule")

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest()
    elif args.command == "train":
        cmd_train()
    elif args.command == "predict":
        cmd_predict(args.smiles)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()