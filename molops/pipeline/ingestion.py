from __future__ import annotations

import logging
import math

import pandas as pd
from chembl_webresource_client.new_client import new_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_CHEMBL_ID = "CHEMBL203"
STANDARD_TYPE = "IC50"
OUTPUT_PATH = "data/processed/egfr_bioactivity.csv"


# ---------------------------------------------------------------------------
# Fetch raw data from ChEMBL
# ---------------------------------------------------------------------------

def fetch_bioactivity(target_id: str = TARGET_CHEMBL_ID) -> pd.DataFrame:
    """
    Pull all IC50 records for a ChEMBL target via the REST API.

    The ChEMBL client handles pagination automatically -- you just filter.
    Returns a raw DataFrame with one row per bioactivity measurement.
    """
    logger.info("Fetching ChEMBL bioactivity for target %s ...", target_id)

    activity = new_client.activity
    records = activity.filter(
        target_chembl_id=target_id,
        standard_type=STANDARD_TYPE,
        standard_relation="=",
    ).only([
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "assay_chembl_id",
    ])

    df = pd.DataFrame.from_records(records)
    logger.info("Fetched %d raw records from ChEMBL", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 2: Curate the data
# ---------------------------------------------------------------------------

def curate(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Remove rows missing the two things we need most
    before = len(df)
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    logger.info("Removed %d rows with missing SMILES or IC50", before - len(df))

    # Convert IC50 to numeric (some values come as strings from the API)
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_value"] > 0]

    # Convert IC50 (nM) to pIC50
    # Formula: pIC50 = -log10(IC50_nM * 1e-9) = -log10(IC50_M)
    df["pIC50"] = df["standard_value"].apply(
        lambda x: -math.log10(x * 1e-9)
    )

    # Keep only physically meaningful range
    df = df[df["pIC50"].between(3, 12)]

    # One entry per compound
    before = len(df)
    df = df.drop_duplicates(subset=["molecule_chembl_id"])
    logger.info("Removed %d duplicate compounds", before - len(df))

    df = df.reset_index(drop=True)
    logger.info("Final curated dataset: %d compounds", len(df))
    return df


# ---------------------------------------------------------------------------
# Run the full pipeline
# ---------------------------------------------------------------------------

def run_ingestion(output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """Download, curate, and save the bioactivity dataset."""
    df_raw = fetch_bioactivity()
    df_clean = curate(df_raw)
    df_clean.to_csv(output_path, index=False)
    logger.info("Saved to %s", output_path)
    return df_clean


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    run_ingestion()