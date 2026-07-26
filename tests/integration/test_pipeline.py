"""
Integration tests -- the full pipeline from raw data to predictions.

"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from molops.pipeline.featurisation import featurise_dataset
from molops.pipeline.ingestion import curate
from molops.pipeline.evaluation import check_applicability_domain, compute_training_fingerprints


KNOWN_DRUGS = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "c1ccc(NC(=O)c2ccccc2)cc1",
    "O=C(O)c1ccccc1O",
    "CC(O)c1ccc(Cl)cc1",
    "c1ccc2c(c1)ccc1ccccc12",
]


def make_synthetic_chembl_df() -> pd.DataFrame:
    """Build a realistic-looking DataFrame with known drug SMILES."""
    rows = []
    for i, smi in enumerate(KNOWN_DRUGS):
        ic50_nm = 10 ** (9 - (i % 5))
        rows.append({
            "molecule_chembl_id": f"CHEMBL{1000 + i}",
            "canonical_smiles": smi,
            "standard_value": ic50_nm,
            "standard_units": "nM",
            "pchembl_value": None,
            "assay_chembl_id": f"ASSAY{i}",
        })
    return pd.DataFrame(rows)


@pytest.mark.integration
def test_full_curation_pipeline() -> None:
    """Curation should produce a clean DataFrame with pIC50 values."""
    raw = make_synthetic_chembl_df()
    clean = curate(raw)
    assert len(clean) > 0
    assert "pIC50" in clean.columns
    assert clean["pIC50"].between(3, 12).all()
    assert clean["canonical_smiles"].notna().all()


@pytest.mark.integration
def test_curation_to_featurisation() -> None:
    """Curated data should featurise without errors."""
    raw = make_synthetic_chembl_df()
    clean = curate(raw)
    fps, desc_df, valid_idx = featurise_dataset(clean)
    assert fps.shape[1] == 2048
    assert len(fps) == len(valid_idx)
    assert len(desc_df) == len(valid_idx)
    assert fps.shape[0] > 0


@pytest.mark.integration
def test_featurised_data_shapes_consistent() -> None:
    """Fingerprint matrix rows must equal pIC50 values count."""
    raw = make_synthetic_chembl_df()
    clean = curate(raw)
    fps, desc_df, valid_idx = featurise_dataset(clean)
    y = clean.iloc[valid_idx]["pIC50"].values
    assert fps.shape[0] == len(y)


@pytest.mark.integration
def test_applicability_domain_on_known_drugs() -> None:
    """
    If we train on aspirin and paracetamol, salicylic acid (very similar
    to aspirin) should be within AD, caffeine should be outside or borderline.
    """
    training = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1"]
    training_fps = compute_training_fingerprints(training)

    sal_acid = "O=C(O)c1ccccc1O"
    max_tan, within = check_applicability_domain(sal_acid, training_fps)
    assert within is True, f"Salicylic acid should be within AD (Tanimoto={max_tan:.3f})"