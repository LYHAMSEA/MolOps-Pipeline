"""Unit tests for the data ingestion and curation pipeline."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from molops.pipeline.ingestion import curate


def make_raw_df(**overrides: object) -> pd.DataFrame:
    """Build a minimal raw ChEMBL-like DataFrame for testing."""
    defaults: dict[str, list] = {
        "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
        "canonical_smiles":   ["CC",       "CCO",      "CCC"],
        "standard_value":     [100.0,      1000.0,     10.0],
        "standard_units":     ["nM",        "nM",       "nM"],
        "pchembl_value":      [None,        None,       None],
        "assay_chembl_id":    ["A1",        "A2",       "A3"],
    }
    defaults.update(overrides)  # type: ignore[arg-type]
    return pd.DataFrame(defaults)


# ---------------------------------------------------------------------------
# Curation logic tests
# ---------------------------------------------------------------------------

def test_curation_removes_null_smiles() -> None:
    df = make_raw_df(canonical_smiles=["CC", None, "CCC"])
    result = curate(df)
    assert result["canonical_smiles"].notna().all()
    assert len(result) == 2


def test_curation_removes_null_ic50() -> None:
    df = make_raw_df(standard_value=[None, 100.0, 50.0])
    result = curate(df)
    assert len(result) == 2


def test_curation_removes_zero_ic50() -> None:
    """IC50 = 0 would give pIC50 = infinity -- must be removed."""
    df = make_raw_df(standard_value=[0.0, 100.0, 50.0])
    result = curate(df)
    assert (result["standard_value"] > 0).all()


def test_curation_computes_pic50_correctly() -> None:
    """1000 nM = 1 uM = 1e-6 M => pIC50 = -log10(1e-6) = 6.0"""
    df = make_raw_df(
        molecule_chembl_id=["A"],
        canonical_smiles=["CC"],
        standard_value=[1000.0],
        standard_units=["nM"],
        pchembl_value=[None],
        assay_chembl_id=["A1"],
    )
    result = curate(df)
    assert abs(result.iloc[0]["pIC50"] - 6.0) < 0.01


def test_curation_filters_unrealistic_pic50() -> None:
    """pIC50 below 3 or above 12 is physically impossible."""
    df = make_raw_df(
        standard_value=[
            1e9,
            100.0,
            1e-3,
        ]
    )
    result = curate(df)
    assert len(result) == 1


def test_curation_removes_duplicates() -> None:
    """Same ChEMBL ID measured twice -- keep only the first."""
    df = make_raw_df(
        molecule_chembl_id=["SAME", "SAME", "DIFF"],
        standard_value=[100.0, 200.0, 50.0],
    )
    result = curate(df)
    assert result["molecule_chembl_id"].nunique() == len(result)
    assert len(result) == 2


def test_curation_resets_index() -> None:
    df = make_raw_df()
    result = curate(df)
    assert list(result.index) == list(range(len(result)))


def test_curate_all_valid_input() -> None:
    """All three default rows should survive curation."""
    df = make_raw_df()
    result = curate(df)
    assert len(result) == 3
    assert "pIC50" in result.columns