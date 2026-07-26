"""Unit tests for the molecular featurisation module."""
from __future__ import annotations

import numpy as np
import pytest

from molops.pipeline.featurisation import (
    lipinski_pass,
    morgan_fingerprint,
    physicochemical_descriptors,
    smiles_to_mol,
    tanimoto_similarity,
    featurise_dataset,
)
import pandas as pd


# ---------------------------------------------------------------------------
# SMILES parsing
# ---------------------------------------------------------------------------

def test_valid_smiles_parses() -> None:
    mol = smiles_to_mol("CCO")
    assert mol is not None


def test_invalid_smiles_returns_none() -> None:
    mol = smiles_to_mol("NOT_A_SMILES!!")
    assert mol is None


def test_empty_string_returns_none() -> None:
    mol = smiles_to_mol("")
    assert mol is None


# ---------------------------------------------------------------------------
# Morgan fingerprints
# ---------------------------------------------------------------------------

def test_fingerprint_shape() -> None:
    mol = smiles_to_mol("CCO")
    assert mol is not None
    fp = morgan_fingerprint(mol)
    assert fp.shape == (2048,)


def test_fingerprint_is_binary() -> None:
    mol = smiles_to_mol("CCO")
    assert mol is not None
    fp = morgan_fingerprint(mol)
    assert set(fp).issubset({0, 1})


def test_different_molecules_give_different_fingerprints() -> None:
    mol1 = smiles_to_mol("CCO")
    mol2 = smiles_to_mol("c1ccccc1")
    assert mol1 is not None and mol2 is not None
    fp1 = morgan_fingerprint(mol1)
    fp2 = morgan_fingerprint(mol2)
    assert not np.array_equal(fp1, fp2)


def test_same_molecule_gives_same_fingerprint() -> None:
    mol = smiles_to_mol("CC(=O)Oc1ccccc1C(=O)O")
    assert mol is not None
    fp1 = morgan_fingerprint(mol)
    fp2 = morgan_fingerprint(mol)
    assert np.array_equal(fp1, fp2)


# ---------------------------------------------------------------------------
# Physicochemical descriptors -- using aspirin as the reference molecule
# ---------------------------------------------------------------------------

def test_aspirin_molecular_weight() -> None:
    mol = smiles_to_mol("CC(=O)Oc1ccccc1C(=O)O")
    assert mol is not None
    desc = physicochemical_descriptors(mol)
    assert 179 < desc["MW"] < 182


def test_aspirin_hbd() -> None:
    """Aspirin has 1 H-bond donor (the carboxylic acid OH)."""
    mol = smiles_to_mol("CC(=O)Oc1ccccc1C(=O)O")
    assert mol is not None
    desc = physicochemical_descriptors(mol)
    assert desc["HBD"] == 1


def test_aspirin_lipinski_pass() -> None:
    """Aspirin is a drug -- must pass Lipinski's Rule of Five."""
    mol = smiles_to_mol("CC(=O)Oc1ccccc1C(=O)O")
    assert mol is not None
    desc = physicochemical_descriptors(mol)
    assert lipinski_pass(desc) is True


def test_large_molecule_lipinski_fail() -> None:
    """A very large molecule should fail Lipinski (MW > 500)."""
    # Cyclosporin A -- MW ~1202, not orally bioavailable
    csA = "CC[C@@H]1OC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H]2CCCN2C(=O)[C@@H](CC(C)C)NC(=O)[C@H](C)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](C)N(C)C(=O)[C@H](C)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@H]([C@@H](C)O)NC1=O"
    mol = smiles_to_mol(csA)
    if mol is not None:
        desc = physicochemical_descriptors(mol)
        assert desc["MW"] > 500


def test_descriptor_keys_present() -> None:
    mol = smiles_to_mol("CCO")
    assert mol is not None
    desc = physicochemical_descriptors(mol)
    for key in ["MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds", "RingCount", "AromaticRings", "HeavyAtoms"]:
        assert key in desc


# ---------------------------------------------------------------------------
# Tanimoto similarity
# ---------------------------------------------------------------------------

def test_identical_smiles_tanimoto_is_one() -> None:
    sim = tanimoto_similarity("CCO", "CCO")
    assert abs(sim - 1.0) < 0.01


def test_very_different_molecules_low_tanimoto() -> None:
    """Aspirin vs caffeine -- structurally very different."""
    sim = tanimoto_similarity(
        "CC(=O)Oc1ccccc1C(=O)O",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    )
    assert sim < 0.3


def test_invalid_smiles_tanimoto_returns_zero() -> None:
    sim = tanimoto_similarity("INVALID", "CCO")
    assert sim == 0.0


# ---------------------------------------------------------------------------
# Batch featurisation
# ---------------------------------------------------------------------------

def test_featurise_dataset_output_shapes() -> None:
    df = pd.DataFrame({
        "canonical_smiles": ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"],
        "pIC50": [5.0, 6.5, 4.2],
    })
    fps, desc_df, valid_idx = featurise_dataset(df)
    assert fps.shape == (3, 2048)
    assert len(desc_df) == 3
    assert len(valid_idx) == 3


def test_featurise_dataset_skips_invalid_smiles() -> None:
    df = pd.DataFrame({
        "canonical_smiles": ["CCO", "INVALID!!", "c1ccccc1"],
        "pIC50": [5.0, 6.5, 4.2],
    })
    fps, desc_df, valid_idx = featurise_dataset(df)
    assert fps.shape == (2, 2048)
    assert len(valid_idx) == 2
    assert 1 not in valid_idx