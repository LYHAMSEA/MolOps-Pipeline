"""Unit tests for the applicability domain evaluation module."""
from __future__ import annotations

import pytest

from molops.pipeline.evaluation import (
    AD_THRESHOLD,
    check_applicability_domain,
    compute_training_fingerprints,
)


TRAINING_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "c1ccc(NC(=O)c2ccccc2)cc1",
]


def test_compute_training_fingerprints_count() -> None:
    fps = compute_training_fingerprints(TRAINING_SMILES)
    assert len(fps) == len(TRAINING_SMILES)


def test_compute_training_fingerprints_skips_invalid() -> None:
    fps = compute_training_fingerprints(["CCO", "INVALID!!", "c1ccccc1"])
    assert len(fps) == 2


def test_similar_molecule_within_ad() -> None:
    """Salicylic acid is very similar to aspirin -- should be within AD."""
    training_fps = compute_training_fingerprints(TRAINING_SMILES)
    salicylic_acid = "OC(=O)c1ccccc1O"
    max_tan, within = check_applicability_domain(salicylic_acid, training_fps)
    assert within is True
    assert max_tan >= AD_THRESHOLD


def test_dissimilar_molecule_outside_ad() -> None:
    """A complex steroid is nothing like NSAIDs -- should be outside AD."""
    training_fps = compute_training_fingerprints(TRAINING_SMILES)
    steroid = "C1CC2CCCC3CC(=O)CCC3(C2(C1)C)C"
    max_tan, within = check_applicability_domain(steroid, training_fps)
    # Max tanimoto to NSAID training set should be low
    assert max_tan < 0.5


def test_invalid_smiles_returns_false() -> None:
    training_fps = compute_training_fingerprints(TRAINING_SMILES)
    max_tan, within = check_applicability_domain("NOT_VALID!!", training_fps)
    assert within is False
    assert max_tan == 0.0


def test_empty_training_set_returns_false() -> None:
    max_tan, within = check_applicability_domain("CCO", [])
    assert within is False
    assert max_tan == 0.0


def test_max_tanimoto_between_zero_and_one() -> None:
    training_fps = compute_training_fingerprints(TRAINING_SMILES)
    max_tan, _ = check_applicability_domain("CCO", training_fps)
    assert 0.0 <= max_tan <= 1.0