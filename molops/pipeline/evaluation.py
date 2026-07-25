from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

AD_THRESHOLD = 0.4   # minimum Tanimoto to be considered "within domain"


def compute_training_fingerprints(smiles_list: list[str]) -> list[DataStructs.ExplicitBitVect]:
    
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
    logger.info("Computed %d training fingerprints for AD", len(fps))
    return fps


def check_applicability_domain(
    query_smiles: str,
    training_fps: list[DataStructs.ExplicitBitVect],
    threshold: float = AD_THRESHOLD,
) -> tuple[float, bool]:
    """
    Check whether a query molecule is within the model's applicability domain.

    """
    mol = Chem.MolFromSmiles(query_smiles)
    if mol is None:
        return 0.0, False

    query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, training_fps)

    if not similarities:
        return 0.0, False

    max_sim = float(max(similarities))
    within_ad = max_sim >= threshold

    if not within_ad:
        logger.warning(
            "Query molecule is OUTSIDE applicability domain "
            "(max Tanimoto=%.3f < threshold=%.1f)",
            max_sim, threshold,
        )

    return max_sim, within_ad