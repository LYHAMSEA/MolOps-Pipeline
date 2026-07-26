from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

logger = logging.getLogger(__name__)

# Morgan fingerprint settings -- ECFP4 is the industry standard
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048


# ---------------------------------------------------------------------------
# Core featurisation functions
# ---------------------------------------------------------------------------

def smiles_to_mol(smiles: str) -> Chem.Mol | None:

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Could not parse SMILES: %s", smiles)
    return mol


def morgan_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """
    Compute an ECFP4 Morgan fingerprint.

    """
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS
    )
    return np.array(fp)


def physicochemical_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """
    Compute Lipinski/Veber physicochemical descriptors.

    """
    return {
        "MW": round(Descriptors.MolWt(mol), 3),
        "LogP": round(Descriptors.MolLogP(mol), 3),
        "HBD": int(Descriptors.NumHDonors(mol)),
        "HBA": int(Descriptors.NumHAcceptors(mol)),
        "TPSA": round(Descriptors.TPSA(mol), 3),
        "RotBonds": int(Descriptors.NumRotatableBonds(mol)),
        "RingCount": int(Descriptors.RingCount(mol)),
        "AromaticRings": int(Descriptors.NumAromaticRings(mol)),
        "HeavyAtoms": int(mol.GetNumHeavyAtoms()),
    }


def lipinski_pass(descriptors: dict[str, float]) -> bool:
    """
    Check Lipinski's Rule of Five for oral drug-likeness.
    
    """
    return (
        descriptors["MW"] <= 500
        and descriptors["LogP"] <= 5
        and descriptors["HBD"] <= 5
        and descriptors["HBA"] <= 10
    )


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    
    mol_a = smiles_to_mol(smiles_a)
    mol_b = smiles_to_mol(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, MORGAN_RADIUS, MORGAN_NBITS)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, MORGAN_RADIUS, MORGAN_NBITS)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


# ---------------------------------------------------------------------------
# Batch featurisation
# ---------------------------------------------------------------------------

def featurise_dataset(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
) -> tuple[np.ndarray, pd.DataFrame, list[int]]:
    
    fingerprints = []
    descriptor_rows = []
    valid_indices = []

    for i, smiles in enumerate(df[smiles_col]):
        mol = smiles_to_mol(smiles)
        if mol is None:
            continue
        fingerprints.append(morgan_fingerprint(mol))
        descriptor_rows.append(physicochemical_descriptors(mol))
        valid_indices.append(i)

    logger.info(
        "Featurised %d / %d molecules (%d invalid SMILES skipped)",
        len(valid_indices),
        len(df),
        len(df) - len(valid_indices),
    )

    fp_matrix = np.vstack(fingerprints)
    desc_df = pd.DataFrame(descriptor_rows)

    return fp_matrix, desc_df, valid_indices