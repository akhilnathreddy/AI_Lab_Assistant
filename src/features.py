"""Turning reaction conditions into numeric feature vectors.

Three encodings are provided:

* ``onehot``      - one indicator per component choice. Knows
                    nothing about chemistry: every pair of ligands is equally
                    different.
* ``descriptors`` - a small set of RDKit physico-chemical descriptors per
                    component, standardised across that component's choices and
                    concatenated. Compact and dense, which suits a GP.
* ``fingerprint`` - Morgan count fingerprints (radius 2) per component,
                    concatenated. High-dimensional; best with tree models.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from .data import components_of

RDLogger.DisableLog("rdApp.*")

ENCODINGS = ("onehot", "descriptors", "fingerprint")

DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "MolMR": Descriptors.MolMR,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "FractionCSP3": Descriptors.FractionCSP3,
    "RingCount": Descriptors.RingCount,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "LabuteASA": Descriptors.LabuteASA,
    "MaxPartialCharge": Descriptors.MaxPartialCharge,
    "MinPartialCharge": Descriptors.MinPartialCharge,
    "MaxAbsEStateIndex": Descriptors.MaxAbsEStateIndex,
    "BertzCT": Descriptors.BertzCT,
}


@lru_cache(maxsize=None)
def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return mol


def one_hot(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    blocks, names = [], []
    for comp in components_of(df):
        dummies = pd.get_dummies(df[comp], prefix=comp, dtype=float)
        blocks.append(dummies.to_numpy())
        names.extend(dummies.columns)
    return np.hstack(blocks), names


def _component_table(smiles: list[str], kind: str, n_bits: int) -> pd.DataFrame:
    """Feature table indexed by SMILES for one component's unique choices."""
    rows = {}
    if kind == "descriptors":
        for s in smiles:
            mol = _mol(s)
            rows[s] = [float(f(mol)) for f in DESCRIPTOR_FUNCS.values()]
        table = pd.DataFrame.from_dict(rows, orient="index", columns=list(DESCRIPTOR_FUNCS))
        # Partial charges are undefined (NaN) for some salts, e.g. CsF.
        table = table.dropna(axis=1)
        table = table.loc[:, table.std(ddof=0) > 1e-9]  # drop constant columns
        return (table - table.mean()) / table.std(ddof=0)
    if kind == "fingerprint":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        for s in smiles:
            rows[s] = gen.GetCountFingerprintAsNumPy(_mol(s)).astype(float)
        table = pd.DataFrame.from_dict(rows, orient="index")
        table = table.loc[:, table.std(ddof=0) > 0]  # keep informative bits
        return np.log1p(table)
    raise ValueError(f"Unknown kind {kind!r}")


def chemistry_features(
    df: pd.DataFrame, kind: str = "descriptors", n_bits: int = 512
) -> tuple[np.ndarray, list[str]]:
    blocks, names = [], []
    for comp in components_of(df):
        col = f"{comp}_smiles"
        table = _component_table(sorted(df[col].unique()), kind, n_bits)
        blocks.append(table.loc[df[col]].to_numpy())
        names.extend(f"{comp}:{c}" for c in table.columns)
    return np.hstack(blocks), names


def encode(df: pd.DataFrame, encoding: str = "onehot") -> tuple[np.ndarray, list[str]]:
    """Encode every row of ``df`` with the chosen encoding."""
    if encoding == "onehot":
        X, names = one_hot(df)
    elif encoding in ("descriptors", "fingerprint"):
        X, names = chemistry_features(df, kind=encoding)
    else:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
    return X.astype(np.float64), names
