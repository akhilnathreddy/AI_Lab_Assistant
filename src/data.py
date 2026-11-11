"""Loading and cleaning the Buchwald-Hartwig C-N coupling dataset.

Source: Ahneman et al., "Predicting reaction performance in C-N cross-coupling
using machine learning", Science 360, 186-190 (2018). The SMILES-annotated table
used here is the copy distributed with the rxn_yields repository
(rxn4chemistry/rxn_yields, MIT licence), sheet ``Plates1-3`` which keeps the
original plate order.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_XLSX = RAW_DIR / "Dreher_and_Doyle_input_data.xlsx"
PROCESSED_CSV = PROCESSED_DIR / "buchwald_hartwig.csv"
RAW_URL = (
    "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/"
    "data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"
)

# The four condition variables (search-space dimensions).
COMPONENTS = ["ligand", "base", "additive", "aryl_halide"]
SMILES_COLS = {c: f"{c}_smiles" for c in COMPONENTS}

LIGAND_NAMES = {
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2": "XPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C=CC=C2": "t-BuXPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC": "t-BuBrettPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)"
    "[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC": "AdBrettPhos",
}
BASE_NAMES = {
    "CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC": "P2Et",
    "CC(C)(C)/N=C(N(C)C)/N(C)C": "BTMG",
    "CN1CCCN2C1=NCCC2": "MTBD",
}
ARYL_HALIDE_NAMES = {
    "ClC1=CC=C(C(F)(F)F)C=C1": "4-CF3-PhCl",
    "BrC1=CC=C(C(F)(F)F)C=C1": "4-CF3-PhBr",
    "IC1=CC=C(C(F)(F)F)C=C1": "4-CF3-PhI",
    "ClC1=CC=C(OC)C=C1": "4-OMe-PhCl",
    "BrC1=CC=C(OC)C=C1": "4-OMe-PhBr",
    "IC1=CC=C(OC)C=C1": "4-OMe-PhI",
    "ClC1=CC=C(CC)C=C1": "4-Et-PhCl",
    "BrC1=CC=C(CC)C=C1": "4-Et-PhBr",
    "IC1=CC=C(CC)C=C1": "4-Et-PhI",
    "ClC1=NC=CC=C1": "2-Cl-pyridine",
    "BrC1=NC=CC=C1": "2-Br-pyridine",
    "IC1=NC=CC=C1": "2-I-pyridine",
    "ClC1=CN=CC=C1": "3-Cl-pyridine",
    "BrC1=CN=CC=C1": "3-Br-pyridine",
    "IC1=CN=CC=C1": "3-I-pyridine",
}


def download_raw(force: bool = False) -> Path:
    """Download the raw spreadsheet into ``data/raw`` (skipped if present)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if force or not RAW_XLSX.exists():
        urllib.request.urlretrieve(RAW_URL, RAW_XLSX)
    return RAW_XLSX


def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, attach readable names, and validate the table."""
    df = raw.rename(
        columns={
            "Ligand": SMILES_COLS["ligand"],
            "Base": SMILES_COLS["base"],
            "Additive": SMILES_COLS["additive"],
            "Aryl halide": SMILES_COLS["aryl_halide"],
            "Output": "yield",
        }
    )[list(SMILES_COLS.values()) + ["yield"]].copy()

    df = df.dropna()
    df = df.drop_duplicates(subset=list(SMILES_COLS.values()))
    df["yield"] = df["yield"].astype(float).clip(0, 100)

    df["ligand"] = df[SMILES_COLS["ligand"]].map(LIGAND_NAMES)
    df["base"] = df[SMILES_COLS["base"]].map(BASE_NAMES)
    df["aryl_halide"] = df[SMILES_COLS["aryl_halide"]].map(ARYL_HALIDE_NAMES)
    # Additives (isoxazoles) have no short common names: number them A01-A22
    # in sorted-SMILES order so labels are stable across runs.
    additive_smiles = sorted(df[SMILES_COLS["additive"]].unique())
    additive_names = {s: f"A{i + 1:02d}" for i, s in enumerate(additive_smiles)}
    df["additive"] = df[SMILES_COLS["additive"]].map(additive_names)

    missing = df[COMPONENTS].isna().sum()
    if missing.any():
        raise ValueError(f"Unmapped SMILES found:\n{missing[missing > 0]}")

    cols = COMPONENTS + list(SMILES_COLS.values()) + ["yield"]
    return df[cols].sort_values(COMPONENTS).reset_index(drop=True)


def build_processed(force_download: bool = False) -> pd.DataFrame:
    """Download (if needed), clean and write ``data/processed``."""
    raw = pd.read_excel(download_raw(force_download), sheet_name="Plates1-3")
    df = clean_raw(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    return df


def load_dataset() -> pd.DataFrame:
    """Return the clean conditions -> yield table (one row per reaction)."""
    if PROCESSED_CSV.exists():
        return pd.read_csv(PROCESSED_CSV)
    return build_processed()


if __name__ == "__main__":
    data = build_processed()
    print(f"Wrote {len(data)} reactions to {PROCESSED_CSV}")
