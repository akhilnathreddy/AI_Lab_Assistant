"""Loading and cleaning the reaction-yield datasets.

Two fully-enumerated high-throughput screens are supported:

``buchwald_hartwig`` (default)
    Ahneman et al., "Predicting reaction performance in C-N cross-coupling
    using machine learning", Science 360, 186-190 (2018). 3,955 reactions:
    4 ligands x 3 bases x 22 additives x 15 aryl halides. SMILES-annotated copy
    from the rxn_yields repository (rxn4chemistry/rxn_yields, MIT licence),
    sheet ``Plates1-3``.

``suzuki_miyaura``
    Perera et al., "A platform for automated nanomole-scale reaction screening
    and micromole-scale synthesis in flow", Science 359, 429-434 (2018).
    3,696 reactions: 4 electrophiles x 3 boron nucleophiles x 11 ligands x
    7 bases x 4 solvents. Component table from the EDBO repository
    (b-shields/edbo, MIT licence) used by Shields et al., Nature (2021); the
    ligand-free and base-free controls of the original screen are excluded.

Every processed table has one column per component (readable name), one
``<component>_smiles`` column per component and a ``yield`` column.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Buchwald-Hartwig (Ahneman et al. 2018)
# ---------------------------------------------------------------------------
BH_LIGANDS = {
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2": "XPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C=CC=C2": "t-BuXPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC": "t-BuBrettPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)"
    "[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC": "AdBrettPhos",
}
BH_BASES = {
    "CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC": "P2Et",
    "CC(C)(C)/N=C(N(C)C)/N(C)C": "BTMG",
    "CN1CCCN2C1=NCCC2": "MTBD",
}
BH_ARYL_HALIDES = {
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
# Kept for backwards compatibility: names used across the Buchwald-Hartwig code.
LIGAND_NAMES, BASE_NAMES, ARYL_HALIDE_NAMES = BH_LIGANDS, BH_BASES, BH_ARYL_HALIDES


def _clean_buchwald_hartwig(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Plates1-3").rename(columns={
        "Ligand": "ligand_smiles", "Base": "base_smiles", "Additive": "additive_smiles",
        "Aryl halide": "aryl_halide_smiles", "Output": "yield",
    })
    # Additives (isoxazoles) have no short common names: number them A01-A22
    # in sorted-SMILES order so labels are stable across runs.
    additives = sorted(raw["additive_smiles"].dropna().unique())
    names = {
        "ligand": BH_LIGANDS, "base": BH_BASES, "aryl_halide": BH_ARYL_HALIDES,
        "additive": {s: f"A{i + 1:02d}" for i, s in enumerate(additives)},
    }
    return _finish(raw, ["ligand", "base", "additive", "aryl_halide"], names)


# ---------------------------------------------------------------------------
# Suzuki-Miyaura (Perera et al. 2018, as tabulated by Shields et al. 2021)
# ---------------------------------------------------------------------------
SM_ELECTROPHILES = {
    "ClC1=CC=C(N=CC=C2)C2=C1": "6-Cl-quinoline",
    "BrC1=CC=C(N=CC=C2)C2=C1": "6-Br-quinoline",
    "IC1=CC=C(N=CC=C2)C2=C1": "6-I-quinoline",
    "O=S(OC1=CC=C(N=CC=C2)C2=C1)(C(F)(F)F)=O": "6-OTf-quinoline",
}
SM_NUCLEOPHILES = {
    "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B(O)O": "boronic acid",
    "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B4OC(C)(C)C(C)(C)O4": "Bpin ester",
    "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1[B-](F)(F)F": "trifluoroborate",
}
SM_LIGANDS = {
    "CC(P(C(C)(C)C)C(C)(C)C)(C)C": "P(t-Bu)3",
    "P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3": "PPh3",
    "CC(C)(C)P(C(C)(C)C)C1=CC=C(N(C)C)C=C1": "AmPhos",
    "P(C1CCCCC1)(C2CCCCC2)C3CCCCC3": "PCy3",
    "CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C": "P(o-Tol)3",
    "CCCCP(C12C[C@@H]3C[C@@H](C[C@H](C2)C3)C1)C45C[C@H]6C[C@@H](C5)C[C@@H](C4)C6": "CataCXium A",
    "COC1=CC=CC(OC)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2": "SPhos",
    "CC(C)(P(C(C)(C)C)[c-]1cccc1)C.CC(C)(P(C(C)(C)C)[c-]2cccc2)C.[Fe+2]": "dtbpf",
    "CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C": "XPhos",
    "[c-]1(P(C2=CC=CC=C2)C3=CC=CC=C3)cccc1.[c-]4(P(C5=CC=CC=C5)C6=CC=CC=C6)cccc4.[Fe+2]": "dppf",
    "CC1(C)C2=C(OC3=C1C=CC=C3P(C4=CC=CC=C4)C5=CC=CC=C5)C(P(C6=CC=CC=C6)C7=CC=CC=C7)=CC=C2": "XantPhos",
}
SM_BASES = {
    "[Na+].[OH-]": "NaOH",
    "OC([O-])=O.[Na+]": "NaHCO3",
    "[Cs+].[F-]": "CsF",
    "O=P([O-])([O-])[O-].[K+].[K+].[K+]": "K3PO4",
    "[K+].[OH-]": "KOH",
    # Perera et al. used LiOt-Bu; the EDBO table encodes the alkoxide as
    # CC([O-])C. The SMILES is kept exactly as published.
    "CC([O-])C.[Li+]": "LiOt-Bu",
    "CCN(CC)CC": "Et3N",
}
SM_SOLVENTS = {"N#CC": "MeCN", "C1COCC1": "THF", "O=CN(C)C": "DMF", "CO": "MeOH"}


def _clean_suzuki_miyaura(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path).rename(columns={
        "Electrophile_SMILES": "electrophile_smiles", "Nucleophile_SMILES": "nucleophile_smiles",
        "Ligand_SMILES": "ligand_smiles", "Base_SMILES": "base_smiles",
        "Solvent_SMILES": "solvent_smiles",
    })
    names = {"electrophile": SM_ELECTROPHILES, "nucleophile": SM_NUCLEOPHILES,
             "ligand": SM_LIGANDS, "base": SM_BASES, "solvent": SM_SOLVENTS}
    return _finish(raw, list(names), names)


# ---------------------------------------------------------------------------
# Shared cleaning + registry
# ---------------------------------------------------------------------------
def _finish(raw: pd.DataFrame, components: list[str], names: dict[str, dict]) -> pd.DataFrame:
    """Drop bad rows, attach readable names, validate, and order columns."""
    smiles_cols = [f"{c}_smiles" for c in components]
    df = raw[smiles_cols + ["yield"]].dropna().drop_duplicates(subset=smiles_cols).copy()
    df["yield"] = df["yield"].astype(float).clip(0, 100)
    for comp in components:
        df[comp] = df[f"{comp}_smiles"].map(names[comp])
    missing = df[components].isna().sum()
    if missing.any():
        raise ValueError(f"Unmapped SMILES found:\n{missing[missing > 0]}")
    return df[components + smiles_cols + ["yield"]].sort_values(components).reset_index(drop=True)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    reaction: str
    components: tuple[str, ...]
    labels: dict[str, str]
    url: str
    raw_file: str
    cleaner: Callable[[Path], pd.DataFrame] = field(repr=False)
    citation: str = ""

    @property
    def raw_path(self) -> Path:
        return RAW_DIR / self.raw_file

    @property
    def processed_path(self) -> Path:
        return PROCESSED_DIR / f"{self.name}.csv"

    @property
    def prefix(self) -> str:
        """Prefix for result files ('' for the default dataset)."""
        return "" if self.name == DEFAULT_DATASET else f"{self.name}_"


DATASETS: dict[str, DatasetSpec] = {
    "buchwald_hartwig": DatasetSpec(
        name="buchwald_hartwig",
        title="Buchwald–Hartwig",
        reaction="Buchwald–Hartwig C–N coupling",
        components=("ligand", "base", "additive", "aryl_halide"),
        labels={"ligand": "Ligand", "base": "Base", "additive": "Additive",
                "aryl_halide": "Aryl halide"},
        url="https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/"
            "data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx",
        raw_file="Dreher_and_Doyle_input_data.xlsx",
        cleaner=_clean_buchwald_hartwig,
        citation="Ahneman et al., Science 2018",
    ),
    "suzuki_miyaura": DatasetSpec(
        name="suzuki_miyaura",
        title="Suzuki–Miyaura",
        reaction="Suzuki–Miyaura C–C coupling",
        components=("electrophile", "nucleophile", "ligand", "base", "solvent"),
        labels={"electrophile": "Electrophile", "nucleophile": "Boron reagent",
                "ligand": "Ligand", "base": "Base", "solvent": "Solvent"},
        url="https://raw.githubusercontent.com/b-shields/edbo/master/"
            "experiments/data/suzuki/experiment_index.csv",
        raw_file="suzuki_experiment_index.csv",
        cleaner=_clean_suzuki_miyaura,
        citation="Perera et al., Science 2018",
    ),
}
DEFAULT_DATASET = "buchwald_hartwig"

# Backwards-compatible module constants for the default dataset.
COMPONENTS = list(DATASETS[DEFAULT_DATASET].components)
SMILES_COLS = {c: f"{c}_smiles" for c in COMPONENTS}
PROCESSED_CSV = DATASETS[DEFAULT_DATASET].processed_path


def get_spec(name: str = DEFAULT_DATASET) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}") from None


def components_of(df: pd.DataFrame) -> list[str]:
    """Component columns of a processed table (those with a ``*_smiles`` twin)."""
    return [c for c in df.columns if f"{c}_smiles" in df.columns]


def download_raw(name: str = DEFAULT_DATASET, force: bool = False) -> Path:
    """Download the raw file into ``data/raw`` (skipped if present)."""
    spec = get_spec(name)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if force or not spec.raw_path.exists():
        urllib.request.urlretrieve(spec.url, spec.raw_path)
    return spec.raw_path


def build_processed(name: str = DEFAULT_DATASET, force_download: bool = False) -> pd.DataFrame:
    """Download (if needed), clean and write ``data/processed/<name>.csv``."""
    spec = get_spec(name)
    df = spec.cleaner(download_raw(name, force_download))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(spec.processed_path, index=False)
    return df


def load_dataset(name: str = DEFAULT_DATASET) -> pd.DataFrame:
    """Return the clean conditions -> yield table (one row per reaction)."""
    spec = get_spec(name)
    if spec.processed_path.exists():
        return pd.read_csv(spec.processed_path)
    return build_processed(name)


if __name__ == "__main__":
    import sys

    for ds in sys.argv[1:] or list(DATASETS):
        data = build_processed(ds)
        print(f"Wrote {len(data)} reactions to {get_spec(ds).processed_path}")
