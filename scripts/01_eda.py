"""Phase 1: exploratory data analysis of the Buchwald-Hartwig screen.

Writes figures to results/figures/ and a summary to results/eda_summary.md.
Run from the repository root:  python scripts/01_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data import COMPONENTS, load_dataset  # noqa: E402
from src.plotting import PALETTE, style_axes  # noqa: E402

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
LABELS = {"ligand": "Ligand", "base": "Base", "additive": "Additive", "aryl_halide": "Aryl halide"}


def main():
    df = load_dataset()
    y = df["yield"]
    n_full = np.prod([df[c].nunique() for c in COMPONENTS])
    top1 = y.quantile(0.99)

    lines = [
        "# EDA summary",
        "",
        f"- Reactions: **{len(df)}** (full factorial would be {n_full}; "
        f"{n_full - len(df)} combinations are absent from the published data)",
        f"- Choices: " + ", ".join(f"{df[c].nunique()} {LABELS[c].lower()}s" for c in COMPONENTS),
        f"- Missing values: {int(df.isna().sum().sum())}; duplicate condition sets: "
        f"{int(df.duplicated(COMPONENTS).sum())}",
        f"- Yield: mean {y.mean():.1f}%, median {y.median():.1f}%, max {y.max():.1f}%",
        f"- {(y == 0).mean():.1%} of reactions gave 0% yield",
        f"- Share above 50%: {(y > 50).mean():.1%}; above 80%: **{(y > 80).mean():.1%}** "
        f"({(y > 80).sum()} reactions); above 90%: **{(y > 90).mean():.1%}** ({(y > 90).sum()})",
        f"- Top-1% threshold: {top1:.1f}% yield ({(y >= top1).sum()} reactions)",
        "",
    ]

    # 1. Yield distribution -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y, bins=50, color=PALETTE[1], alpha=0.85)
    ax.axvline(80, color=PALETTE[2], ls="--", lw=1.5)
    ax.text(81, ax.get_ylim()[1] * 0.9, f"> 80%: {(y > 80).mean():.1%}\nof reactions",
            color=PALETTE[2], fontsize=9, va="top")
    style_axes(ax, "Yield (%)", "Number of reactions",
               "High yields are rare: most conditions give < 50%")
    fig.tight_layout()
    fig.savefig(FIG / "eda_yield_distribution.png", dpi=160)
    plt.close(fig)

    # 2. Yield by each variable --------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), gridspec_kw={"width_ratios": [4, 3, 15, 22]},
                             sharey=True)
    for ax, comp in zip(axes, ["ligand", "base", "aryl_halide", "additive"]):
        order = df.groupby(comp)["yield"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=comp, y="yield", order=order, ax=ax, color=PALETTE[1],
                    fliersize=1.5, linewidth=0.8, boxprops={"alpha": 0.6})
        style_axes(ax, LABELS[comp], "Yield (%)" if comp == "ligand" else "", "")
        ax.tick_params(axis="x", rotation=90, labelsize=8)
    fig.suptitle("Yield by condition variable (sorted by median)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "eda_yield_by_variable.png", dpi=160)
    plt.close(fig)

    # 3. Which variable matters most? Main-effect variance explained ---------
    total = ((y - y.mean()) ** 2).sum()
    eta = {}
    for comp in COMPONENTS:
        means = df.groupby(comp)["yield"].transform("mean")
        eta[LABELS[comp]] = ((means - y.mean()) ** 2).sum() / total
    eta = pd.Series(eta).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh(eta.index, eta.values, color=PALETTE[1])
    for i, v in enumerate(eta.values):
        ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, eta.max() * 1.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    style_axes(ax, "Share of yield variance explained (one-way, η²)", "",
               "Aryl halide dominates; additive comes second")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_variance_explained.png", dpi=160)
    plt.close(fig)
    lines += ["## Variance explained by each variable alone (η²)", ""]
    lines += [f"- {k}: {v:.1%}" for k, v in eta.sort_values(ascending=False).items()]
    lines += [f"- Sum of main effects: {eta.sum():.1%} → the remaining "
              f"{1 - eta.sum():.0%} comes from interactions, which is why a model is needed.", ""]

    # 4. Interaction heatmap: ligand x aryl halide ---------------------------
    pivot = df.pivot_table(index="aryl_halide", columns="ligand", values="yield", aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index,
                      pivot.mean().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "Mean yield (%)"},
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Ligand")
    ax.set_ylabel("Aryl halide")
    ax.set_title("Mean yield: ligand × aryl halide", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "eda_ligand_halide_heatmap.png", dpi=160)
    plt.close(fig)

    # Top reactions --------------------------------------------------------
    top = df.nlargest(10, "yield")[COMPONENTS + ["yield"]].round(1)
    lines += ["## Ten highest-yielding reactions", "", top.to_markdown(index=False), ""]
    (ROOT / "results" / "eda_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
