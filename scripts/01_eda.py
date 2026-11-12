"""Phase 1: exploratory data analysis of the Buchwald-Hartwig screen.

Writes figures to results/figures/ and a summary to results/eda_summary.md
(prefixed with the dataset name for non-default datasets).
Run from the repository root:
    python scripts/01_eda.py                          # Buchwald-Hartwig
    python scripts/01_eda.py --dataset suzuki_miyaura
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data import DATASETS, DEFAULT_DATASET, get_spec, load_dataset  # noqa: E402
from src.plotting import PALETTE, style_axes  # noqa: E402

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main(dataset: str = DEFAULT_DATASET):
    spec = get_spec(dataset)
    COMPONENTS, LABELS, pre = list(spec.components), spec.labels, spec.prefix
    df = load_dataset(dataset)
    y = df["yield"]
    n_full = np.prod([df[c].nunique() for c in COMPONENTS])
    top1 = y.quantile(0.99)

    lines = [
        f"# EDA summary: {spec.reaction} ({spec.citation})",
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
    fig.savefig(FIG / f"{pre}eda_yield_distribution.png", dpi=160)
    plt.close(fig)

    # 2. Yield by each variable --------------------------------------------
    plot_order = sorted(COMPONENTS, key=lambda c: df[c].nunique())
    widths = [max(df[c].nunique(), 3) for c in plot_order]
    fig, axes = plt.subplots(1, len(plot_order), figsize=(min(4 + 0.3 * sum(widths), 18), 5),
                             gridspec_kw={"width_ratios": widths}, sharey=True)
    for ax, comp in zip(axes, plot_order):
        order = df.groupby(comp)["yield"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=comp, y="yield", order=order, ax=ax, color=PALETTE[1],
                    fliersize=1.5, linewidth=0.8, boxprops={"alpha": 0.6})
        style_axes(ax, LABELS[comp], "Yield (%)" if comp == plot_order[0] else "", "")
        ax.tick_params(axis="x", rotation=90, labelsize=8)
    fig.suptitle("Yield by condition variable (sorted by median)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"{pre}eda_yield_by_variable.png", dpi=160)
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
    ranked = eta.sort_values(ascending=False).index
    style_axes(ax, "Share of yield variance explained (one-way, η²)", "",
               f"{ranked[0]} matters most; {ranked[1].lower()} comes second")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / f"{pre}eda_variance_explained.png", dpi=160)
    plt.close(fig)
    lines += ["## Variance explained by each variable alone (η²)", ""]
    lines += [f"- {k}: {v:.1%}" for k, v in eta.sort_values(ascending=False).items()]
    lines += [f"- Sum of main effects: {eta.sum():.1%} → the remaining "
              f"{1 - eta.sum():.0%} comes from interactions, which is why a model is needed.", ""]

    # 4. Interaction heatmap: most important variable x ligand -------------
    by_label = {LABELS[c]: c for c in COMPONENTS}
    rows = by_label[ranked[0]]
    cols = "ligand" if rows != "ligand" else by_label[ranked[1]]
    pivot = df.pivot_table(index=rows, columns=cols, values="yield", aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index,
                      pivot.mean().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(max(6.5, 0.7 * pivot.shape[1] + 3), max(4, 0.4 * pivot.shape[0] + 1.5)))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "Mean yield (%)"},
                linewidths=0.5, ax=ax)
    ax.set_xlabel(LABELS[cols])
    ax.set_ylabel(LABELS[rows])
    ax.set_title(f"Mean yield: {LABELS[cols].lower()} × {LABELS[rows].lower()}", loc="left")
    fig.tight_layout()
    heatmap_name = "eda_ligand_halide_heatmap.png" if not pre else "eda_heatmap.png"
    fig.savefig(FIG / f"{pre}{heatmap_name}", dpi=160)
    plt.close(fig)

    # Top reactions --------------------------------------------------------
    top = df.nlargest(10, "yield")[COMPONENTS + ["yield"]].round(1)
    lines += ["## Ten highest-yielding reactions", "", top.to_markdown(index=False), ""]
    (ROOT / "results" / f"{pre}eda_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 EDA")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    main(parser.parse_args().dataset)
