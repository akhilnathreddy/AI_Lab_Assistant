"""Phase 2: is yield learnable? Random-forest baselines per encoding and split.

Splits
  * random       - 5 repeats of a random 70/30 split (optimistic)
  * additive-out - GroupKFold over additives: test additives never seen in training
  * halide-out   - GroupKFold over aryl halides: test substrates never seen

Writes results/baseline_metrics.md and results/figures/baseline_r2.png.
Run from the repository root:  python scripts/02_baseline_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold, ShuffleSplit  # noqa: E402

from src.data import load_dataset  # noqa: E402
from src.features import encode  # noqa: E402
from src.plotting import PALETTE, style_axes  # noqa: E402

ENCODING_LABELS = {"onehot": "One-hot", "descriptors": "RDKit descriptors",
                   "fingerprint": "Morgan fingerprints"}
SPLIT_LABELS = {"random": "Random 70/30", "additive": "Unseen additives",
                "aryl_halide": "Unseen aryl halides"}


def splits(df):
    yield from (("random", tr, te) for tr, te in
                ShuffleSplit(n_splits=5, test_size=0.3, random_state=0).split(df))
    for group in ("additive", "aryl_halide"):
        for tr, te in GroupKFold(n_splits=5).split(df, groups=df[group]):
            yield group, tr, te


def main():
    df = load_dataset()
    y = df["yield"].to_numpy()
    rows = []
    for enc in ENCODING_LABELS:
        X, _ = encode(df, enc)
        for split, tr, te in splits(df):
            rf = RandomForestRegressor(n_estimators=300, max_features=0.5,
                                       random_state=0, n_jobs=2).fit(X[tr], y[tr])
            pred = rf.predict(X[te])
            rows.append({"encoding": enc, "split": split, "r2": r2_score(y[te], pred),
                         "rmse": mean_squared_error(y[te], pred) ** 0.5})
            print(rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "baseline_metrics.csv", index=False)

    agg = res.groupby(["split", "encoding"], sort=False).agg(
        r2_median=("r2", "median"), r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std")).reset_index()
    table = pd.DataFrame({
        "Split": agg["split"].map(SPLIT_LABELS),
        "Encoding": agg["encoding"].map(ENCODING_LABELS),
        "R² (median)": [f"{m:.2f}" for m in agg.r2_median],
        "R² (mean ± std)": [f"{m:.2f} ± {s:.2f}" for m, s in zip(agg.r2_mean, agg.r2_std)],
        "RMSE (yield %)": [f"{m:.1f} ± {s:.1f}" for m, s in zip(agg.rmse_mean, agg.rmse_std)],
    })
    md = table.to_markdown(index=False)
    (ROOT / "results" / "baseline_metrics.md").write_text(
        "# Random-forest yield prediction\n\n"
        "Mean ± std over 5 folds (random: 5 repeated 70/30 splits; unseen-group "
        "splits: 5-fold GroupKFold).\n\n" + md + "\n")
    print(md)

    # Figure: median R² per split/encoding (bars) with every fold as a dot.
    # Per-fold R² can be very negative when a held-out group has little
    # variance, so the axis is clipped and clipped folds are drawn at the edge.
    plot_results(res, ROOT / "results" / "figures" / "baseline_r2.png")


def plot_results(res: pd.DataFrame, path: Path, floor: float = -1.0) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    splits_order = list(SPLIT_LABELS)
    width = 0.26
    rng = np.random.default_rng(0)
    for i, enc in enumerate(ENCODING_LABELS):
        for j, split in enumerate(splits_order):
            r2 = res[(res.encoding == enc) & (res.split == split)]["r2"].to_numpy()
            x = j + (i - 1) * width
            ax.bar(x, np.median(r2), width * 0.95, color=PALETTE[i + 1], alpha=0.75,
                   label=ENCODING_LABELS[enc] if j == 0 else None)
            jitter = rng.uniform(-width / 4, width / 4, len(r2))
            clipped = r2 < floor
            ax.scatter(x + jitter[~clipped], r2[~clipped], s=12, color="#222", zorder=3)
            ax.scatter(x + jitter[clipped], np.full(clipped.sum(), floor + 0.03), s=22,
                       marker="v", color="#222", zorder=3)
    ax.set_xticks(np.arange(len(splits_order)), [SPLIT_LABELS[s] for s in splits_order])
    ax.set_ylim(floor, 1.05)
    ax.axhline(0, color="#444", lw=0.8)
    style_axes(ax, "", "Test R²",
               "Random splits flatter the model (bar = median, dot = one fold, ▼ = below −1)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    main()
