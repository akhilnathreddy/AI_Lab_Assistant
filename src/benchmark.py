"""Repeated-seed benchmarking of optimization strategies.

Usage (from the repository root)::

    python -m src.benchmark --suite main --seeds 30 --budget 100
    python -m src.benchmark --suite all  --seeds 30

Each suite writes ``results/runs/<suite>.csv`` (one row per experiment of
every run) and ``results/figures/<suite>_*.png`` plus a metrics table
``results/<suite>_metrics.md``.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_dataset
from .features import encode
from .optimizers import BayesOpt, RandomSearch, Strategy, run_campaign
from .virtual_lab import VirtualLab

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results" / "runs"
FIG_DIR = ROOT / "results" / "figures"
RESULTS_DIR = ROOT / "results"

TOP_FRACTION = 0.01  # "top 1 %" target


@dataclass(frozen=True)
class MethodSpec:
    label: str
    strategy: Strategy
    encoding: str = "onehot"
    batch_size: int = 1
    noise_sd: float = 0.0


def suites() -> dict[str, list[MethodSpec]]:
    gp_ei = lambda label: BayesOpt("gp", "ei", label=label)  # noqa: E731
    return {
        # Phase 4: the headline comparison.
        "main": [
            MethodSpec("Random search", RandomSearch()),
            MethodSpec("BO: GP + EI", gp_ei("BO: GP + EI")),
            MethodSpec("BO: RF + EI", BayesOpt("rf", "ei", label="BO: RF + EI")),
        ],
        # Phase 5a: does chemical knowledge in the features help the GP?
        "encodings": [
            MethodSpec("GP-EI, one-hot", gp_ei("GP-EI, one-hot"), "onehot"),
            MethodSpec("GP-EI, RDKit descriptors", gp_ei("GP-EI, RDKit descriptors"), "descriptors"),
            MethodSpec("GP-EI, Morgan fingerprints", gp_ei("GP-EI, Morgan fingerprints"), "fingerprint"),
        ],
        # Phase 5b: plates of 5 in parallel (kriging believer) vs one at a time.
        "batch": [
            MethodSpec("Sequential (q=1)", gp_ei("Sequential (q=1)")),
            MethodSpec("Batch of 5 (kriging believer)", gp_ei("Batch of 5 (kriging believer)"), batch_size=5),
            MethodSpec("Random, batch of 5", RandomSearch(), batch_size=5),
        ],
        # Phase 5d: exploration vs exploitation.
        "acquisition": [
            MethodSpec("Greedy (mean only)", BayesOpt("gp", "greedy", label="Greedy (mean only)")),
            MethodSpec("UCB β=0.1", BayesOpt("gp", "ucb", beta=0.1, label="UCB β=0.1")),
            MethodSpec("UCB β=1", BayesOpt("gp", "ucb", beta=1.0, label="UCB β=1")),
            MethodSpec("UCB β=4", BayesOpt("gp", "ucb", beta=4.0, label="UCB β=4")),
            MethodSpec("UCB β=16", BayesOpt("gp", "ucb", beta=16.0, label="UCB β=16")),
            MethodSpec("EI", gp_ei("EI")),
        ],
        # Extra: what happens when "measurements" are noisy, as in a real lab?
        "noise": [
            MethodSpec("GP-EI, exact yields", gp_ei("GP-EI, exact yields")),
            MethodSpec("GP-EI, ±5% noise", gp_ei("GP-EI, ±5% noise"), noise_sd=5.0),
            MethodSpec("GP-EI, ±10% noise", gp_ei("GP-EI, ±10% noise"), noise_sd=10.0),
            MethodSpec("Random search", RandomSearch()),
        ],
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
_DATA: pd.DataFrame | None = None
_X: dict[str, np.ndarray] = {}


def _init_worker():
    global _DATA, _LIMITS
    # One BLAS thread per worker: otherwise every process spawns a thread per
    # core and the machine thrashes.
    from threadpoolctl import threadpool_limits
    _LIMITS = threadpool_limits(1)
    _DATA = load_dataset()


def _features(encoding: str) -> np.ndarray:
    if encoding not in _X:
        _X[encoding] = encode(_DATA, encoding)[0]
    return _X[encoding]


def _run_one(args) -> pd.DataFrame:
    spec, seed, budget, n_init = args
    lab = VirtualLab(_DATA, noise_sd=spec.noise_sd, seed=seed)
    run_campaign(lab, _features(spec.encoding), spec.strategy, budget=budget,
                 n_init=n_init, batch_size=spec.batch_size, seed=seed)
    idx, measured = lab.observed()
    true_y = _DATA["yield"].to_numpy()[idx]
    return pd.DataFrame({
        "method": spec.label, "seed": seed,
        "experiment": np.arange(1, len(idx) + 1), "row": idx,
        "measured_yield": measured, "yield": true_y,
        "best_so_far": np.maximum.accumulate(true_y),
    })


def run_suite(name: str, specs: list[MethodSpec], seeds: int, budget: int,
              n_init: int, processes: int) -> pd.DataFrame:
    jobs = [(s, seed, budget, n_init) for s in specs for seed in range(seeds)]
    t0 = time.time()
    with Pool(processes, initializer=_init_worker) as pool:
        frames = pool.map(_run_one, jobs, chunksize=1)
    df = pd.concat(frames, ignore_index=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUNS_DIR / f"{name}.csv", index=False)
    print(f"[{name}] {len(jobs)} runs in {time.time() - t0:.0f}s")
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def experiments_to_target(runs: pd.DataFrame, threshold: float) -> pd.Series:
    """First experiment (1-based) whose yield reaches ``threshold``; NaN if never."""
    hit = runs[runs["yield"] >= threshold]
    first = hit.groupby(["method", "seed"])["experiment"].min()
    all_runs = runs.groupby(["method", "seed"]).size().index
    return first.reindex(all_runs)


def metrics_table(runs: pd.DataFrame, threshold: float,
                  checkpoints=(10, 20, 50, 100)) -> pd.DataFrame:
    budget = runs["experiment"].max()
    rows = []
    to_target = experiments_to_target(runs, threshold)
    for method, g in runs.groupby("method", sort=False):
        row = {"method": method}
        for c in checkpoints:
            if c <= budget:
                best = g[g["experiment"] == c]["best_so_far"]
                row[f"best@{c}"] = f"{best.mean():.1f} ± {best.std():.1f}"
        t = to_target.loc[method]
        row["found top-1% (runs)"] = f"{t.notna().mean():.0%}"
        # Median is well defined as long as at least half the runs succeed.
        filled = t.fillna(np.inf)
        med = np.median(filled)
        row["median exps to top-1%"] = f"{med:.0f}" if np.isfinite(med) else f">{budget}"
        rows.append(row)
    return pd.DataFrame(rows)


def random_search_expectation(n_total: int, n_good: int) -> float:
    """Expected draws (without replacement) until the first 'good' item."""
    return (n_total + 1) / (n_good + 1)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_suite(runs: pd.DataFrame, name: str, threshold: float, title: str,
               global_max: float) -> None:
    import matplotlib.pyplot as plt

    from .plotting import PALETTE, style_axes

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    methods = list(dict.fromkeys(runs["method"]))
    budget = runs["experiment"].max()
    to_target = experiments_to_target(runs, threshold)
    for i, m in enumerate(methods):
        color = PALETTE[i % len(PALETTE)]
        g = runs[runs["method"] == m].pivot(index="experiment", columns="seed", values="best_so_far")
        x = g.index.to_numpy()
        axes[0].plot(x, g.mean(axis=1), color=color, lw=2, label=m)
        axes[0].fill_between(x, g.quantile(0.25, axis=1), g.quantile(0.75, axis=1),
                             color=color, alpha=0.15, lw=0)
        t = to_target.loc[m].to_numpy()
        frac = [(t <= k).mean() for k in x]
        axes[1].step(x, frac, where="post", color=color, lw=2, label=m)

    # Reference lines, labelled at the left where the curves are still low.
    for yref, text, ls in ((global_max, "best in dataset", ":"), (threshold, "top 1%", "--")):
        axes[0].axhline(yref, color="#888", ls=ls, lw=1)
        axes[0].text(budget * 0.02, yref + 0.4, text, va="bottom", ha="left",
                     fontsize=8, color="#666")
    style_axes(axes[0], "Experiments run", "Best yield found so far (%)",
               "Mean best yield (band = interquartile range)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    style_axes(axes[1], "Experiments run", "Share of runs",
               "Share of runs that have found a top-1% reaction")
    axes[1].set_ylim(0, 1.02)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_benchmark.png", dpi=160)
    plt.close(fig)


TITLES = {
    "main": "Bayesian optimization vs random search (Buchwald–Hartwig, 3,955 reactions)",
    "encodings": "Does chemistry-aware featurization speed up the search?",
    "batch": "Running plates of 5 in parallel vs one experiment at a time",
    "acquisition": "Exploration vs exploitation: acquisition functions",
    "noise": "Robustness to measurement noise",
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="main", help="suite name or 'all'")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--n-init", type=int, default=5)
    p.add_argument("--processes", type=int, default=4)
    p.add_argument("--plot-only", action="store_true", help="re-plot saved runs")
    args = p.parse_args(argv)

    data = load_dataset()
    lab = VirtualLab(data)
    threshold = lab.quantile_threshold(TOP_FRACTION)
    n_good = int((data["yield"] >= threshold).sum())
    print(f"Top-1% threshold: {threshold:.1f}% yield ({n_good} of {len(data)} reactions)")
    print(f"Random search needs {random_search_expectation(len(data), n_good):.0f} "
          f"experiments on average to hit one (analytic).")

    all_suites = suites()
    names = list(all_suites) if args.suite == "all" else [args.suite]
    for name in names:
        if args.plot_only:
            runs = pd.read_csv(RUNS_DIR / f"{name}.csv")
        else:
            runs = run_suite(name, all_suites[name], args.seeds, args.budget,
                             args.n_init, args.processes)
        table = metrics_table(runs, threshold)
        md = table.to_markdown(index=False)
        (RESULTS_DIR / f"{name}_metrics.md").write_text(
            f"# {TITLES.get(name, name)}\n\n"
            f"{runs['seed'].nunique()} seeds, budget {runs['experiment'].max()}, "
            f"top-1% threshold = {threshold:.1f}% yield. Values are mean ± std of "
            f"the best yield found after N experiments.\n\n{md}\n"
        )
        print(md)
        plot_suite(runs, name, threshold, TITLES.get(name, name), lab.global_max)


if __name__ == "__main__":
    main()
