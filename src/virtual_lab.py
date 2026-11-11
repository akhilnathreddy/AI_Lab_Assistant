"""A lookup-based experiment simulator ("virtual lab").

Every reaction in the dataset has a measured yield, so "running" an
experiment means looking that yield up. The lab keeps track of what has been
run so that optimizers cannot peek at untested results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import COMPONENTS


@dataclass
class VirtualLab:
    """Search space + oracle for a fully-enumerated reaction screen.

    Parameters
    ----------
    table:
        One row per reaction with the condition columns and a yield column.
    yield_col:
        Name of the measured-yield column.
    noise_sd:
        Optional Gaussian measurement noise (in yield %) added to each run, to
        mimic run-to-run variability in a real lab. Default 0 (exact lookup).
    seed:
        Seed for the noise generator.
    """

    table: pd.DataFrame
    yield_col: str = "yield"
    noise_sd: float = 0.0
    seed: int | None = None
    history: list[tuple[int, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.table = self.table.reset_index(drop=True)
        self._yields = self.table[self.yield_col].to_numpy(dtype=float)
        self._rng = np.random.default_rng(self.seed)
        self._tested = np.zeros(len(self.table), dtype=bool)

    # --- search space -------------------------------------------------
    @property
    def n_candidates(self) -> int:
        return len(self.table)

    @property
    def tested_mask(self) -> np.ndarray:
        return self._tested.copy()

    @property
    def untested(self) -> np.ndarray:
        return np.flatnonzero(~self._tested)

    def conditions(self, idx: int) -> dict:
        row = self.table.iloc[int(idx)]
        return {c: row[c] for c in COMPONENTS if c in row}

    # --- running experiments -----------------------------------------
    def run(self, idx: int) -> float:
        """Run the experiment at row ``idx`` and return its (noisy) yield."""
        idx = int(idx)
        if not 0 <= idx < self.n_candidates:
            raise IndexError(f"Experiment {idx} is outside the search space")
        if self._tested[idx]:
            raise ValueError(f"Experiment {idx} has already been run")
        y = self._yields[idx]
        if self.noise_sd > 0:
            y = float(np.clip(y + self._rng.normal(0, self.noise_sd), 0, 100))
        self._tested[idx] = True
        self.history.append((idx, float(y)))
        return float(y)

    def run_batch(self, indices) -> list[float]:
        return [self.run(i) for i in indices]

    def reset(self) -> None:
        self._tested[:] = False
        self.history.clear()

    # --- bookkeeping ---------------------------------------------------
    @property
    def n_run(self) -> int:
        return len(self.history)

    def observed(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.history:
            return np.array([], dtype=int), np.array([], dtype=float)
        idx, y = zip(*self.history)
        return np.array(idx, dtype=int), np.array(y, dtype=float)

    def best_so_far(self) -> np.ndarray:
        """Running maximum of the *true* yield after each experiment."""
        idx, _ = self.observed()
        return np.maximum.accumulate(self._yields[idx]) if len(idx) else np.array([])

    # --- ground truth (for evaluation only, never used by optimizers) ----
    @property
    def global_max(self) -> float:
        return float(self._yields.max())

    def quantile_threshold(self, top_fraction: float = 0.01) -> float:
        """Yield needed to be in the top ``top_fraction`` of the search space."""
        return float(np.quantile(self._yields, 1 - top_fraction))
