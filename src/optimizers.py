"""Experiment-selection strategies: random search and Bayesian optimization.

Every strategy implements ``suggest(X, observed_idx, observed_y, n, rng)``
which returns ``n`` untested row indices of ``X`` to run next. The shared
``run_campaign`` loop drives any strategy against a :class:`VirtualLab`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .virtual_lab import VirtualLab


# ---------------------------------------------------------------------------
# Acquisition functions (all assume we are *maximising* yield)
# ---------------------------------------------------------------------------
def expected_improvement(mu, sigma, best, xi: float = 0.01):
    """EI(x) = E[max(f(x) - best - xi, 0)] under a Gaussian posterior."""
    sigma = np.maximum(sigma, 1e-12)
    imp = mu - best - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def upper_confidence_bound(mu, sigma, beta: float = 2.0):
    """UCB(x) = mu + sqrt(beta) * sigma. Larger beta = more exploration."""
    return mu + np.sqrt(beta) * sigma


# ---------------------------------------------------------------------------
# Surrogate models
# ---------------------------------------------------------------------------
class GPSurrogate:
    """Gaussian process with a Matern-5/2 kernel plus a learned noise term.

    Targets are standardised before fitting, so the acquisition functions
    work in units of "standard deviations of yield seen so far".
    """

    def __init__(self, nu: float = 2.5, n_restarts: int = 2, seed: int = 0):
        self.nu = nu
        self.n_restarts = n_restarts
        self.seed = seed
        self.gp: GaussianProcessRegressor | None = None

    def _kernel(self, n_features: int):
        ls0 = max(1.0, np.sqrt(n_features) / 2)
        return ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=ls0, length_scale_bounds=(1e-2, 1e3), nu=self.nu
        ) + WhiteKernel(1e-2, (1e-6, 1.0))

    def fit(self, X, y, kernel=None):
        self.y_mean, self.y_std = float(np.mean(y)), float(np.std(y)) or 1.0
        z = (np.asarray(y) - self.y_mean) / self.y_std
        fixed = kernel is not None
        self.gp = GaussianProcessRegressor(
            kernel=kernel if fixed else self._kernel(X.shape[1]),
            optimizer=None if fixed else "fmin_l_bfgs_b",
            n_restarts_optimizer=0 if fixed else self.n_restarts,
            normalize_y=False,
            random_state=self.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(X, z)
        return self

    def predict(self, X):
        """Posterior mean and std in standardised units."""
        mu, sd = self.gp.predict(X, return_std=True)
        return mu, sd

    def to_yield(self, mu, sd):
        return mu * self.y_std + self.y_mean, sd * self.y_std


class RFSurrogate:
    """Random forest; uncertainty = spread of the individual trees."""

    def __init__(self, n_estimators: int = 100, seed: int = 0):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X, y):
        self.y_mean, self.y_std = float(np.mean(y)), float(np.std(y)) or 1.0
        z = (np.asarray(y) - self.y_mean) / self.y_std
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators, max_features=0.5,
            random_state=self.seed, n_jobs=1,
        ).fit(X, z)
        return self

    def predict(self, X):
        per_tree = np.stack([t.predict(X) for t in self.rf.estimators_])
        return per_tree.mean(0), per_tree.std(0)

    def to_yield(self, mu, sd):
        return mu * self.y_std + self.y_mean, sd * self.y_std


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
class Strategy:
    name = "strategy"

    def suggest(self, X, observed_idx, observed_y, n, rng) -> np.ndarray:
        raise NotImplementedError


class RandomSearch(Strategy):
    name = "Random search"

    def suggest(self, X, observed_idx, observed_y, n, rng):
        untested = np.setdiff1d(np.arange(len(X)), observed_idx)
        return rng.choice(untested, size=n, replace=False)


@dataclass
class BayesOpt(Strategy):
    """Bayesian optimization over a finite candidate set.

    surrogate:   "gp" or "rf"
    acquisition: "ei", "ucb" or "greedy" (posterior mean only)
    For n > 1 suggestions per round the *kriging believer* heuristic is used:
    pick the best candidate, pretend its yield equals the model's prediction,
    update the model (hyper-parameters fixed) and pick again.
    """

    surrogate: str = "gp"
    acquisition: str = "ei"
    xi: float = 0.01
    beta: float = 2.0
    label: str | None = None

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        acq = {"ei": "EI", "ucb": f"UCB(β={self.beta:g})", "greedy": "greedy"}[self.acquisition]
        return f"BO {self.surrogate.upper()}-{acq}"

    def _model(self, seed):
        return GPSurrogate(seed=seed) if self.surrogate == "gp" else RFSurrogate(seed=seed)

    def _score(self, mu, sd, best):
        if self.acquisition == "ei":
            return expected_improvement(mu, sd, best, self.xi)
        if self.acquisition == "ucb":
            return upper_confidence_bound(mu, sd, self.beta)
        if self.acquisition == "greedy":
            return mu
        raise ValueError(self.acquisition)

    def fit_predict(self, X, observed_idx, observed_y, seed=0):
        """Fit the surrogate; return (model, mu, sd) over all rows, in yield %."""
        model = self._model(seed).fit(X[observed_idx], observed_y)
        mu, sd = model.to_yield(*model.predict(X))
        return model, mu, sd

    def suggest(self, X, observed_idx, observed_y, n, rng):
        seed = int(rng.integers(1 << 31))
        idx = [int(i) for i in observed_idx]
        y = [float(v) for v in observed_y]
        base = self._model(seed).fit(X[idx], np.array(y))
        # Acquisition is computed in the base model's standardised units.
        best = (max(y) - base.y_mean) / base.y_std
        model, chosen, fantasies = base, [], []
        for k in range(n):
            if k > 0:
                # Kriging believer: treat the predicted yield of the reactions
                # already in this batch as if they had been measured, refit
                # (GP hyper-parameters frozen) and choose again.
                X_fit = X[idx + chosen]
                y_fit = np.array(y + fantasies)
                if self.surrogate == "gp":
                    model = GPSurrogate(seed=seed).fit(X_fit, y_fit, kernel=base.gp.kernel_)
                else:
                    model = self._model(seed).fit(X_fit, y_fit)
            mu_y, sd_y = model.to_yield(*model.predict(X))
            mu, sd = (mu_y - base.y_mean) / base.y_std, sd_y / base.y_std
            score = self._score(mu, sd, best)
            score[idx + chosen] = -np.inf
            nxt = int(np.argmax(score))
            chosen.append(nxt)
            fantasies.append(float(mu_y[nxt]))
        return np.array(chosen)


# ---------------------------------------------------------------------------
# Campaign loop
# ---------------------------------------------------------------------------
def run_campaign(
    lab: VirtualLab,
    X: np.ndarray,
    strategy: Strategy,
    budget: int = 50,
    n_init: int = 5,
    batch_size: int = 1,
    seed: int = 0,
) -> VirtualLab:
    """Run ``budget`` experiments: ``n_init`` random ones, then ``strategy``.

    The initial design is drawn from the same seed for every strategy, so
    methods compared at the same seed share their first ``n_init`` reactions.
    """
    lab.reset()
    init_rng = np.random.default_rng(seed)
    lab.run_batch(init_rng.choice(lab.n_candidates, size=n_init, replace=False))
    rng = np.random.default_rng(seed + 10_000)
    while lab.n_run < budget:
        n = min(batch_size, budget - lab.n_run)
        idx, y = lab.observed()
        lab.run_batch(strategy.suggest(X, idx, y, n, rng))
    return lab
