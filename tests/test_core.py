import numpy as np
import pandas as pd
import pytest

from src.benchmark import experiments_to_target, random_search_expectation
from src.data import COMPONENTS, load_dataset
from src.features import encode
from src.optimizers import (BayesOpt, RandomSearch, expected_improvement,
                            run_campaign, upper_confidence_bound)
from src.virtual_lab import VirtualLab


@pytest.fixture(scope="module")
def df():
    return load_dataset()


def test_dataset_is_clean(df):
    assert len(df) == 3955
    assert df[COMPONENTS].notna().all().all()
    assert not df.duplicated(COMPONENTS).any()
    assert df["yield"].between(0, 100).all()
    assert [df[c].nunique() for c in COMPONENTS] == [4, 3, 22, 15]


@pytest.mark.parametrize("encoding", ["onehot", "descriptors", "fingerprint"])
def test_encodings_shape_and_finite(df, encoding):
    X, names = encode(df, encoding)
    assert X.shape[0] == len(df) and X.shape[1] == len(names)
    assert np.isfinite(X).all()


def test_onehot_rows_have_one_choice_per_component(df):
    X, _ = encode(df, "onehot")
    assert (X.sum(1) == len(COMPONENTS)).all()


def test_virtual_lab_lookup_and_no_repeats(df):
    lab = VirtualLab(df)
    assert lab.run(3) == pytest.approx(df["yield"].iloc[3])
    with pytest.raises(ValueError):
        lab.run(3)
    lab.reset()
    assert lab.n_run == 0 and lab.run(3) == pytest.approx(df["yield"].iloc[3])


def test_noisy_lab_stays_in_range(df):
    lab = VirtualLab(df, noise_sd=20, seed=1)
    ys = lab.run_batch(range(200))
    assert all(0 <= v <= 100 for v in ys)


def test_acquisition_functions():
    mu = np.array([0.0, 1.0, 1.0])
    sd = np.array([1.0, 1.0, 0.0])
    ei = expected_improvement(mu, sd, best=0.5)
    assert ei[1] > ei[0] and ei[1] > ei[2] >= 0
    assert np.allclose(upper_confidence_bound(mu, sd, beta=4.0), mu + 2 * sd)


@pytest.mark.parametrize("strategy,batch", [
    (RandomSearch(), 1), (BayesOpt("gp", "ei"), 1), (BayesOpt("gp", "ei"), 4),
    (BayesOpt("rf", "ucb"), 3),
])
def test_campaign_respects_budget_and_uniqueness(df, strategy, batch):
    X, _ = encode(df, "onehot")
    lab = run_campaign(VirtualLab(df), X, strategy, budget=14, n_init=5,
                       batch_size=batch, seed=0)
    idx, _ = lab.observed()
    assert len(idx) == 14 and len(set(idx)) == 14


def test_same_seed_shares_initial_design(df):
    X, _ = encode(df, "onehot")
    a = run_campaign(VirtualLab(df), X, RandomSearch(), budget=8, n_init=5, seed=3)
    b = run_campaign(VirtualLab(df), X, BayesOpt("gp", "ei"), budget=8, n_init=5, seed=3)
    assert list(a.observed()[0][:5]) == list(b.observed()[0][:5])


def test_experiments_to_target():
    runs = pd.DataFrame({"method": ["m"] * 4 + ["m"] * 4, "seed": [0] * 4 + [1] * 4,
                         "experiment": [1, 2, 3, 4] * 2,
                         "yield": [10, 95, 99, 5, 1, 2, 3, 4]})
    t = experiments_to_target(runs, 90)
    assert t.loc[("m", 0)] == 2 and np.isnan(t.loc[("m", 1)])
    assert random_search_expectation(3955, 40) == pytest.approx(3956 / 41)
