import numpy as np
import pandas as pd
import pytest

from src.benchmark import MethodSpec, method_curves, method_summary, track_suite
from src.optimizers import BayesOpt, RandomSearch


def _runs():
    # Two methods x two seeds x four experiments; threshold 90.
    rows = []
    traces = {("A", 0): [10, 95, 99, 5], ("A", 1): [50, 60, 91, 1],
              ("B", 0): [1, 2, 3, 4], ("B", 1): [1, 92, 3, 4]}
    for (m, seed), ys in traces.items():
        best = np.maximum.accumulate(ys)
        for i, (y, b) in enumerate(zip(ys, best), start=1):
            rows.append({"method": m, "seed": seed, "experiment": i, "yield": y,
                         "best_so_far": b})
    return pd.DataFrame(rows)


def test_method_summary_numbers():
    s = method_summary(_runs(), threshold=90, checkpoints=(2, 4))
    assert s["A"]["found_top1_rate"] == 1.0 and s["A"]["median_exps_to_top1"] == 2.5
    assert s["A"]["best_at_2_mean"] == pytest.approx((95 + 60) / 2)
    assert s["B"]["found_top1_rate"] == 0.5
    assert "median_exps_to_top1" not in s["B"]  # median is censored (>budget)


def test_method_curves_shapes():
    c = method_curves(_runs(), threshold=90)
    assert list(c["A"]["share_found_top1"]) == [0, 0.5, 1, 1]
    assert len(c["B"]["mean_best_so_far"]) == 4


def test_track_suite_logs_parent_and_children(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    specs = [MethodSpec("A", BayesOpt("gp", "ei", label="A")), MethodSpec("B", RandomSearch())]
    track_suite(_runs(), suite="unit", name="unit", dataset="toy", title="t",
                threshold=90, specs=specs, n_init=1)
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    exp = client.get_experiment_by_name("benchmark/toy")
    runs = client.search_runs([exp.experiment_id])
    kinds = sorted(r.data.tags["kind"] for r in runs)
    assert kinds == ["benchmark-method", "benchmark-method", "benchmark-suite"]
    child = next(r for r in runs if r.info.run_name == "A")
    assert child.data.params["surrogate"] == "gp"
    assert child.data.metrics["found_top1_rate"] == 1.0
    assert len(client.get_metric_history(child.info.run_id, "mean_best_so_far")) == 4
    assert str(tmp_path) in exp.artifact_location
    mlflow.end_run()
