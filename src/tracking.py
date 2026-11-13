"""Optional experiment tracking with MLflow.

Everything here degrades gracefully: if MLflow is not installed, or tracking
is not requested, the helpers do nothing and the rest of the project works
exactly as before.

By default runs are stored locally (no account or server needed):

* run metadata  -> ``mlflow.db`` (SQLite) in the repository root
* artifacts     -> ``mlartifacts/`` next to that database

Set ``MLFLOW_TRACKING_URI`` to log to another store (for example a shared
MLflow server or Databricks) instead. Browse local runs with::

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"

try:  # MLflow is an optional dependency (requirements-tracking.txt).
    import mlflow
    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient
except ImportError:  # pragma: no cover - exercised only without mlflow
    mlflow = None


def available() -> bool:
    return mlflow is not None


def _setup(experiment: str) -> str:
    """Point MLflow at the store and return the experiment id."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is not None:
        return exp.experiment_id
    kwargs = {}
    if uri.startswith("sqlite:///"):  # local store: keep artifacts next to the db
        artifact_root = Path(uri.removeprefix("sqlite:///")).resolve().parent / "mlartifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        kwargs["artifact_location"] = (artifact_root / experiment.replace("/", "_")).as_uri()
    return mlflow.create_experiment(experiment, **kwargs)


@contextmanager
def run(experiment: str, run_name: str, *, enabled: bool = True, nested: bool = False,
        params: Mapping | None = None, tags: Mapping | None = None) -> Iterator[object | None]:
    """Start an MLflow run, or yield ``None`` when tracking is off/unavailable."""
    if not enabled:
        yield None
        return
    if mlflow is None:
        raise RuntimeError("Tracking requested but MLflow is not installed: "
                           "pip install -r requirements-tracking.txt")
    # Nested runs also need the experiment id, or MLflow files them under
    # the "Default" experiment instead of next to their parent.
    exp_id = _setup(experiment)
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, nested=nested) as active:
        if params:
            mlflow.log_params({k: _param(v) for k, v in params.items()})
        if tags:
            mlflow.set_tags(dict(tags))
        yield active


def _param(value) -> str:
    return "None" if value is None else str(value)


def log_metrics(active, metrics: Mapping[str, float]) -> None:
    if active is None:
        return
    clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}  # drop NaN
    if clean:
        mlflow.log_metrics(clean)


def log_curve(active, key: str, values, start_step: int = 1) -> None:
    """Log a whole learning curve (value per experiment) in one batched call."""
    if active is None:
        return
    import time

    ts = int(time.time() * 1000)
    metrics = [Metric(key, float(v), ts, start_step + i) for i, v in enumerate(values)]
    client = MlflowClient()
    for i in range(0, len(metrics), 1000):  # MLflow's batch limit
        client.log_batch(active.info.run_id, metrics=metrics[i:i + 1000])


def log_artifacts(active, paths) -> None:
    if active is None:
        return
    for p in paths:
        if Path(p).exists():
            mlflow.log_artifact(str(p))
