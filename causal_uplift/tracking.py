"""MLflow tracking: one parent run per pipeline, one child run per model.

    mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs

Each child run records the model's parameters, its validation and test
metrics, the per-budget policy table and its uplift curve figure, so the
learners can be compared in the MLflow UI as well as in docs/results.json.
"""
import json
from contextlib import contextmanager
from pathlib import Path

import mlflow


def setup(tracking_uri, experiment="criteo-uplift"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


@contextmanager
def run(name, nested=False, params=None, tags=None):
    with mlflow.start_run(run_name=name, nested=nested, tags=tags) as active:
        if params:
            mlflow.log_params({k: v for k, v in params.items()
                               if isinstance(v, (int, float, str, bool))})
        yield active


def metrics(values, prefix=""):
    mlflow.log_metrics({f"{prefix}{k}": float(v) for k, v in values.items()
                        if isinstance(v, (int, float)) and v == v})


def artifact_json(name, payload, directory):
    path = Path(directory) / name
    path.write_text(json.dumps(payload, indent=2))
    mlflow.log_artifact(str(path))


def artifact(path):
    mlflow.log_artifact(str(path))
