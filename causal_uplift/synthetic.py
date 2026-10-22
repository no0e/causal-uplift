"""A dataset with the Criteo schema and a known treatment effect.

Twelve features, an 85/15 randomised treatment, a binary visit and a rarer
conversion, and an individual effect that depends on three of the features.
The real file has no ground truth; this one does, which is what lets the tests
check that a metric rewards the right ranking and that a policy estimate is
unbiased, before any of it is run on data where nobody can check.
"""
import numpy as np
import polars as pl

from . import FEATURES, TREATMENT


def true_effect(X):
    """The individual uplift on the visit probability, in [0, 0.2]."""
    return 0.1 * (1 + np.tanh(1.5 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] * X[:, 3]))


def generate(n, seed=0, treated_share=0.85):
    """A Polars frame with the Criteo columns plus `tau`, the true uplift."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES))).astype(np.float32)
    t = (rng.random(n) < treated_share).astype(np.int8)
    base = 1 / (1 + np.exp(-(-2.0 + 0.8 * X[:, 4] - 0.5 * X[:, 5])))
    tau = true_effect(X)
    p_visit = np.clip(base + t * tau, 0, 1)
    visit = (rng.random(n) < p_visit).astype(np.int8)
    conversion = (visit * (rng.random(n) < 0.05 + 0.05 * t)).astype(np.int8)
    frame = pl.DataFrame({f: X[:, i] for i, f in enumerate(FEATURES)})
    return frame.with_columns(
        pl.Series(TREATMENT, t), pl.Series("visit", visit),
        pl.Series("conversion", conversion), pl.Series("tau", tau))


def write_csv(path, n, seed=0):
    """The synthetic set as a CSV shaped like the Criteo download."""
    generate(n, seed).drop("tau").write_csv(path)
    return path
