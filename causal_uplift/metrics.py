"""Uplift curves, the areas under them, and their bootstrap intervals.

On a randomised trial the uplift of the top fraction of a ranking can be read
straight off the data: the outcome rate of the treated users in that fraction
minus the rate of the controls in it. Scaled by the number of users in the
fraction, that is the incremental outcomes a campaign targeting exactly them
would produce. Plotted against the fraction, it is the uplift curve, and
random targeting is the straight line from 0 to the total effect.

AUUC here is the area between the curve and that line, divided by the number
of users, so 0 is random targeting and higher is better, and two models are
compared on the same scale whatever the test size. The Qini curve and its
coefficient are reported too, because that is what most of the literature
quotes.

Ties are broken at random, with a fixed seed. A learner can give large groups
of users exactly the same score (an S-learner whose trees never split on the
treatment gives them all zero), and a stable sort would then order each group
as it sits in the file. Any trend in the file order would leak into the curve,
and a bootstrap resample, which reorders the rows, would see a different
curve from the same model.
"""
import numpy as np


def ranking(score, seed=0):
    """Indices from highest to lowest score, ties in a seeded random order."""
    noise = np.random.default_rng(seed).random(len(score))
    return np.lexsort((noise, -np.asarray(score, dtype=np.float64)))


def _cumulative(score, t, y):
    order = ranking(score)
    t, y = t[order].astype(np.float64), y[order].astype(np.float64)
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    return n_t, n_c, y_t, y_c


def uplift_curve(score, t, y, points=100):
    """(fractions, incremental outcomes) at `points` cut-offs of the ranking."""
    n = len(score)
    n_t, n_c, y_t, y_c = _cumulative(score, t, y)
    cuts = np.unique(np.linspace(1, n, points).astype(int)) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_t = np.where(n_t[cuts] > 0, y_t[cuts] / n_t[cuts], 0.0)
        rate_c = np.where(n_c[cuts] > 0, y_c[cuts] / n_c[cuts], 0.0)
    gain = (rate_t - rate_c) * (cuts + 1)
    return np.concatenate([[0.0], (cuts + 1) / n]), np.concatenate([[0.0], gain])


def qini_curve(score, t, y, points=100):
    """(fractions, Qini) with the control count rescaled to the treated count."""
    n = len(score)
    n_t, n_c, y_t, y_c = _cumulative(score, t, y)
    cuts = np.unique(np.linspace(1, n, points).astype(int)) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = y_t[cuts] - np.where(n_c[cuts] > 0,
                                    y_c[cuts] * n_t[cuts] / n_c[cuts], 0.0)
    return np.concatenate([[0.0], (cuts + 1) / n]), np.concatenate([[0.0], qini])


def _area_over_random(fractions, curve):
    random_line = fractions * curve[-1]
    return float(np.trapezoid(curve - random_line, fractions))


def auuc(score, t, y, points=100):
    """Area between the uplift curve and random targeting, per user."""
    fractions, gain = uplift_curve(score, t, y, points)
    return _area_over_random(fractions, gain) / len(score)


def qini_coefficient(score, t, y, points=100):
    """Area between the Qini curve and random, per treated user."""
    fractions, qini = qini_curve(score, t, y, points)
    return _area_over_random(fractions, qini) / max(int(t.sum()), 1)


def uplift_at(score, t, y, fraction):
    """Treated rate minus control rate in the top `fraction` of the ranking."""
    k = max(1, int(round(fraction * len(score))))
    top = ranking(score)[:k]
    tt, yy = t[top], y[top]
    if tt.sum() == 0 or (1 - tt).sum() == 0:
        return float("nan")
    return float(yy[tt == 1].mean() - yy[tt == 0].mean())


def bootstrap(metric, score, t, y, n=200, seed=0, baseline=None):
    """95% interval of `metric`, or of its difference to `baseline` scores.

    Paired: both rankings are evaluated on the same resample, which is what
    makes a difference between two models on one test set meaningful.
    """
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(score), len(score))
        value = metric(score[idx], t[idx], y[idx])
        if baseline is not None:
            value -= metric(baseline[idx], t[idx], y[idx])
        values.append(value)
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)
