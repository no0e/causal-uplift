"""Metrics and policy values, checked against a dataset with a known effect.

The real data has no ground truth. The synthetic set does, so these pin that
the metrics reward the ranking that is actually right, that random targeting
scores zero, and that the policy estimate recovers the true value.
"""
import numpy as np
import pytest

from causal_uplift.metrics import (
    auuc,
    bootstrap,
    qini_coefficient,
    uplift_at,
    uplift_curve,
)
from causal_uplift.policy import evaluate_budgets, policy_value, top_share
from causal_uplift.synthetic import generate


@pytest.fixture(scope="module")
def trial():
    frame = generate(200_000, seed=3)
    return (frame["treatment"].to_numpy(), frame["visit"].to_numpy(),
            frame["tau"].to_numpy())


def test_the_true_effect_beats_random_by_a_wide_margin(trial):
    t, y, tau = trial
    rng = np.random.default_rng(0)
    assert auuc(tau, t, y) > 0.01
    assert abs(auuc(rng.random(len(t)), t, y)) < 0.004
    assert qini_coefficient(tau, t, y) > qini_coefficient(rng.random(len(t)), t, y)


def test_the_reversed_ranking_scores_below_random(trial):
    t, y, tau = trial
    assert auuc(-tau, t, y) < 0


def test_the_curve_ends_at_the_total_incremental_outcomes(trial):
    t, y, tau = trial
    fractions, gain = uplift_curve(tau, t, y)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert fractions[0] == 0 and fractions[-1] == 1
    assert gain[-1] == pytest.approx(ate * len(t))


def test_uplift_in_the_top_decile_is_above_the_average_effect(trial):
    t, y, tau = trial
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert uplift_at(tau, t, y, 0.1) > ate


def test_the_policy_estimate_recovers_the_true_value(trial):
    """IPW on a randomised trial is unbiased: compare with the known tau."""
    t, y, tau = trial
    treat = top_share(tau, 0.3)
    nobody = policy_value(np.zeros_like(t), t, y, propensity=t.mean())
    gain = policy_value(treat, t, y, propensity=t.mean()) - nobody
    true_gain = tau[treat == 1].sum() / len(t)
    assert gain == pytest.approx(true_gain, abs=0.006)


def test_targeting_by_the_true_effect_captures_more_than_its_budget(trial):
    t, y, tau = trial
    table = evaluate_budgets(tau, t, y, budgets=(0.3,))
    row = table["budgets"][0]
    assert row["captured"] > 0.4
    assert row["incremental_per_user"] > row["random_incremental_per_user"]


def test_the_paired_bootstrap_interval_excludes_zero_for_a_real_gap(trial):
    t, y, tau = trial
    noisy = tau + np.random.default_rng(1).normal(scale=0.2, size=len(tau))
    low, high = bootstrap(auuc, tau, t, y, n=30, baseline=noisy)
    assert low > 0 and high > low


def test_tied_scores_do_not_depend_on_the_order_of_the_rows(trial):
    """A constant score must look like random targeting however rows are sorted.

    With a stable sort, a block of tied users is ranked in file order, so a
    file sorted by outcome would hand a constant score a perfect curve.
    """
    t, y, _ = trial
    order = np.argsort(-y, kind="stable")  # every visitor first
    t_sorted, y_sorted = t[order], y[order]
    constant = np.zeros(len(t))
    assert abs(auuc(constant, t_sorted, y_sorted)) < 0.004
    low, high = bootstrap(auuc, constant, t_sorted, y_sorted, n=20)
    assert low < 0.004 and high > -0.004
