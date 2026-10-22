"""Who to treat under a budget, and what that policy is worth.

A policy here treats the top `budget` share of users by predicted uplift. Its
value is estimated on held-out trial data by inverse propensity weighting:
the users whose actual assignment happens to agree with the policy stand in
for everyone, each weighted by one over the probability of that assignment.
On a randomised trial that probability is known (the treated share), so the
estimate is unbiased with no model of the assignment at all.

The economics are simulated, since the dataset carries no prices: each extra
outcome is worth `value` and each treatment costs `cost`, and the net value
of a policy is what it earns over treating nobody.
"""
import numpy as np

from .metrics import ranking


def policy_value(treat, t, y, propensity):
    """IPW estimate of the mean outcome if `treat` (0/1 per user) were applied."""
    agree = treat == t
    p = np.where(treat == 1, propensity, 1 - propensity)
    return float(np.mean(y * agree / p))


def top_share(score, budget):
    """0/1 per user: the top `budget` share of `score` is treated."""
    k = int(round(budget * len(score)))
    treat = np.zeros(len(score), dtype=np.int8)
    if k:
        treat[ranking(score)[:k]] = 1
    return treat


def evaluate_budgets(score, t, y, budgets=(0.1, 0.2, 0.3, 0.5), value=1.0,
                     cost=0.0, propensity=None, seed=0):
    """Incremental outcomes and net value per budget, against random targeting.

    `captured` is the share of the total incremental outcomes (treat everyone
    versus nobody) that the policy obtains while treating only `budget` of
    the users; random targeting captures `budget` of it in expectation.
    """
    propensity = float(t.mean()) if propensity is None else propensity
    nobody = policy_value(np.zeros_like(t), t, y, propensity)
    everyone = policy_value(np.ones_like(t), t, y, propensity)
    total_gain = everyone - nobody
    rng = np.random.default_rng(seed)

    rows = []
    for budget in budgets:
        treat = top_share(score, budget)
        gain = policy_value(treat, t, y, propensity) - nobody
        random_gain = policy_value(top_share(rng.random(len(t)), budget),
                                   t, y, propensity) - nobody
        rows.append({
            "budget": budget,
            "incremental_per_user": gain,
            "random_incremental_per_user": random_gain,
            "captured": gain / total_gain if total_gain else float("nan"),
            "net_value_per_user": value * gain - cost * budget,
            "net_value_random_per_user": value * random_gain - cost * budget,
        })
    return {"propensity": propensity, "rate_nobody": nobody,
            "rate_everyone": everyone, "budgets": rows}
