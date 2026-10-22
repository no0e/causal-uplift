"""The average effect, identified and stress-tested with DoWhy.

The trial is randomised, so the effect is identified by design; writing it as
a DoWhy causal graph makes that assumption explicit and lets three refuters
try to break the estimate:

- a placebo treatment (shuffled assignment) should give an effect near zero;
- adding a random common cause should leave the estimate where it was;
- re-estimating on random subsets should give the same answer.

If any of those moves the number, something is wrong with the data before
anything is wrong with a model.
"""
import numpy as np

from . import FEATURES, TREATMENT


def identify_and_refute(frame, outcome="visit", simulations=20, seed=0):
    """ATE by propensity-score weighting, and the three refutations."""
    import dowhy
    from dowhy import CausalModel

    data = frame.select(FEATURES + [TREATMENT, outcome]).to_pandas()
    data[TREATMENT] = data[TREATMENT].astype(bool)
    model = CausalModel(data=data, treatment=TREATMENT, outcome=outcome,
                        common_causes=FEATURES)
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand, method_name="backdoor.propensity_score_weighting",
        method_params={"weighting_scheme": "ips_weight"})

    np.random.seed(seed)
    refuters = {
        "placebo_treatment": dict(method_name="placebo_treatment_refuter",
                                  placebo_type="permute"),
        "random_common_cause": dict(method_name="random_common_cause"),
        "data_subset": dict(method_name="data_subset_refuter",
                            subset_fraction=0.5),
    }
    refutations = {}
    for name, kwargs in refuters.items():
        result = model.refute_estimate(estimand, estimate,
                                       num_simulations=simulations, **kwargs)
        refutations[name] = {
            "new_effect": float(result.new_effect),
            "p_value": (float(result.refutation_result["p_value"])
                        if result.refutation_result else None),
        }
    return {"dowhy": dowhy.__version__, "rows": len(data),
            "estimand": str(estimand.estimands["backdoor"]),
            "ate": float(estimate.value), "refutations": refutations}
