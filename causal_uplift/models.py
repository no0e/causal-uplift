"""The uplift models compared, behind one interface: fit(X, t, y), effect(X).

From the simplest to the most structured:

- S-learner: one LightGBM with the treatment as a feature; the uplift is the
  difference of its two predictions. Cheap, and biased towards zero when the
  trees rarely split on the treatment.
- T-learner: one LightGBM per arm. Each model sees only its own arm, so the
  15% control arm gets a model fitted on a seventh of the data.
- X-learner (EconML): imputes each user's effect from the other arm's model
  and regresses those, weighted by the propensity; built for unbalanced arms.
- DR-learner (EconML): a doubly robust pseudo-outcome regressed on X, with
  cross-fitting.
- Causal forest (EconML CausalForestDML): double machine learning residuals
  and an honest forest on them, the one with valid confidence intervals.

The propensity model is a `DummyClassifier` that predicts the treated share:
the trial is randomised, so the true propensity is that constant and fitting
anything richer would only add noise.
"""
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier

DEFAULT_LGBM = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63,
                "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1,
                "colsample_bytree": 0.8, "reg_lambda": 1.0}


def _clf(params, seed):
    return LGBMClassifier(**{**DEFAULT_LGBM, **(params or {})},
                          random_state=seed, n_jobs=-1, verbose=-1)


def _reg(params, seed):
    return LGBMRegressor(**{**DEFAULT_LGBM, **(params or {})},
                         random_state=seed, n_jobs=-1, verbose=-1)


class SLearner:
    name = "s_learner"

    def __init__(self, params=None, seed=0):
        self.model = _clf(params, seed)

    def fit(self, X, t, y):
        self.model.fit(np.column_stack([X, t]), y)
        return self

    def effect(self, X):
        ones, zeros = np.ones(len(X)), np.zeros(len(X))
        return (self.model.predict_proba(np.column_stack([X, ones]))[:, 1]
                - self.model.predict_proba(np.column_stack([X, zeros]))[:, 1])


class TLearner:
    name = "t_learner"

    def __init__(self, params=None, seed=0):
        self.treated, self.control = _clf(params, seed), _clf(params, seed + 1)

    def fit(self, X, t, y):
        self.treated.fit(X[t == 1], y[t == 1])
        self.control.fit(X[t == 0], y[t == 0])
        return self

    def effect(self, X):
        return (self.treated.predict_proba(X)[:, 1]
                - self.control.predict_proba(X)[:, 1])


class _EconML:
    """Adapter: EconML estimators take (Y, T, X=...) and return effect(X)."""

    def fit(self, X, t, y):
        self.estimator.fit(y.astype(np.float64), t, X=X)
        return self

    def effect(self, X):
        return np.asarray(self.estimator.effect(X)).ravel()


class XLearner(_EconML):
    name = "x_learner"

    def __init__(self, params=None, seed=0):
        from econml.metalearners import XLearner as Estimator

        self.estimator = Estimator(
            models=_reg(params, seed), cate_models=_reg(params, seed + 1),
            propensity_model=DummyClassifier(strategy="prior"))


class DRLearner(_EconML):
    name = "dr_learner"

    def __init__(self, params=None, seed=0):
        from econml.dr import DRLearner as Estimator

        self.estimator = Estimator(
            model_propensity=DummyClassifier(strategy="prior"),
            model_regression=_reg(params, seed),
            model_final=_reg({**(params or {}), "n_estimators": 200}, seed + 1),
            min_propensity=0.01, cv=3, random_state=seed)


class CausalForest(_EconML):
    name = "causal_forest"

    def __init__(self, params=None, seed=0, n_estimators=400,
                 min_samples_leaf=500):
        from econml.dml import CausalForestDML

        self.estimator = CausalForestDML(
            model_y=_reg(params, seed),
            model_t=DummyClassifier(strategy="prior"),
            discrete_treatment=True, n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf, max_samples=0.2, cv=3,
            n_jobs=-1, random_state=seed)


class Random:
    """The baseline every curve is drawn against."""

    name = "random"

    def __init__(self, params=None, seed=0):
        self.rng = np.random.default_rng(seed)

    def fit(self, X, t, y):
        return self

    def effect(self, X):
        return self.rng.random(len(X))


MODELS = {cls.name: cls for cls in
          (Random, SLearner, TLearner, XLearner, DRLearner, CausalForest)}


def score_in_batches(model, frames, features):
    """Effects for an iterable of Polars frames, without concatenating inputs."""
    return np.concatenate([model.effect(frame.select(features).to_numpy()
                                        .astype(np.float32))
                           for frame in frames])
