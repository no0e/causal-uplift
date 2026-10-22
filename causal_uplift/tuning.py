"""LightGBM hyperparameters chosen by Optuna on validation AUUC.

The search tunes the T-learner, whose two LightGBM models are the same family
the other learners use for their nuisance and final stages; the best
parameters are then shared with all of them. The objective is the AUUC on the
validation split, never the test split.
"""
import optuna

from .metrics import auuc
from .models import TLearner


def space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 2000,
                                               log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def tune(train, val, n_trials=30, seed=0, callback=None):
    """Best LightGBM params for the T-learner, and the study."""
    X, t, y = train
    Xv, tv, yv = val

    def objective(trial):
        params = space(trial)
        score = auuc(TLearner(params, seed).fit(X, t, y).effect(Xv), tv, yv)
        if callback is not None:
            callback(trial.number, params, score)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study
