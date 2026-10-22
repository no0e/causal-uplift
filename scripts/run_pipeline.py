"""Fit every uplift model, evaluate it on the held-out split, pick a policy.

    python scripts/run_pipeline.py                         # data/criteo.parquet
    python scripts/run_pipeline.py --data data/synthetic.parquet --quick

1. Trial facts out of core with DuckDB: arms, ATE, covariate balance.
2. The ATE identified and refuted with DoWhy on a sample.
3. Train and validation samples drawn with DuckDB.
4. LightGBM tuned by Optuna on validation AUUC (shared by every learner).
5. Each learner fitted on the train sample, then scored on the *whole* test
   split in bounded batches: AUUC, Qini, uplift at 10/30%, a paired bootstrap
   interval against the S-learner, and the budgeted policy table.
6. The best learner (validation AUUC) scores every row of the file, written
   to Parquet batch by batch.

Everything is logged to MLflow (sqlite:///mlflow.db) and summarised in docs/results.json
and docs/uplift_curves.png.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from causal_uplift import (  # noqa: E402
    FEATURES,
    TREATMENT,
    tracking,  # noqa: E402
)
from causal_uplift import data as D  # noqa: E402
from causal_uplift.metrics import (  # noqa: E402
    auuc,
    bootstrap,
    qini_coefficient,
    qini_curve,
    uplift_at,
    uplift_curve,
)
from causal_uplift.models import MODELS, score_in_batches  # noqa: E402
from causal_uplift.policy import evaluate_budgets  # noqa: E402

BASELINE = "s_learner"


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data" / "criteo.parquet"))
    parser.add_argument("--outcome", default="visit")
    parser.add_argument("--train-rows", type=int, default=2_000_000)
    parser.add_argument("--val-rows", type=int, default=1_000_000)
    parser.add_argument("--forest-rows", type=int, default=1_000_000,
                        help="The causal forest is fitted on this many rows.")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--value", type=float, default=1.0,
                        help="Simulated value of one extra visit.")
    parser.add_argument("--cost", type=float, default=0.002,
                        help="Simulated cost of treating one user.")
    parser.add_argument("--dowhy-rows", type=int, default=1_000_000)
    parser.add_argument("--only-dowhy", action="store_true",
                        help="Recompute the DoWhy section of an existing "
                             "results.json and leave the rest alone.")
    parser.add_argument("--batch", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--no-score-all", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Small samples and few trials, for a smoke run.")
    parser.add_argument("--mlflow", default=f"sqlite:///{ROOT / 'mlflow.db'}",
                        help="MLflow tracking URI.")
    parser.add_argument("--out", default=str(ROOT / "docs"))
    args = parser.parse_args()
    if args.quick:
        args.train_rows, args.val_rows, args.forest_rows = 40_000, 20_000, 20_000
        args.trials, args.bootstrap, args.dowhy_rows = 3, 20, 5_000
    return args


def figure(curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, (fractions, gain, qfractions, qini) in curves.items():
        style = {"linestyle": "--", "color": "grey"} if name == "random" else {}
        left.plot(fractions, gain, label=name, **style)
        right.plot(qfractions, qini, label=name, **style)
    left.set(title="Uplift curve (test split)", xlabel="share targeted",
             ylabel="incremental visits")
    right.set(title="Qini curve (test split)", xlabel="share targeted",
              ylabel="Qini")
    left.legend(frameon=False, fontsize=8)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dowhy_section(args, con):
    from causal_uplift.identification import identify_and_refute

    frame = con.sql(f"""
        SELECT {", ".join(FEATURES + [TREATMENT, args.outcome])}
        FROM '{args.data}' WHERE split = 'train'
        USING SAMPLE reservoir({args.dowhy_rows} ROWS) REPEATABLE (7)
    """).pl()
    section = identify_and_refute(frame, args.outcome)
    print(f"DoWhy ATE {section['ate']:+.5f}, refutations "
          + ", ".join(f"{k} {v['new_effect']:+.5f}"
                      for k, v in section["refutations"].items()))
    return section


def main():
    args = parse()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    con = D.connect(args.threads)
    if args.only_dowhy:
        path = out / "results.json"
        results = json.loads(path.read_text())
        results["dowhy"] = dowhy_section(args, con)
        path.write_text(json.dumps(results, indent=2))
        return
    tracking.setup(args.mlflow)
    timings, results = {}, {"data": str(args.data), "outcome": args.outcome}

    with tracking.run("pipeline", params=vars(args)):
        started = time.time()
        facts = D.summary(args.data, con, args.outcome)
        timings["summary_s"] = round(time.time() - started, 1)
        results["trial"] = facts
        print(f"{facts['rows']:,} rows, treated {facts['treated_share']:.3f}, "
              f"ATE {facts['ate']:+.5f}, max |SMD| {facts['max_abs_smd']:.4f}")
        tracking.metrics({"ate": facts["ate"], "rows": facts["rows"],
                          "max_abs_smd": facts["max_abs_smd"]})

        # ---------------------------------------------- DoWhy, on a sample
        if args.dowhy_rows:
            started = time.time()
            results["dowhy"] = dowhy_section(args, con)
            timings["dowhy_s"] = round(time.time() - started, 1)

        # ------------------------------------------------------- samples
        started = time.time()
        train = D.sample(args.data, "train", args.train_rows, 0, con, args.outcome)
        val = D.sample(args.data, "val", args.val_rows, 1, con, args.outcome)
        timings["sampling_s"] = round(time.time() - started, 1)
        print(f"train sample {len(train[0]):,}, val sample {len(val[0]):,}")

        test_frames = list(D.batches(args.data, "test", args.batch,
                                     FEATURES + [TREATMENT, args.outcome]))
        t_test = np.concatenate([f[TREATMENT].to_numpy() for f in test_frames])
        y_test = np.concatenate([f[args.outcome].to_numpy() for f in test_frames])
        results["test_rows"] = int(len(t_test))

        # -------------------------------------------------------- tuning
        params = None
        if args.trials:
            from causal_uplift.tuning import tune

            started = time.time()
            with tracking.run("optuna", nested=True):
                params, study = tune(
                    train, val, args.trials,
                    callback=lambda i, p, s: tracking.metrics({"val_auuc": s},
                                                              prefix=f"trial{i}_"))
                tracking.metrics({"best_val_auuc": study.best_value})
                tracking.artifact_json("best_params.json", params, out)
            timings["tuning_s"] = round(time.time() - started, 1)
            results["tuning"] = {"trials": args.trials, "best_params": params,
                                 "best_val_auuc": study.best_value}
            print(f"Optuna: best val AUUC {study.best_value:.5f} with {params}")

        # -------------------------------------------- every learner, test
        models, scores, curves, rows = {}, {}, {}, {}
        for name in args.models:
            with tracking.run(name, nested=True, params=params or {}):
                X, t, y = train
                if name == "causal_forest" and len(X) > args.forest_rows:
                    X, t, y = X[:args.forest_rows], t[:args.forest_rows], \
                        y[:args.forest_rows]
                started = time.time()
                model = MODELS[name](params, seed=0).fit(X, t, y)
                fit_s = time.time() - started
                val_auuc = auuc(model.effect(val[0]), val[1], val[2])

                started = time.time()
                score = score_in_batches(model, test_frames, FEATURES)
                score_s = time.time() - started
                row = {
                    "fit_rows": int(len(X)), "fit_s": round(fit_s, 1),
                    "test_score_s": round(score_s, 1),
                    "val_auuc": val_auuc,
                    "test_auuc": auuc(score, t_test, y_test),
                    "test_qini": qini_coefficient(score, t_test, y_test),
                    "uplift_at_10": uplift_at(score, t_test, y_test, 0.1),
                    "uplift_at_30": uplift_at(score, t_test, y_test, 0.3),
                }
                row["test_auuc_ci95"] = list(bootstrap(
                    auuc, score, t_test, y_test, n=args.bootstrap))
                row["policy"] = evaluate_budgets(
                    score, t_test, y_test, args.budgets, args.value, args.cost)
                tracking.metrics({k: v for k, v in row.items()
                                  if not isinstance(v, (list, dict))})
                tracking.metrics({f"captured_at_{int(b['budget'] * 100)}":
                                  b["captured"] for b in row["policy"]["budgets"]})
                rows[name], scores[name], models[name] = row, score, model
                f, g = uplift_curve(score, t_test, y_test)
                qf, q = qini_curve(score, t_test, y_test)
                curves[name] = (f, g, qf, q)
                print(f"  {name:14s} val AUUC {val_auuc:.5f}  "
                      f"test AUUC {row['test_auuc']:.5f}  "
                      f"Qini {row['test_qini']:.4f}  fit {fit_s:.0f}s")

        if BASELINE in scores:
            for name, row in rows.items():
                if name in (BASELINE, "random"):
                    continue
                base = rows[BASELINE]["test_auuc"]
                row["auuc_vs_s_learner"] = (row["test_auuc"] - base) / abs(base)
                row["auuc_diff_vs_s_learner_ci95"] = list(bootstrap(
                    auuc, scores[name], t_test, y_test, n=args.bootstrap,
                    baseline=scores[BASELINE]))
        results["models"] = rows

        candidates = {k: v for k, v in rows.items() if k != "random"}
        best = max(candidates, key=lambda k: candidates[k]["val_auuc"])
        results["best_model"] = best
        print(f"Best on validation: {best}")

        figure(curves, out / "uplift_curves.png")
        tracking.artifact(out / "uplift_curves.png")

        # ----------------------------------- score every row, out of core
        if not args.no_score_all:
            target = Path(args.data).with_name("scores.parquet")
            started = time.time()
            writer, written = None, 0
            for frame in D.batches(args.data, None, args.batch,
                                   ["row_id"] + FEATURES):
                effect = models[best].effect(
                    frame.select(FEATURES).to_numpy().astype(np.float32))
                table = pa.table({"row_id": frame["row_id"].to_numpy(),
                                  "uplift": effect.astype(np.float32)})
                writer = writer or pq.ParquetWriter(target, table.schema)
                writer.write_table(table)
                written += len(frame)
            writer.close()
            timings["score_all_s"] = round(time.time() - started, 1)
            results["scored_rows"] = written
            print(f"Scored {written:,} rows with {best} -> {target} "
                  f"in {timings['score_all_s']:.0f}s")

        results["timings"] = timings
        tracking.artifact_json("results.json", results, out)
    print(f"\n{out / 'results.json'}")


if __name__ == "__main__":
    main()
