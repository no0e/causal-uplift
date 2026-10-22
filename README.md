# Causal Uplift on the Criteo Trial

Who should receive an ad, when the question is not "who will visit" but "who will visit *because of* the ad".

The Criteo uplift dataset is a randomised trial: 14M users, 85% shown advertising and 15% held out, with the visit and conversion each user went on to make. On a trial the effect of the treatment can be measured without assumptions about who got it, so this repository fits five uplift models, ranks users by their predicted individual effect, and measures on held-out users how much of the campaign's total effect a targeted campaign keeps while treating only a fraction of them.

Results are filled in below from `docs/results.json` once the full run has finished.

## Pipeline

```
python scripts/fetch_criteo.py      # 14M rows, CSV -> Parquet through DuckDB
python scripts/run_pipeline.py      # everything else, logged to MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

1. **Out of core.** The CSV is streamed through DuckDB into one Parquet file with a row id and a split column (70/10/20 by a hash of the id, so it never depends on file order). Trial facts are SQL aggregates over that file: rates per arm, the average treatment effect with its interval, and the standardised mean difference of every feature between arms. Models are fitted on samples drawn by DuckDB, and the test split and the full table are scored in bounded Polars batches.
2. **Identification.** The average effect is written as a DoWhy causal model and stress-tested with a placebo treatment, a random common cause and data subsets.
3. **Tuning.** LightGBM hyperparameters are chosen by Optuna on validation AUUC and shared by every learner.
4. **Learners.** S-learner, T-learner, X-learner, DR-learner and a causal forest (EconML `CausalForestDML`), against random targeting. The propensity is the known treated share, since the trial is randomised.
5. **Evaluation on the whole test split.** AUUC (area between the uplift curve and random targeting, per user), Qini coefficient, uplift in the top 10% and 30%, and a paired bootstrap interval against the S-learner.
6. **Policy.** Treat the top *k*% by predicted uplift, valued on held-out users by inverse propensity weighting, which is unbiased on a randomised trial. The economics are simulated: one extra visit is worth 1, one treatment costs 0.002.

## Data

Diemert, Betlei, Renaudin, Amini, *A Large Scale Benchmark for Uplift Modeling*, AdKDD 2018. Criteo AI Lab, CC BY-NC-SA 4.0. Downloaded by `scripts/fetch_criteo.py`, never committed.

## Running without the data

```
python scripts/fetch_criteo.py --synthetic 200000
python scripts/run_pipeline.py --data data/synthetic.parquet --quick
pytest -q
```

The synthetic set has the Criteo schema and a known individual effect, which is what the tests use to check that the metrics reward the right ranking and that the policy estimate recovers the true value. CI runs the tests, ruff, the synthetic pipeline end to end and the Docker image on every push.

## Licence

MIT for the code. The data carries its own licence.
