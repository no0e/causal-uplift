"""The data path, out of core from the CSV to the scored table.

The Criteo uplift file is 14M rows. Nothing here loads it whole into Python:

- `prepare` streams the gzipped CSV through DuckDB into one Parquet file,
  adding a row id and a split column derived from a hash of that id, so the
  split is deterministic and needs no shuffled copy.
- `summary` computes the per-arm rates, the average treatment effect and the
  covariate balance as SQL aggregates over the Parquet file.
- `sample` draws a reproducible hash-based sample of one split with DuckDB,
  which is what the models are fitted on.
- `batches` yields a split as Polars frames of a bounded size, which is how
  the fitted models score millions of rows without holding them.
"""
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from . import FEATURES, OUTCOMES, TREATMENT

# The original link, now a fallback: see scripts/fetch_criteo.py.
URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
SPLITS = {"train": (0, 70), "val": (70, 80), "test": (80, 100)}


def connect(threads=None, memory_limit=None):
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    return con


def prepare(source, target, con=None):
    """CSV (gzipped or not) -> Parquet with `row_id` and `split`. Returns rows.

    The split bucket is `hash(row_id) % 100`, so a row's split never depends
    on the order of the file or on anything that was sampled before it.
    """
    con = con or connect()
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    cases = " ".join(
        f"WHEN bucket >= {low} AND bucket < {high} THEN '{name}'"
        for name, (low, high) in SPLITS.items())
    columns = ", ".join(
        [f"CAST({f} AS FLOAT) AS {f}" for f in FEATURES]
        + [f"CAST({TREATMENT} AS TINYINT) AS {TREATMENT}"]
        + [f"CAST({o} AS TINYINT) AS {o}" for o in OUTCOMES])
    con.execute(f"""
        COPY (
            SELECT row_id, CASE {cases} END AS split, * EXCLUDE (row_id, bucket)
            FROM (
                SELECT *, hash(row_id) % 100 AS bucket
                FROM (SELECT row_number() OVER () - 1 AS row_id, {columns}
                      FROM read_csv_auto('{source}', header = true))
            )
        ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
    """)
    return con.execute(f"SELECT count(*) FROM '{target}'").fetchall()[0][0]


def summary(path, con=None, outcome="visit"):
    """Rows per split, rates per arm, the ATE with a 95% interval, balance."""
    con = con or connect()
    splits = dict(con.execute(
        f"SELECT split, count(*) FROM '{path}' GROUP BY split").fetchall())
    arms = con.execute(f"""
        SELECT {TREATMENT}, count(*), avg({outcome}), var_samp({outcome})
        FROM '{path}' GROUP BY {TREATMENT} ORDER BY {TREATMENT}
    """).fetchall()
    (_, n_c, rate_c, var_c), (_, n_t, rate_t, var_t) = arms
    ate = rate_t - rate_c
    se = float(np.sqrt(var_t / n_t + var_c / n_c))

    # Standardised mean difference per feature: a randomised trial should
    # show every one of these near zero, and a large one is a data problem.
    smd = {}
    for f in FEATURES:
        (m0, v0), (m1, v1) = con.execute(f"""
            SELECT avg({f}), var_samp({f}) FROM '{path}'
            GROUP BY {TREATMENT} ORDER BY {TREATMENT}
        """).fetchall()
        smd[f] = float((m1 - m0) / np.sqrt((v0 + v1) / 2 + 1e-12))

    return {
        "rows": int(n_c + n_t), "splits": {k: int(v) for k, v in splits.items()},
        "treated_share": n_t / (n_t + n_c),
        "outcome": outcome,
        "rate_control": rate_c, "rate_treated": rate_t,
        "ate": ate, "ate_ci95": [ate - 1.96 * se, ate + 1.96 * se],
        "relative_lift": ate / rate_c if rate_c else None,
        "max_abs_smd": max(abs(v) for v in smd.values()), "smd": smd,
    }


def sample(path, split, n_rows, seed=0, con=None, outcome="visit"):
    """(X, t, y) for a reproducible random sample of about `n_rows` of `split`.

    A row is kept when a hash of its id and the seed falls under the sampling
    fraction: a Bernoulli sample that is the same on every run and every
    machine, and keeps the trial's allocation ratio in expectation
    (`tests/test_data.py` checks it stays within a point).
    """
    con = con or connect()
    total = con.execute(
        f"SELECT count(*) FROM '{path}' WHERE split = '{split}'").fetchall()[0][0]
    fraction = min(1.0, n_rows / max(total, 1))
    cols = ", ".join(FEATURES + [TREATMENT, outcome])
    frame = con.sql(f"""
        SELECT {cols} FROM '{path}'
        WHERE split = '{split}'
          AND hash(row_id + {int(seed) * 1_000_003}) % 1000000 < {int(fraction * 1e6)}
    """).pl()
    return arrays(frame, outcome)


def arrays(frame, outcome="visit"):
    """A Polars frame -> (X float32, t int8, y int8) numpy arrays."""
    return (frame.select(FEATURES).to_numpy().astype(np.float32),
            frame[TREATMENT].to_numpy().astype(np.int8),
            frame[outcome].to_numpy().astype(np.int8))


def batches(path, split=None, size=1_000_000, columns=None):
    """Yield `split` (or the whole file) as Polars frames of <= `size` rows."""
    lazy = pl.scan_parquet(path)
    if split is not None:
        lazy = lazy.filter(pl.col("split") == split)
    if columns is not None:
        lazy = lazy.select(columns)
    yield from lazy.collect_batches(chunk_size=size)
