"""The out-of-core path and the learners, on a small synthetic CSV."""
import numpy as np
import pytest

from causal_uplift import FEATURES
from causal_uplift import data as D
from causal_uplift.metrics import auuc
from causal_uplift.models import MODELS, score_in_batches
from causal_uplift.synthetic import generate, write_csv

N = 30_000


@pytest.fixture(scope="module")
def parquet(tmp_path_factory):
    directory = tmp_path_factory.mktemp("data")
    source = write_csv(directory / "synthetic.csv", N, seed=5)
    target = directory / "synthetic.parquet"
    assert D.prepare(source, target) == N
    return target


def test_splits_are_disjoint_and_cover_every_row(parquet):
    con = D.connect()
    counts = dict(con.execute(
        f"SELECT split, count(*) FROM '{parquet}' GROUP BY split").fetchall())
    assert sum(counts.values()) == N
    assert set(counts) == {"train", "val", "test"}
    assert counts["train"] / N == pytest.approx(0.7, abs=0.02)
    ids = con.execute(f"SELECT count(DISTINCT row_id) FROM '{parquet}'").fetchall()[0]
    assert ids[0] == N


def test_the_split_is_the_same_on_every_run(parquet, tmp_path):
    again = tmp_path / "again.parquet"
    D.prepare(str(parquet).replace(".parquet", ".csv"), again)
    con = D.connect()
    differ = con.execute(f"""
        SELECT count(*) FROM '{parquet}' a JOIN '{again}' b USING (row_id)
        WHERE a.split <> b.split
    """).fetchall()[0][0]
    assert differ == 0


def test_the_summary_sees_a_balanced_trial_and_a_positive_effect(parquet):
    facts = D.summary(parquet)
    assert facts["treated_share"] == pytest.approx(0.85, abs=0.01)
    assert facts["max_abs_smd"] < 0.05
    assert facts["ate_ci95"][0] > 0


def test_sampling_is_reproducible_and_keeps_the_allocation(parquet):
    X1, t1, _ = D.sample(parquet, "train", 8_000, seed=0)
    X2, t2, _ = D.sample(parquet, "train", 8_000, seed=0)
    X3, _, _ = D.sample(parquet, "train", 8_000, seed=1)
    assert np.array_equal(X1, X2) and np.array_equal(t1, t2)
    assert not np.array_equal(X1[:100], X3[:100])
    assert len(X1) == pytest.approx(8_000, rel=0.1)
    assert t1.mean() == pytest.approx(0.85, abs=0.02)


def test_batches_cover_the_split_without_exceeding_the_size(parquet):
    sizes = [len(frame) for frame in D.batches(parquet, "test", size=1_000)]
    assert max(sizes) <= 1_000
    con = D.connect()
    expected = con.execute(
        f"SELECT count(*) FROM '{parquet}' WHERE split = 'test'").fetchall()[0][0]
    assert sum(sizes) == expected


@pytest.mark.parametrize("name", ["s_learner", "t_learner", "x_learner",
                                  "dr_learner", "causal_forest"])
def test_every_learner_ranks_better_than_random(name):
    frame = generate(40_000, seed=11)
    X = frame.select(FEATURES).to_numpy()
    t, y = frame["treatment"].to_numpy(), frame["visit"].to_numpy()
    fit, test = slice(0, 30_000), slice(30_000, None)
    params = {"n_estimators": 100, "num_leaves": 15, "min_child_samples": 100}
    extra = {"n_estimators": 100, "min_samples_leaf": 50} \
        if name == "causal_forest" else {}
    model = MODELS[name](params, seed=0, **extra).fit(X[fit], t[fit], y[fit])
    score = model.effect(X[test])
    assert score.shape == (10_000,)
    assert auuc(score, t[test], y[test]) > 0.002


def test_batched_scoring_matches_scoring_at_once(parquet):
    X, t, y = D.sample(parquet, "train", 20_000)
    params = {"n_estimators": 50, "num_leaves": 15}
    model = MODELS["t_learner"](params).fit(X, t, y)
    frames = list(D.batches(parquet, "val", size=777))
    batched = score_in_batches(model, frames, FEATURES)
    whole = np.concatenate([f.select(FEATURES).to_numpy() for f in frames])
    assert np.allclose(batched, model.effect(whole.astype(np.float32)))
