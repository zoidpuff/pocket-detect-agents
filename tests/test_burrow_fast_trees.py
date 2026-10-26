import json
from pathlib import Path

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from pocketdetect.burrow.fast.trees import FlatForest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = sorted((ROOT / "results/dev/models").glob("burrow_*.json"))


def cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def boundary_rows(f: FlatForest, n=2000, seed=0):
    """Rows whose values sit exactly on / just around split thresholds (worst case for ties)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, f.n_features), np.float32)
    for j in range(f.n_features):
        t = f.thr[f.feat == j]
        if len(t) == 0:
            X[:, j] = rng.normal(size=n)
            continue
        base = rng.choice(t, n).astype(np.float32)
        jitter = rng.integers(-1, 2, n)
        X[:, j] = np.where(jitter < 0, np.nextafter(base, np.float32(-np.inf)),
                           np.where(jitter > 0, np.nextafter(base, np.float32(np.inf)), base))
    return X


def synthetic_booster(missing: str, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3000, 8)).astype(np.float32)
    y = ((X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + rng.normal(0, 0.5, 3000)) > 0).astype(float)
    if missing == "nan":
        X[rng.random(X.shape) < 0.1] = np.nan
        params = dict(use_missing=True)
    elif missing == "zero":
        X[rng.random(X.shape) < 0.2] = 0.0
        params = dict(use_missing=True, zero_as_missing=True)
    else:
        params = dict(use_missing=False)
    p = dict(objective="binary", num_leaves=31, learning_rate=0.1, verbose=-1, seed=1, deterministic=True,
             num_threads=1, **params)
    return lgb.train(p, lgb.Dataset(X, y), 40), X


@pytest.mark.parametrize("missing", ["none", "nan", "zero"])
def test_synthetic_models_bit_identical(missing):
    b, X = synthetic_booster(missing)
    f = FlatForest.from_booster(b)
    Xt = np.concatenate([X, boundary_rows(f, 500)])
    if missing != "none":
        Xt[::7, 3] = np.nan
        Xt[::5, 4] = 0.0
    ref_raw = b.predict(Xt, raw_score=True, num_threads=1)
    ref = b.predict(Xt, num_threads=1)
    assert np.array_equal(f.predict(Xt, raw_score=True), ref_raw)
    assert np.array_equal(f.predict(Xt), ref)


def test_single_leaf_and_sigmoid_param():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(float)
    b = lgb.train(dict(objective="binary", sigmoid=2.0, num_leaves=4, verbose=-1, min_data_in_leaf=500),
                  lgb.Dataset(X, y), 3)   # min_data_in_leaf > n -> single-leaf trees
    f = FlatForest.from_booster(b)
    assert f.sigmoid == 2.0
    assert np.array_equal(f.predict(X), b.predict(X))


@pytest.mark.skipif(not MODELS, reason="no trained BURROW models")
@pytest.mark.parametrize("path", MODELS[:1])
def test_burrow_models_bit_identical(path):
    d = json.loads(path.read_text())
    for key in ("point_model", "rank_model"):
        if not d.get(key):
            continue
        b = lgb.Booster(model_str=d[key])
        f = FlatForest.from_booster(b)
        X = boundary_rows(f, 3000, seed=1)
        assert np.array_equal(f.predict(X, raw_score=True), b.predict(X, raw_score=True, num_threads=1))
        assert np.array_equal(f.predict(X), b.predict(X, num_threads=1))


@pytest.mark.skipif(not cuda(), reason="CUDA not available")
@pytest.mark.parametrize("missing", ["none", "nan", "zero"])
def test_gpu_matches(missing):
    b, X = synthetic_booster(missing, seed=2)
    f = FlatForest.from_booster(b)
    Xt = np.concatenate([X, boundary_rows(f, 500)])
    tf = f.to_torch("cuda:0")
    raw = tf.predict(Xt, raw_score=True).cpu().numpy()
    ref = b.predict(Xt, raw_score=True, num_threads=1)
    assert np.abs(raw - ref).max() <= 1e-9
    assert np.abs(tf.predict(Xt).cpu().numpy() - b.predict(Xt, num_threads=1)).max() <= 1e-9


@pytest.mark.skipif(not cuda() or not MODELS, reason="CUDA / models not available")
def test_gpu_burrow_model():
    d = json.loads(MODELS[0].read_text())
    b = lgb.Booster(model_str=d["point_model"])
    f = FlatForest.from_booster(b)
    X = boundary_rows(f, 5000, seed=3)
    p = f.to_torch("cuda:0").predict(X).cpu().numpy()
    assert np.abs(p - b.predict(X, num_threads=1)).max() <= 1e-9
