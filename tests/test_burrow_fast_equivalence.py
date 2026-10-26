"""Fast BURROW backend vs the reference implementation (exact equality on dev structures)."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/dev/biolip/inputs_audited.csv"
MODELS = [ROOT / f"results/dev/freeze/models/burrow_all_{p}_s0.json" for p in ("default", "accurate")]
pytestmark = pytest.mark.skipif(not (MANIFEST.exists() and all(m.exists() for m in MODELS)),
                                reason="dev data / frozen models not available")


def cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module", params=MODELS, ids=["default", "accurate"])
def dev(request):
    from pocketdetect.burrow.devdata import read_manifest
    from pocketdetect.burrow.model import BurrowModel
    from pocketdetect.burrow.prep import load_protein
    from pocketdetect.burrow.fast.pipeline import FastModel
    df = read_manifest(MANIFEST).sample(5, random_state=123)
    Ps = [load_protein(str(ROOT / p), structure_id=s) for s, p in zip(df.structure_id, df.structure_path)]
    m = BurrowModel.load(request.param)
    return Ps, m, FastModel.from_model(m)


def same_prediction(a, b):
    return (np.array_equal(a.scores, b.scores) and np.array_equal(a.centers, b.centers)
            and list(a.residues) == list(b.residues)
            and all(np.array_equal(x, y) for x, y in zip(a.points, b.points)))


def test_guard_validated():
    from pocketdetect.burrow.fast.guard import _status
    _status.cache_clear()
    st = _status()
    assert all(st.values()), f"reference code changed since validation: {st}"


def test_features_identical(dev):
    from pocketdetect.burrow.features import point_features as ref
    from pocketdetect.burrow.fast.features import point_features as fast
    Ps, _, fm = dev
    for P in Ps[:3]:
        a, b = ref(P, fm.cfg), fast(P, fm.cfg)
        assert a.names == b.names
        assert np.array_equal(a.feats, b.feats) and np.array_equal(a.points, b.points)
        assert np.array_equal(a.vol, b.vol) and np.array_equal(a.is_metal, b.is_metal)


@pytest.mark.parametrize("ranker", ["learned", "geometry"])
def test_cpu_predictions_identical(dev, ranker):
    from pocketdetect.burrow.pipeline import predict_protein as ref
    from pocketdetect.burrow.fast.pipeline import predict_protein as fast
    Ps, m, fm = dev
    for P in Ps:
        assert same_prediction(ref(P, m, ranker=ranker), fast(P, fm, ranker=ranker)), P.id


def test_fallback_when_reference_changes(dev, monkeypatch):
    """With the drift guard tripped, the fast pipeline must still give reference results."""
    from pocketdetect.burrow.pipeline import predict_protein as ref
    from pocketdetect.burrow.fast import guard, pipeline as FP
    Ps, m, fm = dev
    monkeypatch.setattr(guard, "stage_ok", lambda s: False)
    assert same_prediction(ref(Ps[0], m), FP.predict_protein(Ps[0], fm))


@pytest.mark.skipif(not cuda(), reason="CUDA not available")
def test_gpu_batch_identical(dev):
    from pocketdetect.burrow.pipeline import predict_protein as ref
    from pocketdetect.burrow.fast.gpu import predict_batch
    Ps, m, fm = dev
    out = predict_batch(Ps, fm, device="cuda:0", batch_size=3)
    for P, f in zip(Ps, out):
        assert same_prediction(ref(P, m), f), P.id


@pytest.mark.skipif(not cuda(), reason="CUDA not available")
def test_gpu_features_identical(dev):
    from pocketdetect.burrow.features import point_features as ref
    from pocketdetect.burrow.fast import gpu
    Ps, _, fm = dev
    cfg = fm.cfg
    res = gpu.featurize_batch([gpu.prep(P, cfg) for P in Ps], cfg, "cuda:0")
    for P, (ps, _) in zip(Ps, res):
        a = ref(P, cfg)
        assert a.names == ps.names and np.array_equal(a.feats, ps.feats) and np.array_equal(a.X, ps.X)


def test_af_soft_weights_identical(dev):
    """AF mode: soft occlusion weights through both CPU and GPU fast paths."""
    from pocketdetect.burrow.pipeline import predict_protein as ref
    from pocketdetect.burrow.fast.pipeline import predict_protein as fast
    Ps, m, fm = dev
    P = Ps[1]
    rng = np.random.default_rng(0)
    w = np.clip(rng.uniform(0.2, 1.6, len(P.coords)), 0, 1).astype(np.float32)
    # exact AF weights (pLDDT 60 / 70 -> w = 0.5 / 0.75 -> mu = ln2/2, ln2): first-hit ties at tau == ln2
    w[rng.random(len(w)) < 0.3] = 0.75
    w[rng.random(len(w)) < 0.2] = 0.5
    r = ref(P, m, weights=w)
    assert same_prediction(r, fast(P, fm, weights=w))
    if cuda():
        from pocketdetect.burrow.fast.gpu import predict_batch
        assert same_prediction(r, predict_batch([P], fm, device="cuda:0", weights=[w])[0])
