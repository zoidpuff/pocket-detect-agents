"""Unit-level equivalence of fast BURROW building blocks with their NumPy/SciPy references."""
import numpy as np
import pytest

from pocketdetect.burrow.fast import formation as FF


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pairwise_sum_matches_numpy(dtype):
    rng = np.random.default_rng(0)
    for n in list(range(0, 300)) + [1000, 4097, 9000, 50000]:
        a = (rng.random(n) * rng.choice([1.0, 1e3])).astype(dtype)
        assert FF._pw(a, 0, n) == a.sum()


@pytest.mark.parametrize("wdtype", [np.float32, np.float64])
def test_kmeans_split_matches_reference(wdtype):
    from pocketdetect.burrow.pockets import _kmeans_split
    rng = np.random.default_rng(1)
    for trial in range(40):
        n = int(rng.integers(12, 400))
        X = (rng.normal(size=(n, 3)) * [12, 3, 2]).astype(np.float32).astype(np.float64)
        w = rng.random(n).astype(wdtype)
        k = int(rng.integers(2, 5))
        ref = _kmeans_split(X, w, k)
        got = FF._kmeans_split(X, w, k, w.dtype.type(1e-9), 10)
        assert np.array_equal(ref, got)


def test_extent_close_to_reference():
    from pocketdetect.burrow.pockets import _extent
    rng = np.random.default_rng(2)
    for _ in range(50):
        X = (rng.normal(size=(int(rng.integers(3, 300)), 3)) * [10, 2, 1]).astype(np.float32)
        assert abs(_extent(X) - FF._extent(X.astype(np.float64), np.arange(len(X)))) < 1e-4


def test_point_graph_matches_ckdtree():
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(3)
    X = np.round(rng.uniform(0, 20, (3000, 3)))
    X = np.concatenate([X, rng.uniform(0, 20, (300, 3))]).astype(np.float32)
    ptr, adj = FF.point_graph(X, 2.05)
    pairs = cKDTree(X.astype(np.float64)).query_pairs(2.05, output_type="ndarray")
    ref = {(int(a), int(b)) for a, b in pairs} | {(int(b), int(a)) for a, b in pairs}
    got = {(i, int(j)) for i in range(len(X)) for j in adj[ptr[i]:ptr[i + 1]]}
    assert ref == got


def test_components_match_single_linkage():
    from pocketdetect.geometry.cluster import single_linkage
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 30, (1500, 3)).astype(np.float32)
    ptr, adj = FF.point_graph(X, 2.05)
    mask = rng.random(len(X)) < 0.6
    cptr, members = FF._components(ptr, adj, mask)
    sel = np.flatnonzero(mask)
    lab = single_linkage(X[sel], 2.05)
    ref = [sel[lab == c] for c in range(lab.max() + 1)]
    got = [members[cptr[i]:cptr[i + 1]] for i in range(len(cptr) - 1)]
    assert len(ref) == len(got) and all(np.array_equal(a, b) for a, b in zip(ref, got))


def test_nms_matches_reference():
    from pocketdetect.burrow.pockets import Pocket, nms
    rng = np.random.default_rng(5)
    for trial in range(20):
        K = 60
        idx = [np.unique(rng.integers(0, 400, int(rng.integers(1, 40)))) for _ in range(K)]
        res = [np.unique(rng.integers(0, 80, int(rng.integers(0, 15)))) for _ in range(K)]
        ctr = rng.uniform(0, 40, (K, 3))
        sc = np.round(rng.random(K), 2)            # ties on purpose
        pk = [Pocket(idx=i, strength=0.0, center=c, residues=frozenset(map(str, r))) for i, c, r in zip(idx, ctr, res)]
        c = FF.Candidates(np.concatenate([[0], np.cumsum([len(i) for i in idx])]), np.concatenate(idx),
                          np.zeros(K, np.int64), ctr, np.zeros(K),
                          np.concatenate([[0], np.cumsum([len(r) for r in res])]),
                          np.concatenate(res).astype(np.int64))
        for kw in (dict(dist=5.0, jacc=0.75), dict(dist=10.0, jacc=0.75, overlap=0.3, max_n=50),
                   dict(dist=8.0, jacc=None, min_score=0.3, min_keep=3)):
            assert list(FF.nms(c, sc, **kw)) == nms(pk, sc, **kw)
