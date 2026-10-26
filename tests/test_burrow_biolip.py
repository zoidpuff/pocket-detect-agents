"""BURROW BioLiP2 adapters and vectorised pocket features."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pocketdetect.burrow.model import pocket_features
from pocketdetect.burrow.pockets import Pocket, PointGraph, _extent, components

MIRROR = "/cluster/work/beltrao/shared/rsyncPDB/data/structures/divided/mmCIF"


@pytest.mark.parametrize("pdb,chain", [("6n9i", "A"), ("4hhb", "B"), ("7tgh", "A6")])
def test_write_chain_input_keeps_only_that_chain(tmp_path, pdb, chain):
    path = f"{MIRROR}/{pdb[1:3]}/{pdb}.cif.gz"
    if not os.path.exists(path):
        pytest.skip("PDB mirror not available")
    from pocketdetect.burrow.biolip import write_chain_input
    from pocketdetect.datasets.trainset import TrainEntry
    from pocketdetect.structure import load_structure
    e = TrainEntry(meta=dict(structure_path=path, chain=chain, entry_id="x"), arrays={})
    f = write_chain_input(e, tmp_path / "x.cif")
    s = load_structure(f, protein_only=True)
    ref = load_structure(path, chains=[chain], protein_only=True)
    assert s.chains == [chain]
    assert s.n_atoms == ref.n_atoms
    assert np.allclose(np.sort(s.coords, 0), np.sort(ref.coords, 0))


def test_point_graph_matches_single_linkage():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 15, (800, 3)).astype(np.float32)
    w = np.ones(len(X))
    sel = np.flatnonzero(rng.random(len(X)) < 0.6)
    a = components(X, sel, 2.05, w)
    b = components(X, sel, 2.05, w, graph=PointGraph(X, 2.05))
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_pocket_features_vectorised_values():
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(0, 4, (n, 3)).astype(np.float32)
    names = ["bur10", "bur16", "bur24", "burbb10", "burbb16", "depth", "psp", "nearest", "hit_apolar", "hit_polar",
             "hit_charged", "hit_aromatic", "escape", "metal_cand", "n_coord28", "n_all8", "n_all12", "prot_frac"]
    F = rng.random((n, len(names))).astype(np.float32)
    vol = np.ones(n, np.float32)
    score = rng.random(n).astype(np.float32)
    pk = [Pocket(idx=np.arange(0, 50), strength=1, center=np.zeros(3), residues=frozenset({"A:1", "B:2"}), level=101),
          Pocket(idx=np.arange(100, 400), strength=1, center=np.zeros(3), residues=frozenset({"A:1"}), level=2)]
    Fp, fn = pocket_features(pk, X, vol, F, names, score, 1000)
    c = {k: Fp[:, i] for i, k in enumerate(fn)}
    for j, p in enumerate(pk):
        assert c["n_pts"][j] == len(p.idx)
        assert np.isclose(c["score_max"][j], score[p.idx].max())
        assert np.isclose(c["score_sum"][j], score[p.idx].sum(), rtol=1e-5)
        assert np.isclose(c["score_top10"][j], np.sort(score[p.idx])[::-1][:10].mean(), rtol=1e-5)
        assert np.isclose(c["bur10_mean"][j], F[p.idx, 0].mean(), rtol=1e-5)
        assert np.isclose(c["extent"][j], _extent(X[p.idx]), rtol=1e-4)
    assert list(c["n_chains"]) == [2, 1] and list(c["is_sub"]) == [1, 0] and list(c["level"]) == [1, 2]
