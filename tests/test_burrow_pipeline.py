"""BURROW pipeline tests with known answers (synthetic cavity, metal site), invariances
(HETATM records, rigid rotation), AF-mode no-op, output format."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pocketdetect.burrow.af import af_protein
from pocketdetect.burrow.features import BurrowConfig, metal_candidates, point_features
from pocketdetect.burrow.model import BurrowModel
from pocketdetect.burrow.pipeline import predict_protein, write_prediction
from pocketdetect.burrow.prep import load_protein, protein_from_arrays
from pocketdetect.evaluate import read_predictions
from pocketdetect.structure import load_structure

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/raw/p2rank_datasets/current/chen11/a.001.001.001_1s69a.pdb"  # has HETATM ligands
MODEL = ROOT / "results/dev/models/burrow_chen11_s0.json"


def _blob(radius=16.0, spacing=1.9, cavity_center=(6.0, 0.0, 0.0), cavity_r=6.0, seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(-radius, radius + 1e-6, spacing)
    xyz = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    xyz = xyz[np.linalg.norm(xyz, axis=1) <= radius]
    xyz = xyz[np.linalg.norm(xyz - np.asarray(cavity_center), axis=1) > cavity_r]
    xyz = xyz + rng.normal(0, 0.15, xyz.shape)
    n = len(xyz)
    return protein_from_arrays(xyz, ["C"] * n, ["CB"] * n, ["ALA"] * n, np.arange(1, n + 1), ["A"] * n)


def test_synthetic_buried_cavity_ranked_first():
    c = np.array([6.0, 0.0, 0.0])
    P = _blob(cavity_center=c)
    pred = predict_protein(P, None, ranker="geometry")
    assert len(pred.scores) >= 1
    assert np.linalg.norm(pred.centers[0] - c) < 2.5, pred.centers[:3]


def test_synthetic_cavity_rigid_motion_moves_answer():
    c = np.array([6.0, 0.0, 0.0])
    P = _blob(cavity_center=c)
    R = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
    R *= np.sign(np.linalg.det(R))
    t = np.array([30.0, -12.0, 5.0])
    P2 = replace(P, coords=(P.coords @ R.T + t).astype(np.float32))
    pred = predict_protein(P2, None, ranker="geometry")
    assert np.linalg.norm(pred.centers[0] - (c @ R.T + t)) < 2.5


def test_no_cavity_gives_weak_pockets():
    """Negative control: a cavity-free ball has a much weaker best pocket than a ball with a cavity."""
    s_cav = predict_protein(_blob(), None, ranker="geometry").scores[0]
    s_none = predict_protein(_blob(cavity_r=0.0), None, ranker="geometry").scores
    assert (s_none[0] if len(s_none) else 0.0) < 0.5 * s_cav


def test_metal_candidate_at_engineered_site():
    site = np.array([0.0, 0.0, 0.0])
    dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1]], float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ne2 = site + 2.1 * dirs
    ce1 = site + 3.1 * dirs + 0.8 * np.roll(dirs, 1, axis=1)  # ring carbons further out
    xyz = np.concatenate([ne2, ce1])
    P = protein_from_arrays(xyz, ["N"] * 3 + ["C"] * 3, ["NE2"] * 3 + ["CE1"] * 3, ["HIS"] * 6,
                            [1, 2, 3, 1, 2, 3], ["A"] * 6)
    pts, nco = metal_candidates(P.coords, P, BurrowConfig())
    assert len(pts) == 1
    assert np.linalg.norm(pts[0] - site) < 0.6
    assert nco[0] == 3


@pytest.mark.skipif(not REAL.exists(), reason="P2Rank datasets not available")
@pytest.mark.parametrize("use_model", [False, True])
def test_hetatm_invariance(tmp_path, use_model):
    if use_model and not MODEL.exists():
        pytest.skip("dev model not trained")
    lines = REAL.read_text().splitlines()
    assert any(l.startswith("HETATM") for l in lines)
    stripped = tmp_path / "stripped.pdb"
    stripped.write_text("\n".join(l for l in lines if not l.startswith("HETATM")) + "\n")
    model = BurrowModel.load(MODEL) if use_model else None
    outs = []
    for i, f in enumerate((REAL, stripped)):
        P = load_protein(f, structure_id="x")
        pred = predict_protein(P, model, ranker="learned" if use_model else "geometry")
        d = tmp_path / f"o{i}"
        write_prediction(d, pred)
        outs.append((d / "x.csv").read_text())
    assert outs[0] == outs[1]


@pytest.mark.skipif(not REAL.exists(), reason="P2Rank datasets not available")
def test_rotation_invariance_real():
    """Canonical-frame features: predictions follow a rigid motion (tolerance: top-3 centres
    within 1 Å and residue Jaccard >= 0.8; voxel-boundary round-off may flip a few points)."""
    P = load_protein(REAL)
    model = BurrowModel.load(MODEL) if MODEL.exists() else None
    R = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))[0]
    R *= np.sign(np.linalg.det(R))
    t = np.array([-40.0, 7.0, 13.0])
    P2 = replace(P, coords=(P.coords.astype(np.float64) @ R.T + t).astype(np.float32))
    kw = dict(ranker="learned" if model else "geometry")
    a = predict_protein(P, model, **kw)
    b = predict_protein(P2, model, **kw)
    k = min(3, len(a.scores))
    assert k >= 1
    back = (b.centers[:k] - t) @ R
    assert np.all(np.linalg.norm(back - a.centers[:k], axis=1) < 1.0)
    for ra, rb in zip(a.residues[:k], b.residues[:k]):
        assert len(ra & rb) / max(len(ra | rb), 1) >= 0.8


@pytest.mark.skipif(not REAL.exists(), reason="P2Rank datasets not available")
def test_af_mode_noop_on_high_plddt():
    s = load_structure(REAL, protein_only=True)
    s.b_factor[:] = 95.0
    P, w, st = af_protein(s)
    assert w is None and st["n_res_pruned"] == 0
    P0 = load_protein(REAL)
    assert np.array_equal(P.coords, P0.coords)
    f1 = point_features(P, weights=w)
    f0 = point_features(P0)
    assert np.array_equal(f1.feats, f0.feats)


def test_af_pruning_removes_low_confidence_termini():
    n = 60
    xyz = np.stack([np.arange(n) * 3.8, np.zeros(n), np.zeros(n)], 1)
    P = protein_from_arrays(xyz, ["C"] * n, ["CA"] * n, ["ALA"] * n, np.arange(1, n + 1), ["A"] * n)
    del P
    from pocketdetect.structure import PROTEIN, Structure
    pl = np.full(n, 90.0)
    pl[:10] = 30.0      # terminal low -> pruned
    pl[20:25] = 30.0    # internal stretch of 5 (< 7) -> kept, soft
    pl[30:40] = 30.0    # internal stretch of 10 -> pruned
    s = Structure(id="t", coords=xyz.astype(np.float32), element=np.array(["C"] * n), atom_name=np.array(["CA"] * n),
                  altloc=np.array([""] * n), occupancy=np.ones(n, np.float32), b_factor=pl.astype(np.float32),
                  serial=np.arange(n, dtype=np.int32), res_index=np.arange(n, dtype=np.int32), chain=np.array(["A"] * n),
                  res_name=np.array(["ALA"] * n), res_num=np.arange(1, n + 1, dtype=np.int32), ins_code=np.array([""] * n),
                  is_hetero=np.zeros(n, bool), r_chain=np.array(["A"] * n), r_name=np.array(["ALA"] * n),
                  r_num=np.arange(1, n + 1, dtype=np.int32), r_icode=np.array([""] * n),
                  r_kind=np.full(n, PROTEIN, np.int8), r_hetero=np.zeros(n, bool), r_entity=np.array(["Polymer"] * n))
    P, w, st = af_protein(s)
    assert st["n_res_pruned"] == 20
    assert len(P.coords) == 40 and w is not None and (w == 0).sum() == 5


def test_output_format_roundtrip(tmp_path):
    pred = predict_protein(_blob(), None, ranker="geometry")
    write_prediction(tmp_path, replace(pred, structure_id="blob"))
    pr = read_predictions(tmp_path / "blob.csv", "blob")
    assert pr.n == len(pred.scores)
    assert np.allclose(pr.centers, pred.centers, atol=1e-3)
    pts = (tmp_path / "blob.points.csv").read_text().splitlines()
    assert pts[0] == "rank,x,y,z" and len(pts) - 1 == sum(len(p) for p in pred.points)
