import numpy as np
import pytest

from pocketdetect.geometry import (ATOM_CLASSES, atom_class_matrix, neighbor_features, neighbor_features_torch,
                                   protrusion, protrusion_torch, sas_points, vdw_radii)

from geometry_helpers import blob_protein, load_pdb, needs_cuda

RADII = (4.0, 6.0, 8.0, 10.0, 12.0)
SIGMAS = (2.0, 4.0)


def brute(points, coords, props, radii=RADII, sigmas=SIGMAS):
    d2 = ((points[:, None, :].astype(np.float64) - coords[None].astype(np.float64)) ** 2).sum(-1)
    sums = np.stack([(d2 <= r * r).astype(np.float64) @ props for r in radii], 1)
    gauss = np.stack([(np.exp(-d2 / (2 * s * s)) * (d2 <= (3 * s) ** 2)) @ props for s in sigmas], 1)
    return sums, gauss


def test_neighbor_features_bruteforce():
    rng = np.random.default_rng(0)
    c, _ = blob_protein(12.0)
    props = rng.normal(size=(len(c), 5)).astype(np.float32)
    q = rng.uniform(-18, 18, size=(500, 3)).astype(np.float32)
    nf = neighbor_features(q, c, props, RADII, SIGMAS)
    s, g = brute(q, c, props)
    assert np.allclose(nf.sums, s, atol=1e-3)
    assert np.allclose(nf.gauss, g, atol=1e-3)


def test_counts_and_protrusion():
    c, _ = blob_protein(12.0)
    q = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    nf = neighbor_features(q, c, None, (10.0,))
    d = np.linalg.norm(c - q[0], axis=1)
    assert nf.sums[0, 0, 0] == (d <= 10).sum()
    assert nf.sums[1, 0, 0] == 0
    assert np.array_equal(protrusion(q, c), nf.sums[:, 0, 0])


def test_atom_class_matrix():
    el = np.array(["N", "C", "C", "O", "C", "C", "C", "N", "N", "O", "S"])
    nm = np.array(["N", "CA", "C", "O", "CB", "CG", "CZ", "NZ", "N", "OXT", "SG"])
    rn = np.array(["PHE", "PHE", "PHE", "PHE", "PHE", "PHE", "PHE", "LYS", "PRO", "GLY", "CYS"])
    m, cols = atom_class_matrix(el, nm, rn)
    col = {c: i for i, c in enumerate(cols)}
    assert cols == list(ATOM_CLASSES)
    assert m[:, col["all"]].sum() == len(el)
    assert m[0, col["donor"]] == 1 and m[8, col["donor"]] == 0          # Pro N is no donor
    assert m[3, col["acceptor"]] == 1 and m[9, col["negative"]] == 1     # O / OXT
    assert m[5, col["aromatic"]] == 1 and m[6, col["aromatic"]] == 1 and m[4, col["aromatic"]] == 0
    assert m[4, col["hydrophobic_sc_C"]] == 1 and m[1, col["hydrophobic_sc_C"]] == 0
    assert m[7, col["positive"]] == 1 and m[7, col["sidechain"]] == 1 and m[1, col["backbone"]] == 1
    assert m[10, col["S"]] == 1
    assert np.allclose(vdw_radii(["C", "N", "O", "S", "XX"]), [1.7, 1.55, 1.52, 1.8, 1.8])


@needs_cuda
@pytest.mark.parametrize("use_triton", [False, True])
@pytest.mark.parametrize("pid", [None, "1a28"])
def test_neighbor_cpu_gpu_agree(pid, use_triton):
    if pid is None:
        c, r = blob_protein()
        props = np.random.default_rng(1).normal(size=(len(c), 4)).astype(np.float32)
    else:
        c, el, nm, rn = load_pdb(pid)
        r = vdw_radii(el)
        props, _ = atom_class_matrix(el, nm, rn)
    pts = sas_points(c, r).points
    nf = neighbor_features(pts, c, props, RADII, SIGMAS)
    t = neighbor_features_torch(pts, c, props, radii=RADII, sigmas=SIGMAS, use_triton=use_triton)
    scale = np.abs(nf.sums).max()
    assert np.abs(t["sums"].cpu().numpy() - nf.sums).max() <= 1e-5 * scale + 1e-3
    assert np.abs(t["gauss"].cpu().numpy() - nf.gauss).max() <= 1e-5 * np.abs(nf.gauss).max() + 1e-3
    assert np.array_equal(protrusion_torch(pts, c).cpu().numpy(), protrusion(pts, c)) or pid is None


@needs_cuda
@pytest.mark.parametrize("use_triton", [False, True])
def test_neighbor_torch_batched(use_triton):
    a, _ = blob_protein(10.0, seed=1)
    b, _ = blob_protein(12.0, seed=2)
    qa = np.random.default_rng(0).uniform(-12, 12, (300, 3)).astype(np.float32)
    qb = qa * 1.1
    c = np.concatenate([a, b])
    ab = np.r_[np.zeros(len(a)), np.ones(len(b))].astype(np.int64)
    q = np.concatenate([qa, qb])
    qbatch = np.r_[np.zeros(300), np.ones(300)].astype(np.int64)
    t = neighbor_features_torch(q, c, None, qbatch, ab, RADII, use_triton=use_triton)["sums"].cpu().numpy()
    ref = np.concatenate([neighbor_features(qa, a, None, RADII).sums, neighbor_features(qb, b, None, RADII).sums])
    assert np.abs(t - ref).max() <= 1
