import numpy as np
import pytest

from pocketdetect.geometry import (fibonacci_sphere, grid_candidates, grid_candidates_torch, sas_points,
                                   sas_points_torch)

from geometry_helpers import blob_protein, load_pdb, needs_cuda


def test_fibonacci_sphere_unit_and_symmetric():
    d = fibonacci_sphere(60)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-6)
    assert (d[:, 2] < 0).sum() == 30
    assert np.abs(d.mean(0)).max() < 0.02


def test_single_atom_sas():
    c = np.zeros((1, 3), np.float32)
    s = sas_points(c, [1.7], probe=1.4, n_sphere=50)
    assert len(s.points) == 50
    assert np.allclose(np.linalg.norm(s.points, axis=1), 3.1, atol=1e-5)
    assert np.all(s.atom_index == 0)
    assert np.allclose(s.points, 3.1 * s.normals, atol=1e-5)


def test_two_atoms_buried_cap():
    # two equal atoms at distance d: point of atom 0 removed iff inside sphere of atom 1
    c = np.array([[0, 0, 0], [3.0, 0, 0]], np.float32)
    rp = 1.7 + 1.4
    s = sas_points(c, [1.7, 1.7], n_sphere=2000)
    # analytic: cap of sphere 0 inside sphere 1: x > d/2 -> fraction (1 - (d/2)/rp)/2
    frac = (1 - 1.5 / rp) / 2
    n0 = (s.atom_index == 0).sum()
    assert abs(n0 / 2000 - (1 - frac)) < 0.01
    # all kept points are outside every other expanded sphere
    for p, a in zip(s.points, s.atom_index):
        other = 1 - a
        assert np.linalg.norm(p - c[other]) >= rp - 1e-4


def test_sas_matches_bruteforce():
    c, r = blob_protein(10.0)
    s = sas_points(c, r, n_sphere=30)
    sph = fibonacci_sphere(30)
    rp = r + 1.4
    pts = c[:, None] + rp[:, None, None] * sph[None]
    d2 = ((pts[:, :, None, :] - c[None, None]) ** 2).sum(-1)               # (N,K,N)
    inside = d2 < (rp ** 2)[None, None]
    inside[np.arange(len(c)), :, np.arange(len(c))] = False
    keep = ~inside.any(-1)
    assert keep.sum() == len(s.points)
    ai, ki = np.nonzero(keep)
    assert np.array_equal(ai, s.atom_index)
    assert np.allclose(pts[ai, ki], s.points)


def test_grid_candidates_single_atom():
    c = np.zeros((1, 3), np.float32)
    g = grid_candidates(c, [1.7], spacing=0.5, probe=1.4, max_dist=6.0)
    d = np.linalg.norm(g.points, axis=1)
    assert d.min() >= 3.1 - 1e-5 and d.max() <= 6.0 + 1e-5
    assert np.allclose(d, g.nearest_dist, atol=1e-5)
    vol = 4 / 3 * np.pi * (6.0 ** 3 - 3.1 ** 3)
    assert abs(len(g.points) * 0.125 / vol - 1) < 0.03
    # grid aligned to multiples of spacing (centres at (i + 0.5) * h)
    assert np.allclose(np.mod(g.points / 0.5 - 0.5, 1.0) * (1 - np.mod(g.points / 0.5 - 0.5, 1.0)), 0, atol=1e-4)


def test_grid_candidates_bruteforce():
    c, r = blob_protein(8.0)
    g = grid_candidates(c, r, spacing=1.0, probe=1.4, max_dist=6.0)
    d = np.linalg.norm(g.points[:, None] - c[None], axis=-1)
    assert np.all(d >= r[None] + 1.4 - 1e-4)
    assert np.allclose(d.min(1), g.nearest_dist, atol=1e-4)
    assert np.all(d.min(1) <= 6.0 + 1e-4)
    # completeness: a dense superset grid restricted to the same lattice gives the same count
    lo = np.floor((c.min(0) - 6.0) / 1.0) - 1
    hi = np.ceil(c.max(0) + 6.0) + 1
    ax = [np.arange(lo[i], hi[i]) + 0.5 for i in range(3)]
    allp = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    dd = np.linalg.norm(allp[:, None] - c[None], axis=-1)
    ok = (dd >= r[None] + 1.4).all(1) & (dd.min(1) <= 6.0)
    assert ok.sum() == len(g.points)


@needs_cuda
@pytest.mark.parametrize("use_triton", [False, True])
@pytest.mark.parametrize("pid", [None, "1a28"])
def test_sas_cpu_gpu_agree(pid, use_triton):
    if pid is None:
        c, r = blob_protein()
    else:
        from pocketdetect.geometry import vdw_radii
        c, el, _, _ = load_pdb(pid)
        r = vdw_radii(el)
    s = sas_points(c, r)
    t = sas_points_torch(c, r, use_triton=use_triton)
    assert t["points"].shape[0] == len(s.points)
    assert np.array_equal(t["atom_index"].cpu().numpy(), s.atom_index)
    assert np.allclose(t["points"].cpu().numpy(), s.points, atol=1e-5)
    g = grid_candidates(c, r)
    gt = grid_candidates_torch(c, r)
    pc = {tuple(x) for x in np.round(g.points, 3)}
    pg = {tuple(x) for x in np.round(gt["points"].cpu().numpy(), 3)}
    # float32 (GPU) vs float64 (CPU) can flip voxels lying exactly on a boundary
    assert len(pc ^ pg) <= max(2, 1e-4 * len(pc))
    if len(pc ^ pg) == 0:
        assert np.allclose(gt["nearest_dist"].cpu().numpy(), g.nearest_dist, atol=1e-4)


@needs_cuda
@pytest.mark.parametrize("use_triton", [False, True])
def test_sas_torch_batched_equals_individual(use_triton):
    import torch
    a, ra = blob_protein(10.0, seed=1)
    b, rb = blob_protein(12.0, seed=2)
    c = np.concatenate([a, b + 0.5])          # overlapping in space, separated by batch
    r = np.concatenate([ra, rb])
    batch = np.r_[np.zeros(len(a)), np.ones(len(b))].astype(np.int64)
    t = sas_points_torch(c, r, batch, use_triton=use_triton)
    sa, sb = sas_points(a, ra), sas_points(b + 0.5, rb)
    tb = t["batch"].cpu().numpy()
    assert (tb == 0).sum() == len(sa.points) and (tb == 1).sum() == len(sb.points)
    assert np.allclose(t["points"][torch.from_numpy(tb == 1).cuda()].cpu().numpy(), sb.points, atol=1e-5)
