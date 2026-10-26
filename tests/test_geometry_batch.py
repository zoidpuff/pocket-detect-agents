import numpy as np
import pytest

from pocketdetect.geometry import (FeatureConfig, atom_class_matrix, compute_point_features,
                                   compute_point_features_batch_torch, vdw_radii)

from geometry_helpers import blob_protein, cuda_available, load_pdb, needs_cuda, rod_protein


def synthetic_structs():
    out = []
    for i, size in enumerate((9.0, 12.0, 15.0)):
        c, r = blob_protein(size, seed=i)
        props = np.random.default_rng(i).integers(0, 2, (len(c), 6)).astype(np.float32)
        w = np.random.default_rng(i).uniform(0, 1, len(c)).astype(np.float32)
        out.append(dict(coords=c, radii=r, props=props, weights=w))
    return out


def match_points(a, b, tol=1e-3):
    """Index arrays (ia, ib) pairing identical points of two point sets."""
    from scipy.spatial import cKDTree
    d, j = cKDTree(b).query(a)
    ok = d < tol
    return np.nonzero(ok)[0], j[ok]


@pytest.mark.parametrize("points", ["sas", "grid", "cascade"])
def test_cpu_pipeline_shapes(points):
    s = synthetic_structs()[0]
    cfg = FeatureConfig(points=points)
    f = compute_point_features(s["coords"], s["radii"], s["props"], s["weights"], cfg)
    p = len(f.points)
    assert p > 0
    assert f.blocked.shape == (p, len(cfg.ray_lengths)) and f.depth.shape == (p,) and f.psp.shape == (p,)
    assert f.sums.shape == (p, len(cfg.nb_radii), 6) and f.gauss.shape == (p, len(cfg.sigmas), 6)
    assert {"points", "prefilter", "buriedness", "neighbors", "total"} <= set(f.timings)
    assert f.psp.min() >= 0 and f.psp.max() <= cfg.psp_dirs


def test_cascade_is_subset_of_grid():
    s = synthetic_structs()[2]
    g = compute_point_features(s["coords"], s["radii"], s["props"], None, FeatureConfig(points="grid"))
    c = compute_point_features(s["coords"], s["radii"], s["props"], None, FeatureConfig(points="cascade"))
    ia, ib = match_points(c.points, g.points)
    assert len(ia) == len(c.points) and len(c.points) < len(g.points)
    assert np.array_equal(c.psp[ia], g.psp[ib])
    assert np.allclose(c.blocked[ia], g.blocked[ib], atol=1e-6)
    # every enclosed point (psp >= 2) survives the cascade
    assert (g.psp >= 2).sum() <= len(c.points)


@pytest.mark.parametrize("points", ["sas", "cascade"])
def test_bricking_matches_unbricked(points):
    c, r = rod_protein()
    props = np.random.default_rng(0).integers(0, 2, (len(c), 3)).astype(np.float32)
    ref = compute_point_features(c, r, props, None, FeatureConfig(points=points))
    cfg = FeatureConfig(points=points, max_voxels={"sas": 60_000, "cascade": 100_000}[points])
    br = compute_point_features(c, r, props, None, cfg)
    assert br.timings["n_bricks"] >= 2 and ref.timings["n_bricks"] == 1
    assert len(br.points) == len(ref.points)
    ia, ib = match_points(br.points, ref.points)
    assert len(ia) == len(ref.points)
    assert np.array_equal(br.atom_index[ia], ref.atom_index[ib])
    assert np.allclose(br.blocked[ia], ref.blocked[ib], atol=1e-5)
    assert np.allclose(br.sums[ia], ref.sums[ib])
    assert np.array_equal(br.psp[ia], ref.psp[ib])


@needs_cuda
@pytest.mark.parametrize("points", ["sas", "grid", "cascade"])
def test_batched_gpu_equals_cpu(points):
    structs = synthetic_structs()
    cfg = FeatureConfig(points=points)
    gpu = compute_point_features_batch_torch(structs, cfg)
    assert len(gpu) == len(structs)
    for s, g in zip(structs, gpu):
        f = compute_point_features(s["coords"], s["radii"], s["props"], s["weights"], cfg)
        assert abs(len(g["points"]) - len(f.points)) <= 2
        ia, ib = match_points(g["points"], f.points)
        assert len(ia) >= len(f.points) - 2
        assert np.mean(g["atom_index"][ia] == f.atom_index[ib]) > 0.999
        assert np.mean(g["psp"][ia] == f.psp[ib]) > 0.995
        assert np.abs(g["blocked"][ia] - f.blocked[ib]).mean() < 1e-3   # soft weights: rare boundary flips
        assert np.abs(g["sums"][ia] - f.sums[ib]).max() <= 1.0
        assert np.allclose(g["gauss"][ia], f.gauss[ib], atol=1e-2, rtol=1e-4)


@needs_cuda
def test_batched_gpu_real_structures():
    structs = []
    for pid in ("1a28", "1hsg"):
        c, el, nm, rn = load_pdb(pid)
        structs.append(dict(coords=c, radii=vdw_radii(el), props=atom_class_matrix(el, nm, rn)[0]))
    gpu = compute_point_features_batch_torch(structs)
    for s, g in zip(structs, gpu):
        f = compute_point_features(s["coords"], s["radii"], s["props"])
        assert np.array_equal(g["atom_index"], f.atom_index)
        assert np.abs(g["blocked"] - f.blocked).mean() < 1e-4
        assert np.abs(g["blocked"] - f.blocked).max() < 0.035      # <= 2 of 60 rays flip on a voxel boundary
        assert np.abs(g["sums"] - f.sums).max() <= 1.0


@needs_cuda
@pytest.mark.parametrize("points", ["sas", "cascade"])
def test_gpu_bricking_matches_unbricked(points):
    c, r = rod_protein()
    s = [dict(coords=c, radii=r, props=np.ones((len(c), 6), np.float32)), synthetic_structs()[0]]
    budget = {"sas": 60_000, "cascade": 100_000}[points]
    ref = compute_point_features_batch_torch(s, FeatureConfig(points=points))
    br = compute_point_features_batch_torch(s, FeatureConfig(points=points, max_voxels_gpu=budget))
    for a, b in zip(br, ref):
        assert len(a["points"]) == len(b["points"])
        ia, ib = match_points(a["points"], b["points"])
        assert len(ia) == len(b["points"])
        assert np.array_equal(a["atom_index"][ia], b["atom_index"][ib])
        assert np.abs(a["blocked"][ia] - b["blocked"][ib]).max() < 1e-5
        assert np.allclose(a["sums"][ia], b["sums"][ib])


@pytest.mark.skipif(not cuda_available() or __import__("torch").cuda.device_count() < 2, reason="needs 2 GPUs")
def test_second_gpu_device():
    """Triton kernels must launch on the tensors' device, not the current one (cuda:0)."""
    import torch
    import pocketdetect.geometry as g
    s = synthetic_structs()[1]
    r0 = compute_point_features_batch_torch([s], device="cuda:0")[0]
    r1 = compute_point_features_batch_torch([s], device="cuda:1")[0]
    assert torch.cuda.current_device() == 0
    for k in r0:
        assert np.allclose(r0[k], r1[k], atol=1e-5), k
    c = torch.from_numpy(s["coords"]).to("cuda:1")
    r = torch.from_numpy(s["radii"]).to("cuda:1")
    sp = g.sas_points_torch(c, r)
    assert sp["points"].device == torch.device("cuda:1")
    b = g.ray_buriedness_torch(sp["points"], coords=c, radii=r)["blocked"]
    nf = g.neighbor_features_torch(sp["points"], c)["sums"]
    gc = g.grid_candidates_torch(c, r)["points"]
    occ = g.TorchOccupancyGrid(c, r)
    ps = g.psp_counts_torch(gc, None, occ)
    ref = g.ray_buriedness(sp["points"].cpu().numpy(), s["coords"], s["radii"]).blocked
    assert np.abs(b.cpu().numpy() - ref).max() < 1e-4
    assert nf.device == ps.device == torch.device("cuda:1")
