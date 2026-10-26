import numpy as np
import pytest

from pocketdetect.geometry import (FeatureConfig, atom_class_matrix, compute_point_features,
                                   compute_point_features_batch_torch, vdw_radii)

from geometry_helpers import blob_protein, load_pdb, needs_cuda


def synthetic_structs():
    out = []
    for i, size in enumerate((9.0, 12.0, 15.0)):
        c, r = blob_protein(size, seed=i)
        props = np.random.default_rng(i).integers(0, 2, (len(c), 6)).astype(np.float32)
        w = np.random.default_rng(i).uniform(0, 1, len(c)).astype(np.float32)
        out.append(dict(coords=c, radii=r, props=props, weights=w))
    return out


def test_cpu_pipeline_shapes():
    s = synthetic_structs()[0]
    cfg = FeatureConfig()
    f = compute_point_features(s["coords"], s["radii"], s["props"], s["weights"], cfg)
    p = len(f.points)
    assert f.blocked.shape == (p, len(cfg.ray_lengths)) and f.depth.shape == (p,)
    assert f.sums.shape == (p, len(cfg.nb_radii), 6) and f.gauss.shape == (p, len(cfg.sigmas), 6)
    assert set(f.timings) == {"points", "buriedness", "neighbors", "total"}


@needs_cuda
@pytest.mark.parametrize("points", ["sas", "grid"])
def test_batched_gpu_equals_cpu(points):
    structs = synthetic_structs()
    cfg = FeatureConfig(points=points, grid_max_dist=5.0)
    gpu = compute_point_features_batch_torch(structs, cfg)
    assert len(gpu) == len(structs)
    for s, g in zip(structs, gpu):
        f = compute_point_features(s["coords"], s["radii"], s["props"], s["weights"], cfg)
        if points == "sas":
            assert np.array_equal(g["atom_index"], f.atom_index)
        assert abs(len(g["points"]) - len(f.points)) <= 2
        if len(g["points"]) == len(f.points):
            assert np.allclose(g["points"], f.points, atol=1e-4)
            assert np.abs(g["blocked"] - f.blocked).mean() < 1e-3  # soft weights: rare voxel-boundary flips
            assert np.abs(g["sums"] - f.sums).max() <= 1.0
            assert np.allclose(g["gauss"], f.gauss, atol=1e-2, rtol=1e-4)


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
        assert np.abs(g["blocked"] - f.blocked).max() < 1e-3
        assert np.abs(g["sums"] - f.sums).max() <= 1.0
