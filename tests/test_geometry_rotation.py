"""Rotation invariance of the point features.

Without a canonical frame the Fibonacci ray set and the voxel grid are fixed
in the input frame, so a rigid rotation changes ray buriedness by ~0.02 on
average (angular sampling noise, ~1/sqrt(n_rays)).  The pipeline therefore
works in the PCA frame of the structure (``canonical_frame=True``), which
makes all features invariant up to float32 round-off.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pocketdetect.geometry import (FeatureConfig, canonical_frame, compute_point_features,
                                   compute_point_features_batch_torch, ray_buriedness, sas_points,
                                   smooth_point_values, to_frame, vdw_radii)

from geometry_helpers import blob_protein, load_pdb, needs_cuda

ROTS = Rotation.random(5, random_state=7).as_matrix()


def structure():
    try:
        c, el, _, _ = load_pdb("3ptb")
        return c, vdw_radii(el)
    except pytest.skip.Exception:
        return blob_protein(14.0)


def test_canonical_frame_is_rotation_covariant():
    c, _ = structure()
    ref = to_frame(c, *canonical_frame(c))
    for M in ROTS:
        cr = (c @ M.T + np.array([5.0, -3.0, 2.0])).astype(np.float32)
        R, ctr = canonical_frame(cr)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10) and np.linalg.det(R) > 0
        assert np.abs(to_frame(cr, R, ctr) - ref).max() < 1e-3


@pytest.mark.parametrize("points", ["sas", "cascade"])
def test_pipeline_rotation_invariant(points):
    c, r = structure()
    cfg = FeatureConfig(points=points)
    ref = compute_point_features(c, r, None, None, cfg)
    for M in ROTS:
        f = compute_point_features(c @ M.T, r, None, None, cfg)
        assert abs(len(f.points) - len(ref.points)) <= 0.001 * len(ref.points) + 2
        # points are returned in the input frame: compare with the rotated reference points
        from scipy.spatial import cKDTree
        d, j = cKDTree(ref.points @ M.T).query(f.points)
        ok = d < 1e-2
        assert ok.mean() > 0.998
        db = np.abs(f.blocked[ok] - ref.blocked[j[ok]])
        assert db.mean() < 2e-3 and np.quantile(db, 0.999) < 0.05
        assert np.mean(f.psp[ok] == ref.psp[j[ok]]) > 0.99
        assert np.abs(f.sums[ok] - ref.sums[j[ok]]).max() <= 1.0


def test_raw_buriedness_rotation_noise_bounded_and_smoothing_helps():
    """Same physical points, lab-frame rays: documented noise level, reduced by 2 Å smoothing."""
    c, r = structure()
    pts = sas_points(c, r).points
    out = []
    for M in ROTS[:2]:
        out.append(ray_buriedness((pts @ M.T).astype(np.float32), (c @ M.T).astype(np.float32), r).blocked[:, 1])
    raw = np.abs(out[0] - out[1]).mean()
    sm = np.abs(smooth_point_values(pts, out[0], 2.0) - smooth_point_values(pts, out[1], 2.0)).mean()
    assert 0.003 < raw < 0.04
    assert sm < 0.7 * raw


@needs_cuda
def test_gpu_pipeline_rotation_invariant():
    c, r = structure()
    structs = [dict(coords=(c @ M.T).astype(np.float32), radii=r) for M in ROTS[:3]]
    res = compute_point_features_batch_torch(structs, FeatureConfig(points="cascade"))
    from scipy.spatial import cKDTree
    for M, f in zip(ROTS[1:3], res[1:]):
        d, j = cKDTree(res[0]["points"] @ ROTS[0] @ M.T).query(f["points"])
        ok = d < 1e-2
        assert ok.mean() > 0.998
        assert np.abs(f["blocked"][ok] - res[0]["blocked"][j[ok]]).mean() < 2e-3
