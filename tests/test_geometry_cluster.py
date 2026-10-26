import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, linkage

from pocketdetect.geometry import (canonical_labels, mean_shift, mean_shift_torch, quick_shift, quick_shift_torch,
                                   sas_points, single_linkage, single_linkage_torch)

from geometry_helpers import blob_protein, needs_cuda


def blobs(seed=0, n=150):
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0], [15, 0, 0], [0, 15, 0]], float)
    pts = np.concatenate([c + rng.normal(0, 1.0, (n, 3)) for c in centres]).astype(np.float32)
    truth = np.repeat(np.arange(3), n)
    return pts, truth


def same_partition(a, b):
    return np.array_equal(canonical_labels(a), canonical_labels(b))


def test_canonical_labels():
    assert np.array_equal(canonical_labels([5, 5, 2, 7, 2]), [0, 0, 1, 2, 1])


@pytest.mark.parametrize("d", [0.8, 1.5, 3.0])
def test_single_linkage_vs_scipy(d):
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 20, (400, 3))
    ref = fcluster(linkage(pts, "single"), t=d, criterion="distance")
    assert same_partition(single_linkage(pts, d), ref)


def test_single_linkage_batch_separates():
    pts, _ = blobs()
    two = np.concatenate([pts, pts])
    batch = np.r_[np.zeros(len(pts)), np.ones(len(pts))]
    lab = single_linkage(two, 3.0, batch=batch)
    assert lab.max() + 1 == 2 * (single_linkage(pts, 3.0).max() + 1)
    assert not set(lab[:len(pts)]) & set(lab[len(pts):])


def test_mean_shift_and_quick_shift_recover_blobs():
    pts, truth = blobs()
    assert same_partition(mean_shift(pts, 2.0), truth)
    assert same_partition(quick_shift(pts, 1.5, 4.0), truth)


@needs_cuda
def test_cluster_cpu_gpu_agree():
    c, r = blob_protein()
    pts = sas_points(c, r).points
    for d in (1.0, 2.0, 3.0):
        assert np.array_equal(single_linkage(pts, d), single_linkage_torch(pts, d).cpu().numpy())
    assert np.array_equal(mean_shift(pts, 3.0), mean_shift_torch(pts, 3.0).cpu().numpy())
    assert np.array_equal(quick_shift(pts, 2.0, 4.0), quick_shift_torch(pts, 2.0, 4.0).cpu().numpy())
    two = np.concatenate([pts, pts])
    batch = np.r_[np.zeros(len(pts)), np.ones(len(pts))].astype(np.int64)
    assert np.array_equal(single_linkage(two, 2.0, batch), single_linkage_torch(two, 2.0, batch).cpu().numpy())
    assert np.array_equal(quick_shift(two, 2.0, 4.0, batch=batch),
                          quick_shift_torch(two, 2.0, 4.0, batch=batch).cpu().numpy())
