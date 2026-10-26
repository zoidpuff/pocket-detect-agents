import numpy as np
import pytest

from pocketdetect.geometry import (OccupancyGrid, absorption_from_weights, fibonacci_sphere, occupancy_grid,
                                   ray_buriedness, ray_buriedness_torch, sas_points, vdw_radii)

from geometry_helpers import blob_protein, load_pdb, needs_cuda


def shell(radius=10.0, n=3000):
    return (fibonacci_sphere(n) * radius).astype(np.float32), np.full(n, 1.7, np.float32)


def slab(half=45.0, a=1.5):
    g = np.arange(-half, half + 1e-6, a)
    xy = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    layers = [np.c_[xy, np.full(len(xy), z)] for z in (0.0, -a)]
    c = np.concatenate(layers).astype(np.float32)
    return c, np.full(len(c), 1.7, np.float32)


def test_cavity_center_fully_buried_and_far_point_zero():
    c, r = shell()
    q = np.array([[0, 0, 0], [60, 0, 0]], np.float32)
    b = ray_buriedness(q, c, r, n_rays=100, lengths=(8.0, 12.0, 20.0))
    # shell inner surface at 10 - 1.7 = 8.3 Å: L=8 sees nothing, L>=12 everything
    assert b.blocked[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert b.blocked[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert b.blocked[0, 2] == pytest.approx(1.0, abs=1e-6)
    assert np.all(b.blocked[1] == 0)
    assert b.depth[0] == pytest.approx(8.3, abs=0.6)
    assert b.depth[1] == pytest.approx(20.0)


def test_slab_half_blocked():
    c, r = slab()
    h = 0.3
    q = np.array([[0.3, 0.2, 1.7 + h]], np.float32)
    for L in (8.0, 12.0, 20.0):
        b = ray_buriedness(q, c, r, n_rays=1000, lengths=(L,), spacing=0.5, step=0.25).blocked[0, 0]
        expected = 0.5 * (1 - h / L)          # downward rays that reach the slab surface within L
        assert abs(b - expected) < 0.025, (L, b, expected)


def test_transparency_zero_weight_no_occlusion():
    c, r = shell()
    q = np.zeros((1, 3), np.float32)
    b0 = ray_buriedness(q, c, r, weights=np.zeros(len(c)), lengths=(12.0, 20.0)).blocked
    assert np.all(b0 == 0)
    prev = 0.0
    for w in (0.1, 0.3, 0.6, 0.9, 1.0):
        bw = ray_buriedness(q, c, r, weights=np.full(len(c), w), lengths=(12.0,)).blocked[0, 0]
        assert bw > prev - 1e-7
        prev = bw
    assert prev == pytest.approx(1.0, abs=1e-6)
    # upper half of the shell opaque, lower half transparent: rays with z > 0 are blocked, plus
    # slightly downward rays grazing opaque atoms sitting at z ~ 0 (angular radius asin(1.7/10))
    w = (c[:, 2] > 0).astype(np.float32)
    bh = ray_buriedness(q, c, r, weights=w, n_rays=400, lengths=(12.0,)).blocked[0, 0]
    assert 0.5 - 0.01 < bh < 0.5 + 0.5 * np.sin(np.arcsin(1.7 / 10)) + 0.02


def test_beer_lambert_reference_length():
    """Uniform medium of weight w: a ray of length ref_length is transmitted with prob 1-w."""
    for w in (0.2, 0.5, 0.9):
        mu = absorption_from_weights([w], ref_length=2.0)[0]
        grid = OccupancyGrid(np.full((40, 40, 40), mu, np.float32), np.zeros(3), 0.5)
        q = np.full((1, 3), 10.0, np.float32)
        b = ray_buriedness(q, grid=grid, n_rays=30, lengths=(2.0, 4.0), step=0.5)
        assert b.blocked[0, 0] == pytest.approx(w, abs=1e-5)
        assert b.blocked[0, 1] == pytest.approx(1 - (1 - w) ** 2, abs=1e-5)


def test_absorption_limits():
    mu = absorption_from_weights([0.0, 1.0, 0.5])
    assert mu[0] == 0 and mu[1] >= 100 and mu[2] == pytest.approx(np.log(2) / 2.0, rel=1e-6)


def test_occupancy_grid_bruteforce():
    c, r = blob_protein(6.0)
    g = occupancy_grid(c, r, spacing=0.7)
    idx = np.argwhere(np.ones(g.mu.shape, bool))
    centres = g.origin + (idx + 0.5) * 0.7
    d = np.linalg.norm(centres[:, None] - c[None], axis=-1)
    expect = (d <= r[None]).any(1)
    got = g.mu.reshape(-1) > 0
    assert (expect != got).sum() <= 2  # boundary ties only


def test_multiscale_monotone_and_pocket_more_buried():
    c, r = blob_protein(14.0)
    s = sas_points(c, r)
    b = ray_buriedness(s.points, c, r, lengths=(4.0, 8.0, 12.0, 20.0))
    assert np.all(np.diff(b.blocked, axis=1) >= -1e-6)
    pocket = np.linalg.norm(s.points - np.array([14.0 * 0.6, 0, 0]), axis=1) < 4.0
    assert pocket.sum() > 5
    assert b.blocked[pocket, 2].mean() > b.blocked[~pocket, 2].mean() + 0.15


@needs_cuda
@pytest.mark.parametrize("use_triton", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_buriedness_cpu_gpu_agree(use_triton, weighted):
    c, el, _, _ = load_pdb("1a28")
    r = vdw_radii(el)
    w = np.random.default_rng(0).uniform(0, 1, len(c)).astype(np.float32) if weighted else None
    s = sas_points(c, r)
    b = ray_buriedness(s.points, c, r, w)
    t = ray_buriedness_torch(s.points, coords=c, radii=r, weights=w, use_triton=use_triton)
    d = np.abs(t["blocked"].cpu().numpy() - b.blocked)
    assert d.mean() < 1e-5 and d.max() < 0.01
    dd = np.abs(t["depth"].cpu().numpy() - b.depth)
    assert dd.mean() < 1e-3 and np.quantile(dd, 0.999) < 0.5


@needs_cuda
def test_buriedness_torch_batched():
    a, ra = blob_protein(10.0, seed=1)
    bb, rb = blob_protein(13.0, seed=2)
    sa, sb = sas_points(a, ra), sas_points(bb, rb)
    c = np.concatenate([a, bb])
    r = np.concatenate([ra, rb])
    ab = np.r_[np.zeros(len(a)), np.ones(len(bb))].astype(np.int64)
    pts = np.concatenate([sa.points, sb.points])
    pb = np.r_[np.zeros(len(sa.points)), np.ones(len(sb.points))].astype(np.int64)
    t = ray_buriedness_torch(pts, pb, c, r, None, ab)["blocked"].cpu().numpy()
    ref = np.concatenate([ray_buriedness(sa.points, a, ra).blocked, ray_buriedness(sb.points, bb, rb).blocked])
    assert np.abs(t - ref).max() < 1e-4
