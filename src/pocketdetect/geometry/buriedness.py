"""Multi-scale ray-casting buriedness with soft (transparent) occlusion.

Model
-----
Atoms are voxelised into an absorption grid: voxel ``v`` gets the absorption
coefficient ``mu_v = max_a mu_a`` over atoms whose blocking sphere
(``radius_scale * r_vdw + radius_pad``) contains the voxel centre, where for an atom of
opacity weight ``w`` in [0, 1]::

    mu_a = -ln(1 - w) / ref_length      (w >= 1  ->  opaque, mu = 1e3 / Å)

so that a ray crossing ``ref_length`` Å of material of weight ``w`` is
transmitted with probability ``1 - w`` (Beer–Lambert).  ``w = 0`` atoms are
fully transparent; ``w = 1`` atoms block deterministically (binary case).

For every query point, ``n_rays`` Fibonacci-sphere rays are marched with step
``step`` (samples at ``(s + 1/2) * step``, nearest-voxel lookup).  For each ray
length ``L`` in ``lengths`` the output is the *expected blocked fraction*::

    buriedness_L(p) = mean_rays [1 - exp(-sum_{samples t < L} mu(p + t d) step)]

All lengths are computed in one march (multi-scale for free).  ``depth`` is
the mean over rays of the first-hit distance (first sample where the
transmittance drops below 1/2), capped at ``max(lengths)``.

Orientation: the ray set and the voxel grid are fixed in the frame of the
input coordinates, so results change slightly (mean |Δ| ~ 0.02 for 60 rays,
mostly angular-sampling noise ~ 1/sqrt(n_rays)) when a structure is rotated.
:func:`pocketdetect.geometry.canonical_frame` (used by the pipeline by
default) removes this by computing everything in a PCA frame of the structure.

CPU: Numba, parallel over points.  GPU: Triton ray-march kernel (fallback:
PyTorch gather of all (point, ray, sample) positions of a chunk) over a flat
batch of per-structure grids.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from ._grid import TorchGridFrames, grid_frame, stamp_offsets_torch
from .sphere import fibonacci_sphere

OPAQUE_MU = 1.0e3
TAU_STOP = 30.0
LN2 = math.log(2.0)


def absorption_from_weights(weights, ref_length: float = 2.0) -> np.ndarray:
    """Per-atom absorption coefficient (1/Å) from opacity weights in [0, 1]."""
    w = np.clip(np.asarray(weights, np.float64), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        mu = -np.log1p(-np.minimum(w, 1.0 - 1e-12)) / ref_length
    mu[w >= 1.0 - 1e-9] = OPAQUE_MU
    return mu.astype(np.float32)


@dataclass
class OccupancyGrid:
    mu: np.ndarray       # (nx, ny, nz) float32 absorption coefficient (1/Å)
    origin: np.ndarray   # (3,) float64, corner of voxel (0,0,0)
    spacing: float
    ref_length: float = 2.0


@dataclass
class Buriedness:
    blocked: np.ndarray  # (P, n_lengths) float32 expected blocked fraction
    depth: np.ndarray    # (P,) float32 mean first-hit distance (Å)
    lengths: tuple


# --------------------------------------------------------------------------- CPU
@njit(parallel=True, cache=True)
def _occupancy_kernel(cs_sorted, mu_sorted, rb_sorted, origin, dims, h, rmax):
    nx, ny, nz = dims[0], dims[1], dims[2]
    grid = np.zeros((nx, ny, nz), np.float32)
    xs = cs_sorted[:, 0]
    for ix in prange(nx):
        x = origin[0] + (ix + 0.5) * h
        lo = np.searchsorted(xs, x - rmax)
        hi = np.searchsorted(xs, x + rmax, side="right")
        for a in range(lo, hi):
            mu = mu_sorted[a]
            if mu <= 0.0:
                continue
            r2 = rb_sorted[a] * rb_sorted[a]
            dx2 = (x - xs[a]) ** 2
            rem = r2 - dx2
            if rem < 0:
                continue
            ya = cs_sorted[a, 1]
            za = cs_sorted[a, 2]
            ry = math.sqrt(rem)
            iy0 = max(0, int(math.ceil((ya - ry - origin[1]) / h - 0.5)))
            iy1 = min(ny - 1, int(math.floor((ya + ry - origin[1]) / h - 0.5)))
            for iy in range(iy0, iy1 + 1):
                y = origin[1] + (iy + 0.5) * h
                dy2 = (y - ya) ** 2
                rem2 = rem - dy2
                if rem2 < 0:
                    continue
                rz = math.sqrt(rem2)
                iz0 = max(0, int(math.ceil((za - rz - origin[2]) / h - 0.5)))
                iz1 = min(nz - 1, int(math.floor((za + rz - origin[2]) / h - 0.5)))
                for iz in range(iz0, iz1 + 1):
                    z = origin[2] + (iz + 0.5) * h
                    if dx2 + dy2 + (z - za) ** 2 <= r2 and mu > grid[ix, iy, iz]:
                        grid[ix, iy, iz] = mu
    return grid


def occupancy_grid(coords, radii, weights=None, spacing: float = 1.0, ref_length: float = 2.0,
                   radius_scale: float = 1.0, radius_pad: float = 0.0, pad: float = 0.0) -> OccupancyGrid:
    """Voxelise atoms into an absorption grid (CPU / Numba).

    Blocking radius = ``radius_scale * radii + radius_pad`` (inflate to close
    sub-probe gaps); ``pad`` extends the grid box beyond the atoms.
    """
    coords = np.asarray(coords, np.float32).astype(np.float64)
    rb = np.asarray(radii, np.float64) * radius_scale + radius_pad
    w = np.ones(len(coords)) if weights is None else weights
    mu = absorption_from_weights(w, ref_length).astype(np.float64)
    if len(coords) == 0:
        return OccupancyGrid(np.zeros((1, 1, 1), np.float32), np.zeros(3), spacing, ref_length)
    rmax = float(rb.max())
    origin, dims = grid_frame(coords.min(0), coords.max(0), spacing, rmax + pad)
    o = np.argsort(coords[:, 0], kind="stable")
    grid = _occupancy_kernel(np.ascontiguousarray(coords[o]), mu[o], rb[o], origin, dims,
                             float(spacing), rmax)
    return OccupancyGrid(grid, origin, float(spacing), float(ref_length))


@njit(parallel=True, fastmath=True, cache=True)
def _ray_kernel(points, dirs, grid, origin, h, step, n_steps, ck, lmax):
    p_n = points.shape[0]
    k_n = dirs.shape[0]
    n_l = ck.shape[0]
    nx, ny, nz = grid.shape
    flat = grid.reshape(-1)
    out = np.zeros((p_n, n_l), np.float32)
    depth = np.zeros(p_n, np.float32)
    inv_h = 1.0 / h
    for p in prange(p_n):
        px = (points[p, 0] - origin[0]) * inv_h
        py = (points[p, 1] - origin[1]) * inv_h
        pz = (points[p, 2] - origin[2]) * inv_h
        acc = np.zeros(n_l, np.float64)
        dsum = 0.0
        for k in range(k_n):
            dx = dirs[k, 0] * inv_h
            dy = dirs[k, 1] * inv_h
            dz = dirs[k, 2] * inv_h
            tau = 0.0
            l = 0
            hit_t = lmax
            hit = False
            entered = False
            for s in range(n_steps):
                t = (s + 0.5) * step
                fx = px + dx * t
                fy = py + dy * t
                fz = pz + dz * t
                if fx < 0.0 or fy < 0.0 or fz < 0.0 or fx >= nx or fy >= ny or fz >= nz:
                    if entered:
                        break      # left the (convex) grid box: nothing more to hit
                    v = 0.0        # not yet entered (query point outside the grid)
                else:
                    entered = True
                    v = flat[(np.int64(fx) * ny + np.int64(fy)) * nz + np.int64(fz)]
                if v > 0.0:
                    tau += v * step
                    if not hit and tau > 0.6931471805599453:
                        hit = True
                        hit_t = t
                    if tau > 30.0:
                        break
                while l < n_l and ck[l] == s + 1:
                    acc[l] += 1.0 - math.exp(-tau)
                    l += 1
            rest = 1.0 - math.exp(-tau)
            while l < n_l:
                acc[l] += rest
                l += 1
            dsum += hit_t
        for l in range(n_l):
            out[p, l] = acc[l] / k_n
        depth[p] = dsum / k_n
    return out, depth


def _steps(lengths, step):
    lengths = tuple(float(x) for x in lengths)
    ck = np.array([int(round(L / step)) for L in lengths], np.int64)
    if np.any(np.diff(ck) < 0):
        raise ValueError("lengths must be ascending")
    return lengths, ck


def ray_buriedness(points, coords=None, radii=None, weights=None, *, grid: OccupancyGrid | None = None,
                   n_rays: int = 60, lengths=(8.0, 12.0, 20.0), spacing: float = 1.0,
                   step: float = 1.0, ref_length: float = 2.0, radius_scale: float = 1.0,
                   radius_pad: float = 0.0) -> Buriedness:
    """Multi-scale ray-casting buriedness of query ``points`` (CPU / Numba).

    Either pass ``coords, radii[, weights]`` (grid is built) or a prebuilt
    ``grid`` from :func:`occupancy_grid` (reuse across query sets).
    """
    if grid is None:
        grid = occupancy_grid(coords, radii, weights, spacing, ref_length, radius_scale, radius_pad)
    lengths, ck = _steps(lengths, step)
    dirs = fibonacci_sphere(n_rays).astype(np.float64)
    pts = np.ascontiguousarray(points, np.float32)
    blocked, depth = _ray_kernel(pts, dirs, grid.mu, grid.origin, float(grid.spacing), float(step),
                                 int(ck[-1]), ck, float(max(lengths)))
    return Buriedness(blocked, depth, lengths)


# --------------------------------------------------------------------------- GPU
class TorchOccupancyGrid(TorchGridFrames):
    """Batch of per-structure absorption grids stored in one flat float32 tensor ``mu``."""

    def __init__(self, coords, radii, weights=None, batch=None, spacing: float = 1.0,
                 ref_length: float = 2.0, radius_scale: float = 1.0, n_batch: int | None = None,
                 radius_pad: float = 0.0, max_elems: int = 1 << 25, use_triton: bool | None = None):
        import torch
        from ._triton_kernels import stamp_triton, triton_available
        from .torch_utils import as_tensor

        coords = as_tensor(coords, torch.float32)
        dev = coords.device
        n = coords.shape[0]
        rb = as_tensor(radii, torch.float32, dev) * radius_scale + radius_pad
        w = np.ones(n, np.float32) if weights is None else (
            weights.detach().cpu().numpy() if isinstance(weights, torch.Tensor) else weights)
        mu = torch.as_tensor(absorption_from_weights(w, ref_length), device=dev)
        batch = torch.zeros(n, dtype=torch.long, device=dev) if batch is None else as_tensor(batch, torch.long, dev)
        if n_batch is None:
            n_batch = int(batch.max().item()) + 1 if n else 1
        rmax = float(rb.max().item()) if n else 0.0
        super().__init__(coords, batch, n_batch, spacing, rmax)
        self.ref_length = float(ref_length)
        grid = torch.zeros(self.total, dtype=torch.float32, device=dev)
        off = stamp_offsets_torch(rmax, spacing, dev)
        if use_triton is None:
            use_triton = triton_available(dev)
        if use_triton and n > 0:
            frac, c0 = self.atom_voxels(coords, batch)
            stamp_triton(frac, c0, self.base[batch].contiguous(), self.dims[batch].to(torch.int32).contiguous(),
                         off, (rb * rb).contiguous(), None, spacing, mu=mu.contiguous(), grid=grid, mode=1)
        else:
            offc = off.float() + 0.5
            chunk = max(1, max_elems // off.shape[0])
            for s in range(0, n, chunk):
                b = batch[s:s + chunk]
                frac, c0 = self.atom_voxels(coords[s:s + chunk], b)
                dv = (offc[None] - frac[:, None, :]) * spacing                # (A,O,3) float32
                d2 = (dv * dv).sum(-1)
                vox = c0[:, None, :] + off[None]
                d = self.dims[b].to(torch.int32)[:, None, :]
                r2 = (rb[s:s + chunk] ** 2)[:, None]
                ok = ((vox >= 0) & (vox < d)).all(-1) & (d2 <= r2) & (mu[s:s + chunk] > 0)[:, None]
                lin = self.base[b][:, None] + ((vox[..., 0] * d[..., 1] + vox[..., 1]) * d[..., 2] + vox[..., 2]).long()
                vals = mu[s:s + chunk][:, None].expand_as(d2)
                grid.scatter_reduce_(0, lin[ok], vals[ok], "amax")
        self.mu = grid


def ray_buriedness_torch(points, point_batch=None, coords=None, radii=None, weights=None,
                         atom_batch=None, *, grid: TorchOccupancyGrid | None = None,
                         n_rays: int = 60, lengths=(8.0, 12.0, 20.0), spacing: float = 1.0,
                         step: float = 1.0, ref_length: float = 2.0, radius_scale: float = 1.0,
                         radius_pad: float = 0.0, n_batch: int | None = None, max_elems: int = 1 << 24,
                         use_triton: bool | None = None):
    """Batched GPU ray-casting buriedness; returns dict ``blocked (P,nL)``, ``depth (P,)``.

    ``use_triton=None`` uses the fused Triton ray-march kernel when available
    (CUDA + triton), otherwise the pure-PyTorch gather implementation.
    """
    import torch
    from ._triton_kernels import ray_march_triton, triton_available
    from .torch_utils import as_tensor

    if grid is None:
        grid = TorchOccupancyGrid(coords, radii, weights, atom_batch, spacing, ref_length,
                                  radius_scale, n_batch, radius_pad)
    dev = grid.mu.device
    pts = as_tensor(points, torch.float32, dev)
    p_n = pts.shape[0]
    pb = torch.zeros(p_n, dtype=torch.long, device=dev) if point_batch is None else as_tensor(point_batch, torch.long, dev)
    lengths, ck = _steps(lengths, step)
    n_steps = int(ck[-1])
    lmax = float(max(lengths))
    dirs = torch.from_numpy(fibonacci_sphere(n_rays)).to(dev)
    if use_triton is None:
        use_triton = triton_available(dev)
    if use_triton:
        rel = ((pts - grid.origin32[pb]) / grid.spacing).contiguous()
        dims = grid.dims[pb].to(torch.int32).contiguous()
        base = grid.base[pb].contiguous()
        ck32 = torch.as_tensor(ck, dtype=torch.int32, device=dev)
        blocked, depth = ray_march_triton(rel, dims, base, (dirs / grid.spacing).contiguous(), grid.mu,
                                          ck32, step, lmax)
        return {"blocked": blocked, "depth": depth, "lengths": lengths}
    t = (torch.arange(n_steps, device=dev, dtype=torch.float32) + 0.5) * step
    ray = dirs[:, None, :] * t[None, :, None] / grid.spacing                   # (K,S,3) voxel units
    ck_t = torch.as_tensor(ck - 1, device=dev)
    blocked = torch.empty((p_n, len(lengths)), dtype=torch.float32, device=dev)
    depth = torch.empty(p_n, dtype=torch.float32, device=dev)
    chunk = max(1, max_elems // (n_rays * n_steps))
    for s in range(0, p_n, chunk):
        b = pb[s:s + chunk]
        rel = (pts[s:s + chunk] - grid.origin32[b]) / grid.spacing             # (P,3)
        f = rel[:, None, None, :] + ray[None]                                  # (P,K,S,3)
        idx = torch.floor(f).long()
        d = grid.dims[b][:, None, None, :]
        ok = ((f >= 0) & (idx < d)).all(-1)
        lin = grid.base[b][:, None, None] + (idx[..., 0] * d[..., 1] + idx[..., 1]) * d[..., 2] + idx[..., 2]
        lin = torch.where(ok, lin, torch.zeros_like(lin))
        v = torch.where(ok, grid.mu[lin], torch.zeros((), device=dev))
        cum = torch.cumsum(v * step, dim=-1)                                   # (P,K,S)
        blocked[s:s + chunk] = (1.0 - torch.exp(-cum[..., ck_t])).mean(1)
        hitm = cum > LN2
        first = torch.argmax(hitm.to(torch.uint8), dim=-1)
        hit_t = torch.where(hitm.any(-1), t[first], torch.full_like(t[first], lmax))
        depth[s:s + chunk] = hit_t.mean(1)
    return {"blocked": blocked, "depth": depth, "lengths": lengths}
