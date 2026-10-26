"""Grid-based candidate points and the cheap LIGSITE-style prefilter (cascade).

Cascade (``FeatureConfig(points="cascade")``)::

    grid_candidates      voxel centres (1 Å, absolute-aligned) where a probe fits
                         and the nearest atom centre is <= max_dist (default 5 Å)
    psp_counts           LIGSITE "protein-solvent-protein" events: for each of 7
                         (or 13) line directions, is the point enclosed by protein
                         on both sides within ``length``?  (O(points * dirs * L))
    cascade_mask         keep points with psp >= min_psp, plus a coarse 2 Å
                         sub-lattice of all others (shallow sites stay covered)

Only the survivors are passed to the (expensive) multi-scale ray casting.
All voxel arrays are float32/int32 (<= 9 bytes/voxel on CPU).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from ._grid import TorchGridFrames, grid_frame, stamp_offsets_torch


@dataclass
class GridPoints:
    points: np.ndarray        # (M, 3) float32 voxel centres
    nearest_atom: np.ndarray  # (M,) int32
    nearest_dist: np.ndarray  # (M,) float32, distance to nearest atom centre


# 3 axes + 4 body diagonals (the 7 LIGSITEcs directions) + 6 face diagonals
_D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
               [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
               [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]], np.float64)
PSP_DIRECTIONS = (_D / np.linalg.norm(_D, axis=1, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------------- CPU
@njit(parallel=True, cache=True)
def _grid_stamp_kernel(cs_sorted, idx_sorted, rclash, origin, dims, h, dmax):
    nx, ny, nz = dims[0], dims[1], dims[2]
    best = np.full(nx * ny * nz, np.inf, np.float32)
    arg = np.full(nx * ny * nz, -1, np.int32)
    clash = np.zeros(nx * ny * nz, np.bool_)
    xs = cs_sorted[:, 0]
    dmax2 = dmax * dmax
    for ix in prange(nx):
        x = origin[0] + (ix + 0.5) * h
        lo = np.searchsorted(xs, x - dmax)
        hi = np.searchsorted(xs, x + dmax, side="right")
        for a in range(lo, hi):
            ai = idx_sorted[a]
            ya = cs_sorted[a, 1]
            za = cs_sorted[a, 2]
            dx2 = (x - cs_sorted[a, 0]) ** 2
            rem = dmax2 - dx2
            if rem < 0:
                continue
            ry = math.sqrt(rem)
            iy0 = max(0, int(math.ceil((ya - ry - origin[1]) / h - 0.5)))
            iy1 = min(ny - 1, int(math.floor((ya + ry - origin[1]) / h - 0.5)))
            rc2 = rclash[a] * rclash[a]
            for iy in range(iy0, iy1 + 1):
                y = origin[1] + (iy + 0.5) * h
                dy2 = (y - ya) ** 2
                rem2 = rem - dy2
                if rem2 < 0:
                    continue
                rz = math.sqrt(rem2)
                iz0 = max(0, int(math.ceil((za - rz - origin[2]) / h - 0.5)))
                iz1 = min(nz - 1, int(math.floor((za + rz - origin[2]) / h - 0.5)))
                row = (ix * ny + iy) * nz
                dxy2 = dx2 + dy2
                for iz in range(iz0, iz1 + 1):
                    v = row + iz
                    if clash[v]:
                        continue            # discarded anyway: skip the argmin bookkeeping
                    dz = origin[2] + (iz + 0.5) * h - za
                    d2 = dxy2 + dz * dz
                    if d2 > dmax2:
                        continue
                    if d2 < rc2:
                        clash[v] = True
                        continue
                    d2f = np.float32(d2)
                    b = best[v]
                    if d2f < b or (d2f == b and ai < arg[v]):
                        best[v] = d2f
                        arg[v] = ai
    return best.reshape(nx, ny, nz), arg.reshape(nx, ny, nz), clash.reshape(nx, ny, nz)


def grid_candidates(coords, radii, spacing: float = 1.0, probe: float = 1.4,
                    max_dist: float = 5.0) -> GridPoints:
    """Grid points in the solvent shell around the protein (CPU / Numba).

    A voxel centre ``x`` is kept iff ``|x - a| >= r_a + probe`` for all atoms
    (a probe sphere fits) and ``min_a |x - a| <= max_dist``.  Points are in C
    order of the (ix, iy, iz) grid whose origin is aligned to multiples of
    ``spacing``.  Memory: 9 bytes per voxel of the bounding box + max_dist
    (use :func:`compute_point_features`, which bricks huge inputs).
    """
    coords = np.ascontiguousarray(coords, np.float32)
    radii = np.asarray(radii, np.float32)
    if coords.shape[0] == 0:
        return GridPoints(np.zeros((0, 3), np.float32), np.zeros(0, np.int32), np.zeros(0, np.float32))
    c64 = coords.astype(np.float64)
    origin, dims = grid_frame(c64.min(0), c64.max(0), spacing, max_dist)
    o = np.argsort(c64[:, 0], kind="stable")
    rclash = (radii.astype(np.float64) + probe)[o]
    best, arg, clash = _grid_stamp_kernel(np.ascontiguousarray(c64[o]), o.astype(np.int64), rclash,
                                          origin, dims, float(spacing), float(max_dist))
    sel = (arg >= 0) & ~clash
    ijk = np.argwhere(sel)
    pts = (origin[None, :] + (ijk + 0.5) * spacing).astype(np.float32)
    return GridPoints(pts, arg[sel], np.sqrt(best[sel]))


@njit(parallel=True, cache=True)
def _psp_kernel(points, dirs, grid, origin, h, n_steps, min_mu):
    p_n = points.shape[0]
    nx, ny, nz = grid.shape
    out = np.zeros(p_n, np.int32)
    for p in prange(p_n):
        px = (points[p, 0] - origin[0]) / h
        py = (points[p, 1] - origin[1]) / h
        pz = (points[p, 2] - origin[2]) / h
        cnt = 0
        for k in range(dirs.shape[0]):
            both = 1
            for sgn in (1.0, -1.0):
                dx = sgn * dirs[k, 0]
                dy = sgn * dirs[k, 1]
                dz = sgn * dirs[k, 2]
                hit = 0
                for s in range(1, n_steps + 1):
                    fx = px + dx * s
                    fy = py + dy * s
                    fz = pz + dz * s
                    if fx < 0.0 or fy < 0.0 or fz < 0.0 or fx >= nx or fy >= ny or fz >= nz:
                        continue
                    if grid[int(fx), int(fy), int(fz)] > min_mu:
                        hit = 1
                        break
                if hit == 0:
                    both = 0
                    break
            cnt += both
        out[p] = cnt
    return out


def _psp_setup(length, spacing, step, min_weight, ref_length):
    from .buriedness import absorption_from_weights
    step = spacing if step is None else step
    n_steps = int(round(length / step))
    min_mu = float(absorption_from_weights([min_weight], ref_length)[0]) * (1 - 1e-6)
    dirs = PSP_DIRECTIONS.astype(np.float64) * (step / spacing)      # voxel units per step
    return dirs, n_steps, min_mu


def psp_counts(points, grid, length: float = 8.0, n_dirs: int = 7, step: float | None = None,
               min_weight: float = 0.5) -> np.ndarray:
    """LIGSITE-style enclosure count (0..n_dirs) of query points (CPU / Numba).

    ``grid`` is an :class:`OccupancyGrid`; a voxel counts as protein if its atom
    weight is >= ``min_weight``.  For each of the first ``n_dirs`` (7: axes +
    body diagonals, or 13: + face diagonals) line directions the point scores 1
    if protein is met within ``length`` Å in *both* senses.
    """
    dirs, n_steps, min_mu = _psp_setup(length, grid.spacing, step, min_weight, grid.ref_length)
    dirs = dirs[:n_dirs]
    pts = np.ascontiguousarray(points, np.float32)
    return _psp_kernel(pts, dirs, grid.mu, grid.origin, float(grid.spacing), n_steps, min_mu)


def sublattice_mask(points, spacing: float = 1.0, factor: int = 2) -> np.ndarray:
    """True for grid points whose absolute voxel indices are all multiples of ``factor``."""
    idx = np.rint(np.asarray(points, np.float64) / spacing - 0.5).astype(np.int64)
    return np.all(idx % factor == 0, axis=1)


def cascade_mask(points, psp, spacing: float = 1.0, min_psp: int = 2, subsample: int = 2):
    """Survivors of the cascade: ``psp >= min_psp`` or on the coarse sub-lattice."""
    keep = np.asarray(psp) >= min_psp
    if subsample and subsample > 1:
        keep |= sublattice_mask(points, spacing, subsample)
    return keep


# --------------------------------------------------------------------------- GPU
def grid_candidates_torch(coords, radii, batch=None, spacing: float = 1.0, probe: float = 1.4,
                          max_dist: float = 5.0, n_batch: int | None = None,
                          max_elems: int = 1 << 25, use_triton: bool | None = None):
    """Batched GPU version of :func:`grid_candidates`.

    Returns dict ``points (M,3)``, ``nearest_atom (M,)`` (global index),
    ``nearest_dist (M,)``, ``batch (M,)``; per structure in the CPU order.
    Fast path: Triton per-atom stamping with 64-bit atomic-min keys
    ``(quantised d^2 << 31) | atom``; fallback: torch ``scatter_reduce``.
    """
    import torch
    from ._triton_kernels import stamp_triton, triton_available
    from .torch_utils import as_tensor

    coords = as_tensor(coords, torch.float32)
    dev = coords.device
    radii = as_tensor(radii, torch.float32, dev)
    n = coords.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=dev) if batch is None else as_tensor(batch, torch.long, dev)
    if n_batch is None:
        n_batch = int(batch.max().item()) + 1 if n else 1
    fr = TorchGridFrames(coords, batch, n_batch, spacing, max_dist)
    INIT = torch.iinfo(torch.int64).max
    key = torch.full((fr.total,), INIT, dtype=torch.int64, device=dev)
    off = stamp_offsets_torch(max_dist, spacing, dev)
    dmax2 = max_dist * max_dist
    if use_triton is None:
        use_triton = triton_available(dev)
    if use_triton and n > 0:
        clash = torch.zeros(fr.total, dtype=torch.int32, device=dev)
        frac, c0 = fr.atom_voxels(coords, batch)
        stamp_triton(frac, c0, fr.base[batch].contiguous(), fr.dims[batch].to(torch.int32).contiguous(), off,
                     torch.full((n,), dmax2, dtype=torch.float32, device=dev),
                     ((radii + probe) ** 2).contiguous(), spacing, key=key, clash=clash, mode=0)
        clash = clash.bool()
    else:
        clash = torch.zeros(fr.total, dtype=torch.bool, device=dev)
        offc = off.float() + 0.5
        chunk = max(1, max_elems // off.shape[0])
        for s in range(0, n, chunk):
            b = batch[s:s + chunk]
            frac, c0 = fr.atom_voxels(coords[s:s + chunk], b)
            dv = (offc[None] - frac[:, None, :]) * spacing
            d2 = (dv * dv).sum(-1)
            vox = c0[:, None, :] + off[None]
            d = fr.dims[b].to(torch.int32)[:, None, :]
            ok = ((vox >= 0) & (vox < d)).all(-1) & (d2 <= dmax2)
            lin = fr.base[b][:, None] + ((vox[..., 0] * d[..., 1] + vox[..., 1]) * d[..., 2] + vox[..., 2]).long()
            q = torch.floor(d2 * float(1 << 20)).long()
            kk = (q << 31) | torch.arange(s, s + b.shape[0], device=dev)[:, None]
            key.scatter_reduce_(0, lin[ok], kk[ok], "amin")
            rc = (radii[s:s + chunk] + probe)[:, None]
            clash[lin[ok & (d2 < rc * rc)]] = True
    flat = torch.nonzero((key != INIT) & ~clash, as_tuple=True)[0]
    vb, centre = fr.decode(flat)
    near = key[flat] & ((1 << 31) - 1)
    dist = torch.sqrt(((centre - coords[near].double()) ** 2).sum(1))
    return {"points": centre.float(), "nearest_atom": near, "nearest_dist": dist.float(), "batch": vb}


def psp_counts_torch(points, point_batch, grid, length: float = 8.0, n_dirs: int = 7, step: float | None = None,
                     min_weight: float = 0.5, use_triton: bool | None = None, max_elems: int = 1 << 24):
    """GPU version of :func:`psp_counts` on a :class:`TorchOccupancyGrid`."""
    import torch
    from ._triton_kernels import psp_triton, triton_available
    from .torch_utils import as_tensor
    dev = grid.mu.device
    pts = as_tensor(points, torch.float32, dev)
    p_n = pts.shape[0]
    pb = torch.zeros(p_n, dtype=torch.long, device=dev) if point_batch is None else as_tensor(point_batch, torch.long, dev)
    dirs, n_steps, min_mu = _psp_setup(length, grid.spacing, step, min_weight, grid.ref_length)
    dirs_t = torch.as_tensor(dirs[:n_dirs], dtype=torch.float32, device=dev).contiguous()
    rel = ((pts - grid.origin32[pb]) / grid.spacing).contiguous()
    dims = grid.dims[pb].to(torch.int32).contiguous()
    base = grid.base[pb].contiguous()
    if use_triton is None:
        use_triton = triton_available(dev)
    if use_triton:
        return psp_triton(rel, dims, base, dirs_t, grid.mu, n_steps, min_mu)
    out = torch.zeros(p_n, dtype=torch.int32, device=dev)
    s_idx = torch.arange(1, n_steps + 1, device=dev, dtype=torch.float32)
    nd = dirs_t.shape[0]
    chunk = max(1, max_elems // (2 * nd * n_steps))
    for s in range(0, p_n, chunk):
        sl = slice(s, s + chunk)
        hits = []
        for sgn in (1.0, -1.0):
            f = rel[sl][:, None, None, :] + sgn * dirs_t[None, :, None, :] * s_idx[None, None, :, None]
            idx = torch.floor(f).long()
            d = dims[sl].long()[:, None, None, :]
            ok = ((f >= 0) & (idx < d)).all(-1)
            lin = base[sl][:, None, None] + (idx[..., 0] * d[..., 1] + idx[..., 1]) * d[..., 2] + idx[..., 2]
            v = torch.where(ok, grid.mu[torch.where(ok, lin, torch.zeros_like(lin))], torch.zeros((), device=dev))
            hits.append((v > min_mu).any(-1))
        out[sl] = (hits[0] & hits[1]).sum(1).to(torch.int32)
    return out


def sublattice_mask_torch(points, spacing: float = 1.0, factor: int = 2):
    import torch
    idx = torch.round(points.double() / spacing - 0.5).long()
    return (torch.remainder(idx, factor) == 0).all(1)


def cascade_mask_torch(points, psp, spacing: float = 1.0, min_psp: int = 2, subsample: int = 2):
    keep = psp >= min_psp
    if subsample and subsample > 1:
        keep = keep | sublattice_mask_torch(points, spacing, subsample)
    return keep
