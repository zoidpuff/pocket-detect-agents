"""Dual-grid ray casting with first-hit chemistry (CPU / Numba).

One absorption grid for all heavy atoms (``mu``), one for backbone+CB only (``mu_bb``,
side-chain-tolerant buriedness) and a per-voxel label = class of the atom owning the voxel
(nearest by d - r_block).  A single march per (point, ray) gives, for each length L:

* ``blocked[L]``    expected blocked fraction, full atoms (Beer-Lambert soft occlusion)
* ``blocked_bb[L]`` same with backbone+CB only
* ``depth``         mean first-hit distance (transmittance < 1/2), capped at max(L)
* ``hit[c]``        fraction of rays whose first hit (within max(L)) is an atom of class c;
                    ``1 - sum(hit)`` is the escape fraction (mouth openness)

Opacity weights (AF soft occlusion) map to absorption exactly as in
:mod:`pocketdetect.geometry.buriedness` (w = 1 -> opaque).  The grid is built in the frame of the
coordinates passed in (the pipeline passes canonical-frame coordinates).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from ..geometry._grid import grid_frame
from ..geometry.buriedness import OccupancyGrid, absorption_from_weights
from ..geometry.sphere import fibonacci_sphere


@dataclass
class DualGrid:
    mu: np.ndarray        # (nx,ny,nz) float32
    mu_bb: np.ndarray     # (nx,ny,nz) float32
    label: np.ndarray     # (nx,ny,nz) int8 hit class (-1 empty)
    origin: np.ndarray
    spacing: float
    ref_length: float

    def occupancy(self) -> OccupancyGrid:
        return OccupancyGrid(self.mu, self.origin, self.spacing, self.ref_length)


@njit(parallel=True, cache=True)
def _dual_kernel(cs, mu_a, bb_a, cls_a, rb, origin, dims, h, rmax):
    nx, ny, nz = dims[0], dims[1], dims[2]
    mu = np.zeros((nx, ny, nz), np.float32)
    mub = np.zeros((nx, ny, nz), np.float32)
    lab = np.full((nx, ny, nz), -1, np.int8)
    best = np.full((nx, ny, nz), np.inf, np.float32)
    xs = cs[:, 0]
    for ix in prange(nx):
        x = origin[0] + (ix + 0.5) * h
        lo = np.searchsorted(xs, x - rmax)
        hi = np.searchsorted(xs, x + rmax, side="right")
        for a in range(lo, hi):
            m = mu_a[a]
            if m <= 0.0:
                continue
            r = rb[a]
            r2 = r * r
            dx2 = (x - xs[a]) ** 2
            rem = r2 - dx2
            if rem < 0:
                continue
            ya = cs[a, 1]
            za = cs[a, 2]
            ry = math.sqrt(rem)
            iy0 = max(0, int(math.ceil((ya - ry - origin[1]) / h - 0.5)))
            iy1 = min(ny - 1, int(math.floor((ya + ry - origin[1]) / h - 0.5)))
            for iy in range(iy0, iy1 + 1):
                y = origin[1] + (iy + 0.5) * h
                rem2 = rem - (y - ya) ** 2
                if rem2 < 0:
                    continue
                rz = math.sqrt(rem2)
                iz0 = max(0, int(math.ceil((za - rz - origin[2]) / h - 0.5)))
                iz1 = min(nz - 1, int(math.floor((za + rz - origin[2]) / h - 0.5)))
                for iz in range(iz0, iz1 + 1):
                    z = origin[2] + (iz + 0.5) * h
                    d2 = dx2 + (y - ya) ** 2 + (z - za) ** 2
                    if d2 > r2:
                        continue
                    if m > mu[ix, iy, iz]:
                        mu[ix, iy, iz] = m
                    if bb_a[a] and m > mub[ix, iy, iz]:
                        mub[ix, iy, iz] = m
                    sc = np.float32(math.sqrt(d2) - r)
                    if sc < best[ix, iy, iz]:
                        best[ix, iy, iz] = sc
                        lab[ix, iy, iz] = cls_a[a]
    return mu, mub, lab


def dual_grid(coords, radii, hit_class, bb_mask, weights=None, spacing: float = 1.0,
              ref_length: float = 2.0, radius_scale: float = 1.0, radius_pad: float = 0.5) -> DualGrid:
    c = np.asarray(coords, np.float64)
    rb = np.asarray(radii, np.float64) * radius_scale + radius_pad
    w = np.ones(len(c)) if weights is None else np.asarray(weights, np.float64)
    mu = absorption_from_weights(w, ref_length).astype(np.float64)
    if len(c) == 0:
        z = np.zeros((1, 1, 1), np.float32)
        return DualGrid(z, z.copy(), np.full((1, 1, 1), -1, np.int8), np.zeros(3), spacing, ref_length)
    rmax = float(rb.max())
    origin, dims = grid_frame(c.min(0), c.max(0), spacing, rmax)
    o = np.argsort(c[:, 0], kind="stable")
    mu_g, mub_g, lab = _dual_kernel(np.ascontiguousarray(c[o]), mu[o], np.asarray(bb_mask, np.bool_)[o],
                                    np.asarray(hit_class, np.int8)[o], rb[o], origin, dims, float(spacing), rmax)
    return DualGrid(mu_g, mub_g, lab, origin, float(spacing), float(ref_length))


@njit(parallel=True, fastmath=True, cache=True)
def _march_kernel(points, dirs, mu, mub, lab, origin, h, step, n_steps, ck, n_cls):
    p_n = points.shape[0]
    k_n = dirs.shape[0]
    n_l = ck.shape[0]
    nx, ny, nz = mu.shape
    fm = mu.reshape(-1)
    fb = mub.reshape(-1)
    fl = lab.reshape(-1)
    lmax = n_steps * step
    blocked = np.zeros((p_n, n_l), np.float32)
    blocked_bb = np.zeros((p_n, n_l), np.float32)
    depth = np.zeros(p_n, np.float32)
    hit = np.zeros((p_n, n_cls), np.float32)
    inv_h = 1.0 / h
    for p in prange(p_n):
        px = (points[p, 0] - origin[0]) * inv_h
        py = (points[p, 1] - origin[1]) * inv_h
        pz = (points[p, 2] - origin[2]) * inv_h
        acc = np.zeros(n_l, np.float64)
        accb = np.zeros(n_l, np.float64)
        hc = np.zeros(n_cls, np.float64)
        dsum = 0.0
        for k in range(k_n):
            dx = dirs[k, 0] * inv_h
            dy = dirs[k, 1] * inv_h
            dz = dirs[k, 2] * inv_h
            tau = 0.0
            taub = 0.0
            l = 0
            hit_t = lmax
            got = False
            entered = False
            for s in range(n_steps):
                t = (s + 0.5) * step
                fx = px + dx * t
                fy = py + dy * t
                fz = pz + dz * t
                if fx < 0.0 or fy < 0.0 or fz < 0.0 or fx >= nx or fy >= ny or fz >= nz:
                    if entered:
                        break
                    v = 0.0
                    vb = 0.0
                    li = -1
                else:
                    entered = True
                    idx = (np.int64(fx) * ny + np.int64(fy)) * nz + np.int64(fz)
                    v = fm[idx]
                    vb = fb[idx]
                    li = fl[idx]
                if v > 0.0:
                    tau += v * step
                    if not got and tau > 0.6931471805599453:
                        got = True
                        hit_t = t
                        if li >= 0:
                            hc[li] += 1.0
                if vb > 0.0:
                    taub += vb * step
                while l < n_l and ck[l] == s + 1:
                    acc[l] += 1.0 - math.exp(-tau)
                    accb[l] += 1.0 - math.exp(-taub)
                    l += 1
                if tau > 30.0 and taub > 30.0:
                    break
            rest = 1.0 - math.exp(-tau)
            restb = 1.0 - math.exp(-taub)
            while l < n_l:
                acc[l] += rest
                accb[l] += restb
                l += 1
            dsum += hit_t
        for l in range(n_l):
            blocked[p, l] = acc[l] / k_n
            blocked_bb[p, l] = accb[l] / k_n
        for c in range(n_cls):
            hit[p, c] = hc[c] / k_n
        depth[p] = dsum / k_n
    return blocked, blocked_bb, depth, hit


def march(points, grid: DualGrid, n_rays: int = 60, lengths=(6.0, 10.0, 16.0, 24.0), step: float = 1.0,
          n_cls: int = 4):
    lengths = tuple(float(x) for x in lengths)
    ck = np.array([int(round(L / step)) for L in lengths], np.int64)
    dirs = fibonacci_sphere(n_rays).astype(np.float64)
    pts = np.ascontiguousarray(points, np.float32)
    if len(pts) == 0:
        z = np.zeros((0, len(lengths)), np.float32)
        return z, z.copy(), np.zeros(0, np.float32), np.zeros((0, n_cls), np.float32)
    return _march_kernel(pts, dirs, grid.mu, grid.mu_bb, grid.label, grid.origin, float(grid.spacing),
                         float(step), int(ck[-1]), ck, n_cls)
