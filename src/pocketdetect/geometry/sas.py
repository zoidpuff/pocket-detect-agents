"""Solvent-accessible surface (SAS) points and grid-based candidate points.

* :func:`sas_points` – Shrake–Rupley: ``n_sphere`` Fibonacci points on every
  atom sphere of radius ``r_vdw + probe``; a point is kept iff it is not
  strictly inside the expanded sphere of any *other* atom.
* :func:`grid_candidates` – voxel centres of a regular grid (aligned to
  multiples of ``spacing`` in absolute coordinates) that are (a) not inside any
  expanded sphere ``r_vdw + probe`` (i.e. a probe sphere fits) and (b) within
  ``max_dist`` of at least one atom centre.  Each point gets its nearest atom.

Both have a Numba CPU and a batched PyTorch implementation (``*_torch``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from ._cells import build_cell_list, cell_range
from .sphere import fibonacci_sphere


@dataclass
class SASPoints:
    points: np.ndarray      # (M, 3) float32
    atom_index: np.ndarray  # (M,) int32, owning atom
    normals: np.ndarray     # (M, 3) float32, outward unit normal (sphere direction)


@dataclass
class GridPoints:
    points: np.ndarray       # (M, 3) float32 voxel centres
    nearest_atom: np.ndarray  # (M,) int32
    nearest_dist: np.ndarray  # (M,) float32, distance to nearest atom centre


def _n_sphere(n_sphere, density, radii, probe):
    if density is None:
        return int(n_sphere)
    rp = float(np.mean(radii)) + probe
    return max(4, int(round(density * 4.0 * math.pi * rp * rp)))


# --------------------------------------------------------------------------- CPU
@njit(cache=True)
def _sas_one_atom(i, coords, rp, pts, cell_size, origin, dims, start, order, rp_max, keep_row):
    xi = coords[i, 0]
    yi = coords[i, 1]
    zi = coords[i, 2]
    rcut = rp[i] + rp_max
    nb = np.empty(64, np.int64)
    nn = 0
    x0, x1 = cell_range(xi, origin[0], cell_size, rcut, dims[0])
    y0, y1 = cell_range(yi, origin[1], cell_size, rcut, dims[1])
    z0, z1 = cell_range(zi, origin[2], cell_size, rcut, dims[2])
    for cx in range(x0, x1):
        for cy in range(y0, y1):
            for cz in range(z0, z1):
                c = (cx * dims[1] + cy) * dims[2] + cz
                for t in range(start[c], start[c + 1]):
                    j = order[t]
                    if j == i:
                        continue
                    dx = coords[j, 0] - xi
                    dy = coords[j, 1] - yi
                    dz = coords[j, 2] - zi
                    rr = rp[i] + rp[j]
                    if dx * dx + dy * dy + dz * dz < rr * rr:
                        if nn == nb.shape[0]:
                            nb2 = np.empty(nn * 2, np.int64)
                            nb2[:nn] = nb
                            nb = nb2
                        nb[nn] = j
                        nn += 1
    if nn == 0:
        return
    last = 0
    for k in range(pts.shape[1]):
        px = pts[i, k, 0]
        py = pts[i, k, 1]
        pz = pts[i, k, 2]
        for tt in range(nn):
            t = tt + last
            if t >= nn:
                t -= nn
            j = nb[t]
            dx = coords[j, 0] - px
            dy = coords[j, 1] - py
            dz = coords[j, 2] - pz
            if dx * dx + dy * dy + dz * dz < rp[j] * rp[j]:
                keep_row[k] = False
                last = t
                break


@njit(parallel=True, cache=True)
def _sas_keep_kernel(coords, rp, pts, cell_size, origin, dims, start, order, rp_max):
    n = coords.shape[0]
    keep = np.ones((n, pts.shape[1]), np.bool_)
    for i in prange(n):
        _sas_one_atom(i, coords, rp, pts, cell_size, origin, dims, start, order, rp_max, keep[i])
    return keep


def sas_points(coords, radii, probe: float = 1.4, n_sphere: int = 30,
               density: float | None = None) -> SASPoints:
    """Shrake–Rupley SAS points (CPU / Numba).

    Parameters
    ----------
    coords : (N, 3) float32 atom coordinates (Å)
    radii : (N,) vdW radii (Å)
    probe : probe radius (Å)
    n_sphere : points per atom sphere (ignored if ``density`` is given)
    density : optional points per Å^2 of SAS (n_sphere = density·4π(r̄+probe)²)
    """
    coords = np.ascontiguousarray(coords, np.float32)
    radii = np.asarray(radii, np.float32)
    k = _n_sphere(n_sphere, density, radii, probe)
    sph = fibonacci_sphere(k)
    rp = (radii + np.float32(probe)).astype(np.float32)
    pts = coords[:, None, :] + rp[:, None, None] * sph[None, :, :]
    if coords.shape[0] == 0:
        e = np.zeros((0, 3), np.float32)
        return SASPoints(e, np.zeros(0, np.int32), e)
    rp_max = float(rp.max())
    cs = 2.0 * rp_max
    origin, dims, start, order = build_cell_list(coords.astype(np.float64), cs)
    keep = _sas_keep_kernel(coords.astype(np.float64), rp.astype(np.float64), pts, cs,
                            origin, dims, start, order, rp_max)
    ai, ki = np.nonzero(keep)
    return SASPoints(pts[ai, ki], ai.astype(np.int32), sph[ki])


def _grid_frame(lo, hi, spacing, max_dist):
    origin = np.floor((lo - max_dist) / spacing) * spacing - spacing
    dims = np.ceil((hi + max_dist - origin) / spacing).astype(np.int64) + 1
    return origin, dims


@njit(parallel=True, cache=True)
def _grid_stamp_kernel(cs_sorted, idx_sorted, rclash, origin, dims, h, dmax):
    nx, ny, nz = dims[0], dims[1], dims[2]
    best = np.full((nx, ny, nz), np.inf, np.float64)
    arg = np.full((nx, ny, nz), -1, np.int64)
    clash = np.zeros((nx, ny, nz), np.bool_)
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
                for iz in range(iz0, iz1 + 1):
                    z = origin[2] + (iz + 0.5) * h
                    d2 = dx2 + dy2 + (z - za) ** 2
                    if d2 > dmax2:
                        continue
                    b = best[ix, iy, iz]
                    if d2 < b or (d2 == b and ai < arg[ix, iy, iz]):
                        best[ix, iy, iz] = d2
                        arg[ix, iy, iz] = ai
                    if d2 < rc2:
                        clash[ix, iy, iz] = True
    return best, arg, clash


def grid_candidates(coords, radii, spacing: float = 1.0, probe: float = 1.4,
                    max_dist: float = 8.0) -> GridPoints:
    """Grid points in the solvent region near the protein (CPU / Numba).

    A voxel centre ``x`` is kept iff ``|x - a| >= r_a + probe`` for all atoms
    (a probe of radius ``probe`` centred at x does not overlap any vdW sphere)
    and ``min_a |x - a| <= max_dist``.  Points are returned in C order of the
    (ix, iy, iz) grid whose origin is aligned to multiples of ``spacing``.
    """
    coords = np.ascontiguousarray(coords, np.float32)
    radii = np.asarray(radii, np.float32)
    if coords.shape[0] == 0:
        return GridPoints(np.zeros((0, 3), np.float32), np.zeros(0, np.int32), np.zeros(0, np.float32))
    c64 = coords.astype(np.float64)
    origin, dims = _grid_frame(c64.min(0), c64.max(0), spacing, max_dist)
    o = np.argsort(c64[:, 0], kind="stable")
    rclash = (radii.astype(np.float64) + probe)[o]
    best, arg, clash = _grid_stamp_kernel(np.ascontiguousarray(c64[o]), o.astype(np.int64), rclash,
                                          origin, dims, float(spacing), float(max_dist))
    sel = (arg >= 0) & ~clash
    ijk = np.argwhere(sel)
    pts = (origin[None, :] + (ijk + 0.5) * spacing).astype(np.float32)
    return GridPoints(pts, arg[sel].astype(np.int32), np.sqrt(best[sel]).astype(np.float32))


# --------------------------------------------------------------------------- GPU
def sas_points_torch(coords, radii, batch=None, probe: float = 1.4, n_sphere: int = 30,
                     density: float | None = None, n_batch: int | None = None,
                     use_triton: bool | None = None):
    """Batched SAS points on a torch device.

    ``coords`` (N,3) / ``radii`` (N,) tensors (or arrays, moved to CUDA if
    available), ``batch`` (N,) structure ids.  Returns a dict of tensors
    ``points (M,3)``, ``atom_index (M,)`` (global atom index), ``normals``,
    ``batch (M,)``.  Point order equals the CPU version (atom-major).
    """
    import torch
    from ._torch_cells import TorchCellList
    from .torch_utils import as_tensor

    coords = as_tensor(coords, torch.float32)
    dev = coords.device
    radii = as_tensor(radii, torch.float32, dev)
    n = coords.shape[0]
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=dev)
    else:
        batch = as_tensor(batch, torch.long, dev)
    k = _n_sphere(n_sphere, density, radii.cpu().numpy(), probe)
    sph = torch.from_numpy(fibonacci_sphere(k)).to(dev)
    rp = radii + probe
    pts = coords[:, None, :] + rp[:, None, None] * sph[None, :, :]   # (N,K,3)
    from ._triton_kernels import sas_keep_triton, triton_available
    if use_triton is None:
        use_triton = triton_available(dev)
    if use_triton and n > 0:
        keep = sas_keep_triton(coords, rp, pts, batch,
                               n_batch if n_batch is not None else int(batch.max().item()) + 1)
        ai, ki = torch.nonzero(keep, as_tuple=True)
        return {"points": pts[ai, ki], "atom_index": ai, "normals": sph[ki], "batch": batch[ai]}
    buried = torch.zeros((n, k), dtype=torch.int32, device=dev)
    if n > 0:
        rp_max = float(rp.max().item())
        cl = TorchCellList(coords, batch, rp_max, n_batch)
        for qi, aj, d2 in cl.pairs(coords, batch, 2.0 * rp_max):
            rr = rp[qi] + rp[aj]
            m = (qi != aj) & (d2 < rr * rr)
            qi, aj = qi[m], aj[m]
            step = max(1, (1 << 24) // k)
            for s in range(0, qi.shape[0], step):
                q, a = qi[s:s + step], aj[s:s + step]
                diff = pts[q] - coords[a][:, None, :]                  # (E,K,3)
                hit = ((diff * diff).sum(-1) < (rp[a] * rp[a])[:, None]).to(torch.int32)
                buried.index_add_(0, q, hit)
    keep = buried == 0
    ai, ki = torch.nonzero(keep, as_tuple=True)
    return {"points": pts[ai, ki], "atom_index": ai, "normals": sph[ki], "batch": batch[ai]}


def grid_candidates_torch(coords, radii, batch=None, spacing: float = 1.0, probe: float = 1.4,
                          max_dist: float = 8.0, n_batch: int | None = None,
                          max_elems: int = 1 << 25):
    """Batched GPU version of :func:`grid_candidates`.

    Returns dict ``points (M,3)``, ``nearest_atom (M,)`` (global index),
    ``nearest_dist (M,)``, ``batch (M,)``; per structure in the CPU order.
    """
    import torch
    from .torch_utils import as_tensor, batch_bounds

    coords = as_tensor(coords, torch.float32)
    dev = coords.device
    radii = as_tensor(radii, torch.float32, dev)
    n = coords.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=dev) if batch is None else as_tensor(batch, torch.long, dev)
    if n_batch is None:
        n_batch = int(batch.max().item()) + 1
    lo, hi = batch_bounds(coords, batch, n_batch)
    # frame computed in float64 on host for bit-identical origin with the CPU path
    lo64, hi64 = lo.double().cpu().numpy(), hi.double().cpu().numpy()
    frames = [_grid_frame(lo64[b], hi64[b], spacing, max_dist) for b in range(n_batch)]
    origin = torch.tensor(np.stack([f[0] for f in frames]), dtype=torch.float64, device=dev)
    dims = torch.tensor(np.stack([f[1] for f in frames]), dtype=torch.long, device=dev)
    nvox = dims.prod(1)
    base = torch.cumsum(nvox, 0) - nvox
    total = int(nvox.sum().item())
    INIT = torch.iinfo(torch.int64).max
    key = torch.full((total,), INIT, dtype=torch.int64, device=dev)
    clash = torch.zeros(total, dtype=torch.bool, device=dev)

    m = int(math.ceil(max_dist / spacing)) + 1
    rng = torch.arange(-m, m + 1, device=dev)
    off = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    gap = (off.abs().double() - 0.5).clamp(min=0) * spacing
    off = off[(gap * gap).sum(1) <= max_dist * max_dist + 1e-9]
    dmax2 = max_dist * max_dist
    c64 = coords.double()
    off32 = off.to(torch.int32)
    offc = off.float() + 0.5
    chunk = max(1, max_elems // off.shape[0])
    for s in range(0, n, chunk):
        b = batch[s:s + chunk]
        rel = (c64[s:s + chunk] - origin[b]) / spacing                     # float64, exact frame
        c0 = torch.floor(rel)
        frac = (rel - c0).float()
        c0 = c0.to(torch.int32)
        dv = ((offc[None] - frac[:, None, :]) * spacing)                   # (A,O,3) float32
        d2 = (dv * dv).sum(-1)
        vox = c0[:, None, :] + off32[None]
        d = dims[b].to(torch.int32)[:, None, :]
        ok = ((vox >= 0) & (vox < d)).all(-1) & (d2 <= dmax2)
        lin = base[b][:, None] + ((vox[..., 0] * d[..., 1] + vox[..., 1]) * d[..., 2] + vox[..., 2]).long()
        q = torch.floor(d2 * float(1 << 20)).long()
        kk = (q << 31) | torch.arange(s, s + b.shape[0], device=dev)[:, None]
        key.scatter_reduce_(0, lin[ok], kk[ok], "amin")
        rc = (radii[s:s + chunk] + probe)[:, None]
        clash[lin[ok & (d2 < rc * rc)]] = True
    sel = (key != INIT) & ~clash
    flat = torch.nonzero(sel, as_tuple=True)[0]
    vb = torch.searchsorted(base, flat, right=True) - 1
    loc = flat - base[vb]
    dv = dims[vb]
    iz = loc % dv[:, 2]
    iy = (loc // dv[:, 2]) % dv[:, 1]
    ix = loc // (dv[:, 2] * dv[:, 1])
    ijk = torch.stack([ix, iy, iz], 1).double()
    pts = (origin[vb] + (ijk + 0.5) * spacing)
    near = key[flat] & ((1 << 31) - 1)
    dist = torch.sqrt(((pts - c64[near]) ** 2).sum(1))
    return {"points": pts.float(), "nearest_atom": near, "nearest_dist": dist.float(), "batch": vb}
