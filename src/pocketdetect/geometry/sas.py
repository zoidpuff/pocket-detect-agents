"""Solvent-accessible surface (SAS) points (Shrake–Rupley).

``n_sphere`` Fibonacci points on every atom sphere of radius ``r_vdw + probe``;
a point is kept iff it is not strictly inside the expanded sphere of any
*other* atom.  Numba CPU implementation and batched PyTorch/Triton version.
Grid-based candidates live in :mod:`pocketdetect.geometry.candidates`.
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
