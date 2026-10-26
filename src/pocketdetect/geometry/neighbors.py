"""Neighbourhood accumulation of per-atom properties around query points.

Given atom ``coords (N,3)`` and a generic property matrix ``props (N,F)``
(e.g. :func:`atom_class_matrix` one-hot classes, pLDDT, charges, ...), compute
for every query point

* ``sums[p, r, f]  = sum_{a : |p - a| <= radii[r]} props[a, f]``   (nested radii)
* ``gauss[p, s, f] = sum_{a : |p - a| <= 3 sigma_s} exp(-|p-a|^2 / (2 sigma_s^2)) props[a, f]``

With the ``all`` (ones) column of :func:`atom_class_matrix`, ``sums`` gives
plain atom counts; the count within 10 Å is P2Rank's *protrusion*
(:func:`protrusion`).  Both outputs come from one pass over the neighbour
pairs within ``max(max(radii), 3·max(sigmas))``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from ._cells import box_dist2, build_cell_list, cell_range

GAUSS_CUTOFF = 3.0  # Gaussian kernels truncated at 3 sigma


@dataclass
class NeighborFeatures:
    sums: np.ndarray    # (P, R, F) float32
    gauss: np.ndarray   # (P, S, F) float32 (S may be 0)
    radii: tuple
    sigmas: tuple


@njit(parallel=True, fastmath=True, cache=True)
def _nf_kernel(points, coords, props, r2s, inv2s2, g2s, rcut, cell_size, origin, dims, start, order):
    p_n = points.shape[0]
    f_n = props.shape[1]
    r_n = r2s.shape[0]
    s_n = inv2s2.shape[0]
    sums = np.zeros((p_n, r_n, f_n), np.float32)
    gauss = np.zeros((p_n, s_n, f_n), np.float32)
    rcut2 = rcut * rcut
    for p in prange(p_n):
        x = np.float64(points[p, 0])
        y = np.float64(points[p, 1])
        z = np.float64(points[p, 2])
        acc = np.zeros((r_n, f_n), np.float64)
        gac = np.zeros((s_n, f_n), np.float64)
        x0, x1 = cell_range(x, origin[0], cell_size, rcut, dims[0])
        y0, y1 = cell_range(y, origin[1], cell_size, rcut, dims[1])
        z0, z1 = cell_range(z, origin[2], cell_size, rcut, dims[2])
        for cx in range(x0, x1):
            for cy in range(y0, y1):
                for cz in range(z0, z1):
                    if box_dist2(x, y, z, origin, cell_size, cx, cy, cz) > rcut2:
                        continue
                    c = (cx * dims[1] + cy) * dims[2] + cz
                    for t in range(start[c], start[c + 1]):
                        a = order[t]
                        dx = coords[a, 0] - x
                        dy = coords[a, 1] - y
                        dz = coords[a, 2] - z
                        d2 = dx * dx + dy * dy + dz * dz
                        if d2 > rcut2:
                            continue
                        b = 0
                        while b < r_n and d2 > r2s[b]:
                            b += 1
                        if b < r_n:
                            for f in range(f_n):
                                acc[b, f] += props[a, f]
                        for s in range(s_n):
                            if d2 <= g2s[s]:
                                w = np.exp(-d2 * inv2s2[s])
                                for f in range(f_n):
                                    gac[s, f] += w * props[a, f]
        for f in range(f_n):
            run = 0.0
            for b in range(r_n):
                run += acc[b, f]
                sums[p, b, f] = run
            for s in range(s_n):
                gauss[p, s, f] = gac[s, f]
    return sums, gauss


def _prep(radii, sigmas):
    radii = tuple(float(r) for r in radii)
    sigmas = tuple(float(s) for s in (sigmas or ()))
    if list(radii) != sorted(radii):
        raise ValueError("radii must be ascending")
    rcut = max(list(radii) + [GAUSS_CUTOFF * s for s in sigmas])
    return radii, sigmas, rcut


def neighbor_features(points, coords, props=None, radii=(4.0, 6.0, 8.0, 10.0, 12.0),
                      sigmas=None) -> NeighborFeatures:
    """Radius-summed and Gaussian-weighted per-atom properties (CPU / Numba).

    ``props`` defaults to a single ones column (atom counts).
    """
    pts = np.ascontiguousarray(points, np.float32)
    coords = np.ascontiguousarray(coords, np.float32)
    if props is None:
        props = np.ones((len(coords), 1), np.float32)
    props = np.ascontiguousarray(props, np.float32)
    if props.ndim == 1:
        props = props[:, None]
    radii, sigmas, rcut = _prep(radii, sigmas)
    r2s = np.array(radii, np.float64) ** 2
    inv2s2 = np.array([1.0 / (2.0 * s * s) for s in sigmas], np.float64)
    g2s = np.array([(GAUSS_CUTOFF * s) ** 2 for s in sigmas], np.float64)
    if len(coords) == 0:
        return NeighborFeatures(np.zeros((len(pts), len(radii), props.shape[1]), np.float32),
                                np.zeros((len(pts), len(sigmas), props.shape[1]), np.float32), radii, sigmas)
    cs = rcut / 2.0
    origin, dims, start, order = build_cell_list(coords.astype(np.float64), cs)
    sums, gauss = _nf_kernel(pts, coords, props, r2s, inv2s2, g2s, rcut, cs, origin, dims, start, order)
    return NeighborFeatures(sums, gauss, radii, sigmas)


def protrusion(points, coords, radius: float = 10.0) -> np.ndarray:
    """Number of atoms within ``radius`` of each point (P2Rank 'protrusion')."""
    return neighbor_features(points, coords, None, (radius,)).sums[:, 0, 0]


# --------------------------------------------------------------------------- GPU
def neighbor_features_torch(points, coords, props=None, point_batch=None, atom_batch=None,
                            radii=(4.0, 6.0, 8.0, 10.0, 12.0), sigmas=None,
                            n_batch: int | None = None, cell_list=None, use_triton: bool | None = None):
    """Batched GPU version; returns dict ``sums (P,R,F)``, ``gauss (P,S,F)``.

    Pairs within the cutoff come from :class:`TorchCellList`; each pair is
    binned into its innermost radius shell with ``searchsorted`` and scattered
    with a single ``index_add_``, followed by a cumulative sum over shells.
    """
    import torch
    from ._torch_cells import TorchCellList
    from .torch_utils import as_tensor

    coords = as_tensor(coords, torch.float32)
    dev = coords.device
    pts = as_tensor(points, torch.float32, dev)
    if props is None:
        props = torch.ones((coords.shape[0], 1), dtype=torch.float32, device=dev)
    props = as_tensor(props, torch.float32, dev)
    if props.ndim == 1:
        props = props[:, None]
    radii, sigmas, rcut = _prep(radii, sigmas)
    p_n, f_n, r_n, s_n = pts.shape[0], props.shape[1], len(radii), len(sigmas)
    sums = torch.zeros((p_n * (r_n + 1), f_n), dtype=torch.float32, device=dev)
    gauss = torch.zeros((p_n, s_n, f_n), dtype=torch.float32, device=dev)
    if coords.shape[0] == 0 or p_n == 0:
        return {"sums": sums.view(p_n, r_n + 1, f_n)[:, :r_n], "gauss": gauss}
    if use_triton is None:
        from ._triton_kernels import triton_available
        use_triton = triton_available(dev)
    if use_triton and r_n + s_n <= 8:
        from ._triton_kernels import neighbor_sums_triton
        ab = (torch.zeros(coords.shape[0], dtype=torch.long, device=dev) if atom_batch is None
              else as_tensor(atom_batch, torch.long, dev))
        pb = (torch.zeros(p_n, dtype=torch.long, device=dev) if point_batch is None
              else as_tensor(point_batch, torch.long, dev))
        nbt = n_batch if n_batch is not None else int(max(ab.max().item(), pb.max().item())) + 1
        th = [r * r for r in radii] + [(GAUSS_CUTOFF * s) ** 2 for s in sigmas]
        co = [0.0] * r_n + [1.0 / (2.0 * s * s) for s in sigmas]
        out = neighbor_sums_triton(pts, pb, coords, ab, props, th, co, rcut, nbt, n_hard=r_n)
        return {"sums": out[:, :r_n], "gauss": out[:, r_n:]}
    if cell_list is None:
        cell_list = TorchCellList(coords, atom_batch if atom_batch is None else as_tensor(atom_batch, torch.long, dev),
                                  rcut / 2.0, n_batch)
    pb = None if point_batch is None else as_tensor(point_batch, torch.long, dev)
    r2s = torch.tensor(radii, device=dev, dtype=torch.float32) ** 2
    sig = torch.tensor(sigmas, device=dev, dtype=torch.float32)
    for qi, aj, d2 in cell_list.pairs(pts, pb, rcut):
        pa = props[aj]
        b = torch.searchsorted(r2s, d2.contiguous())          # first shell with d2 <= r^2 (r_n = outside)
        sums.index_add_(0, qi * (r_n + 1) + b, pa)
        if s_n:
            w = torch.exp(-d2[:, None] / (2.0 * sig * sig)[None]) * (d2[:, None] <= (GAUSS_CUTOFF * sig)[None] ** 2)
            gauss.view(p_n, s_n * f_n).index_add_(0, qi, (w[:, :, None] * pa[:, None, :]).reshape(-1, s_n * f_n))
    sums = torch.cumsum(sums.view(p_n, r_n + 1, f_n)[:, :r_n], dim=1)
    return {"sums": sums, "gauss": gauss}


def protrusion_torch(points, coords, radius: float = 10.0, point_batch=None, atom_batch=None):
    return neighbor_features_torch(points, coords, None, point_batch, atom_batch, (radius,))["sums"][:, 0, 0]


def smooth_point_values(points, values, radius: float = 2.0, batch=None) -> np.ndarray:
    """Mean of ``values`` (P,) or (P,F) over all points within ``radius`` (self included).

    Cheap denoiser for per-point features such as ray buriedness (whose
    angular-sampling noise is ~ 1/sqrt(n_rays)).  ``batch`` keeps structures apart.
    """
    pts = np.ascontiguousarray(points, np.float32)
    v = np.asarray(values, np.float32)
    one_d = v.ndim == 1
    v2 = np.c_[v.reshape(len(pts), -1), np.ones(len(pts), np.float32)]
    if batch is None:
        s = neighbor_features(pts, pts, v2, (radius,)).sums[:, 0]
    else:
        batch = np.asarray(batch)
        s = np.empty_like(v2)
        for b in np.unique(batch):
            m = np.nonzero(batch == b)[0]
            s[m] = neighbor_features(pts[m], pts[m], v2[m], (radius,)).sums[:, 0]
    out = s[:, :-1] / s[:, -1:]
    return out[:, 0] if one_d else out


def smooth_point_values_torch(points, values, radius: float = 2.0, batch=None, n_batch=None):
    """GPU version of :func:`smooth_point_values`."""
    import torch
    from .torch_utils import as_tensor
    pts = as_tensor(points, torch.float32)
    v = as_tensor(values, torch.float32, pts.device)
    one_d = v.ndim == 1
    v2 = torch.cat([v.reshape(pts.shape[0], -1), torch.ones((pts.shape[0], 1), device=pts.device)], 1)
    s = neighbor_features_torch(pts, pts, v2, batch, batch, (radius,), None, n_batch)["sums"][:, 0]
    out = s[:, :-1] / s[:, -1:]
    return out[:, 0] if one_d else out
