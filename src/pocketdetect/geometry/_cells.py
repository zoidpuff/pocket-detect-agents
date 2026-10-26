"""Numba uniform-grid cell list used by all CPU neighbour kernels.

A cell list sorts atoms into cubic cells of edge ``cell_size``.  Kernels then
visit only the cells overlapping a query sphere.  The layout is the classic
counting-sort CSR layout::

    atoms of cell c  ==  order[start[c]:start[c + 1]]

Cells are indexed ``(ix * ny + iy) * nz + iz``.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def build_cell_list(coords, cell_size):
    """Build a cell list over ``coords`` (N, 3).

    Returns ``origin (3,) f8, dims (3,) i8, start (ncell+1,) i8, order (N,) i8``.
    """
    n = coords.shape[0]
    origin = np.empty(3, np.float64)
    dims = np.empty(3, np.int64)
    for d in range(3):
        lo = np.inf
        hi = -np.inf
        for i in range(n):
            v = coords[i, d]
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        if n == 0:
            lo = 0.0
            hi = 0.0
        origin[d] = lo
        dims[d] = int(np.floor((hi - lo) / cell_size)) + 1
    ncell = dims[0] * dims[1] * dims[2]
    cell_of = np.empty(n, np.int64)
    counts = np.zeros(ncell + 1, np.int64)
    for i in range(n):
        ix = min(int((coords[i, 0] - origin[0]) / cell_size), dims[0] - 1)
        iy = min(int((coords[i, 1] - origin[1]) / cell_size), dims[1] - 1)
        iz = min(int((coords[i, 2] - origin[2]) / cell_size), dims[2] - 1)
        c = (ix * dims[1] + iy) * dims[2] + iz
        cell_of[i] = c
        counts[c + 1] += 1
    start = np.cumsum(counts)
    fill = start[:-1].copy()
    order = np.empty(n, np.int64)
    for i in range(n):
        c = cell_of[i]
        order[fill[c]] = i
        fill[c] += 1
    return origin, dims, start, order


@njit(cache=True, inline="always")
def cell_range(x, origin, cell_size, r, dim):
    """Inclusive-exclusive index range of cells overlapping [x-r, x+r]."""
    lo = int(np.floor((x - r - origin) / cell_size))
    hi = int(np.floor((x + r - origin) / cell_size)) + 1
    if lo < 0:
        lo = 0
    if hi > dim:
        hi = dim
    return lo, hi


@njit(cache=True, inline="always")
def box_dist2(x, y, z, origin, cell_size, ix, iy, iz):
    """Squared distance from point to the axis-aligned box of cell (ix,iy,iz)."""
    d2 = 0.0
    lo = origin[0] + ix * cell_size
    if x < lo:
        d2 += (lo - x) ** 2
    elif x > lo + cell_size:
        d2 += (x - lo - cell_size) ** 2
    lo = origin[1] + iy * cell_size
    if y < lo:
        d2 += (lo - y) ** 2
    elif y > lo + cell_size:
        d2 += (y - lo - cell_size) ** 2
    lo = origin[2] + iz * cell_size
    if z < lo:
        d2 += (lo - z) ** 2
    elif z > lo + cell_size:
        d2 += (z - lo - cell_size) ** 2
    return d2
