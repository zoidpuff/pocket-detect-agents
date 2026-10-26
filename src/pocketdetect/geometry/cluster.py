"""Point clustering: single linkage, mean shift, quick shift (density peaks).

All functions return canonical integer labels: clusters are numbered
0, 1, 2, ... in order of their lowest point index, so CPU and GPU results can
be compared with ``np.array_equal``.  An optional ``batch`` vector prevents
points of different structures from being clustered together.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ._cells import build_cell_list, cell_range

GAUSS_CUTOFF = 3.0


def canonical_labels(labels) -> np.ndarray:
    """Relabel so that clusters are numbered by their first (lowest) index."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.astype(np.int64)
    _, first, inv = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(len(first), np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return rank[inv.reshape(-1)]


def _batched(fn, points, batch, **kw):
    """Apply a single-structure clustering to each batch and offset labels."""
    if batch is None:
        return fn(points, **kw)
    batch = np.asarray(batch)
    labels = np.empty(len(points), np.int64)
    off = 0
    for b in np.unique(batch):
        m = np.nonzero(batch == b)[0]
        lab = fn(points[m], **kw)
        labels[m] = lab + off
        off += int(lab.max()) + 1 if len(lab) else 0
    return canonical_labels(labels)


# --------------------------------------------------------------------------- single linkage
def _single_linkage_one(points, dist):
    n = len(points)
    if n == 0:
        return np.zeros(0, np.int64)
    pairs = cKDTree(points).query_pairs(dist, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return canonical_labels(lab)


def single_linkage(points, dist: float, batch=None) -> np.ndarray:
    """Connected components of the graph ``|x_i - x_j| <= dist`` (CPU, cKDTree)."""
    pts = np.asarray(points, np.float64)
    return _batched(_single_linkage_one, pts, batch, dist=dist)


# --------------------------------------------------------------------------- mean shift
@njit(parallel=True, fastmath=True, cache=True)
def _mean_shift_kernel(points, weights, bw, max_iter, tol, cell_size, origin, dims, start, order):
    n = points.shape[0]
    modes = points.copy()
    inv2 = 1.0 / (2.0 * bw * bw)
    rcut = GAUSS_CUTOFF * bw
    rcut2 = rcut * rcut
    tol2 = tol * tol
    for i in prange(n):
        mx = modes[i, 0]
        my = modes[i, 1]
        mz = modes[i, 2]
        for it in range(max_iter):
            sx = 0.0
            sy = 0.0
            sz = 0.0
            sw = 0.0
            x0, x1 = cell_range(mx, origin[0], cell_size, rcut, dims[0])
            y0, y1 = cell_range(my, origin[1], cell_size, rcut, dims[1])
            z0, z1 = cell_range(mz, origin[2], cell_size, rcut, dims[2])
            for cx in range(x0, x1):
                for cy in range(y0, y1):
                    for cz in range(z0, z1):
                        c = (cx * dims[1] + cy) * dims[2] + cz
                        for t in range(start[c], start[c + 1]):
                            j = order[t]
                            dx = points[j, 0] - mx
                            dy = points[j, 1] - my
                            dz = points[j, 2] - mz
                            d2 = dx * dx + dy * dy + dz * dz
                            if d2 <= rcut2:
                                w = weights[j] * np.exp(-d2 * inv2)
                                sx += w * points[j, 0]
                                sy += w * points[j, 1]
                                sz += w * points[j, 2]
                                sw += w
            if sw <= 0.0:
                break
            nx = sx / sw
            ny = sy / sw
            nz = sz / sw
            sh = (nx - mx) ** 2 + (ny - my) ** 2 + (nz - mz) ** 2
            mx = nx
            my = ny
            mz = nz
            if sh < tol2:
                break
        modes[i, 0] = mx
        modes[i, 1] = my
        modes[i, 2] = mz
    return modes


def _mean_shift_one(points, bandwidth, weights, max_iter, tol, merge_dist):
    n = len(points)
    if n == 0:
        return np.zeros(0, np.int64)
    w = np.ones(n) if weights is None else np.asarray(weights, np.float64)
    cs = GAUSS_CUTOFF * bandwidth / 2.0
    origin, dims, start, order = build_cell_list(points, cs)
    modes = _mean_shift_kernel(points, w, float(bandwidth), int(max_iter), float(tol), cs,
                               origin, dims, start, order)
    return _single_linkage_one(modes, merge_dist)


def mean_shift(points, bandwidth: float, weights=None, batch=None, max_iter: int = 50,
               tol: float = 1e-2, merge_dist: float | None = None) -> np.ndarray:
    """Gaussian mean shift (kernel truncated at 3·bandwidth) on CPU.

    Every point climbs to a mode of the (weighted) kernel density; modes closer
    than ``merge_dist`` (default ``bandwidth / 2``) are merged by single linkage.
    """
    pts = np.asarray(points, np.float64)
    md = bandwidth / 2.0 if merge_dist is None else merge_dist
    if batch is None:
        return _mean_shift_one(pts, bandwidth, weights, max_iter, tol, md)
    w = None if weights is None else np.asarray(weights, np.float64)
    batch = np.asarray(batch)
    labels = np.empty(len(pts), np.int64)
    off = 0
    for b in np.unique(batch):
        m = np.nonzero(batch == b)[0]
        lab = _mean_shift_one(pts[m], bandwidth, None if w is None else w[m], max_iter, tol, md)
        labels[m] = lab + off
        off += int(lab.max()) + 1
    return canonical_labels(labels)


# --------------------------------------------------------------------------- quick shift
def _density(points, weights, sigma, tree):
    pairs = tree.query_pairs(GAUSS_CUTOFF * sigma, output_type="ndarray")
    d2 = ((points[pairs[:, 0]] - points[pairs[:, 1]]) ** 2).sum(1)
    k = np.exp(-d2 / (2 * sigma * sigma))
    rho = weights.copy()
    np.add.at(rho, pairs[:, 0], k * weights[pairs[:, 1]])
    np.add.at(rho, pairs[:, 1], k * weights[pairs[:, 0]])
    return rho


def _quick_shift_one(points, sigma, max_link, weights):
    n = len(points)
    if n == 0:
        return np.zeros(0, np.int64)
    w = np.ones(n) if weights is None else np.asarray(weights, np.float64)
    tree = cKDTree(points)
    rho = _density(points, w, sigma, tree)
    pairs = tree.query_pairs(max_link, output_type="ndarray")
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    up = (rho[j] > rho[i]) | ((rho[j] == rho[i]) & (j < i))
    i, j = i[up], j[up]
    d2 = ((points[i] - points[j]) ** 2).sum(1)
    # nearest higher-density neighbour: sort by (i, d2, j) and take first per i
    o = np.lexsort((j, d2, i))
    i, j = i[o], j[o]
    first = np.r_[True, i[1:] != i[:-1]] if len(i) else np.zeros(0, bool)
    parent = np.arange(n)
    parent[i[first]] = j[first]
    for _ in range(64):
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    return canonical_labels(parent)


def quick_shift(points, sigma: float, max_link: float, weights=None, batch=None) -> np.ndarray:
    """Quick shift / density-peak clustering on CPU.

    Density ``rho_i = sum_j w_j exp(-d_ij^2 / 2 sigma^2)`` (cut at 3 sigma, self
    included).  Each point links to its nearest neighbour within ``max_link``
    of strictly higher density (ties broken by lower index); trees rooted at
    density peaks are the clusters.
    """
    pts = np.asarray(points, np.float64)
    if batch is None:
        return _quick_shift_one(pts, sigma, max_link, weights)
    w = None if weights is None else np.asarray(weights, np.float64)
    batch = np.asarray(batch)
    labels = np.empty(len(pts), np.int64)
    off = 0
    for b in np.unique(batch):
        m = np.nonzero(batch == b)[0]
        lab = _quick_shift_one(pts[m], sigma, max_link, None if w is None else w[m])
        labels[m] = lab + off
        off += int(lab.max()) + 1
    return canonical_labels(labels)


# --------------------------------------------------------------------------- GPU
def _canonical_torch(root):
    """Torch equivalent of :func:`canonical_labels` (first-occurrence order)."""
    import torch
    _, inv = torch.unique(root, sorted=True, return_inverse=True)
    n_c = int(inv.max().item()) + 1 if inv.numel() else 0
    first = torch.full((n_c,), inv.numel(), dtype=torch.long, device=inv.device)
    first.scatter_reduce_(0, inv, torch.arange(inv.numel(), device=inv.device), "amin")
    rank = torch.empty_like(first)
    rank[torch.argsort(first)] = torch.arange(n_c, device=inv.device)
    return rank[inv]


def _components_torch(n, qi, aj, dev):
    """Connected components by min-label propagation + pointer jumping."""
    import torch
    lab = torch.arange(n, device=dev)
    while True:
        new = lab.clone()
        new.scatter_reduce_(0, qi, lab[aj], "amin")
        new.scatter_reduce_(0, aj, lab[qi], "amin")
        for _ in range(8):
            new = new[new]
        if torch.equal(new, lab):
            return lab
        lab = new


def single_linkage_torch(points, dist: float, batch=None, n_batch=None):
    """GPU single linkage; labels equal the CPU :func:`single_linkage`."""
    import torch
    from ._torch_cells import TorchCellList
    from .torch_utils import as_tensor
    pts = as_tensor(points, torch.float32)
    dev = pts.device
    b = None if batch is None else as_tensor(batch, torch.long, dev)
    if pts.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=dev)
    cl = TorchCellList(pts, b, dist, n_batch)
    qs, as_ = [], []
    for qi, aj, _ in cl.pairs(pts, b, dist):
        m = qi < aj
        qs.append(qi[m])
        as_.append(aj[m])
    root = _components_torch(pts.shape[0], torch.cat(qs), torch.cat(as_), dev)
    return _canonical_torch(root)


def mean_shift_torch(points, bandwidth: float, weights=None, batch=None, max_iter: int = 50,
                     tol: float = 1e-2, merge_dist: float | None = None, n_batch=None):
    """GPU mean shift (same algorithm as :func:`mean_shift`)."""
    import torch
    from ._torch_cells import TorchCellList
    from .torch_utils import as_tensor
    pts = as_tensor(points, torch.float32)
    dev = pts.device
    n = pts.shape[0]
    b = torch.zeros(n, dtype=torch.long, device=dev) if batch is None else as_tensor(batch, torch.long, dev)
    w = torch.ones(n, device=dev) if weights is None else as_tensor(weights, torch.float32, dev)
    rcut = GAUSS_CUTOFF * bandwidth
    cl = TorchCellList(pts, b, rcut / 2.0, n_batch)
    modes = pts.clone()
    active = torch.arange(n, device=dev)
    for _ in range(max_iter):
        if active.numel() == 0:
            break
        num = torch.zeros((active.numel(), 3), device=dev)
        den = torch.zeros(active.numel(), device=dev)
        for qi, aj, d2 in cl.pairs(modes[active], b[active], rcut):
            k = w[aj] * torch.exp(-d2 / (2 * bandwidth * bandwidth))
            num.index_add_(0, qi, k[:, None] * pts[aj])
            den.index_add_(0, qi, k)
        ok = den > 0
        new = torch.where(ok[:, None], num / den.clamp(min=1e-30)[:, None], modes[active])
        sh = ((new - modes[active]) ** 2).sum(1)
        modes[active] = new
        active = active[(sh >= tol * tol) & ok]
    md = bandwidth / 2.0 if merge_dist is None else merge_dist
    return single_linkage_torch(modes, md, b, n_batch)


def quick_shift_torch(points, sigma: float, max_link: float, weights=None, batch=None, n_batch=None):
    """GPU quick shift (same algorithm as :func:`quick_shift`)."""
    import torch
    from ._torch_cells import TorchCellList
    from .torch_utils import as_tensor
    pts = as_tensor(points, torch.float32)
    dev = pts.device
    n = pts.shape[0]
    b = None if batch is None else as_tensor(batch, torch.long, dev)
    w = torch.ones(n, device=dev, dtype=torch.float64) if weights is None else as_tensor(weights, torch.float64, dev)
    rc = GAUSS_CUTOFF * sigma
    cl = TorchCellList(pts, b, max(rc, max_link) / 2.0, n_batch)
    rho = torch.zeros(n, device=dev, dtype=torch.float64)
    for qi, aj, d2 in cl.pairs(pts, b, rc):
        rho.index_add_(0, qi, w[aj] * torch.exp(-d2.double() / (2 * sigma * sigma)))
    INF = torch.iinfo(torch.int64).max
    best = torch.full((n,), INF, dtype=torch.int64, device=dev)
    for qi, aj, d2 in cl.pairs(pts, b, max_link):
        up = (rho[aj] > rho[qi]) | ((rho[aj] == rho[qi]) & (aj < qi))
        qi, aj, d2 = qi[up], aj[up], d2[up]
        # pack (d2 quantised, j) so that amin picks nearest, ties -> lower j
        key = (torch.floor(d2.double() * float(1 << 20)).long() << 31) | aj
        best.scatter_reduce_(0, qi, key, "amin")
    parent = torch.where(best == INF, torch.arange(n, device=dev), best & ((1 << 31) - 1))
    for _ in range(64):
        nxt = parent[parent]
        if torch.equal(nxt, parent):
            break
        parent = nxt
    return _canonical_torch(parent)
