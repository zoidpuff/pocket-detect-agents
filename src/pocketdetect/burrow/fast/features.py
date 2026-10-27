"""Fast BURROW point featurizer (CPU), numerically identical to ``burrow.features.point_features``.

Identical by construction for the shared parts (canonical frame, grid candidates, dual grid, PSP,
ray march are the very same functions).  Re-implemented (Numba, exact arithmetic replication):

* metal candidates (reference: cKDTree + Python loops)
* nearest-atom distance of metal candidates
* neighbour sums over sparse per-atom properties (adding a zero is exact, so skipping zeros and
  keeping the reference's atom visiting order gives identical float64 sums)
* metal-coordination geometry (n_coord28, coord_spread; float32 NumPy semantics)

``supports(cfg)`` returns False when the reference featurizer's source changed since this port was
validated (see ``guard.py``); the pipeline then falls back to the reference implementation.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

from ...geometry._cells import box_dist2, build_cell_list, cell_range
from ...geometry.candidates import grid_candidates, psp_counts, sublattice_mask
from ...geometry.frames import canonical_frame, from_frame, to_frame
from ...geometry.neighbors import neighbor_features
from ..features import _METAL_DIRS, _METAL_SHELLS, BurrowConfig, PointSet
from ..prep import HIT_CLASSES
from ..raycast import dual_grid, march


def supports(cfg) -> bool:
    from .guard import stage_ok
    return stage_ok("features")


# --------------------------------------------------------------------------- cell-list queries
@njit(cache=True)
def _count_within(q, pts, r, origin, dims, start, order, cs):
    """Number of pts within r (d^2 <= r^2) of each query."""
    n = q.shape[0]
    out = np.zeros(n, np.int64)
    r2 = r * r
    for i in range(n):
        x, y, z = q[i, 0], q[i, 1], q[i, 2]
        x0, x1 = cell_range(x, origin[0], cs, r, dims[0])
        y0, y1 = cell_range(y, origin[1], cs, r, dims[1])
        z0, z1 = cell_range(z, origin[2], cs, r, dims[2])
        c = 0
        for cx in range(x0, x1):
            for cy in range(y0, y1):
                for cz in range(z0, z1):
                    cc = (cx * dims[1] + cy) * dims[2] + cz
                    for t in range(start[cc], start[cc + 1]):
                        j = order[t]
                        dx = q[i, 0] - pts[j, 0]
                        dy = q[i, 1] - pts[j, 1]
                        dz = q[i, 2] - pts[j, 2]
                        if dx * dx + dy * dy + dz * dz <= r2:
                            c += 1
        out[i] = c
    return out


@njit(cache=True)
def _within_csr(q, pts, r, origin, dims, start, order, cs):
    """Sorted indices of pts within r of each query (CSR), like cKDTree.query_ball_point."""
    n = q.shape[0]
    r2 = r * r
    ptr = np.zeros(n + 1, np.int64)
    out = np.empty(0, np.int64)
    for pas in range(2):
        for i in range(n):
            x, y, z = q[i, 0], q[i, 1], q[i, 2]
            x0, x1 = cell_range(x, origin[0], cs, r, dims[0])
            y0, y1 = cell_range(y, origin[1], cs, r, dims[1])
            z0, z1 = cell_range(z, origin[2], cs, r, dims[2])
            k = 0
            for cx in range(x0, x1):
                for cy in range(y0, y1):
                    for cz in range(z0, z1):
                        cc = (cx * dims[1] + cy) * dims[2] + cz
                        for t in range(start[cc], start[cc + 1]):
                            j = order[t]
                            dx = q[i, 0] - pts[j, 0]
                            dy = q[i, 1] - pts[j, 1]
                            dz = q[i, 2] - pts[j, 2]
                            if dx * dx + dy * dy + dz * dz <= r2:
                                if pas == 1:
                                    out[ptr[i] + k] = j
                                k += 1
            if pas == 0:
                ptr[i + 1] = ptr[i] + k
            else:
                out[ptr[i]:ptr[i] + k] = np.sort(out[ptr[i]:ptr[i] + k])
        if pas == 0:
            out = np.empty(ptr[n], np.int64)
    return ptr, out


@njit(cache=True)
def _any_within_unmasked(q, pts, mask, r, origin, dims, start, order, cs):
    """True where some pts[j] with mask[j] == False lies within r (d^2 <= r^2) of the query."""
    n = q.shape[0]
    out = np.zeros(n, np.bool_)
    r2 = r * r
    for i in range(n):
        x, y, z = q[i, 0], q[i, 1], q[i, 2]
        x0, x1 = cell_range(x, origin[0], cs, r, dims[0])
        y0, y1 = cell_range(y, origin[1], cs, r, dims[1])
        z0, z1 = cell_range(z, origin[2], cs, r, dims[2])
        hit = False
        for cx in range(x0, x1):
            for cy in range(y0, y1):
                for cz in range(z0, z1):
                    cc = (cx * dims[1] + cy) * dims[2] + cz
                    for t in range(start[cc], start[cc + 1]):
                        j = order[t]
                        if mask[j]:
                            continue
                        dx = q[i, 0] - pts[j, 0]
                        dy = q[i, 1] - pts[j, 1]
                        dz = q[i, 2] - pts[j, 2]
                        if dx * dx + dy * dy + dz * dz <= r2:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                break
        out[i] = hit
    return out


@njit(cache=True)
def _nearest_dist(q, pts, origin, dims, start, order, cs):
    """sqrt(min squared distance) to pts (expanding cell shells), float64."""
    n = q.shape[0]
    out = np.empty(n)
    for i in range(n):
        best = np.inf
        r = cs
        while True:
            x0, x1 = cell_range(q[i, 0], origin[0], cs, r, dims[0])
            y0, y1 = cell_range(q[i, 1], origin[1], cs, r, dims[1])
            z0, z1 = cell_range(q[i, 2], origin[2], cs, r, dims[2])
            for cx in range(x0, x1):
                for cy in range(y0, y1):
                    for cz in range(z0, z1):
                        cc = (cx * dims[1] + cy) * dims[2] + cz
                        for t in range(start[cc], start[cc + 1]):
                            j = order[t]
                            dx = q[i, 0] - pts[j, 0]
                            dy = q[i, 1] - pts[j, 1]
                            dz = q[i, 2] - pts[j, 2]
                            d2 = dx * dx + dy * dy + dz * dz
                            if d2 < best:
                                best = d2
            if best <= r * r or (x0 == 0 and y0 == 0 and z0 == 0 and x1 == dims[0] and y1 == dims[1]
                                 and z1 == dims[2]):
                break
            r *= 2.0
        out[i] = math.sqrt(best)
    return out


class _Tree:
    """Cell list over points (float64 copy, as cKDTree uses)."""

    def __init__(self, pts, cs):
        self.p = np.ascontiguousarray(pts, np.float64)
        self.cs = float(cs)
        self.origin, self.dims, self.start, self.order = build_cell_list(self.p, self.cs)

    def count(self, q, r):
        return _count_within(q, self.p, float(r), self.origin, self.dims, self.start, self.order, self.cs)

    def ball(self, q, r):
        return _within_csr(q, self.p, float(r), self.origin, self.dims, self.start, self.order, self.cs)

    def any_unmasked(self, q, r, mask):
        return _any_within_unmasked(q, self.p, mask, float(r), self.origin, self.dims, self.start, self.order, self.cs)

    def nearest(self, q):
        return _nearest_dist(q, self.p, self.origin, self.dims, self.start, self.order, self.cs)


# --------------------------------------------------------------------------- metal candidates
@njit(cache=True)
def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@njit(cache=True)
def _metal_select(cand, nco, lig_ptr, lig, dd_ptr, dd):
    """Greedy dedup (priority: n_coord desc, index asc) then one candidate per connected cluster
    of shared coordinating atoms (union by min root, as the reference's union-find)."""
    n = cand.shape[0]
    order = np.argsort(-nco, kind="mergesort")       # == lexsort((arange, -nco))
    taken = np.zeros(n, np.bool_)
    sel = np.empty(n, np.int64)
    ns = 0
    for oi in range(n):
        i = order[oi]
        if taken[i]:
            continue
        sel[ns] = i
        ns += 1
        for t in range(dd_ptr[i], dd_ptr[i + 1]):
            taken[dd[t]] = True
    n_atoms = 0
    for t in range(lig.shape[0]):
        n_atoms = max(n_atoms, lig[t] + 1)
    parent = np.arange(max(n_atoms, 1))
    for si in range(ns):
        i = sel[si]
        a0 = lig[lig_ptr[i]]
        for t in range(lig_ptr[i] + 1, lig_ptr[i + 1]):
            ra = _find(parent, a0)
            rb = _find(parent, lig[t])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    best = np.full(max(n_atoms, 1), -1, np.int64)
    for si in range(ns):
        i = sel[si]
        g = _find(parent, lig[lig_ptr[i]])
        if best[g] < 0:
            best[g] = i
    return np.sort(best[best >= 0])


def metal_candidates(X, P, cfg: BurrowConfig):
    """Numba equivalent of ``burrow.features.metal_candidates`` (same points, same order)."""
    sc = np.flatnonzero(P.coord_sc_mask)
    co = np.flatnonzero(P.coord_mask)
    if len(sc) == 0 or len(co) < 2:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
    cand = (X[sc][:, None, None, :].astype(np.float64)
            + np.array(_METAL_SHELLS)[None, :, None, None] * _METAL_DIRS[None, None, :, :]).reshape(-1, 3)
    tco = _Tree(X[co], cfg.metal_rmax)
    cand = cand[tco.count(cand, cfg.metal_rmax) >= 2]
    if len(cand) == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
    tall = _Tree(X, cfg.metal_clash)
    ok = tall.nearest(cand) >= cfg.metal_rmin
    bad = tall.any_unmasked(cand, cfg.metal_clash, np.ascontiguousarray(P.coord_mask, np.bool_))
    cand = cand[ok & ~bad]
    if len(cand) == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
    lptr, lig = tco.ball(cand, cfg.metal_rmax)
    nco = np.diff(lptr)
    scm = np.asarray(P.coord_sc_mask, bool)[co]
    has_sc = np.zeros(len(cand), bool)
    np.logical_or.at(has_sc, np.repeat(np.arange(len(cand)), nco), scm[lig])
    keep = (nco >= 2) & has_sc
    if not keep.any():
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
    kidx = np.flatnonzero(keep)
    cand = cand[kidx]
    nco = nco[kidx]
    lens = np.diff(lptr)[kidx]
    lig = np.concatenate([lig[lptr[i]:lptr[i + 1]] for i in kidx]) if len(kidx) else np.zeros(0, np.int64)
    lptr = np.concatenate([[0], np.cumsum(lens)])
    dptr, dd = _Tree(cand, cfg.metal_dedup).ball(cand, cfg.metal_dedup)
    idx = _metal_select(cand, nco.astype(np.int64), lptr.astype(np.int64), lig.astype(np.int64), dptr, dd)
    return cand[idx].astype(np.float32), nco[idx].astype(np.int32)


# --------------------------------------------------------------------------- neighbour sums (sparse props)
@njit(fastmath=True, cache=True)   # fastmath like the reference kernel (same d2 contraction)
def _nf_sparse(points, coords, nz_ptr, nz_col, nz_val, n_f, r2s, rcut, cell_size, origin, dims, start, order):
    """Same visiting order and float64 accumulation as geometry.neighbors._nf_kernel (radii only)."""
    p_n = points.shape[0]
    r_n = r2s.shape[0]
    sums = np.zeros((p_n, r_n, n_f), np.float32)
    rcut2 = rcut * rcut
    acc = np.zeros((r_n, n_f), np.float64)
    for p in range(p_n):
        x = np.float64(points[p, 0])
        y = np.float64(points[p, 1])
        z = np.float64(points[p, 2])
        acc[:, :] = 0.0
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
                            for e in range(nz_ptr[a], nz_ptr[a + 1]):
                                acc[b, nz_col[e]] += nz_val[e]
        for f in range(n_f):
            run = 0.0
            for b in range(r_n):
                run += acc[b, f]
                sums[p, b, f] = run
    return sums


def neighbor_sums(points, coords, props, radii):
    """Identical to ``geometry.neighbor_features(points, coords, props, radii, None).sums``."""
    pts = np.ascontiguousarray(points, np.float32)
    coords = np.ascontiguousarray(coords, np.float32)
    props = np.ascontiguousarray(props, np.float32)
    radii = tuple(float(r) for r in radii)
    rcut = max(radii)
    if len(coords) == 0:
        return np.zeros((len(pts), len(radii), props.shape[1]), np.float32)
    nzr, nzc = np.nonzero(props)
    nz_ptr = np.zeros(len(coords) + 1, np.int64)
    np.cumsum(np.bincount(nzr, minlength=len(coords)), out=nz_ptr[1:])
    cs = rcut / 2.0
    origin, dims, start, order = build_cell_list(coords.astype(np.float64), cs)
    return _nf_sparse(pts, coords, nz_ptr, nzc.astype(np.int64), props[nzr, nzc].astype(np.float32),
                      props.shape[1], np.array(radii, np.float64) ** 2, rcut, cs, origin, dims, start, order)


# --------------------------------------------------------------------------- metal geometry
def norm1d_f32(M):
    """Row-wise ``np.linalg.norm(v)`` of float32 3-vectors as NumPy computes it for 1-D input:
    sqrt(float32(sdot)), where OpenBLAS sdot sums float32 products in float64 (verified bit-exact)."""
    p = (M * M).astype(np.float64)
    return np.sqrt(((p[:, 0] + p[:, 1]) + p[:, 2]).astype(np.float32))


@njit(cache=True)
def _seg_sum_f32(V, starts, counts):
    out = np.zeros((starts.shape[0], 3), np.float32)
    for k in range(starts.shape[0]):
        for t in range(starts[k], starts[k] + counts[k]):
            for d in range(3):
                out[k, d] = out[k, d] + V[t, d]
    return out


def metal_geometry(pts, Xco, r: float = 2.8):
    """n_coord28 and coord_spread exactly as the reference loop (float32 NumPy semantics)."""
    n = len(pts)
    n_co = np.zeros(n, np.float32)
    spread = np.zeros(n, np.float32)
    if len(Xco) == 0 or n == 0:
        return n_co, spread
    ptr, lst = _Tree(Xco, r).ball(np.ascontiguousarray(pts, np.float64), r)
    cnt = np.diff(ptr)
    has = np.flatnonzero(cnt)
    if len(has) == 0:
        return n_co, spread
    rows = np.repeat(np.arange(n), cnt)
    V = Xco[lst] - pts[rows]                                         # float32, as X[co[l]] - pts[i]
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9             # row-wise: identical per row
    n_co[has] = cnt[has]
    starts = ptr[:-1][has]
    sums = _seg_sum_f32(V, starts, cnt[has])                         # sequential float32 rows (== mean's sum)
    means = sums / cnt[has].astype(np.float32)[:, None]              # == mean(0) (float32 true_divide)
    spread[has] = 1.0 - norm1d_f32(means)
    return n_co, spread


# --------------------------------------------------------------------------- full featurizer
def point_features(P, cfg: BurrowConfig = BurrowConfig(), weights=None) -> PointSet:
    tm = {}
    t0 = time.perf_counter()
    coords = P.coords
    if cfg.canonical:
        R, c = canonical_frame(coords)
        X = to_frame(coords, R, c)
    else:
        R, c, X = np.eye(3), np.zeros(3), coords
    if weights is not None:
        cm = np.asarray(weights) >= 0.5
        g = grid_candidates(X[cm], P.radii[cm], cfg.spacing, cfg.probe, cfg.max_dist)
    else:
        g = grid_candidates(X, P.radii, cfg.spacing, cfg.probe, cfg.max_dist)
    t1 = time.perf_counter()
    grid = dual_grid(X, P.radii, P.hit_class, P.bb_mask, weights, cfg.spacing, radius_pad=cfg.radius_pad)
    psp = psp_counts(g.points, grid.occupancy(), cfg.psp_length, cfg.psp_dirs)
    dense = psp >= cfg.min_psp
    keep = dense.copy()
    if cfg.subsample and cfg.subsample > 1:
        keep |= sublattice_mask(g.points, cfg.spacing, cfg.subsample)
    pts = g.points[keep]
    vol = np.where(dense[keep], 1.0, float(cfg.subsample ** 3)).astype(np.float32)
    psp_k = psp[keep].astype(np.float32)
    ndist = g.nearest_dist[keep].astype(np.float32)
    t2 = time.perf_counter()
    mp = metal_candidates(X, P, cfg)[0] if cfg.metal else np.zeros((0, 3), np.float32)
    if len(mp):
        mpsp = psp_counts(mp, grid.occupancy(), cfg.psp_length, cfg.psp_dirs).astype(np.float32)
        md = _Tree(X, 4.0).nearest(np.ascontiguousarray(mp, np.float64))
        pts = np.concatenate([pts, mp])
        vol = np.concatenate([vol, np.zeros(len(mp), np.float32)])
        psp_k = np.concatenate([psp_k, mpsp])
        ndist = np.concatenate([ndist, md.astype(np.float32)])
    is_metal = np.zeros(len(pts), bool)
    is_metal[len(pts) - len(mp):] = True
    t3 = time.perf_counter()
    blocked, blocked_bb, depth, hit = march(pts, grid, cfg.n_rays, cfg.lengths)
    t4 = time.perf_counter()
    S = neighbor_features(pts, X, P.props, cfg.nb_radii, None).sums
    t5 = time.perf_counter()
    co = np.flatnonzero(P.coord_mask)
    n_co, spread = metal_geometry(pts, X[co])
    names, cols = [], []

    def add(n, v):
        names.append(n)
        cols.append(np.asarray(v, np.float32))

    # ---- assembly: verbatim from the reference
    for i, L in enumerate(cfg.lengths):
        add(f"bur{int(L)}", blocked[:, i])
    for i, L in enumerate(cfg.lengths):
        add(f"burbb{int(L)}", blocked_bb[:, i])
    add("depth", depth)
    for i, h in enumerate(HIT_CLASSES):
        add(f"hit_{h}", hit[:, i])
    add("escape", 1.0 - hit.sum(1))
    add("psp", psp_k)
    add("nearest", ndist)
    add("metal_cand", is_metal)
    add("n_coord28", n_co)
    add("coord_spread", spread)
    pn = P.prop_names
    ia = pn.index("all")
    for ri, r in enumerate(cfg.nb_radii):
        add(f"n_all{int(r)}", S[:, ri, ia])
    r4 = list(cfg.nb_radii).index(4.0)
    for fi, f in enumerate(pn):
        if f in ("all", "wcn"):
            continue
        add(f"n4_{f}", S[:, r4, fi])
    for r in (8.0, 12.0):
        ri = list(cfg.nb_radii).index(r)
        den = np.maximum(S[:, ri, ia], 1.0)
        for fi, f in enumerate(pn):
            if f == "all":
                continue
            add(f"f{int(r)}_{f}", S[:, ri, fi] / den)
    add("prot_frac", S[:, list(cfg.nb_radii).index(10.0), ia] / max(len(coords), 1))
    feats = np.stack(cols, 1) if cols else np.zeros((len(pts), 0), np.float32)
    out_pts = from_frame(pts, R, c) if cfg.canonical else pts.copy()
    tm.update(points=(t1 - t0) * 1e3, prefilter=(t2 - t1) * 1e3, metal=(t3 - t2) * 1e3,
              rays=(t4 - t3) * 1e3, neighbors=(t5 - t4) * 1e3, misc=(time.perf_counter() - t5) * 1e3)
    return PointSet(out_pts, pts, vol, is_metal, feats, names, tm)
