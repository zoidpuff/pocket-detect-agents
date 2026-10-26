"""BURROW candidate points (DESIGN §2.2) and point features (§2.3), CPU.

    canonical PCA frame -> solvent-shell grid (>= r_vdW + probe, <= max_dist)
    -> dual absorption grid -> LIGSITE PSP prefilter (cascade) -> + capped metal candidates
    -> multi-scale dual-grid rays with first-hit chemistry -> neighbourhood sums -> feature matrix

All geometry is computed in the canonical frame (rotation invariance); points are returned in
the input frame.  Each grid point is a volume element (``vol`` = 1 Å^3 for dense survivors,
``subsample^3`` for coarse sub-lattice points kept by the cascade, 0 for metal candidates).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.candidates import grid_candidates, psp_counts, sublattice_mask
from ..geometry.frames import canonical_frame, from_frame, to_frame
from ..geometry.neighbors import neighbor_features
from ..geometry.sphere import fibonacci_sphere
from .prep import HIT_CLASSES, RES_GROUPS, Protein
from .raycast import dual_grid, march


@dataclass(frozen=True)
class BurrowConfig:
    spacing: float = 1.0
    probe: float = 1.0                 # shell: |x - a| >= r_vdW + probe
    max_dist: float = 6.0              # shell: nearest atom centre <= max_dist
    psp_length: float = 8.0
    psp_dirs: int = 7
    min_psp: int = 2                   # cascade: dense survivors
    subsample: int = 2                 # + coarse sub-lattice (0 = off)
    n_rays: int = 60
    lengths: tuple = (6.0, 10.0, 16.0, 24.0)
    radius_pad: float = 0.5            # occlusion radius inflation
    nb_radii: tuple = (4.0, 8.0, 10.0, 12.0)
    metal: bool = True
    metal_rmin: float = 1.8
    metal_rmax: float = 2.6
    metal_clash: float = 2.4           # non-coordinating heavy atoms must be >= this
    metal_dedup: float = 1.5
    canonical: bool = True


@dataclass
class PointSet:
    points: np.ndarray       # (P,3) input frame
    X: np.ndarray            # (P,3) canonical frame
    vol: np.ndarray          # (P,) float32 volume weight
    is_metal: np.ndarray     # (P,) bool
    feats: np.ndarray        # (P,F) float32
    names: list
    timings: dict = field(default_factory=dict)

    def col(self, name):
        return self.feats[:, self.names.index(name)]


# --------------------------------------------------------------------------- metal candidates
_METAL_DIRS = fibonacci_sphere(48).astype(np.float64)
_METAL_SHELLS = (1.95, 2.15, 2.35)


def metal_candidates(X, P: Protein, cfg: BurrowConfig):
    """Points 1.8-2.6 Å from >= 2 potential coordinating atoms (>= 1 side-chain), capped to one
    per coordinating-atom cluster after 1.5 Å dedup.  Returns (points (M,3), n_coord (M,))."""
    sc = np.flatnonzero(P.coord_sc_mask)
    co = np.flatnonzero(P.coord_mask)
    if len(sc) == 0 or len(co) < 2:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
    cand = (X[sc][:, None, None, :].astype(np.float64)
            + np.array(_METAL_SHELLS)[None, :, None, None] * _METAL_DIRS[None, None, :, :]).reshape(-1, 3)
    tco = cKDTree(X[co])
    near = tco.query_ball_point(cand, cfg.metal_rmax, return_length=True)
    ok = near >= 2
    cand = cand[ok]
    if len(cand) == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
    tall = cKDTree(X)
    # clash: any atom closer than rmin, or any non-coordinating atom closer than metal_clash
    d1, _ = tall.query(cand, k=1)
    ok = d1 >= cfg.metal_rmin
    lst = tall.query_ball_point(cand, cfg.metal_clash)
    cm = P.coord_mask
    ok &= np.array([all(cm[j] for j in l) for l in lst], bool)
    cand = cand[ok]
    if len(cand) == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
    lig = tco.query_ball_point(cand, cfg.metal_rmax)
    nco = np.array([len(l) for l in lig])
    has_sc = np.array([any(P.coord_sc_mask[co[j]] for j in l) for l in lig])
    keep = (nco >= 2) & has_sc
    cand, lig, nco = cand[keep], [lig[i] for i in np.flatnonzero(keep)], nco[keep]
    if len(cand) == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
    # greedy dedup by (n_coord desc, index)
    order = np.lexsort((np.arange(len(cand)), -nco))
    tc = cKDTree(cand)
    taken = np.zeros(len(cand), bool)
    sel = []
    for i in order:
        if taken[i]:
            continue
        sel.append(i)
        taken[tc.query_ball_point(cand[i], cfg.metal_dedup)] = True
    # one per coordinating-atom cluster (union-find over shared coordinating atoms)
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in sel:
        atoms = lig[i]
        for a in atoms[1:]:
            ra, rb = find(atoms[0]), find(a)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    best = {}
    for i in sel:  # sel is already in priority order
        g = find(lig[i][0])
        if g not in best:
            best[g] = i
    idx = np.array(sorted(best.values()), np.int64)
    return cand[idx].astype(np.float32), nco[idx].astype(np.int32)


# --------------------------------------------------------------------------- features
def point_features(P: Protein, cfg: BurrowConfig = BurrowConfig(), weights=None) -> PointSet:
    tm = {}
    t0 = time.perf_counter()
    coords = P.coords
    if cfg.canonical:
        R, c = canonical_frame(coords)
        X = to_frame(coords, R, c)
    else:
        R, c, X = np.eye(3), np.zeros(3), coords
    if weights is not None:  # AF soft occlusion: low-confidence atoms never exclude candidate points
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
    if cfg.metal:
        mp, _ = metal_candidates(X, P, cfg)
    else:
        mp = np.zeros((0, 3), np.float32)
    if len(mp):
        from ..geometry.buriedness import OccupancyGrid  # noqa: F401
        mpsp = psp_counts(mp, grid.occupancy(), cfg.psp_length, cfg.psp_dirs).astype(np.float32)
        md, _ = cKDTree(X).query(mp, k=1)
        pts = np.concatenate([pts, mp])
        vol = np.concatenate([vol, np.zeros(len(mp), np.float32)])
        psp_k = np.concatenate([psp_k, mpsp])
        ndist = np.concatenate([ndist, md.astype(np.float32)])
    is_metal = np.zeros(len(pts), bool)
    is_metal[len(pts) - len(mp):] = True
    t3 = time.perf_counter()
    blocked, blocked_bb, depth, hit = march(pts, grid, cfg.n_rays, cfg.lengths)
    t4 = time.perf_counter()
    nf = neighbor_features(pts, X, P.props, cfg.nb_radii, None)
    S = nf.sums  # (P, R, F)
    t5 = time.perf_counter()
    # metal-coordination geometry: # coordinating atoms within 2.8 Å and angular spread
    co = np.flatnonzero(P.coord_mask)
    n_co = np.zeros(len(pts), np.float32)
    spread = np.zeros(len(pts), np.float32)
    if len(co) and len(pts):
        tco = cKDTree(X[co])
        lst = tco.query_ball_point(pts, 2.8)
        for i, l in enumerate(lst):
            if l:
                v = X[co[l]] - pts[i]
                v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
                n_co[i] = len(l)
                spread[i] = 1.0 - np.linalg.norm(v.mean(0))
    names, cols = [], []

    def add(n, v):
        names.append(n)
        cols.append(np.asarray(v, np.float32))

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


__all__ = ["BurrowConfig", "PointSet", "point_features", "metal_candidates", "RES_GROUPS"]
