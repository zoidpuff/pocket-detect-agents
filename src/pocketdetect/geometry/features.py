"""End-to-end per-point geometric feature pipeline (CPU and batched GPU).

    canonical PCA frame  ->  [bricks if huge]  ->  query points (SAS | grid | cascade)
    ->  absorption grid  ->  LIGSITE PSP count  ->  [cascade filter]
    ->  multi-scale ray buriedness  ->  neighbourhood property sums (incl. protrusion)

This is the timing unit reported in ``docs/GEOMETRY_BENCH.md``.  Output
points are returned in the *input* frame; all features are rotation
invariant up to float32 round-off when ``canonical_frame=True``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .buriedness import occupancy_grid, ray_buriedness
from .candidates import cascade_mask, grid_candidates, psp_counts
from .frames import brick_boxes, canonical_frame, from_frame, in_box, to_frame
from .neighbors import GAUSS_CUTOFF, neighbor_features
from .sas import sas_points


@dataclass
class FeatureConfig:
    points: str = "sas"                 # "sas" | "grid" | "cascade"
    probe: float = 1.4
    n_sphere: int = 30                  # SAS points per atom
    grid_spacing: float = 1.0           # grid / cascade candidates
    grid_max_dist: float = 5.0          # solvent shell: nearest atom centre <= this
    psp_length: float = 8.0             # LIGSITE scan length (Å)
    psp_dirs: int = 7                   # 7 (axes + body diagonals) or 13 (+ face diagonals)
    min_psp: int = 2                    # cascade: keep psp >= min_psp ...
    subsample: int = 2                  # ... plus every 2nd voxel (2 Å sub-lattice) of the rest
    n_rays: int = 60
    ray_lengths: tuple = (8.0, 12.0, 20.0)
    occ_spacing: float = 1.0            # absorption-grid voxel (Å)
    ray_step: float = 1.0
    ref_length: float = 2.0
    radius_scale: float = 1.0
    radius_pad: float = 0.0             # occlusion radius inflation (Å)
    nb_radii: tuple = (4.0, 6.0, 8.0, 10.0, 12.0)
    sigmas: tuple = (2.0, 4.0)
    canonical_frame: bool = True
    max_voxels: int = 1 << 24           # CPU per-brick voxel budget (~150 MB of grids)
    max_voxels_gpu: int = 1 << 27       # GPU per-call voxel budget


@dataclass
class PointFeatures:
    points: np.ndarray        # (P,3) in the input frame
    atom_index: np.ndarray    # (P,) owning (SAS) or nearest (grid) atom
    psp: np.ndarray           # (P,) LIGSITE enclosure count 0..psp_dirs
    blocked: np.ndarray       # (P, n_lengths)
    depth: np.ndarray         # (P,)
    sums: np.ndarray          # (P, R, F)
    gauss: np.ndarray         # (P, S, F)
    timings: dict = field(default_factory=dict)


def _halo(cfg: FeatureConfig, radii) -> float:
    rb = float(np.max(radii)) * cfg.radius_scale + cfg.radius_pad if len(radii) else 0.0
    rcut = max(list(cfg.nb_radii) + [GAUSS_CUTOFF * s for s in cfg.sigmas])
    return max(max(cfg.ray_lengths) + rb + cfg.occ_spacing, cfg.psp_length + cfg.occ_spacing + rb, rcut,
               2 * (float(np.max(radii, initial=0)) + cfg.probe), cfg.grid_max_dist + cfg.grid_spacing)


def _pad(cfg: FeatureConfig, radii) -> float:
    return max(cfg.grid_max_dist if cfg.points != "sas" else 0.0,
               float(np.max(radii, initial=0)) * cfg.radius_scale + cfg.radius_pad) + 2.0


def _features_one(X, radii, props, weights, cfg, core, tm):
    lo, hi = core
    bricked = bool(np.isfinite(np.r_[lo, hi]).any())
    t0 = time.perf_counter()
    if cfg.points == "sas":
        s = sas_points(X, radii, cfg.probe, cfg.n_sphere)
        pts, aidx = s.points, s.atom_index
        m = in_box(X[aidx], lo, hi) if bricked else None
    else:
        g = grid_candidates(X, radii, cfg.grid_spacing, cfg.probe, cfg.grid_max_dist)
        pts, aidx = g.points, g.nearest_atom
        m = in_box(pts, lo, hi) if bricked else None
    if m is not None:
        pts, aidx = pts[m], aidx[m]
    t1 = time.perf_counter()
    occ = occupancy_grid(X, radii, weights, cfg.occ_spacing, cfg.ref_length, cfg.radius_scale, cfg.radius_pad)
    psp = psp_counts(pts, occ, cfg.psp_length, cfg.psp_dirs)
    if cfg.points == "cascade":
        keep = cascade_mask(pts, psp, cfg.grid_spacing, cfg.min_psp, cfg.subsample)
        pts, aidx, psp = pts[keep], aidx[keep], psp[keep]
    t2 = time.perf_counter()
    bur = ray_buriedness(pts, grid=occ, n_rays=cfg.n_rays, lengths=cfg.ray_lengths, step=cfg.ray_step)
    t3 = time.perf_counter()
    nf = neighbor_features(pts, X, props, cfg.nb_radii, cfg.sigmas)
    t4 = time.perf_counter()
    for k, v in (("points", t1 - t0), ("prefilter", t2 - t1), ("buriedness", t3 - t2), ("neighbors", t4 - t3)):
        tm[k] = tm.get(k, 0.0) + v * 1e3
    return pts, aidx, psp, bur.blocked, bur.depth, nf.sums, nf.gauss


def compute_point_features(coords, radii, props=None, weights=None,
                           cfg: FeatureConfig = FeatureConfig()) -> PointFeatures:
    """Full CPU feature computation for one structure (timings in ms).

    Huge inputs are processed in overlapping boxes (bricks) so that no voxel
    grid exceeds ``cfg.max_voxels``; point order is then brick-major.
    """
    t_start = time.perf_counter()
    coords = np.ascontiguousarray(coords, np.float32)
    radii = np.asarray(radii, np.float32)
    if props is None:
        props = np.ones((len(coords), 1), np.float32)
    props = np.asarray(props, np.float32)
    if cfg.canonical_frame:
        R, c = canonical_frame(coords)
        X = to_frame(coords, R, c)
    else:
        R, c, X = np.eye(3), np.zeros(3), coords
    tm = {}
    halo = _halo(cfg, radii)
    boxes = brick_boxes(X, min(cfg.occ_spacing, cfg.grid_spacing), _pad(cfg, radii), halo, cfg.max_voxels)
    parts = []
    for lo, hi in boxes:
        idx = np.arange(len(X)) if len(boxes) == 1 else np.nonzero(in_box(X, lo, hi, halo))[0]
        if len(idx) == 0:
            continue
        w = None if weights is None else np.asarray(weights)[idx]
        out = _features_one(X[idx], radii[idx], props[idx], w, cfg, (lo, hi), tm)
        parts.append((out[0], idx[out[1]].astype(np.int32)) + out[2:])
    cat = [np.concatenate([p[i] for p in parts]) for i in range(7)]
    pts = from_frame(cat[0], R, c) if cfg.canonical_frame else cat[0]
    tm["total"] = (time.perf_counter() - t_start) * 1e3
    tm["n_bricks"] = len(boxes)
    return PointFeatures(pts, cat[1], cat[2], cat[3], cat[4], cat[5], cat[6], tm)


# --------------------------------------------------------------------------- GPU
def concat_structures(structures):
    """Concatenate a list of dicts with ``coords, radii[, props, weights]``.

    Returns concatenated numpy arrays plus the atom batch vector.
    """
    coords = np.concatenate([np.asarray(s["coords"], np.float32) for s in structures])
    radii = np.concatenate([np.asarray(s["radii"], np.float32) for s in structures])
    batch = np.concatenate([np.full(len(s["coords"]), i, np.int64) for i, s in enumerate(structures)])
    props = None
    if structures[0].get("props") is not None:
        props = np.concatenate([np.asarray(s["props"], np.float32) for s in structures])
    weights = None
    if any(s.get("weights") is not None for s in structures):
        weights = np.concatenate([np.ones(len(s["coords"]), np.float32) if s.get("weights") is None
                                  else np.asarray(s["weights"], np.float32) for s in structures])
    return coords, radii, props, weights, batch


def _gpu_items(items, cfg, dev):
    """Run the pipeline on a list of (already framed / bricked) items in ONE batched pass."""
    import torch
    from ._torch_cells import TorchCellList
    from .buriedness import TorchOccupancyGrid, ray_buriedness_torch
    from .candidates import cascade_mask_torch, grid_candidates_torch, psp_counts_torch
    from .neighbors import _prep, neighbor_features_torch
    from .sas import sas_points_torch

    coords, radii, props, weights, batch = concat_structures(items)
    nb = len(items)
    c = torch.from_numpy(coords).to(dev)
    r = torch.from_numpy(radii).to(dev)
    b = torch.from_numpy(batch).to(dev)
    p = torch.from_numpy(props).to(dev)
    if cfg.points == "sas":
        s = sas_points_torch(c, r, b, cfg.probe, cfg.n_sphere, n_batch=nb)
        pts, aidx, pb = s["points"], s["atom_index"], s["batch"]
        core_coord = c[aidx]
    else:
        g = grid_candidates_torch(c, r, b, cfg.grid_spacing, cfg.probe, cfg.grid_max_dist, n_batch=nb)
        pts, aidx, pb = g["points"], g["nearest_atom"], g["batch"]
        core_coord = pts
    cores = [it["core"] for it in items]
    if any(np.isfinite(np.r_[lo, hi]).any() for lo, hi in cores):
        lo = torch.tensor(np.stack([lo for lo, _ in cores]), device=dev, dtype=torch.float32)[pb]
        hi = torch.tensor(np.stack([hi for _, hi in cores]), device=dev, dtype=torch.float32)[pb]
        m = ((core_coord >= lo) & (core_coord < hi)).all(1)
        pts, aidx, pb = pts[m], aidx[m], pb[m]
    grid = TorchOccupancyGrid(c, r, weights, b, cfg.occ_spacing, cfg.ref_length, cfg.radius_scale, nb,
                              cfg.radius_pad)
    psp = psp_counts_torch(pts, pb, grid, cfg.psp_length, cfg.psp_dirs)
    if cfg.points == "cascade":
        keep = cascade_mask_torch(pts, psp, cfg.grid_spacing, cfg.min_psp, cfg.subsample)
        pts, aidx, pb, psp = pts[keep], aidx[keep], pb[keep], psp[keep]
    bur = ray_buriedness_torch(pts, pb, grid=grid, n_rays=cfg.n_rays, lengths=cfg.ray_lengths,
                               step=cfg.ray_step)
    _, _, rcut = _prep(cfg.nb_radii, cfg.sigmas)
    cl = TorchCellList(c, b, rcut / 2.0, nb)
    nf = neighbor_features_torch(pts, c, p, pb, b, cfg.nb_radii, cfg.sigmas, nb, cell_list=cl)
    atom_off = torch.from_numpy(np.concatenate([[0], np.cumsum([len(it["coords"]) for it in items])])).to(dev)
    out = {"points": pts, "atom_index": aidx - atom_off[pb], "psp": psp, "blocked": bur["blocked"],
           "depth": bur["depth"], "sums": nf["sums"], "gauss": nf["gauss"]}
    counts = torch.bincount(pb, minlength=nb).tolist()
    out = {k: v.cpu().numpy() for k, v in out.items()}
    split = np.cumsum(counts)[:-1]
    parts = {k: np.split(v, split) for k, v in out.items()}
    return [{k: parts[k][i] for k in parts} for i in range(nb)]


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _voxels(coords, spacing, pad):
    ext = (coords.max(0) - coords.min(0) + 2 * pad) / spacing + 3
    return float(np.prod(ext))


def compute_point_features_batch_torch(structures, cfg: FeatureConfig = FeatureConfig(), device=None):
    """Batched GPU feature computation for MANY structures.

    ``structures``: list of dicts with ``coords (N,3)``, ``radii (N,)`` and
    optional ``props (N,F)`` / ``weights (N,)``.  Structures are framed (PCA),
    bricked if a single one exceeds ``cfg.max_voxels_gpu``, and packed into as
    few calls as the voxel budget allows; inside a call every primitive runs
    once on the whole batch (flat per-structure grids / cell lists), so the
    number of kernel launches does not grow with the number of structures.
    Returns per-structure dicts of numpy arrays with the fields of
    :class:`PointFeatures` (``atom_index`` local to the structure, points in
    the input frame).
    """
    import torch
    from .torch_utils import default_device
    dev = torch.device(device) if device is not None else default_device()
    sp = min(cfg.occ_spacing, cfg.grid_spacing)
    items, frames = [], []
    for si, s in enumerate(structures):
        coords = np.asarray(s["coords"], np.float32)
        radii = np.asarray(s["radii"], np.float32)
        props = np.ones((len(coords), 1), np.float32) if s.get("props") is None else np.asarray(s["props"], np.float32)
        w = s.get("weights")
        if cfg.canonical_frame:
            R, c = canonical_frame(coords)
            X = to_frame(coords, R, c)
        else:
            R, c, X = np.eye(3), np.zeros(3), coords
        frames.append((R, c))
        halo, pad = _halo(cfg, radii), _pad(cfg, radii)
        boxes = brick_boxes(X, sp, pad, halo, cfg.max_voxels_gpu)
        w = None if w is None else np.asarray(w, np.float32)
        if len(boxes) == 1:   # common case: no copies
            items.append(dict(coords=X, radii=radii, props=props, weights=w, core=boxes[0], sid=si,
                              idx=None, vox=_voxels(X, sp, pad)))
            continue
        for lo, hi in boxes:
            idx = np.nonzero(in_box(X, lo, hi, halo))[0]
            if len(idx) == 0:
                continue
            items.append(dict(coords=X[idx], radii=radii[idx], props=props[idx],
                              weights=None if w is None else w[idx],
                              core=(lo, hi), sid=si, idx=idx, vox=_voxels(X[idx], sp, pad)))
    results = [[] for _ in structures]
    group, gvox = [], 0.0
    for it in items + [None]:
        if it is None or (group and gvox + it["vox"] > cfg.max_voxels_gpu):
            with torch.cuda.device(dev) if dev.type == "cuda" else _nullctx():
                res_list = _gpu_items(group, cfg, dev)
            for g_it, res in zip(group, res_list):
                if g_it["idx"] is not None:
                    res["atom_index"] = g_it["idx"][res["atom_index"]].astype(np.int32)
                results[g_it["sid"]].append(res)
            group, gvox = [], 0.0
        if it is not None:
            group.append(it)
            gvox += it["vox"]
    out = []
    for si, parts in enumerate(results):
        d = parts[0] if len(parts) == 1 else {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        if cfg.canonical_frame:
            d["points"] = from_frame(d["points"], *frames[si])
        out.append(d)
    return out
