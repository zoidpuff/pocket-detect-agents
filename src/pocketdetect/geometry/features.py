"""End-to-end per-point geometric feature pipeline (CPU and batched GPU).

query points (SAS or grid)  ->  ray buriedness (multi-scale)  ->  neighbourhood
property sums / Gaussian sums (incl. protrusion).  This is the timing unit
reported in ``docs/GEOMETRY_BENCH.md``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .buriedness import occupancy_grid, ray_buriedness
from .neighbors import neighbor_features
from .sas import grid_candidates, sas_points


@dataclass
class FeatureConfig:
    points: str = "sas"                 # "sas" | "grid"
    probe: float = 1.4
    n_sphere: int = 30                  # SAS points per atom
    grid_spacing: float = 1.0           # grid candidates
    grid_max_dist: float = 8.0
    n_rays: int = 60
    ray_lengths: tuple = (8.0, 12.0, 20.0)
    occ_spacing: float = 1.0            # absorption-grid voxel (Å)
    ray_step: float = 0.5
    ref_length: float = 2.0
    radius_scale: float = 1.0
    nb_radii: tuple = (4.0, 6.0, 8.0, 10.0, 12.0)
    sigmas: tuple = (2.0, 4.0)


@dataclass
class PointFeatures:
    points: np.ndarray        # (P,3)
    atom_index: np.ndarray    # (P,) owning (SAS) or nearest (grid) atom
    blocked: np.ndarray       # (P, n_lengths)
    depth: np.ndarray         # (P,)
    sums: np.ndarray          # (P, R, F)
    gauss: np.ndarray         # (P, S, F)
    timings: dict = field(default_factory=dict)


def compute_point_features(coords, radii, props=None, weights=None,
                           cfg: FeatureConfig = FeatureConfig()) -> PointFeatures:
    """Full CPU feature computation for one structure (timings in ms)."""
    tm = {}
    t0 = time.perf_counter()
    if cfg.points == "sas":
        s = sas_points(coords, radii, cfg.probe, cfg.n_sphere)
        pts, aidx = s.points, s.atom_index
    else:
        g = grid_candidates(coords, radii, cfg.grid_spacing, cfg.probe, cfg.grid_max_dist)
        pts, aidx = g.points, g.nearest_atom
    t1 = time.perf_counter()
    grid = occupancy_grid(coords, radii, weights, cfg.occ_spacing, cfg.ref_length, cfg.radius_scale)
    bur = ray_buriedness(pts, grid=grid, n_rays=cfg.n_rays, lengths=cfg.ray_lengths, step=cfg.ray_step)
    t2 = time.perf_counter()
    nf = neighbor_features(pts, coords, props, cfg.nb_radii, cfg.sigmas)
    t3 = time.perf_counter()
    tm.update(points=(t1 - t0) * 1e3, buriedness=(t2 - t1) * 1e3, neighbors=(t3 - t2) * 1e3,
              total=(t3 - t0) * 1e3)
    return PointFeatures(pts, aidx, bur.blocked, bur.depth, nf.sums, nf.gauss, tm)


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


def compute_point_features_batch_torch(structures, cfg: FeatureConfig = FeatureConfig(), device=None,
                                       to_numpy: bool = True):
    """Batched GPU feature computation for MANY structures in one call.

    ``structures``: list of dicts with ``coords (N,3)``, ``radii (N,)`` and
    optional ``props (N,F)`` / ``weights (N,)``.  All structures are
    concatenated with a batch index; every primitive works on the whole batch
    at once (per-structure cell lists / voxel grids share flat tensors), so the
    number of kernel launches is independent of the batch size.  Returns a list
    of per-structure dicts (numpy arrays if ``to_numpy``) with the same fields
    as :class:`PointFeatures` (``atom_index`` is local to the structure).
    """
    import torch
    from ._torch_cells import TorchCellList
    from .buriedness import TorchOccupancyGrid, ray_buriedness_torch
    from .neighbors import _prep, neighbor_features_torch
    from .sas import grid_candidates_torch, sas_points_torch
    from .torch_utils import default_device

    dev = torch.device(device) if device is not None else default_device()
    coords, radii, props, weights, batch = concat_structures(structures)
    nb = len(structures)
    c = torch.from_numpy(coords).to(dev, non_blocking=True)
    r = torch.from_numpy(radii).to(dev)
    b = torch.from_numpy(batch).to(dev)
    p = None if props is None else torch.from_numpy(props).to(dev)
    if cfg.points == "sas":
        s = sas_points_torch(c, r, b, cfg.probe, cfg.n_sphere, n_batch=nb)
        pts, aidx, pb = s["points"], s["atom_index"], s["batch"]
    else:
        g = grid_candidates_torch(c, r, b, cfg.grid_spacing, cfg.probe, cfg.grid_max_dist, n_batch=nb)
        pts, aidx, pb = g["points"], g["nearest_atom"], g["batch"]
    grid = TorchOccupancyGrid(c, r, weights, b, cfg.occ_spacing, cfg.ref_length, cfg.radius_scale, nb)
    bur = ray_buriedness_torch(pts, pb, grid=grid, n_rays=cfg.n_rays, lengths=cfg.ray_lengths,
                               step=cfg.ray_step)
    _, _, rcut = _prep(cfg.nb_radii, cfg.sigmas)
    cl = TorchCellList(c, b, rcut / 2.0, nb)
    nf = neighbor_features_torch(pts, c, p, pb, b, cfg.nb_radii, cfg.sigmas, nb, cell_list=cl)
    atom_off = torch.from_numpy(np.concatenate([[0], np.cumsum([len(s["coords"]) for s in structures])])).to(dev)
    aidx_local = aidx - atom_off[pb]
    out = {"points": pts, "atom_index": aidx_local, "blocked": bur["blocked"], "depth": bur["depth"],
           "sums": nf["sums"], "gauss": nf["gauss"]}
    counts = torch.bincount(pb, minlength=nb).tolist()
    if to_numpy:
        out = {k: v.cpu().numpy() for k, v in out.items()}
        split = np.cumsum(counts)[:-1]
        parts = {k: np.split(v, split) for k, v in out.items()}
    else:
        parts = {k: torch.split(v, counts) for k, v in out.items()}
    return [{k: parts[k][i] for k in parts} for i in range(nb)]
