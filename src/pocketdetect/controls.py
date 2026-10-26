"""Evaluation controls (DESIGN §3: run on every evaluation): oracle, protein centroid, random
surface points, surface tiling. Each control writes a normal prediction directory.

* ``oracle``   — one pocket per reference site/ligand at its centre (site centre / ligand centroid),
                 ranked in table order: upper bound of the protocol.
* ``centroid`` — a single pocket at the protein heavy-atom centroid.
* ``random``   — k random points in the solvent shell (nearest protein heavy atom 2.5–3.5 Å),
                 k = 10 per structure, random ranks (seed from SEEDS).
* ``tiling``   — points of the same shell thinned to a ≥ 12 Å spacing (greedy, random order) and
                 ranked randomly: a method that "covers" the surface uniformly without ranking.
Inputs: list of (structure_id, structure_path, chains or None).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .evaluate import write_predictions
from .seeds import SEEDS
from .structure import load_structure

__all__ = ["shell_points", "tiling_points", "write_control"]


def shell_points(xyz: np.ndarray, rng: np.random.Generator, n: int, shell=(2.5, 3.5), batch: int = 20000) -> np.ndarray:
    tree = cKDTree(xyz)
    lo, hi = xyz.min(0) - shell[1], xyz.max(0) + shell[1]
    out = np.zeros((0, 3))
    for _ in range(200):
        if len(out) >= n:
            break
        c = rng.uniform(lo, hi, size=(batch, 3))
        d, _ = tree.query(c)
        out = np.vstack([out, c[(d >= shell[0]) & (d <= shell[1])]])
    return out[:n]


def tiling_points(xyz: np.ndarray, rng: np.random.Generator, spacing: float = 12.0) -> np.ndarray:
    cand = shell_points(xyz, rng, max(200, int(len(xyz) / 2)))
    rng.shuffle(cand)
    kept: list = []
    for p in cand:
        if not kept or np.min(np.linalg.norm(np.asarray(kept) - p, axis=1)) >= spacing:
            kept.append(p)
    pts = np.asarray(kept).reshape(-1, 3)
    return pts[rng.permutation(len(pts))]


def write_control(kind: str, structures: Iterable[tuple[str, str, Sequence[str] | None]], out_dir: str | Path,
                  centers: dict | None = None, seed: int = SEEDS[0], k_random: int = 10) -> int:
    """Write control predictions for ``structures`` into ``out_dir``. ``centers`` (oracle only):
    structure_id -> (k,3) reference centres in rank order."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, (sid, path, chains) in enumerate(structures):
        if kind == "oracle":
            c = np.asarray((centers or {}).get(sid, np.zeros((0, 3))), float).reshape(-1, 3)
        else:
            s = load_structure(path, chains=list(chains) if chains else None, protein_only=True)
            xyz = s.coords.astype(float)
            if len(xyz) == 0:
                c = np.zeros((0, 3))
            elif kind == "centroid":
                c = xyz.mean(0, keepdims=True)
            elif kind == "random":
                c = shell_points(xyz, np.random.default_rng([seed, i]), k_random)
            elif kind == "tiling":
                c = tiling_points(xyz, np.random.default_rng([seed, i]))
            else:
                raise ValueError(kind)
        write_predictions(out / f"{sid}.csv", c)
        n += 1
    return n
