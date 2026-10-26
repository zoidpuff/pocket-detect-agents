"""Canonical (PCA) frame and spatial bricking helpers.

``canonical_frame`` gives every structure an intrinsic, deterministic
orientation, so that grid- and ray-based features become invariant to rigid
rotations of the input (the voxel grid and the Fibonacci ray set are fixed in
the canonical frame instead of the arbitrary PDB frame).

``brick_boxes`` tiles a huge structure into overlapping boxes (bricks) so
that per-brick voxel grids stay below a memory budget.
"""
from __future__ import annotations

import numpy as np


def canonical_frame(coords):
    """Return ``(R, centre)`` with ``x_canon = (x - centre) @ R.T`` (proper rotation, float64).

    Rows of R are the principal axes (descending variance).  Signs of the
    first two axes are fixed by the third moment of the projections (ties ->
    +), the third axis is their cross product.  For exactly rotated inputs the
    canonical coordinates agree to floating-point precision; for near-
    degenerate inertia (globular, symmetric oligomers) the frame is still
    deterministic but may jump under small coordinate changes.
    """
    x = np.asarray(coords, np.float64)
    if len(x) < 3:
        return np.eye(3), (x.mean(0) if len(x) else np.zeros(3))
    c = x.mean(0)
    y = x - c
    w, v = np.linalg.eigh(y.T @ y)
    axes = v[:, ::-1].T                              # rows, descending eigenvalue
    for k in range(2):
        m3 = np.sum((y @ axes[k]) ** 3)
        if m3 < 0:
            axes[k] = -axes[k]
    axes[2] = np.cross(axes[0], axes[1])
    return axes, c


def to_frame(x, R, centre):
    return ((np.asarray(x, np.float64) - centre) @ R.T).astype(np.float32)


def from_frame(x, R, centre):
    return (np.asarray(x, np.float64) @ R + centre).astype(np.float32)


def brick_boxes(coords, spacing: float, pad: float, halo: float, max_voxels: int):
    """Tile the bounding box into a grid of boxes whose padded grids fit ``max_voxels``.

    Returns a list of ``(lo (3,), hi (3,))`` core boxes (outermost faces are
    +-inf); a single unbounded box if everything fits.  A brick must contain
    all atoms within ``halo`` of its core box; points are assigned to the core
    box containing them (or their owning atom), so every point is computed
    exactly once with the same neighbourhood as without bricking.
    """
    x = np.asarray(coords, np.float64)
    lo, hi = x.min(0), x.max(0)
    ext = hi - lo
    if np.prod((ext + 2 * pad) / spacing + 3) <= max_voxels:
        return [(np.full(3, -np.inf), np.full(3, np.inf))]

    # split the k longest axes (smallest feasible k) into cores of edge b
    order = np.argsort(ext)[::-1]
    n = None
    for k in (1, 2, 3):
        split, unsplit = order[:k], order[k:]
        fixed = np.prod((ext[unsplit] + 2 * pad) / spacing + 3)
        per_axis = (max_voxels / fixed) ** (1.0 / k)                 # voxels per split axis
        b = (per_axis - 3) * spacing - 2 * (pad + halo)
        if b >= 4 * spacing:
            n = np.ones(3, int)
            n[split] = np.maximum(1, np.ceil(ext[split] / b).astype(int))
            break
    if n is None:
        raise MemoryError("max_voxels too small for the halo of this configuration")
    edges = [np.linspace(lo[i], hi[i], n[i] + 1) for i in range(3)]
    for e in edges:
        e[0], e[-1] = -np.inf, np.inf
    boxes = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                boxes.append((np.array([edges[0][i], edges[1][j], edges[2][k]]),
                              np.array([edges[0][i + 1], edges[1][j + 1], edges[2][k + 1]])))
    return boxes


def in_box(x, lo, hi, margin: float = 0.0):
    x = np.asarray(x)
    return np.all((x >= lo - margin) & (x < hi + margin), axis=1)
