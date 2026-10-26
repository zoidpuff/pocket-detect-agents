"""Deterministic quasi-uniform point sets on the unit sphere."""
from __future__ import annotations

import numpy as np


def fibonacci_sphere(n: int, dtype=np.float32) -> np.ndarray:
    """``n`` unit vectors on a Fibonacci (golden-angle) spiral, shape (n, 3).

    The z coordinates are ``1 - (2i + 1) / n`` and therefore symmetric about 0:
    exactly half of the directions point into ``z < 0`` for even ``n``.
    """
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - (2.0 * i + 1.0) / n
    rho = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1).astype(dtype)
