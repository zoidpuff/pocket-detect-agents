"""Shared fixtures/helpers for the geometry tests (synthetic + optional real structures)."""
from __future__ import annotations

import os

import numpy as np
import pytest

PDB_MIRROR = "/cluster/work/beltrao/shared/rsyncPDB/data/structures/divided/mmCIF"


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


needs_cuda = pytest.mark.skipif(not cuda_available(), reason="CUDA not available")


def blob_protein(n_side: float = 14.0, spacing: float = 1.9, seed: int = 0, pocket=True):
    """Jittered cubic lattice of 'atoms' inside a sphere, optionally with a carved pocket."""
    rng = np.random.default_rng(seed)
    g = np.arange(-n_side, n_side + 1e-6, spacing)
    xyz = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    xyz = xyz[np.linalg.norm(xyz, axis=1) <= n_side]
    if pocket:
        c = np.array([n_side * 0.6, 0, 0])
        xyz = xyz[np.linalg.norm(xyz - c, axis=1) > 5.0]
    xyz = xyz + rng.normal(0, 0.2, xyz.shape)
    radii = rng.choice([1.52, 1.55, 1.7, 1.8], len(xyz)).astype(np.float32)
    return xyz.astype(np.float32), radii


def load_pdb(pid: str):
    """Protein heavy atoms of the first model via gemmi (skips test if unavailable)."""
    path = f"{PDB_MIRROR}/{pid[1:3]}/{pid}.cif.gz"
    if not os.path.exists(path):
        pytest.skip(f"PDB mirror not available ({path})")
    import gemmi
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    xyz, el, nm, rn = [], [], [], []
    for ch in st[0]:
        for res in ch:
            for a in res:
                xyz.append(a.pos.tolist())
                el.append(a.element.name)
                nm.append(a.name)
                rn.append(res.name)
    return np.array(xyz, np.float32), np.array(el), np.array(nm), np.array(rn)


def rod_protein(length: float = 120.0, radius: float = 8.0, spacing: float = 1.9, seed: int = 0):
    """Elongated jittered lattice (forces spatial bricking with small voxel budgets)."""
    rng = np.random.default_rng(seed)
    gx = np.arange(-length / 2, length / 2 + 1e-6, spacing)
    gy = np.arange(-radius, radius + 1e-6, spacing)
    xyz = np.stack(np.meshgrid(gx, gy, gy, indexing="ij"), -1).reshape(-1, 3)
    xyz = xyz[np.hypot(xyz[:, 1], xyz[:, 2]) <= radius]
    # a few grooves so that buriedness varies along the rod
    groove = (np.abs(np.sin(xyz[:, 0] / 6.0)) > 0.8) & (xyz[:, 1] > radius - 4)
    xyz = xyz[~groove] + rng.normal(0, 0.2, (int((~groove).sum()), 3))
    radii = rng.choice([1.52, 1.55, 1.7, 1.8], len(xyz)).astype(np.float32)
    return xyz.astype(np.float32), radii
