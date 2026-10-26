"""AF-mode input variants for ablations and controls (DESIGN §2.6, §9).

One function, :func:`af_input`, turns a loaded AF model (B-factor = pLDDT) into the
``(Protein, weights, stats)`` triple that :func:`pocketdetect.burrow.pipeline.predict_protein`
consumes.  Mechanisms are switched by :class:`AFConfig`:

* ``prune``      ``none`` | ``af`` (spaghetti pruning, :func:`af.prune_mask`) |
                 ``rand_seg`` (control: same number of residues removed as contiguous runs of the
                 same lengths, placed at random positions of the same chain) |
                 ``rand_res`` (control: random single residues removed until >= the same number of
                 heavy atoms is removed) | ``crop70`` (oracle crop: keep residues with pLDDT >= 70)
* ``soft``       soft occlusion w = clip((pLDDT - soft_lo) / (soft_hi - soft_lo), 0, 1), default 40/80
                 (atoms with w < 0.5 do not exclude candidate points; see features.point_features);
                 soft_hi <= 90 keeps the no-op guarantee
* ``lam``        side-chain tolerance: atoms beyond Cβ get w × λ ...
* ``lam_scope``  ... for all residues (``all``) or only residues with pLDDT < ``lam_below`` (``low``;
                 keeps the no-op guarantee for confident models)
* ``const_plddt`` overwrite pLDDT with a constant (no-op check on PDB input)

Random controls are seeded per structure id (sha256) and :data:`pocketdetect.seeds.SEEDS`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..seeds import SEEDS
from ..structure import Structure
from .af import LOW, prune_mask, residue_plddt
from .prep import BACKBONE_CB, Protein, protein_from_structure

__all__ = ["AFConfig", "CONFIGS", "af_input", "soft_weights", "keep_mask"]


@dataclass(frozen=True)
class AFConfig:
    prune: str = "none"
    soft: bool = False
    lam: float = 1.0
    lam_scope: str = "all"
    lam_below: float = 90.0
    const_plddt: float | None = None
    soft_lo: float = 40.0
    soft_hi: float = 80.0


CONFIGS = {
    "naive": AFConfig(),
    "prune": AFConfig(prune="af"),
    "soft": AFConfig(soft=True),
    "prune_soft": AFConfig(prune="af", soft=True),
    "prune_soft_lam0.5": AFConfig(prune="af", soft=True, lam=0.5),
    "prune_soft_lam0.25": AFConfig(prune="af", soft=True, lam=0.25),
    "prune_soft_lam0.5low": AFConfig(prune="af", soft=True, lam=0.5, lam_scope="low"),
    "prune_soft_lam0.25low": AFConfig(prune="af", soft=True, lam=0.25, lam_scope="low"),
    "prune_soft30_70": AFConfig(prune="af", soft=True, soft_lo=30.0, soft_hi=70.0),   # gentler (more opaque)
    "prune_soft50_90": AFConfig(prune="af", soft=True, soft_lo=50.0, soft_hi=90.0),   # stronger (more transparent)
    "rand_seg": AFConfig(prune="rand_seg"),
    "rand_res": AFConfig(prune="rand_res"),
    "crop70": AFConfig(prune="crop70"),
}


def _rng(structure_id: str, salt: int = 0) -> np.random.Generator:
    h = int(hashlib.sha256(structure_id.encode()).hexdigest()[:8], 16)
    return np.random.default_rng([SEEDS[0], h, salt])


def _runs(mask: np.ndarray):
    """(start, end) of runs of True in a 1-D bool array."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def keep_mask(s: Structure, cfg: AFConfig) -> np.ndarray:
    """Per-residue keep mask of a protein-heavy-atom structure."""
    n = s.n_residues
    if cfg.prune == "none":
        return np.ones(n, bool)
    if cfg.prune == "crop70":
        return residue_plddt(s) >= 70.0
    af_keep = prune_mask(s)
    if cfg.prune == "af":
        return af_keep
    rng = _rng(str(s.id), 1 if cfg.prune == "rand_seg" else 2)
    keep = np.ones(n, bool)
    if (~af_keep).sum() == 0:
        return keep
    if cfg.prune == "rand_seg":
        for ch in dict.fromkeys(s.r_chain.tolist()):
            idx = np.flatnonzero(s.r_chain == ch)
            lens = sorted((b - a for a, b in _runs(~af_keep[idx])), reverse=True)
            free = np.ones(len(idx), bool)
            for L in lens:
                if L >= len(idx):
                    free[:] = False
                    continue
                starts = [a for a in range(len(idx) - L + 1) if free[a:a + L].all()]
                if not starts:  # no free slot of that length: take the longest free stretches
                    starts = list(range(len(idx) - L + 1))
                a = int(rng.choice(starts))
                free[a:a + L] = False
            keep[idx[~free]] = False
        return keep
    if cfg.prune == "rand_res":
        cnt = np.bincount(s.res_index, minlength=n)
        target = int(cnt[~af_keep].sum())
        order = rng.permutation(n)
        cs = np.cumsum(cnt[order])
        k = int(np.searchsorted(cs, target)) + 1
        keep[order[:k]] = False
        return keep
    raise ValueError(cfg.prune)


def soft_weights(P: Protein, cfg: AFConfig) -> np.ndarray | None:
    w = np.ones(len(P.coords), np.float32)
    if cfg.soft:
        w = np.clip((P.plddt - cfg.soft_lo) / (cfg.soft_hi - cfg.soft_lo), 0.0, 1.0).astype(np.float32)
    if cfg.lam != 1.0:
        sc = ~P.bb_mask
        if cfg.lam_scope == "low":
            sc &= P.plddt < cfg.lam_below
        w = np.where(sc, w * np.float32(cfg.lam), w).astype(np.float32)
    return None if (w >= 1).all() else w


def af_input(s: Structure, cfg: AFConfig) -> tuple[Protein, np.ndarray | None, dict]:
    """(pruned Protein, per-atom opacity weights or None, stats).  ``s`` = loaded AF model."""
    s = s.select(s.protein_heavy_mask())
    if cfg.const_plddt is not None:
        from dataclasses import replace
        s = replace(s, b_factor=np.full(len(s.b_factor), cfg.const_plddt, np.float32))
    keep = keep_mask(s, cfg)
    s2 = s.select(keep[s.res_index])
    P = protein_from_structure(s2)
    w = soft_weights(P, cfg)
    cnt = np.bincount(s.res_index, minlength=s.n_residues)
    stats = dict(n_res=int(s.n_residues), n_atoms=int(len(s.coords)), n_res_removed=int((~keep).sum()),
                 n_atoms_removed=int(cnt[~keep].sum()),
                 n_soft_atoms=int((w < 1).sum()) if w is not None else 0,
                 n_res_low=int((residue_plddt(s) < LOW).sum()))
    return P, w, stats
