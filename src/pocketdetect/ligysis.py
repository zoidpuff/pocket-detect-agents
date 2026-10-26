"""LIGYSIS (Utgés et al. 2024, LBS-comparison v2.0.0) data import.

Inputs: ``data/raw/ligysis/lbs_comparison_results`` (Zenodo 14645504; pandas pickles). Pickles are
untrusted: they are read with a restricted unpickler that only resolves pandas/numpy/builtin
container classes.

Reference sites (the 2,775 representative chains of the published benchmark) come from the LIGYSIS
rows of ``ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl`` (6,882 sites). As in notebook
``12_BENCHMARK_pocket_level`` the scored set is ``filter_redundant_pockets2(LIGYSIS, 0.75, 5.0)``
(6,721 sites), while N (top-N) counts all 6,882 sites of a chain. Site centre = centroid of the CA
atoms of the site residues on the representative chain, stored by the authors in a transformed
frame ``x_T = R (x − c)`` (``PDB_rot_matrices_ALL_CLUST.pkl``, ``PDB_orig_centroids_ALL_CLUST.pkl``);
:func:`to_original_frame` inverts it.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import Predictions

__all__ = [
    "LIGYSIS_RESULTS",
    "safe_unpickle",
    "filter_redundant",
    "load_reference_sites",
    "load_transforms",
    "to_original_frame",
    "predictions_from_comparisons",
    "predictions_from_frame",
]

LIGYSIS_RESULTS = Path(__file__).resolve().parents[2] / "data" / "raw" / "ligysis" / "lbs_comparison_results"

_ALLOWED_ROOTS = {"pandas", "numpy", "collections", "builtins", "datetime", "copyreg", "_codecs"}
_BLOCKED_BUILTINS = {"eval", "exec", "open", "__import__", "compile", "getattr", "setattr", "globals", "input", "breakpoint"}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        root = module.split(".")[0]
        if root not in _ALLOWED_ROOTS or (module == "builtins" and name in _BLOCKED_BUILTINS):
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def safe_unpickle(path: str | Path):
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


# --------------------------------------------------------------------------- redundancy filter
def filter_redundant(df: pd.DataFrame, score_col: str, jaccard: float = 0.75, distance: float = 5.0,
                     group_cols=("origin", "rep_chain"), res_col="up_aas", centre_col="centre") -> pd.DataFrame:
    """Faithful port of LBS-comparison ``filter_redundant_pockets2``: within each (origin, chain)
    group, in the given row order, pocket j is dropped if it overlaps an earlier kept pocket i
    (residue Jaccard ≥ ``jaccard`` or centre distance ≤ ``distance``) and score_i ≥ score_j; if
    score_i < score_j, i is dropped instead and the scan of i stops."""
    keep_all = np.ones(len(df), dtype=bool)
    pos = 0
    for _, g in df.groupby(list(group_cols), sort=True):
        idx = g.index.to_numpy()
        res = [set(x) for x in g[res_col]]
        cen = np.array([np.asarray(c, dtype=float) for c in g[centre_col]])
        sc = g[score_col].to_numpy(dtype=float)
        keep = np.ones(len(g), dtype=bool)
        for i in range(len(g)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(g)):
                u = len(res[i] | res[j])
                jac = len(res[i] & res[j]) / u if u else 0.0
                d = float(np.linalg.norm(cen[i] - cen[j]))
                if jac >= jaccard or d <= distance:
                    if sc[i] >= sc[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        keep_all[df.index.get_indexer(idx)] = keep
    return df[keep_all]


# --------------------------------------------------------------------------- reference sites
def load_reference_sites(results_dir: str | Path = LIGYSIS_RESULTS) -> pd.DataFrame:
    """Site table (pocketdetect.sites format) of the published LIGYSIS benchmark, TRANS frame.
    ``residues`` are UniProt numbers ``U:<n>``; ``residues_pdb`` are the authors' PDB residue ids (``aas``) of the
    representative chain; ``structure_id`` = representative chain (``1a52_A``)."""
    results_dir = Path(results_dir)
    comb = safe_unpickle(results_dir / "ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl")
    lig = comb[comb.origin == "LIGYSIS"].sort_values(["rep_chain", "ID"]).reset_index(drop=True)
    nr = filter_redundant(lig, score_col="n_aas")  # LIGYSIS has no score: larger site kept
    nr_keys = set(zip(nr.rep_chain, nr.ID))
    meta = safe_unpickle(results_dir / "LIGYSIS_sites_DEF_TRANS.pkl")[["rep_chain", "ID", "ACC", "SEG_ID", "SITE_NAME", "pdb_id", "auth_asym_id", "struct_asym_id"]]
    lig = lig.merge(meta, on=["rep_chain", "ID"], how="left")
    c = np.array([np.asarray(x, dtype=float) for x in lig.centre])
    out = pd.DataFrame({
        "structure_id": lig.rep_chain,
        "site_id": lig.ID.astype(str),
        "center_x": c[:, 0], "center_y": c[:, 1], "center_z": c[:, 2],
        "residues": [" ".join(f"U:{int(r)}" for r in sorted(x)) for x in lig.up_aas],
        "n_sites": lig.groupby("rep_chain")["ID"].transform("size").astype(int),
        "scored": [(r, i) in nr_keys for r, i in zip(lig.rep_chain, lig.ID)],
        "residues_pdb": [" ".join(str(r) for r in x) for x in lig.aas],
        "uniprot": lig.ACC, "seg_id": lig.SEG_ID, "site_name": lig.SITE_NAME,
        "pdb_id": lig.pdb_id, "auth_asym_id": lig.auth_asym_id, "struct_asym_id": lig.struct_asym_id,
        "ion_prop": lig.ion_prop, "n_aas": lig.n_aas,
    })
    return out


def load_transforms(results_dir: str | Path = LIGYSIS_RESULTS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    results_dir = Path(results_dir)
    R = safe_unpickle(results_dir / "PDB_rot_matrices_ALL_CLUST.pkl")
    c0 = safe_unpickle(results_dir / "PDB_orig_centroids_ALL_CLUST.pkl")
    return {k: (np.asarray(R[k], dtype=float), np.asarray(c0[k], dtype=float)) for k in R if k in c0}


def to_original_frame(xyz_trans: np.ndarray, R: np.ndarray, c0: np.ndarray) -> np.ndarray:
    """Inverse of the authors' ``apply_rotation``: x_T = R (x − c0)  ⇒  x = Rᵀ x_T + c0."""
    return np.asarray(xyz_trans, dtype=float) @ R + c0


# --------------------------------------------------------------------------- published predictions
def predictions_from_frame(df: pd.DataFrame, chain_col="rep_chain", centre_col="centre", rank_col="RANK",
                           score_col="SCORE", res_col="up_aas") -> dict[str, Predictions]:
    out = {}
    for chain, g in df.groupby(chain_col, sort=False):
        g = g.sort_values(rank_col, kind="stable")
        ranks = g[rank_col].to_numpy(dtype=np.int64)
        if len(np.unique(ranks)) != len(ranks):
            raise ValueError(f"{chain}: duplicate ranks in published predictions")
        cen = np.array([np.asarray(x, dtype=float) for x in g[centre_col]])
        res = [frozenset(f"U:{int(r)}" for r in x) if isinstance(x, (list, tuple, np.ndarray)) else frozenset() for x in g[res_col]]
        out[chain] = Predictions(chain, cen, g[score_col].to_numpy(dtype=float), ranks, res)
    return out


def predictions_from_comparisons(comp: pd.DataFrame, method: str) -> dict[str, Predictions]:
    """Per-chain predictions of one published method variant, as scored by the authors
    (rows of ``ALL_COMPARISONS_EXTENDED6.pkl``: centre_pred / RANK_pred / SCORE_pred / up_aas_pred)."""
    g = comp[comp.origin_pred == method].drop_duplicates(["rep_chain_pred", "ID_pred"])
    g = g.rename(columns={"rep_chain_pred": "rep_chain", "centre_pred": "centre", "RANK_pred": "RANK", "SCORE_pred": "SCORE", "up_aas_pred": "up_aas"})
    return predictions_from_frame(g)
