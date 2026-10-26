"""AlphaFold-model evaluation utilities (PREREG O5).

* :func:`pocket_plddt` — mean pLDDT of a predicted pocket on the (full, unpruned) model: mean over
  the pocket's residues of the residue pLDDT (CA B-factor, else mean atom B-factor); pockets
  without residues use the residues with any heavy atom ≤ 6 Å of the centre (PREREG residue rule
  for centre-only methods). NaN if no residue qualifies.
* :func:`drop_low_plddt` — the "naive + drop pockets with mean pLDDT < 70" baseline transform:
  removes pockets below the threshold and re-ranks the rest in their original order.
* :func:`hallucination_counts` — hallucinated pocket = mean pLDDT < 50 and matching no reference
  site of the model (DCC ≤ 12 Å, rounded to 0.01 Å, to any site, scored or not). Counted over all
  pockets and over the top-(N+2); always evaluated against the pLDDT of the unpruned model.
* :func:`summarize_hallucination` — pooled rates with cluster-bootstrap CIs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .evaluate import Predictions, _bootstrap_weights, write_predictions
from .seeds import BOOTSTRAP_SEED
from .sites import Site
from .structure import Structure, load_structure

__all__ = ["residue_plddt", "pocket_plddt", "drop_low_plddt", "hallucination_counts", "summarize_hallucination", "transform_pred_dir"]

CENTRE_RES_DIST = 6.0


def residue_plddt(s: Structure) -> dict[str, float]:
    ids = s.residue_ids
    out = {}
    for r in range(s.n_residues):
        a, b = s.r_start[r], s.r_end[r]
        ca = np.flatnonzero(s.atom_name[a:b] == "CA")
        out[ids[r]] = float(s.b_factor[a + ca[0]]) if ca.size else float(s.b_factor[a:b].mean())
    return out


def pocket_plddt(pred: Predictions, s: Structure, plddt: dict | None = None, radius: float = CENTRE_RES_DIST) -> np.ndarray:
    plddt = plddt or residue_plddt(s)
    out = np.full(pred.n, np.nan)
    tree = None
    heavy = np.flatnonzero(s.is_protein & ~s.is_hydrogen)
    ids = s.residue_ids
    for j in range(pred.n):
        res = [r for r in pred.residues[j] if r in plddt]
        if not res:
            if tree is None:
                tree = cKDTree(s.coords[heavy])
            hit = tree.query_ball_point(pred.centers[j], r=radius)
            res = list({ids[s.res_index[heavy[i]]] for i in hit})
        vals = [plddt[r] for r in res if r in plddt]
        if vals:
            out[j] = float(np.mean(vals))
    return out


def drop_low_plddt(pred: Predictions, pocket_pl: np.ndarray, threshold: float = 70.0) -> Predictions:
    """Keep pockets with mean pLDDT ≥ threshold (NaN pLDDT kept), original order, ranks 1..k."""
    keep = ~(pocket_pl < threshold)
    idx = np.flatnonzero(keep)
    return replace(pred, centers=pred.centers[idx], scores=pred.scores[idx], ranks=np.arange(1, len(idx) + 1),
                   residues=[pred.residues[i] for i in idx])


def hallucination_counts(pred: Predictions, pocket_pl: np.ndarray, sites: list[Site], threshold: float = 50.0,
                         dcc: float = 12.0, round_decimals: int | None = 2) -> dict:
    m = pred.n
    N = int(sites[0].n_sites) if sites else 0
    if m and sites:
        D = cdist(pred.centers, np.stack([s.center for s in sites]))
        if round_decimals is not None:
            D = np.round(D, round_decimals)
        matched = (D <= dcc).any(axis=1)
    else:
        matched = np.zeros(m, dtype=bool)
    low = pocket_pl < threshold
    hall = low & ~matched
    k = min(N + 2, m)
    return {"n_pockets": m, "n_low": int(low.sum()), "n_hallucinated": int(hall.sum()),
            "n_pockets_topn2": k, "n_low_topn2": int(low[:k].sum()), "n_hallucinated_topn2": int(hall[:k].sum())}


def summarize_hallucination(df: pd.DataFrame, groups: Mapping[str, str] | None = None, n_boot: int = 1000,
                            seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """df: one row per structure with :func:`hallucination_counts` columns + ``structure_id``."""
    units = df.structure_id.map(lambda s: (groups or {}).get(s, s)).to_numpy()
    un, ui = np.unique(units, return_inverse=True)
    W = _bootstrap_weights(len(un), n_boot, seed) if n_boot and len(un) else None
    rows = []
    for name, num, den in (("hallucination_rate", "n_hallucinated", "n_pockets"),
                           ("hallucination_rate_topn2", "n_hallucinated_topn2", "n_pockets_topn2"),
                           ("low_plddt_pocket_rate", "n_low", "n_pockets")):
        a = np.bincount(ui, weights=df[num].to_numpy(float), minlength=len(un))
        b = np.bincount(ui, weights=df[den].to_numpy(float), minlength=len(un))
        v = a.sum() / b.sum() if b.sum() else np.nan
        lo = hi = np.nan
        if W is not None and b.sum():
            with np.errstate(invalid="ignore", divide="ignore"):
                bs = (W @ a) / (W @ b)
            bs = bs[np.isfinite(bs)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"metric": name, "value": v, "ci_low": lo, "ci_high": hi, "n": int(b.sum())})
    rows.append({"metric": "n_hallucinated", "value": float(df.n_hallucinated.sum()), "ci_low": np.nan, "ci_high": np.nan, "n": len(df)})
    return pd.DataFrame(rows)


def transform_pred_dir(pred_dir: str | Path, out_dir: str | Path, structures: Mapping[str, str], threshold: float = 70.0) -> pd.DataFrame:
    """Apply :func:`drop_low_plddt` to every prediction file of ``pred_dir`` (structure_id ->
    model path in ``structures``); write the filtered predictions to ``out_dir``. Returns a log."""
    from .evaluate import read_predictions

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for sid, path in structures.items():
        p = Path(pred_dir) / f"{sid}.csv"
        if not p.exists():
            log.append({"structure_id": sid, "n_in": 0, "n_out": 0, "missing": True})
            continue
        pred = read_predictions(p, sid)
        s = load_structure(path)
        pl = pocket_plddt(pred, s)
        f = drop_low_plddt(pred, pl, threshold)
        write_predictions(out_dir / f"{sid}.csv", f.centers, f.scores, [" ".join(sorted(r)) for r in f.residues])
        log.append({"structure_id": sid, "n_in": pred.n, "n_out": f.n, "missing": False})
    return pd.DataFrame(log)
