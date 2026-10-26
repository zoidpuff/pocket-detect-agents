"""Pocket prediction evaluation: success rates (DCA / DCC, top-k / top-n / top-(n+2)), residue
overlap, residue-level classification metrics, pocket-level precision, AP, bootstrap CIs and paired
method comparison. Exact definitions: docs/EVALUATION.md.

Prediction format (one CSV per structure, ``<pred_dir>/<structure_id>.csv``)::

    rank,score,center_x,center_y,center_z,residues
    1,0.93,10.1,-3.2,22.0,A:45 A:46 A:52A B:12

``rank`` (1 = best) is authoritative for ordering; ranks must be unique integers. ``residues``
uses ``chain:resnum[icode]`` (author numbering), space separated, may be empty.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

from .ligands import GroundTruth
from .seeds import BOOTSTRAP_SEED

__all__ = [
    "Predictions",
    "Criterion",
    "parse_criterion",
    "DEFAULT_CRITERIA",
    "TOPK",
    "read_predictions",
    "write_predictions",
    "empty_predictions",
    "evaluate_structure",
    "evaluate",
    "EvalResult",
    "summarize",
    "compare_methods",
    "pooled_average_precision",
]

PRED_COLUMNS = ["rank", "score", "center_x", "center_y", "center_z", "residues"]
TOPK: tuple[int, ...] = (1, 3, 5, 10)
RELATIVE_K: tuple[int, ...] = (0, 2)  # top-(n+0), top-(n+2)


# --------------------------------------------------------------------------- predictions
@dataclass
class Predictions:
    structure_id: str
    centers: np.ndarray  # (m,3) float64, sorted by rank
    scores: np.ndarray  # (m,)
    ranks: np.ndarray  # (m,) original rank values (sorted ascending)
    residues: list  # list[frozenset[str]]
    missing: bool = False  # prediction file absent

    @property
    def n(self) -> int:
        return int(self.centers.shape[0])

    @property
    def has_residues(self) -> bool:
        return any(len(r) > 0 for r in self.residues)


def empty_predictions(structure_id: str, missing: bool = False) -> Predictions:
    return Predictions(structure_id, np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64), [], missing)


def _parse_residues(val) -> frozenset:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return frozenset()
    return frozenset(str(val).split())


def predictions_from_frame(df: pd.DataFrame, structure_id: str) -> Predictions:
    missing_cols = [c for c in PRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{structure_id}: prediction table lacks columns {missing_cols}")
    if len(df) == 0:
        return empty_predictions(structure_id)
    ranks = pd.to_numeric(df["rank"], errors="raise").to_numpy()
    if not np.all(np.isfinite(ranks)) or not np.all(ranks == np.round(ranks)):
        raise ValueError(f"{structure_id}: ranks must be integers")
    ranks = ranks.astype(np.int64)
    if len(np.unique(ranks)) != len(ranks):
        raise ValueError(f"{structure_id}: duplicate ranks {sorted(ranks.tolist())}")
    order = np.argsort(ranks, kind="stable")
    centers = df[["center_x", "center_y", "center_z"]].to_numpy(dtype=np.float64)[order]
    if not np.all(np.isfinite(centers)):
        raise ValueError(f"{structure_id}: non-finite pocket centers")
    scores = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=np.float64)[order]
    res = [_parse_residues(v) for v in df["residues"].to_numpy(dtype=object)[order]]
    return Predictions(structure_id, centers, scores, ranks[order], res)


def read_predictions(path: str | Path, structure_id: str | None = None) -> Predictions:
    path = Path(path)
    sid = structure_id or path.name.removesuffix(".csv")
    df = pd.read_csv(path, dtype={"residues": str}, keep_default_na=False, na_values={"score": ["", "nan", "NaN"]})
    return predictions_from_frame(df, sid)


def write_predictions(path: str | Path, centers, scores=None, residues=None) -> None:
    """Write the common prediction format (rank = 1..m in the given order)."""
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    m = len(centers)
    scores = np.arange(m, 0, -1, dtype=float) if scores is None else np.asarray(scores, dtype=float)
    residues = [""] * m if residues is None else [r if isinstance(r, str) else " ".join(r) for r in residues]
    pd.DataFrame(
        {
            "rank": np.arange(1, m + 1),
            "score": scores,
            "center_x": centers[:, 0],
            "center_y": centers[:, 1],
            "center_z": centers[:, 2],
            "residues": residues,
        }
    ).to_csv(path, index=False, float_format="%.4f")


# --------------------------------------------------------------------------- criteria
@dataclass(frozen=True)
class Criterion:
    """``dca``: min distance pocket centre -> any ligand atom <= t (P2Rank DCA).
    ``dcc``: distance pocket centre -> ligand centre <= t (any pocket within the top-k counts).
    ``ldcc``: LIGYSIS-style DCC: only the pocket *closest* (by DCC) to the site counts; success at k
    iff that pocket is within t and its rank <= k (Utgés et al. 2024 benchmark_lib)."""

    name: str
    kind: str
    threshold: float


_CRIT_RE = re.compile(r"^(DCA|DCC|LDCC)(\d+(?:\.\d+)?)$", re.I)


def parse_criterion(name: str | Criterion) -> Criterion:
    if isinstance(name, Criterion):
        return name
    m = _CRIT_RE.match(name.strip())
    if not m:
        raise ValueError(f"bad criterion {name!r}; use e.g. DCA4, DCC4, LDCC12")
    return Criterion(name.strip().upper(), m.group(1).lower(), float(m.group(2)))


DEFAULT_CRITERIA: tuple[str, ...] = ("DCA4", "DCC4")


def _k_labels(n_lig: int, topk=TOPK, rel=RELATIVE_K) -> list[tuple[str, int]]:
    out = [(f"top{k}", k) for k in topk]
    out += [("topn" if r == 0 else f"topn{r}", n_lig + r) for r in rel]
    return out


# --------------------------------------------------------------------------- per structure
def _residue_counts(pred: set, true: set, universe: set) -> dict:
    """Confusion counts over protein residues. Predicted IDs that are not protein residues of the
    structure (e.g. wrong chain naming) count as false positives but cannot be true negatives."""
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    tn = len(universe) - len((pred & universe) | true)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _cls_metrics(tp, fp, fn, tn) -> dict:
    tp, fp, fn, tn = (float(x) for x in (tp, fp, fn, tn))
    prec = tp / (tp + fp) if tp + fp > 0 else float("nan")
    rec = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "mcc": mcc}


def evaluate_structure(
    gt: GroundTruth,
    pred: Predictions,
    criteria: Sequence[str | Criterion] = DEFAULT_CRITERIA,
    primary: str | None = None,
    topk: Sequence[int] = TOPK,
    rel_k: Sequence[int] = RELATIVE_K,
) -> tuple[list[dict], dict, pd.DataFrame]:
    """Evaluate one structure. Returns (ligand rows, protein row, pocket table)."""
    crits = [parse_criterion(c) for c in criteria]
    prim = parse_criterion(primary) if primary else crits[0]
    if prim.name not in [c.name for c in crits]:
        crits.append(prim)
    ligs = gt.ligands
    n, m = len(ligs), pred.n
    P = pred.centers
    dca = np.full((n, m), np.inf)
    dcc = np.full((n, m), np.inf)
    if n and m:
        for j, lig in enumerate(ligs):
            dca[j] = cdist(P, lig.coords).min(axis=1)
            dcc[j] = np.linalg.norm(P - lig.center[None, :], axis=1)
    dist_of = {"dca": dca, "dcc": dcc, "ldcc": dcc}
    klabels = _k_labels(n, topk, rel_k)
    has_res = pred.has_residues
    n2 = n + 2

    def first_rank(c: Criterion) -> np.ndarray:
        """1-based rank of the pocket identifying each ligand, -1 if none."""
        D = dist_of[c.kind]
        if m == 0:
            return np.full(n, -1)
        if c.kind == "ldcc":
            i = np.argmin(D, axis=1)  # ties -> best rank
            ok = D[np.arange(n), i] <= c.threshold
            return np.where(ok, i + 1, -1)
        hit = D <= c.threshold
        i = np.argmax(hit, axis=1)
        return np.where(hit.any(axis=1), i + 1, -1)

    ranks = {c.name: first_rank(c) for c in crits}

    lig_rows = []
    for j, lig in enumerate(ligs):
        row = {
            "structure_id": gt.structure_id,
            "ligand_id": lig.ligand_id,
            "ligand_name": lig.name,
            "ligand_chains": lig.chains,
            "ligand_n_atoms": lig.n_atoms,
            "n_ligands": n,
            "n_pred": m,
            "pred_missing": pred.missing,
            "dca_top1": dca[j, 0] if m else np.nan,
            "dcc_top1": dcc[j, 0] if m else np.nan,
            "dca_min": dca[j].min() if m else np.nan,
            "dcc_min": dcc[j].min() if m else np.nan,
        }
        for c in crits:
            r = int(ranks[c.name][j])
            row[f"rank_{c.name}"] = r
            for lab, k in klabels:
                row[f"success_{c.name}_{lab}"] = bool(0 < r <= k)
        # residue overlap with the pocket matched by the primary criterion within top-(n+2)
        T = set(lig.binding_residues)
        if has_res and T:
            r = int(ranks[prim.name][j])
            if 0 < r <= n2:
                Pm = pred.residues[r - 1]
                inter = len(Pm & T)
                row["res_jaccard_matched"] = inter / len(Pm | T)
                row["res_irel_matched"] = inter / len(T)
                row["res_precision_matched"] = inter / len(Pm) if Pm else np.nan
            else:
                row["res_jaccard_matched"] = 0.0
                row["res_irel_matched"] = 0.0
                row["res_precision_matched"] = np.nan
            row["res_matched"] = bool(0 < r <= n2)
            js = [len(R & T) / len(R | T) for R in pred.residues[: min(n2, m)]]
            row["res_jaccard_best_topn2"] = max(js) if js else 0.0
        else:
            for k_ in ("res_jaccard_matched", "res_irel_matched", "res_precision_matched", "res_jaccard_best_topn2"):
                row[k_] = np.nan
            row["res_matched"] = np.nan
        row["n_binding_residues"] = len(T)
        lig_rows.append(row)

    # ---- pocket level (primary criterion; pocket "positive" if it identifies any relevant ligand)
    Dp = dist_of[prim.kind]
    pocket_hit = (Dp <= prim.threshold).any(axis=0) if n else np.zeros(m, dtype=bool)
    # greedy one-to-one matching in rank order (duplicates of an already matched ligand = FP)
    tp = np.zeros(m, dtype=bool)
    matched_lig = np.full(m, -1)
    taken = np.zeros(n, dtype=bool)
    for i in range(m):
        cand = np.flatnonzero((Dp[:, i] <= prim.threshold) & ~taken)
        if cand.size:
            jbest = cand[np.argmin(Dp[cand, i])]
            taken[jbest] = True
            tp[i] = True
            matched_lig[i] = jbest
    prot = {
        "structure_id": gt.structure_id,
        "n_ligands": n,
        "n_pred": m,
        "pred_missing": pred.missing,
        "has_residues": has_res,
        "n_other_ligands": len(gt.other_ligands),
        "n_protein_residues": len(gt.protein_residues),
        "score_monotonic": bool(m < 2 or np.all(np.diff(pred.scores) <= 1e-12) or np.all(~np.isfinite(pred.scores))),
    }
    for lab, k in klabels:
        kk = min(k, m)
        prot[f"pocket_precision_{lab}"] = float(pocket_hit[:kk].mean()) if (kk > 0 and n > 0) else np.nan
    if n > 0:
        prec_at = np.cumsum(tp) / np.arange(1, m + 1) if m else np.zeros(0)
        prot["ap"] = float((prec_at * tp).sum() / n)
    else:
        prot["ap"] = np.nan

    # ---- residue-level classification (union of binding residues of relevant ligands)
    T_all = set().union(*[set(l.binding_residues) for l in ligs]) if ligs else set()
    U = set(gt.protein_residues.tolist())
    pred_union_all = set().union(*pred.residues) if m else set()
    prot["n_pred_res_unknown"] = len(pred_union_all - U)
    for scope, kk in (("topn2", min(n2, m)), ("all", m)):
        if has_res and n > 0:
            Pu = set().union(*pred.residues[:kk]) if kk else set()
            cnt = _residue_counts(Pu, T_all, U)
            for key, v in cnt.items():
                prot[f"res_{scope}_{key}"] = v
            for key, v in _cls_metrics(**cnt).items():
                prot[f"res_{scope}_{key}"] = v
        else:
            for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "mcc"):
                prot[f"res_{scope}_{key}"] = np.nan

    pockets = pd.DataFrame(
        {
            "structure_id": gt.structure_id,
            "rank": np.arange(1, m + 1),
            "score": pred.scores,
            "hit_any": pocket_hit,
            "tp_greedy": tp,
            "matched_ligand": [ligs[j].ligand_id if j >= 0 else "" for j in matched_lig],
        }
    )
    return lig_rows, prot, pockets


# --------------------------------------------------------------------------- dataset
@dataclass
class EvalResult:
    ligands: pd.DataFrame
    proteins: pd.DataFrame
    pockets: pd.DataFrame
    summary: pd.DataFrame
    info: dict = field(default_factory=dict)

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.ligands.to_csv(out / "ligands.csv", index=False)
        self.proteins.to_csv(out / "proteins.csv", index=False)
        self.pockets.to_csv(out / "pockets.csv", index=False)
        self.summary.to_csv(out / "summary.csv", index=False)


def load_prediction_dir(pred_dir: str | Path, structure_ids: Iterable[str]) -> dict[str, Predictions]:
    pred_dir = Path(pred_dir)
    out = {}
    for sid in structure_ids:
        p = pred_dir / f"{sid}.csv"
        out[sid] = read_predictions(p, sid) if p.exists() else empty_predictions(sid, missing=True)
    return out


def evaluate(
    gts: Sequence[GroundTruth],
    predictions: Mapping[str, Predictions] | str | Path,
    criteria: Sequence[str] = DEFAULT_CRITERIA,
    primary: str | None = None,
    n_boot: int = 1000,
    seed: int = BOOTSTRAP_SEED,
    topk: Sequence[int] = TOPK,
) -> EvalResult:
    """Evaluate predictions for a dataset. Structures without prediction file count as
    zero predictions (``pred_missing``). Structures with no relevant ligand are kept in the
    protein table but excluded from all summary statistics."""
    ids = [g.structure_id for g in gts]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate structure ids in ground truth")
    if isinstance(predictions, (str, Path)):
        predictions = load_prediction_dir(predictions, ids)
    lig_rows, prot_rows, pockets = [], [], []
    for g in gts:
        pred = predictions.get(g.structure_id) or empty_predictions(g.structure_id, missing=True)
        lr, pr, pk = evaluate_structure(g, pred, criteria, primary, topk)
        lig_rows += lr
        prot_rows.append(pr)
        pockets.append(pk)
    ligands = pd.DataFrame(lig_rows)
    proteins = pd.DataFrame(prot_rows)
    pockets_df = pd.concat(pockets, ignore_index=True) if pockets else pd.DataFrame()
    summary = summarize(ligands, proteins, pockets_df, n_boot=n_boot, seed=seed)
    info = {
        "n_structures": len(gts),
        "n_structures_no_ligands": int((proteins["n_ligands"] == 0).sum()) if len(proteins) else 0,
        "n_pred_missing": int(proteins["pred_missing"].sum()) if len(proteins) else 0,
        "n_ligands": int(len(ligands)),
        "criteria": list(criteria),
        "primary": primary or criteria[0],
        "n_boot": n_boot,
        "seed": seed,
        "rules": sorted({g.rules for g in gts}),
    }
    return EvalResult(ligands, proteins, pockets_df, summary, info)


# --------------------------------------------------------------------------- statistics
def _bootstrap_weights(n_units: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_units) multiplicity matrix of a nonparametric bootstrap over units."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_units, size=(n_boot, n_units))
    W = np.zeros((n_boot, n_units), dtype=np.float64)
    np.add.at(W, (np.repeat(np.arange(n_boot), n_units), idx.ravel()), 1.0)
    return W


def _ratio_with_ci(num: np.ndarray, den: np.ndarray, W: np.ndarray | None) -> tuple[float, float, float]:
    tot = den.sum()
    val = num.sum() / tot if tot > 0 else np.nan
    if W is None or tot == 0:
        return float(val), np.nan, np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (W @ num) / (W @ den)
    bs = bs[np.isfinite(bs)]
    if bs.size == 0:
        return float(val), np.nan, np.nan
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(val), float(lo), float(hi)


def pooled_average_precision(scores: np.ndarray, tp: np.ndarray, n_pos: float, weights: np.ndarray | None = None) -> float:
    """Non-interpolated AP over a pooled pocket list ranked by ``score`` (descending). Tied
    scores form one threshold. Recall denominator = ``n_pos`` (all relevant ligands, so missed
    ligands cap recall). ``weights``: per-pocket multiplicities (bootstrap)."""
    if n_pos <= 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    w = np.ones(len(scores)) if weights is None else weights
    order = np.argsort(-scores, kind="stable")
    s, t, w = scores[order], tp[order].astype(float), w[order]
    ctp = np.cumsum(t * w)
    cnt = np.cumsum(w)
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]  # last index of each tie group
    prec = np.divide(ctp[last], cnt[last], out=np.zeros(len(last)), where=cnt[last] > 0)
    rec = ctp[last] / n_pos
    drec = np.diff(np.r_[0.0, rec])
    return float((drec * prec).sum())


def summarize(
    ligands: pd.DataFrame,
    proteins: pd.DataFrame,
    pockets: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Summary table: one row per (metric, average) with value and percentile bootstrap 95% CI
    from resampling proteins (structures with >=1 relevant ligand) with replacement."""
    prots = proteins[proteins["n_ligands"] > 0].reset_index(drop=True)
    sids = prots["structure_id"].to_numpy()
    n_p = len(sids)
    pos = {s: i for i, s in enumerate(sids)}
    W = _bootstrap_weights(n_p, n_boot, seed) if (n_boot and n_p) else None
    rows = []

    def add(metric, average, num, den, n_units):
        v, lo, hi = _ratio_with_ci(num, den, W)
        rows.append({"metric": metric, "average": average, "value": v, "ci_low": lo, "ci_high": hi, "n": int(n_units)})

    lig = ligands[ligands["structure_id"].isin(pos)] if len(ligands) else ligands
    lig_metrics = [c for c in lig.columns if c.startswith("success_")] + [
        "res_jaccard_matched",
        "res_irel_matched",
        "res_jaccard_best_topn2",
        "dcc_top1",
        "dca_top1",
    ]
    if len(lig):
        pidx = lig["structure_id"].map(pos).to_numpy()
        for c in lig_metrics:
            if c not in lig.columns:
                continue
            x = lig[c].to_numpy(dtype=float)
            ok = np.isfinite(x)
            if not ok.any():
                continue
            num = np.bincount(pidx[ok], weights=x[ok], minlength=n_p)
            den = np.bincount(pidx[ok], minlength=n_p).astype(float)
            add(c, "ligand", num, den, ok.sum())
            has = den > 0
            pm = np.divide(num, den, out=np.zeros(n_p), where=has)
            add(c, "protein", pm, has.astype(float), has.sum())
        # DVO-style overlap restricted to successfully located ligands
        if "res_matched" in lig.columns:
            mm = lig["res_matched"].to_numpy(dtype=float) == 1.0
            x = lig["res_jaccard_matched"].to_numpy(dtype=float)
            ok = mm & np.isfinite(x)
            if ok.any():
                num = np.bincount(pidx[ok], weights=x[ok], minlength=n_p)
                den = np.bincount(pidx[ok], minlength=n_p).astype(float)
                add("res_jaccard_matched_given_success", "ligand", num, den, ok.sum())

    prot_metrics = [c for c in prots.columns if c.startswith("pocket_precision_")] + ["ap"]
    for scope in ("topn2", "all"):
        prot_metrics += [f"res_{scope}_{k}" for k in ("precision", "recall", "f1", "mcc")]
    for c in prot_metrics:
        if c not in prots.columns:
            continue
        x = prots[c].to_numpy(dtype=float)
        ok = np.isfinite(x)
        if not ok.any():
            continue
        add(c, "protein", np.where(ok, x, 0.0), ok.astype(float), ok.sum())

    # micro-averaged residue metrics (pooled confusion counts)
    for scope in ("topn2", "all"):
        cols = [f"res_{scope}_{k}" for k in ("tp", "fp", "fn", "tn")]
        if not all(c in prots.columns for c in cols):
            continue
        C = prots[cols].to_numpy(dtype=float)
        ok = np.all(np.isfinite(C), axis=1)
        if not ok.any():
            continue
        C = np.where(ok[:, None], C, 0.0)

        def micro(Cs):
            return _cls_metrics(*Cs)

        point = micro(C.sum(axis=0))
        bs = {k: [] for k in point}
        if W is not None:
            S = W @ C
            for b in range(S.shape[0]):
                for k, v in micro(S[b]).items():
                    bs[k].append(v)
        for k, v in point.items():
            arr = np.asarray(bs[k], dtype=float)
            arr = arr[np.isfinite(arr)]
            lo, hi = (np.percentile(arr, [2.5, 97.5]) if arr.size else (np.nan, np.nan))
            rows.append({"metric": f"res_{scope}_{k}", "average": "micro", "value": v, "ci_low": float(lo), "ci_high": float(hi), "n": int(ok.sum())})

    # pooled AP across all pockets of all structures (ranked by score; needs comparable scores)
    if len(pockets) and n_p:
        pk = pockets[pockets["structure_id"].isin(pos)]
        sc = pk["score"].to_numpy(dtype=float)
        if np.all(np.isfinite(sc)):
            tpv = pk["tp_greedy"].to_numpy(dtype=bool)
            ppos = pk["structure_id"].map(pos).to_numpy()
            nl = prots["n_ligands"].to_numpy(dtype=float)
            val = pooled_average_precision(sc, tpv, nl.sum())
            lo = hi = np.nan
            if W is not None:
                bs = np.array([pooled_average_precision(sc, tpv, W[b] @ nl, W[b, ppos]) for b in range(W.shape[0])])
                bs = bs[np.isfinite(bs)]
                if bs.size:
                    lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"metric": "ap_pooled", "average": "pooled", "value": val, "ci_low": float(lo), "ci_high": float(hi), "n": int(len(pk))})
        else:
            warnings.warn("non-finite scores: pooled AP skipped")

    rows.append({"metric": "n_proteins", "average": "count", "value": float(n_p), "ci_low": np.nan, "ci_high": np.nan, "n": n_p})
    rows.append({"metric": "n_ligands", "average": "count", "value": float(len(lig)), "ci_low": np.nan, "ci_high": np.nan, "n": len(lig)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- paired comparison
def compare_methods(
    lig_a: pd.DataFrame,
    lig_b: pd.DataFrame,
    metric: str = "success_DCA4_topn2",
    n_boot: int = 10000,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired comparison of two methods on the same ligands (same ground truth required).

    Returns rates (ligand- and protein-averaged), difference A-B with paired cluster-bootstrap
    95% CI and two-sided bootstrap p-value (resampling proteins, keeping pairs), and an exact
    McNemar test on per-ligand discordant pairs (assumes ligand independence; prefer the
    bootstrap when proteins have several ligands)."""
    key = ["structure_id", "ligand_id"]
    a = lig_a[key + [metric]].rename(columns={metric: "a"})
    b = lig_b[key + [metric]].rename(columns={metric: "b"})
    m = a.merge(b, on=key, how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        bad = m[m["_merge"] != "both"]
        raise ValueError(f"ligand sets differ between methods ({len(bad)} unmatched rows); evaluate both with the same ground truth")
    m = m.dropna(subset=["a", "b"])
    xa = m["a"].to_numpy(dtype=float)
    xb = m["b"].to_numpy(dtype=float)
    sids, pidx = np.unique(m["structure_id"].to_numpy(), return_inverse=True)
    n_p = len(sids)
    na = np.bincount(pidx, weights=xa, minlength=n_p)
    nb = np.bincount(pidx, weights=xb, minlength=n_p)
    den = np.bincount(pidx, minlength=n_p).astype(float)
    W = _bootstrap_weights(n_p, n_boot, seed)
    d_lig = (W @ (na - nb)) / (W @ den)
    d_prot = (W @ ((na - nb) / den)) / W.sum(axis=1)
    diff_lig = (na - nb).sum() / den.sum()
    diff_prot = float(np.mean((na - nb) / den))

    def pval(d):  # two-sided percentile-bootstrap p-value for H0: diff = 0
        return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))

    out = {
        "metric": metric,
        "n_ligands": int(len(m)),
        "n_proteins": int(n_p),
        "a_ligand": float(xa.mean()),
        "b_ligand": float(xb.mean()),
        "diff_ligand": float(diff_lig),
        "diff_ligand_ci": tuple(np.percentile(d_lig, [2.5, 97.5]).tolist()),
        "p_bootstrap_ligand": pval(d_lig),
        "a_protein": float(np.mean(na / den)),
        "b_protein": float(np.mean(nb / den)),
        "diff_protein": diff_prot,
        "diff_protein_ci": tuple(np.percentile(d_prot, [2.5, 97.5]).tolist()),
        "p_bootstrap_protein": pval(d_prot),
    }
    binary = np.all(np.isin(xa, (0.0, 1.0))) and np.all(np.isin(xb, (0.0, 1.0)))
    if binary:
        b10 = int(((xa == 1) & (xb == 0)).sum())
        b01 = int(((xa == 0) & (xb == 1)).sum())
        out["mcnemar_a_only"] = b10
        out["mcnemar_b_only"] = b01
        out["p_mcnemar_exact"] = float(stats.binomtest(b10, b10 + b01, 0.5).pvalue) if b10 + b01 > 0 else 1.0
    return out
