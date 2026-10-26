"""Site-defined benchmark evaluation (LIGYSIS LBS-comparison protocol and stricter variants).

Reference = site table (``pocketdetect.sites``), predictions = common per-structure CSVs
(``pocketdetect.evaluate.Predictions``). Definitions follow Utgés et al. 2024 /
bartongroup/LBS-comparison v2.0.0 (``ANALYSIS/benchmark_lib.py``, notebook
``12_BENCHMARK_pocket_level``); see docs/EVALUATION.md §10 for the exact mapping.

Per scored site s of a structure with N = ``n_sites`` and pockets p (rank order):

* DCC(s,p) = ‖centre_p − centre_s‖, rounded to ``round_decimals`` (2, as their code rounds the
  ``distance`` column before thresholding).
* **closest pocket** (their ``BEST`` selection): minimal rounded DCC, ties → larger IRel → better rank.
  IRel = |res_p ∩ res_s| / |res_s|.
* recall@top-N+k (LIGYSIS) = closest pocket has DCC ≤ t and rank ≤ N + k; ``max recall`` = no rank cap.
* ``anyp`` variant: any pocket within rank ≤ N + k with DCC ≤ t (P2Rank-style).
* ``1to1`` variant: maximum one-to-one matching (Hungarian / max-cardinality) between scored sites and
  the top-(N+k) pockets with DCC ≤ t; a pocket can recall at most one site.
* DCA variants (if ``ligand_atoms`` present): DCA = min distance pocket centre → site ligand heavy
  atom; ``ldca`` = pocket with minimal DCA, success if DCA ≤ ``dca`` and rank ≤ N + k.
* Precision (pooled over the dataset): all pockets of all structures sorted by score (desc; ties →
  structure_id, rank); a pocket is TP if DCC ≤ t to *any* scored site of its structure.
  ``P1K`` = TP fraction among the first 1,000; ``TP@100FP`` = TPs before the 101st FP;
  ``P1K_1to1`` = same but a pocket is TP only if it hits a site not yet claimed by a higher-ranked
  pocket of the pooled list.
* RRO = mean IRel of closest pockets with DCC ≤ t (any rank) and IRel > 0 (their "Overlap" mean).
* Residue level per structure: true = union of residues of all sites (scored or not), predicted =
  union of residues of all pockets; F1/MCC with sklearn conventions (0 when undefined); needs
  ``n_residues_structure`` for MCC.

Structures without a prediction file count with zero pockets (all their sites are misses).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .evaluate import Predictions, empty_predictions, load_prediction_dir, _bootstrap_weights
from .seeds import BOOTSTRAP_SEED
from .sites import Site

__all__ = ["SiteProtocol", "SiteEvalResult", "evaluate_sites", "summarize_sites", "compare_site_methods", "pooled_precision"]


@dataclass(frozen=True)
class SiteProtocol:
    dcc: float = 12.0
    dca: float = 4.0
    round_decimals: int | None = 2
    rank_offsets: tuple = (0, 2)
    p_at: int = 1000
    fp_at: int = 100


@dataclass
class SiteEvalResult:
    sites: pd.DataFrame  # one row per scored site
    structures: pd.DataFrame  # one row per structure
    pockets: pd.DataFrame  # one row per predicted pocket
    protocol: SiteProtocol = field(default_factory=SiteProtocol)

    def save(self, out_dir) -> None:
        from pathlib import Path

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.sites.to_csv(out / "sites_eval.csv", index=False)
        self.structures.to_csv(out / "structures_eval.csv", index=False)
        self.pockets.to_csv(out / "pockets_eval.csv", index=False)


def _r(d: np.ndarray, dec) -> np.ndarray:
    return d if dec is None else np.round(d, dec)


def _f1_mcc(tp, fp, fn, tn):
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if tn is not None else 0.0
    mcc = (tp * tn - fp * fn) / den if (tn is not None and den > 0) else (0.0 if tn is not None else np.nan)
    return f1, mcc


def evaluate_sites(
    sites: Mapping[str, list[Site]],
    predictions: Mapping[str, Predictions] | str,
    protocol: SiteProtocol = SiteProtocol(),
    groups: Mapping[str, str] | None = None,
) -> SiteEvalResult:
    sids = list(sites)
    if isinstance(predictions, str) or hasattr(predictions, "joinpath"):
        predictions = load_prediction_dir(predictions, sids)
    P = protocol
    site_rows, struct_rows, pocket_rows = [], [], []
    for sid in sids:
        all_sites = sites[sid]
        scored = [s for s in all_sites if s.scored]
        pred = predictions.get(sid) or empty_predictions(sid, missing=True)
        m = pred.n
        N = int(all_sites[0].n_sites) if all_sites else 0
        grp = (groups or {}).get(sid) or (all_sites[0].meta.get("group") if all_sites and all_sites[0].meta.get("group") not in (None, "") else sid)
        S = len(scored)
        if S and m:
            C = np.stack([s.center for s in scored])
            D = _r(cdist(C, pred.centers), P.round_decimals)  # (S, m)
        else:
            D = np.full((S, m), np.inf)
        irel = np.zeros((S, m))
        for i, s in enumerate(scored):
            if s.residues:
                for j in range(m):
                    irel[i, j] = len(s.residues & pred.residues[j]) / len(s.residues)
        ranks = np.arange(1, m + 1)
        hit = D <= P.dcc
        # one-to-one matchings per rank cap
        o2o = {}
        for k in P.rank_offsets:
            kk = min(N + k, m)
            matched = np.zeros(S, dtype=bool)
            if S and kk:
                Hk = hit[:, :kk]
                if Hk.any():
                    r_, c_ = linear_sum_assignment(-Hk.astype(float))
                    ok = Hk[r_, c_]
                    matched[r_[ok]] = True
            o2o[k] = matched
        for i, s in enumerate(scored):
            row = {"structure_id": sid, "site_id": s.site_id, "group": grp, "n_sites": N, "n_pred": m, "pred_missing": pred.missing, "n_site_residues": len(s.residues)}
            if m:
                order = np.lexsort((ranks, -irel[i], D[i]))  # min DCC, max IRel, best rank
                j = order[0]
                row.update(dcc_closest=float(D[i, j]), rank_closest=int(j + 1), irel_closest=float(irel[i, j]))
                anyhit = np.flatnonzero(hit[i])
                row["rank_first_hit"] = int(anyhit[0] + 1) if anyhit.size else -1
            else:
                row.update(dcc_closest=np.nan, rank_closest=-1, irel_closest=np.nan, rank_first_hit=-1)
            ok_close = m > 0 and row["dcc_closest"] <= P.dcc
            for k in P.rank_offsets:
                lab = "topn" if k == 0 else f"topn{k}"
                row[f"recall_{lab}"] = bool(ok_close and row["rank_closest"] <= N + k)
                row[f"recall_anyp_{lab}"] = bool(0 < row["rank_first_hit"] <= N + k)
                row[f"recall_1to1_{lab}"] = bool(o2o[k][i])
            row["recall_max"] = bool(ok_close)
            if s.ligand_atoms is not None and len(s.ligand_atoms) and m:
                dca = _r(cdist(pred.centers, s.ligand_atoms).min(axis=1), P.round_decimals)
                jb = int(np.lexsort((ranks, dca))[0])
                row.update(dca_closest=float(dca[jb]), rank_closest_dca=jb + 1)
                fh = np.flatnonzero(dca <= P.dca)
                for k in P.rank_offsets:
                    lab = "topn" if k == 0 else f"topn{k}"
                    row[f"recall_ldca_{lab}"] = bool(dca[jb] <= P.dca and jb + 1 <= N + k)
                    row[f"recall_anyp_dca_{lab}"] = bool(fh.size and fh[0] + 1 <= N + k)
            elif s.ligand_atoms is not None and len(s.ligand_atoms):
                row.update(dca_closest=np.nan, rank_closest_dca=-1)
                for k in P.rank_offsets:
                    lab = "topn" if k == 0 else f"topn{k}"
                    row[f"recall_ldca_{lab}"] = False
                    row[f"recall_anyp_dca_{lab}"] = False
            site_rows.append(row)
        # pockets: TP if within dcc of any scored site
        tp_any = hit.any(axis=0) if S else np.zeros(m, dtype=bool)
        nearest = np.where(tp_any, np.argmin(np.where(hit, D, np.inf), axis=0) if S else -1, -1)
        for j in range(m):
            pocket_rows.append(
                {
                    "structure_id": sid,
                    "group": grp,
                    "rank": j + 1,
                    "score": pred.scores[j],
                    "tp": bool(tp_any[j]),
                    # hit sites ordered by distance (one-to-one P1K claims the closest unclaimed one)
                    "hit_sites": " ".join(scored[i].site_id for i in sorted(np.flatnonzero(hit[:, j]), key=lambda i: D[i, j])) if S else "",
                }
            )
        # residue level
        true_res = set().union(*[s.residues for s in all_sites]) if all_sites else set()
        pred_res = set().union(*pred.residues) if m else set()
        nres = all_sites[0].meta.get("n_residues_structure") if all_sites else None
        nres = None if nres in (None, "") or (isinstance(nres, float) and np.isnan(nres)) else int(nres)
        tp = len(true_res & pred_res)
        fp = len(pred_res - true_res)
        fn = len(true_res - pred_res)
        tn = (nres - len(true_res | pred_res)) if nres is not None else None
        f1, mcc = _f1_mcc(tp, fp, fn, tn)
        if m and not pred.has_residues:  # method reports no residues: residue metrics undefined, not 0
            f1 = mcc = np.nan
        struct_rows.append(
            {"structure_id": sid, "group": grp, "n_sites": N, "n_scored": S, "n_pred": m, "pred_missing": pred.missing,
             "res_tp": tp, "res_fp": fp, "res_fn": fn, "res_tn": tn, "res_f1": f1 if (true_res or pred_res) else np.nan, "res_mcc": mcc,
             "has_residues": pred.has_residues}
        )
    pockets = pd.DataFrame(pocket_rows, columns=["structure_id", "group", "rank", "score", "tp", "hit_sites"])
    return SiteEvalResult(pd.DataFrame(site_rows), pd.DataFrame(struct_rows), pockets, P)


# --------------------------------------------------------------------------- pooled precision
def _pooled_order(pockets: pd.DataFrame) -> np.ndarray:
    """Pooled ranking: score desc, ties → structure_id asc, rank asc."""
    return np.lexsort((pockets["rank"].to_numpy(), pockets["structure_id"].to_numpy().astype(str), -pockets["score"].to_numpy(dtype=float)))


def _one_to_one_tp(pockets: pd.DataFrame, order: np.ndarray) -> np.ndarray:
    claimed: set = set()
    out = np.zeros(len(pockets), dtype=bool)
    sid = pockets["structure_id"].to_numpy()
    hits = pockets["hit_sites"].to_numpy()
    for i in order:
        h = hits[i]
        if not h:
            continue
        for site in h.split():  # hit_sites ordered by distance: claim the closest unclaimed site
            key = (sid[i], site)
            if key not in claimed:
                claimed.add(key)
                out[i] = True
                break
    return out


def pooled_precision(tp_sorted: np.ndarray, w_sorted: np.ndarray | None, p_at: int, fp_at: int) -> tuple[float, float]:
    """P@p_at and #TP before the (fp_at+1)-th FP on a pre-sorted pooled list with integer
    multiplicities ``w_sorted`` (None = 1). Returns (precision, n_tp)."""
    t = tp_sorted.astype(float)
    w = np.ones_like(t) if w_sorted is None else w_sorted.astype(float)
    cw = np.cumsum(w)
    ctp = np.cumsum(w * t)
    if cw.size == 0 or cw[-1] <= 0:
        return np.nan, np.nan
    if cw[-1] >= p_at:
        i = int(np.searchsorted(cw, p_at))  # first index with cw >= p_at
        prev_w = cw[i - 1] if i > 0 else 0.0
        prev_t = ctp[i - 1] if i > 0 else 0.0
        p = (prev_t + (p_at - prev_w) * t[i]) / p_at
    else:
        p = ctp[-1] / cw[-1]
    cfp = np.cumsum(w * (1 - t))
    if cfp[-1] > fp_at:
        i = int(np.searchsorted(cfp, fp_at + 1))  # first index where FP count reaches fp_at+1
        ntp = ctp[i - 1] if i > 0 else 0.0
    else:
        ntp = ctp[-1]
    return float(p), float(ntp)


# --------------------------------------------------------------------------- summary + bootstrap
def _site_metric_cols(sites: pd.DataFrame) -> list[str]:
    return [c for c in sites.columns if c.startswith("recall_")]


def summarize_sites(res: SiteEvalResult, n_boot: int = 1000, seed: int = BOOTSTRAP_SEED, groups_col: str = "group") -> pd.DataFrame:
    """Point estimates + percentile 95 % CIs from resampling ``group`` units (cluster bootstrap)."""
    S, T, K, P = res.sites, res.structures, res.pockets, res.protocol
    gnames = pd.Index(sorted(set(T[groups_col].astype(str))))
    gi_sites = gnames.get_indexer(S[groups_col].astype(str)) if len(S) else np.zeros(0, int)
    gi_struct = gnames.get_indexer(T[groups_col].astype(str))
    G = len(gnames)
    W = _bootstrap_weights(G, n_boot, seed) if n_boot and G else None
    rows = []

    def ratio(name, num_site, den_site, gi):
        num = np.bincount(gi, weights=num_site, minlength=G)
        den = np.bincount(gi, weights=den_site, minlength=G)
        v = num.sum() / den.sum() if den.sum() > 0 else np.nan
        lo = hi = np.nan
        if W is not None and den.sum() > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                bs = (W @ num) / (W @ den)
            bs = bs[np.isfinite(bs)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"metric": name, "value": v, "ci_low": lo, "ci_high": hi, "n": int(den.sum())})

    for c in _site_metric_cols(S):
        x = S[c].astype(float).to_numpy()
        ok = np.isfinite(x)
        ratio(c, np.where(ok, x, 0), ok.astype(float), gi_sites)
    # RRO
    if len(S):
        sel = (S["dcc_closest"] <= P.dcc) & (S["irel_closest"] > 0)
        ratio("RRO", np.where(sel, S["irel_closest"].fillna(0), 0.0), sel.astype(float).to_numpy(), gi_sites)
    # residue metrics (mean over structures with defined values)
    for c in ("res_f1", "res_mcc"):
        x = T[c].astype(float).to_numpy()
        ok = np.isfinite(x)
        if ok.any():
            ratio(c, np.where(ok, x, 0), ok.astype(float), gi_struct)
    # pooled precision
    if len(K):
        order = _pooled_order(K)
        tp = K["tp"].to_numpy(dtype=bool)[order]
        tp11 = _one_to_one_tp(K, order)[order]
        gk = gnames.get_indexer(K[groups_col].astype(str))[order]
        for name, t in (("P1K", tp), ("P1K_1to1", tp11)):
            p, ntp = pooled_precision(t, None, P.p_at, P.fp_at)
            plo = phi = nlo = nhi = np.nan
            if W is not None:
                bsp, bsn = [], []
                for b in range(W.shape[0]):
                    pb, nb = pooled_precision(t, W[b, gk], P.p_at, P.fp_at)
                    bsp.append(pb)
                    bsn.append(nb)
                plo, phi = np.nanpercentile(bsp, [2.5, 97.5])
                nlo, nhi = np.nanpercentile(bsn, [2.5, 97.5])
            rows.append({"metric": name, "value": p, "ci_low": plo, "ci_high": phi, "n": len(K)})
            suffix = "" if name == "P1K" else "_1to1"
            rows.append({"metric": f"TP@{P.fp_at}FP{suffix}", "value": ntp, "ci_low": nlo, "ci_high": nhi, "n": len(K)})
    rows.append({"metric": "n_scored_sites", "value": float(len(S)), "ci_low": np.nan, "ci_high": np.nan, "n": len(S)})
    rows.append({"metric": "n_structures", "value": float(len(T)), "ci_low": np.nan, "ci_high": np.nan, "n": len(T)})
    rows.append({"metric": "n_groups", "value": float(G), "ci_low": np.nan, "ci_high": np.nan, "n": G})
    rows.append({"metric": "n_pred_missing", "value": float(T["pred_missing"].sum()), "ci_low": np.nan, "ci_high": np.nan, "n": len(T)})
    return pd.DataFrame(rows)


def compare_site_methods(a: SiteEvalResult, b: SiteEvalResult, metrics: Sequence[str] = ("recall_topn2",), n_boot: int = 1000,
                         seed: int = BOOTSTRAP_SEED, groups_col: str = "group", include_p1k: bool = True) -> pd.DataFrame:
    """Paired cluster-bootstrap comparison A − B on identical site sets (missing sites = misses).
    Also compares pooled P1K recomputed in every resample with the same weights."""
    key = ["structure_id", "site_id"]
    sa, sb = a.sites, b.sites
    m = sa[key + [groups_col] + list(metrics)].merge(sb[key + list(metrics)], on=key, how="outer", suffixes=("_a", "_b"), indicator=True)
    n_unpaired = int((m["_merge"] != "both").sum())
    if n_unpaired:
        # sites missing in one evaluation (should not happen with identical site tables) → misses
        if m[groups_col].isna().any():
            gb = sb.set_index(key)[groups_col]
            m[groups_col] = m[groups_col].fillna(pd.Series([gb.get(tuple(k)) for k in m[key].to_numpy()], index=m.index))
    gnames = pd.Index(sorted(set(m[groups_col].astype(str))))
    gi = gnames.get_indexer(m[groups_col].astype(str))
    G = len(gnames)
    W = _bootstrap_weights(G, n_boot, seed)
    rows = []
    for c in metrics:
        xa = m[f"{c}_a"].astype(float).fillna(0.0).to_numpy()
        xb = m[f"{c}_b"].astype(float).fillna(0.0).to_numpy()
        na = np.bincount(gi, weights=xa, minlength=G)
        nb_ = np.bincount(gi, weights=xb, minlength=G)
        den = np.bincount(gi, minlength=G).astype(float)
        d = (W @ (na - nb_)) / (W @ den)
        diff = (na - nb_).sum() / den.sum()
        b10 = int(((xa == 1) & (xb == 0)).sum())
        b01 = int(((xa == 0) & (xb == 1)).sum())
        from scipy import stats

        rows.append({
            "metric": c, "a": na.sum() / den.sum(), "b": nb_.sum() / den.sum(), "diff": diff,
            "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "p_boot": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            "p_mcnemar": float(stats.binomtest(b10, b10 + b01, 0.5).pvalue) if b10 + b01 else 1.0,
            "n_sites": int(den.sum()), "n_groups": G, "n_unpaired_as_miss": n_unpaired,
        })
    if include_p1k:
        P = a.protocol
        res = []
        for r in (a, b):
            K = r.pockets
            order = _pooled_order(K)
            gk = gnames.get_indexer(K[groups_col].astype(str))[order]
            if (gk < 0).any():  # pockets of structures whose groups have no scored site: never resampled
                keep = gk >= 0
            else:
                keep = np.ones(len(gk), dtype=bool)
            res.append((K["tp"].to_numpy(dtype=bool)[order][keep], gk[keep], *pooled_precision(K["tp"].to_numpy(dtype=bool)[order], None, P.p_at, P.fp_at)))
        d = np.array([pooled_precision(res[0][0], W[i, res[0][1]], P.p_at, P.fp_at)[0] - pooled_precision(res[1][0], W[i, res[1][1]], P.p_at, P.fp_at)[0] for i in range(W.shape[0])])
        rows.append({"metric": "P1K", "a": res[0][2], "b": res[1][2], "diff": res[0][2] - res[1][2],
                     "ci_low": float(np.nanpercentile(d, 2.5)), "ci_high": float(np.nanpercentile(d, 97.5)),
                     "p_boot": float(min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))), "p_mcnemar": np.nan,
                     "n_sites": np.nan, "n_groups": G, "n_unpaired_as_miss": n_unpaired})
    return pd.DataFrame(rows)
