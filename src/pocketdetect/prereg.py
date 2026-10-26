"""Pre-registered analysis (docs/PREREG.md, amendments A1–A3): endpoint definitions, paired
cluster-bootstrap comparisons on evaluation output directories, and the decision rules.

Every comparison works on *evaluation directories* written by ``benchmarks.run_benchmark`` (the
same files for baselines and BURROW):
  sites benchmarks : sites_eval.csv (+ group = 30 % cluster unit), pockets_eval.csv
  ligand benchmarks: ligands.csv (units from pocketdetect.clusters)
  dimer            : sites.csv (eval_dimer rows; units from pocketdetect.clusters over complexes)

Paired comparison of methods A, B on one benchmark: rows are matched on the reference key
(site / ligand); a reference missing from one method's table counts as a miss for that method;
rows undefined for both (e.g. DCA for sites without ligand atoms, PREREG A2) are dropped. The
statistic is the pooled rate (Σ successes / Σ rows); its bootstrap distribution resamples units
with replacement (B = 1000, seed BOOTSTRAP_SEED), the same weights for A and B. P1K is recomputed
on the pooled pocket list of each method in every resample.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .clusters import groups_for_structures
from .evaluate import _bootstrap_weights
from .seeds import BOOTSTRAP_SEED
from .site_eval import SiteProtocol, _pooled_order, pooled_precision

__all__ = ["Endpoint", "FAMILY1", "FAMILY2", "load_rows", "paired", "holm", "decide_noninferiority"]


@dataclass(frozen=True)
class Endpoint:
    name: str
    benchmark: str  # registry name
    metric: str
    stratum: str | None = None  # dimer: site_class filter ("interface")
    ranker: str = "all"
    af_mode: bool = False
    comparators: tuple = ()  # baseline variants (baselines/outputs/<bench_dir>/<variant>)
    bench_dir: str = ""  # baselines/outputs/<bench_dir>
    margin: float = -2.0  # non-inferiority margin (points)
    note: str = ""


# Family 1 (fixed sequence) — comparators: published per-chain predictions + our re-run (A3)
FAMILY1 = [
    Endpoint("C1", "ligysis", "recall_topn2", ranker="all", comparators=("published:fpocket+PRANK", "fpocket_4.2.3+PRANK_2.4.2"),
             bench_dir="ligysis", note="Δ ≥ +2.0 pts and paired CI of Δ excludes 0, vs both comparators, each seed"),
    Endpoint("C1-strict", "ligysis", "recall_ldca_topn2", ranker="all", comparators=("published:fpocket+PRANK", "fpocket_4.2.3+PRANK_2.4.2"),
             bench_dir="ligysis", note="CI of Δ excludes 0 in our favour (A2 endpoint)"),
    Endpoint("C2", "ligysis", "P1K", ranker="all", comparators=("published:fpocket+PRANK", "fpocket_4.2.3+PRANK_2.4.2"),
             bench_dir="ligysis", margin=-3.0, note="non-inferiority: lower CI bound of Δ > −3 pts"),
]
FAMILY2 = [
    Endpoint("O2a", "coach420_mlig", "success_DCA4_topn", ranker="small", comparators=("p2rank_2.5.1", "fpocket_4.2.3+PRANK_2.5.1"), bench_dir="coach420_mlig"),
    Endpoint("O2b", "holo4k_mlig", "success_DCA4_topn", ranker="small", comparators=("p2rank_2.5.1", "fpocket_4.2.3+PRANK_2.5.1"), bench_dir="holo4k_mlig"),
    Endpoint("O3", "pocketwatch", "recall_topn2", ranker="small", af_mode=True,
             comparators=("p2rank_2.5.1", "p2rank_2.5.1_af", "fpocket_4.2.3+PRANK_2.5.1"), bench_dir="pocketwatch"),
    Endpoint("O4", "cryptobench", "recall_topn2", ranker="small", comparators=("p2rank_2.5.1", "fpocket_4.2.3+PRANK_2.5.1"), bench_dir="cryptobench"),
    Endpoint("O5", "af_set", "recall_topn2", ranker="all", af_mode=True,
             comparators=("p2rank_2.5.1_af", "p2rank_2.5.1", "fpocket_4.2.3+PRANK_2.5.1"), bench_dir="af_set"),
    Endpoint("O6", "dimer", "success_DCC12_topn2", stratum="interface", ranker="all",
             comparators=("p2rank_2.5.1", "fpocket_4.2.3+PRANK_2.5.1"), bench_dir="dimer_assembly"),
]
CLUSTER_SOURCES = {"ligysis": ("ligysis", "ligysis_extra"), "ligysis_ni": ("ligysis", "ligysis_extra"), "coach420_mlig": ("coach420",),
                   "holo4k_mlig": ("holo4k",), "pocketwatch": ("pocketwatch_test",), "cryptobench": ("cryptobench",),
                   "af_set": ("af_set",), "af_set_allsites": ("af_set",), "dimer": ("dimer",)}
INTERFACE = ("homo_interface", "hetero_interface")


# --------------------------------------------------------------------------- row tables
@dataclass
class Rows:
    kind: str  # sites | ligands | dimer
    rows: pd.DataFrame  # key columns + metric columns + group
    keys: list
    pockets: pd.DataFrame | None = None  # sites only (P1K)
    eval_dir: str = ""


def load_rows(benchmark: str, eval_dir: str | Path) -> Rows:
    d = Path(eval_dir)
    if (d / "sites_eval.csv").exists():
        s = pd.read_csv(d / "sites_eval.csv", dtype={"structure_id": str, "site_id": str, "group": str})
        k = pd.read_csv(d / "pockets_eval.csv", dtype={"structure_id": str, "group": str, "hit_sites": str}, keep_default_na=False)
        return Rows("sites", s, ["structure_id", "site_id"], k, str(d))
    if (d / "ligands.csv").exists():
        L = pd.read_csv(d / "ligands.csv", dtype={"structure_id": str, "ligand_id": str})
        g = groups_for_structures(sorted(L.structure_id.unique()), CLUSTER_SOURCES[benchmark])
        L["group"] = L.structure_id.map(g)
        return Rows("ligands", L, ["structure_id", "ligand_id"], None, str(d))
    if (d / "sites.csv").exists():
        S = pd.read_csv(d / "sites.csv", dtype={"structure_id": str, "complex_id": str})
        g = groups_for_structures(sorted(S.complex_id.unique()), CLUSTER_SOURCES[benchmark])
        S["group"] = S.complex_id.map(g)
        S["site"] = S["site"].astype(str)
        return Rows("dimer", S, ["complex_id", "site"], None, str(d))
    raise FileNotFoundError(f"no evaluation tables in {d}")


def _units(groups: pd.Series):
    names, idx = np.unique(groups.astype(str).to_numpy(), return_inverse=True)
    return names, idx


def paired(a: Rows, b: Rows, metric: str, stratum: str | None = None, n_boot: int = 1000, seed: int = BOOTSTRAP_SEED,
           common_subset: bool = False, keep_draws: bool = False) -> dict:
    """A − B on one metric (points = ×100 for rates). See module docstring."""
    if metric == "P1K":
        return _paired_p1k(a, b, n_boot, seed, keep_draws)
    ra, rb = a.rows, b.rows
    if stratum == "interface":
        ra = ra[ra.site_class.isin(INTERFACE)]
        rb = rb[rb.site_class.isin(INTERFACE)]
    cols = a.keys + ["group", metric]
    m = ra[cols].merge(rb[b.keys + [metric]], on=a.keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    n_unpaired = int((m["_merge"] != "both").sum())
    if m["group"].isna().any():
        gb = rb.set_index(b.keys)["group"]
        m["group"] = m["group"].fillna(pd.Series([gb.get(tuple(r)) for r in m[a.keys].to_numpy()], index=m.index))
    xa = m[f"{metric}_a"].astype(float)
    xb = m[f"{metric}_b"].astype(float)
    undefined = xa.isna() & xb.isna()
    if common_subset:
        sa = set(a.rows.loc[~a.rows.get("pred_missing", pd.Series(False, index=a.rows.index)).astype(bool), a.keys[0]])
        sb = set(b.rows.loc[~b.rows.get("pred_missing", pd.Series(False, index=b.rows.index)).astype(bool), b.keys[0]])
        undefined |= ~m[a.keys[0]].isin(sa & sb)
    m = m[~undefined]
    xa = m[f"{metric}_a"].astype(float).fillna(0.0).to_numpy()  # missing / failed = miss
    xb = m[f"{metric}_b"].astype(float).fillna(0.0).to_numpy()
    names, ui = _units(m["group"])
    G = len(names)
    na = np.bincount(ui, weights=xa, minlength=G)
    nb = np.bincount(ui, weights=xb, minlength=G)
    den = np.bincount(ui, minlength=G).astype(float)
    W = _bootstrap_weights(G, n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        da = (W @ na) / (W @ den)
        db = (W @ nb) / (W @ den)
    d = 100 * (da - db)
    out = {
        "metric": metric, "stratum": stratum or "", "a": 100 * na.sum() / den.sum(), "b": 100 * nb.sum() / den.sum(),
        "diff": 100 * (na.sum() - nb.sum()) / den.sum(), "ci_low": float(np.nanpercentile(d, 2.5)), "ci_high": float(np.nanpercentile(d, 97.5)),
        "a_ci": [float(np.nanpercentile(100 * da, 2.5)), float(np.nanpercentile(100 * da, 97.5))],
        "n_rows": int(len(m)), "n_units": G, "n_unpaired_as_miss": n_unpaired, "n_undefined_dropped": int(undefined.sum()),
        "common_subset": common_subset,
    }
    if keep_draws:
        out["_draws"] = d
    return out


def _paired_p1k(a: Rows, b: Rows, n_boot: int, seed: int, keep_draws: bool) -> dict:
    P = SiteProtocol()
    units = sorted(set(a.rows.group.astype(str)) | set(b.rows.group.astype(str)))
    pos = {u: i for i, u in enumerate(units)}
    W = _bootstrap_weights(len(units), n_boot, seed)
    res = []
    for r in (a, b):
        K = r.pockets
        o = _pooled_order(K)
        t = K["tp"].astype(str).str.lower().isin(["true", "1"]).to_numpy()[o]
        gk = K["group"].astype(str).map(pos).fillna(-1).astype(int).to_numpy()[o]
        keep = gk >= 0  # pockets of structures without scored sites are never resampled
        res.append((t[keep], gk[keep], pooled_precision(t, None, P.p_at, P.fp_at)[0]))
    d = np.array([pooled_precision(res[0][0], W[i, res[0][1]], P.p_at, P.fp_at)[0] - pooled_precision(res[1][0], W[i, res[1][1]], P.p_at, P.fp_at)[0]
                  for i in range(W.shape[0])]) * 100
    out = {"metric": "P1K", "stratum": "", "a": 100 * res[0][2], "b": 100 * res[1][2], "diff": 100 * (res[0][2] - res[1][2]),
           "ci_low": float(np.nanpercentile(d, 2.5)), "ci_high": float(np.nanpercentile(d, 97.5)), "n_units": len(units),
           "n_rows": int(len(a.pockets)), "n_unpaired_as_miss": 0, "n_undefined_dropped": 0, "common_subset": False}
    if keep_draws:
        out["_draws"] = d
    return out


# --------------------------------------------------------------------------- decisions
def decide_noninferiority(c: dict, margin: float) -> str:
    if c["ci_low"] > margin:
        return "pass"
    if c["ci_high"] < margin:
        return "fail"
    return "inconclusive"


def holm(pvals: dict, alpha: float = 0.025) -> dict:
    """Holm step-down on one-sided p-values (α = 0.025 one-sided ≙ two-sided 95 % CIs)."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    out, stop = {}, False
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        rej = (not stop) and pvals[k] <= thr
        if not rej:
            stop = True
        out[k] = {"p": pvals[k], "threshold": thr, "rejected": rej}
    return out
