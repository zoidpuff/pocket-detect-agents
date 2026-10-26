#!/usr/bin/env python
"""Dimer-benchmark evaluation adaptor (symmetric-copy aware). Spec: docs/DIMER_BENCHMARK.md §5,
docs/PREREG.md (O6, amendment A1).

Per complex (assembly mode) or isolated chain (chain mode):
  1. Ground truth = ligand copies from data/processed/dimer/gt/<pdb>.json, grouped into unique sites
     (default ``j50``: connected components of entity-level residue-set Jaccard >= 0.5, PREREG A1;
     ``strict``: identical entity-level residue sets). In chain mode only copies relevant to the
     chain are kept (>= min(3, max residues on any chain) contact residues on that chain).
     N = number of unique sites.
  2. Label-free prediction collapse (every method): predictions are visited in rank order; a
     prediction is dropped if its entity-level residue set (entity id + author resnum/icode) has
     Jaccard >= 0.5 with an already *kept* prediction. Predictions without residues get residues
     with any heavy atom <= 6 A of the centre (PREREG residue rule for centre-only methods) when
     the input structure is available; empty sets never merge.
  3. A prediction hits a unique site if it hits ANY copy: DCC12 = centre <= 12 A from that copy's
     ligand centroid; DCA4 = centre <= 4 A from any heavy atom of that copy. Site success at k
     (top-N, top-N+2, all) if any of the first k kept predictions hits it (not one-to-one, as in
     LIGYSIS). LDCC12 (LIGYSIS 'closest prediction' rule) is reported as a descriptive variant.
  4. Precision within the top-(N+2) kept predictions, rank order: a prediction hitting a not yet
     matched unique site is TP; one hitting only already matched sites is neutral (not FP); one
     hitting nothing is FP.

Predictions: <pred-dir>/<structure_id>.csv in the pocketdetect.evaluate format
(rank,score,center_x,center_y,center_z,residues); structure_id = <pdb> (assembly mode) or
<pdb>_<chain> (chain mode). Missing file = zero predictions (all sites missed).

Usage:
  python scripts/eval_dimer.py --pred-dir results/preds/X/dimer_assembly --mode assembly --out OUT
  python scripts/eval_dimer.py --pred-dir results/preds/X/dimer_chains --mode chain [--chain-set all]
  python scripts/eval_dimer.py --oracle --mode assembly --out OUT      # ceiling control
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketdetect.evaluate import Predictions, empty_predictions, read_predictions  # noqa: E402
from pocketdetect.seeds import BOOTSTRAP_SEED  # noqa: E402
from pocketdetect.structure import load_structure, parse_residue_id  # noqa: E402

DATA = ROOT / "data/processed/dimer"
COLLAPSE_JACCARD = 0.5
DCC_T, DCA_T = 12.0, 4.0
CENTRE_RES_DIST = 6.0
CRITERIA = ("DCC12", "DCA4", "LDCC12")
INTERFACE = ("homo_interface", "hetero_interface")


# ----------------------------------------------------------------------------- helpers
def jaccard(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def entity_set(residues, chain_entity: dict) -> frozenset:
    """'chain:resnum[icode]' ids -> entity-level ids 'E<entity>:<resnum><icode>' (unknown chain
    ids are kept as '?<chain>:<resnum>' so they can still match each other but never GT)."""
    out = set()
    for rid in residues:
        try:
            c, n, i = parse_residue_id(rid)
        except ValueError:
            continue
        e = chain_entity.get(c)
        out.add(f"E{e}:{n}{i}" if e is not None else f"?{c}:{n}{i}")
    return frozenset(out)


def group_sets(sets: list[frozenset], mode: str) -> list[int]:
    if mode == "strict":
        seen: dict = {}
        return [seen.setdefault(s, len(seen)) for s in sets]
    if mode != "j50":
        raise ValueError(f"unknown grouping {mode!r}")
    n = len(sets)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(sets[i], sets[j]) >= COLLAPSE_JACCARD:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    order: dict = {}
    return [order.setdefault(find(i), len(order)) for i in range(n)]


def chain_relevant(chain_counts: dict, chain: str) -> bool:
    if not chain_counts:
        return False
    return chain_counts.get(chain, 0) >= min(3, max(chain_counts.values()))


def residues_near(centers: np.ndarray, coords: np.ndarray, res_ids: np.ndarray, cutoff: float = CENTRE_RES_DIST) -> list[frozenset]:
    """Residues (ids per atom in ``res_ids``) with any atom within ``cutoff`` of each centre."""
    if len(coords) == 0:
        return [frozenset() for _ in centers]
    tree = cKDTree(coords)
    return [frozenset(res_ids[np.asarray(h, dtype=np.int64)].tolist()) for h in tree.query_ball_point(centers, r=cutoff)]


# ----------------------------------------------------------------------------- ground truth
@dataclass
class SiteGT:
    complex_id: str
    structure_id: str
    chain_entity: dict
    copy_site: np.ndarray  # (c,) unique-site index per copy
    copy_centers: np.ndarray  # (c,3)
    copy_coords: list  # list of (k,3)
    copy_ids: list
    sites: pd.DataFrame  # one row per unique site: site, site_class, is_ion, n_copies, ccd

    @property
    def n_sites(self) -> int:
        return len(self.sites)


_RANK = {"hetero_interface": 3, "homo_interface": 2, "minor_interface": 1, "single_chain": 0}


def site_gt(gt: dict, grouping: str = "j50", chain: str | None = None) -> SiteGT:
    chain_entity = {c["chain"]: c["entity_id"] for c in gt["chains"]}
    copies = [c for c in gt["copies"] if chain is None or chain_relevant(c["chain_counts"], chain)]
    groups = group_sets([frozenset(c["entity_residues"]) for c in copies], grouping)
    rows = []
    for g in range(max(groups) + 1 if groups else 0):
        cs = [c for c, gg in zip(copies, groups) if gg == g]
        classes = {c["site_class"] for c in cs}
        rows.append(
            {
                "site": g,
                "site_class": max(classes, key=_RANK.get),
                "class_consistent": len(classes) == 1,
                "is_ion": all(c["is_ion"] for c in cs),
                "n_copies": len(cs),
                "ccd": "&".join(sorted({c["ccd"] for c in cs})),
                "copies": " ".join(c["ligand_id"] for c in cs),
            }
        )
    pid = gt["pdb_id"]
    return SiteGT(
        complex_id=pid,
        structure_id=pid if chain is None else f"{pid}_{chain}",
        chain_entity=chain_entity,
        copy_site=np.asarray(groups, dtype=np.int64),
        copy_centers=np.asarray([c["center"] for c in copies], dtype=np.float64).reshape(-1, 3),
        copy_coords=[np.asarray(c["coords"], dtype=np.float64).reshape(-1, 3) for c in copies],
        copy_ids=[c["ligand_id"] for c in copies],
        sites=pd.DataFrame(rows, columns=["site", "site_class", "class_consistent", "is_ion", "n_copies", "ccd", "copies"]),
    )


# ----------------------------------------------------------------------------- predictions
def collapse_predictions(pred: Predictions, chain_entity: dict, thr: float = COLLAPSE_JACCARD) -> tuple[np.ndarray, list]:
    """Indices (rank order) of predictions kept by the label-free collapse, and entity sets."""
    esets = [entity_set(r, chain_entity) for r in pred.residues]
    kept: list[int] = []
    for i, e in enumerate(esets):
        if e and any(jaccard(e, esets[j]) >= thr for j in kept):
            continue
        kept.append(i)
    return np.asarray(kept, dtype=np.int64), esets


def fill_missing_residues(pred: Predictions, structure_path: Path | None) -> Predictions:
    """PREREG rule for centre-only methods: residues with any heavy atom <= 6 A of the centre."""
    if pred.n == 0 or all(len(r) for r in pred.residues) or structure_path is None or not Path(structure_path).exists():
        return pred
    s = load_structure(str(structure_path), protein_only=True)
    rid = s.atom_residue_ids() if callable(getattr(s, "atom_residue_ids", None)) else s.atom_residue_ids
    near = residues_near(pred.centers, s.coords.astype(np.float64), np.asarray(rid, dtype=object))
    res = [r if len(r) else near[i] for i, r in enumerate(pred.residues)]
    return Predictions(pred.structure_id, pred.centers, pred.scores, pred.ranks, res, pred.missing)


# ----------------------------------------------------------------------------- scoring
def evaluate_unit(g: SiteGT, pred: Predictions, collapse: bool = True) -> tuple[list[dict], dict]:
    """Score one structure (complex or isolated chain). Returns (site rows, unit row)."""
    if collapse:
        kept, _ = collapse_predictions(pred, g.chain_entity)
    else:
        kept = np.arange(pred.n, dtype=np.int64)
    P = pred.centers[kept] if pred.n else np.zeros((0, 3))
    m, n_s, n_c = len(P), g.n_sites, len(g.copy_ids)
    dcc = np.full((m, n_c), np.inf)
    dca = np.full((m, n_c), np.inf)
    for j in range(n_c):
        if m:
            dcc[:, j] = np.linalg.norm(P - g.copy_centers[j], axis=1)
            dca[:, j] = np.min(np.linalg.norm(P[:, None, :] - g.copy_coords[j][None, :, :], axis=2), axis=1)
    # site-level hit matrices (prediction x unique site): hit if any copy is hit
    S = np.zeros((n_c, n_s), dtype=bool)
    if n_c:
        S[np.arange(n_c), g.copy_site] = True

    def site_hits(D, t):
        H = (D <= t).astype(np.int64)
        return (H @ S.astype(np.int64)) > 0  # (m, n_s)

    hits = {"DCC12": site_hits(dcc, DCC_T), "DCA4": site_hits(dca, DCA_T)}
    # site-level min distance over copies (for LDCC and reporting)
    sdcc = np.full((m, n_s), np.inf)
    for s in range(n_s):
        if m:
            sdcc[:, s] = dcc[:, g.copy_site == s].min(axis=1)
    ks = {"topn": n_s, "topn2": n_s + 2, "all": m}
    rows = []
    for s in range(n_s):
        r = {"structure_id": g.structure_id, "complex_id": g.complex_id, **g.sites.iloc[s].to_dict(), "n_sites": n_s, "n_pred_kept": m}
        for c in ("DCC12", "DCA4"):
            col = hits[c][:, s]
            first = int(np.argmax(col)) + 1 if col.any() else -1
            r[f"rank_{c}"] = first
            for lab, k in ks.items():
                r[f"success_{c}_{lab}"] = bool(0 < first <= k)
        if m:
            i = int(np.argmin(sdcc[:, s]))
            ok = sdcc[i, s] <= DCC_T
            r["rank_LDCC12"] = i + 1 if ok else -1
            r["min_dcc"] = float(sdcc[i, s])
        else:
            r["rank_LDCC12"], r["min_dcc"] = -1, np.nan
        for lab, k in ks.items():
            r[f"success_LDCC12_{lab}"] = bool(0 < r["rank_LDCC12"] <= k)
        rows.append(r)
    # precision within top-(N+2), primary criterion DCC12
    tp = fp = neutral = 0
    matched = np.zeros(n_s, dtype=bool)
    H = hits["DCC12"]
    for i in range(min(m, n_s + 2)):
        hs = np.flatnonzero(H[i])
        if hs.size == 0:
            fp += 1
            continue
        free = hs[~matched[hs]]
        if free.size:
            matched[free[np.argmin(sdcc[i, free])]] = True
            tp += 1
        else:
            neutral += 1
    unit = {
        "structure_id": g.structure_id,
        "complex_id": g.complex_id,
        "n_sites": n_s,
        "n_copies": n_c,
        "n_pred_raw": pred.n,
        "n_pred_kept": m,
        "n_pred_collapsed": pred.n - m,
        "pred_missing": pred.missing,
        "tp_topn2": tp,
        "fp_topn2": fp,
        "neutral_topn2": neutral,
    }
    return rows, unit


def oracle_predictions(g: SiteGT, gt: dict) -> Predictions:
    """Ceiling control: one prediction per GT copy (centroid, copy binding residues)."""
    by_id = {c["ligand_id"]: c for c in gt["copies"]}
    res = [frozenset(by_id[i]["binding_residues"]) for i in g.copy_ids]
    m = len(res)
    return Predictions(g.structure_id, g.copy_centers.copy(), np.arange(m, 0, -1.0), np.arange(1, m + 1), res)


# ----------------------------------------------------------------------------- summary
def _boot(num: np.ndarray, den: np.ndarray, W: np.ndarray | None):
    tot = den.sum()
    v = num.sum() / tot if tot else np.nan
    if W is None or not tot:
        return v, np.nan, np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (W @ num) / (W @ den)
    b = b[np.isfinite(b)]
    lo, hi = np.percentile(b, [2.5, 97.5]) if b.size else (np.nan, np.nan)
    return float(v), float(lo), float(hi)


def summarize(sites: pd.DataFrame, units: pd.DataFrame, n_boot: int = 1000, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """Site-pooled recall per class/criterion/k and top-(N+2) precision; percentile bootstrap CI
    over complexes (one complex per LIGYSIS UniProt / hetero complex)."""
    cids = np.array(sorted(units["complex_id"].unique()))
    pos = {c: i for i, c in enumerate(cids)}
    rng = np.random.default_rng(seed)
    W = None
    if n_boot and len(cids):
        idx = rng.integers(0, len(cids), size=(n_boot, len(cids)))
        W = np.zeros((n_boot, len(cids)))
        np.add.at(W, (np.repeat(np.arange(n_boot), len(cids)), idx.ravel()), 1.0)
    strata = {
        "all": np.ones(len(sites), dtype=bool),
        "interface": sites["site_class"].isin(INTERFACE).to_numpy(),
        "homo_interface": (sites["site_class"] == "homo_interface").to_numpy(),
        "hetero_interface": (sites["site_class"] == "hetero_interface").to_numpy(),
        "single_chain": (sites["site_class"] == "single_chain").to_numpy(),
        "minor_interface": (sites["site_class"] == "minor_interface").to_numpy(),
        "ion": sites["is_ion"].to_numpy(dtype=bool),
        "non_ion": ~sites["is_ion"].to_numpy(dtype=bool),
    }
    rows = []
    cix = sites["complex_id"].map(pos).to_numpy() if len(sites) else np.zeros(0, int)
    for c in CRITERIA:
        for lab in ("topn", "topn2", "all"):
            col = f"success_{c}_{lab}"
            x = sites[col].to_numpy(dtype=float) if len(sites) else np.zeros(0)
            for sname, mask in strata.items():
                num = np.bincount(cix[mask], weights=x[mask], minlength=len(cids))
                den = np.bincount(cix[mask], minlength=len(cids)).astype(float)
                v, lo, hi = _boot(num, den, W)
                rows.append({"metric": f"recall_{c}_{lab}", "stratum": sname, "value": v, "ci_low": lo, "ci_high": hi, "n_sites": int(mask.sum())})
    ucix = units["complex_id"].map(pos).to_numpy()
    num = np.bincount(ucix, weights=units["tp_topn2"].to_numpy(float), minlength=len(cids))
    den = np.bincount(ucix, weights=(units["tp_topn2"] + units["fp_topn2"]).to_numpy(float), minlength=len(cids))
    v, lo, hi = _boot(num, den, W)
    rows.append({"metric": "precision_DCC12_topn2", "stratum": "all", "value": v, "ci_low": lo, "ci_high": hi, "n_sites": int(len(sites))})
    for k in ("n_pred_raw", "n_pred_kept", "n_pred_collapsed", "neutral_topn2"):
        rows.append({"metric": f"mean_{k}", "stratum": "all", "value": float(units[k].mean()) if len(units) else np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_sites": int(len(sites))})
    rows.append({"metric": "n_units", "stratum": "all", "value": float(len(units)), "ci_low": np.nan, "ci_high": np.nan, "n_sites": int(len(sites))})
    rows.append({"metric": "n_pred_missing", "stratum": "all", "value": float(units["pred_missing"].sum()), "ci_low": np.nan, "ci_high": np.nan, "n_sites": int(len(sites))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- driver
def load_gts(data: Path, complexes: list[str] | None = None) -> dict[str, dict]:
    ids = complexes or pd.read_csv(data / "complexes.tsv", sep="\t", dtype=str)["complex_id"].tolist()
    return {i: json.loads((data / "gt" / f"{i}.json").read_text()) for i in ids}


def run(
    gts: dict[str, dict],
    mode: str,
    pred_dir: Path | None,
    grouping: str = "j50",
    chain_set: str = "primary",
    collapse: bool = True,
    oracle: bool = False,
    data: Path = DATA,
    n_boot: int = 1000,
):
    site_rows, unit_rows = [], []
    for pid, gt in gts.items():
        if mode == "assembly":
            units = [(None, data / "inputs/assembly" / f"{pid}.pdb")]
        elif mode == "chain":
            chains = [gt["primary_chain"]] if chain_set == "primary" else [c["chain"] for c in gt["chains"]]
            units = [(c, data / "inputs/chains" / f"{pid}_{c}.pdb") for c in chains]
        else:
            raise ValueError(mode)
        for chain, inp in units:
            g = site_gt(gt, grouping, chain)
            if oracle:
                pred = oracle_predictions(g, gt)
            else:
                p = Path(pred_dir) / f"{g.structure_id}.csv"
                pred = read_predictions(p, g.structure_id) if p.exists() else empty_predictions(g.structure_id, missing=True)
                pred = fill_missing_residues(pred, inp)
            sr, ur = evaluate_unit(g, pred, collapse=collapse)
            site_rows += sr
            unit_rows.append({**ur, "chain": chain or ""})
    sites = pd.DataFrame(site_rows)
    units = pd.DataFrame(unit_rows)
    summary = summarize(sites, units, n_boot=n_boot)
    return sites, units, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path)
    ap.add_argument("--oracle", action="store_true", help="evaluate GT-copy oracle predictions (ceiling control)")
    ap.add_argument("--mode", choices=["assembly", "chain"], default="assembly")
    ap.add_argument("--chain-set", choices=["primary", "all"], default="primary")
    ap.add_argument("--gt-grouping", choices=["j50", "strict"], default="j50", help="j50 = PREREG amendment A1 (default)")
    ap.add_argument("--no-collapse", action="store_true", help="diagnostic only: skip the label-free prediction collapse")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if not a.oracle and a.pred_dir is None:
        ap.error("--pred-dir or --oracle required")
    gts = load_gts(a.data)
    sites, units, summary = run(gts, a.mode, a.pred_dir, a.gt_grouping, a.chain_set, not a.no_collapse, a.oracle, a.data, a.n_boot)
    a.out.mkdir(parents=True, exist_ok=True)
    sites.to_csv(a.out / "sites.csv", index=False)
    units.to_csv(a.out / "units.csv", index=False)
    summary.to_csv(a.out / "summary.csv", index=False)
    info = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    (a.out / "run_info.json").write_text(json.dumps(info, indent=2))
    show = summary[summary.metric.isin(["recall_DCC12_topn2", "recall_DCA4_topn2", "precision_DCC12_topn2", "n_units", "n_pred_missing"])]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
