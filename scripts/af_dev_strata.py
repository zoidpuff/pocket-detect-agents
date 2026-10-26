#!/usr/bin/env python
"""Stratified AF-mode evaluation (dev / train-OOF only).

Site strata:
  ``near_low``  site within 8 Å (heavy atom to heavy atom) of a prunable segment (pLDDT < 50 run removed by
                the spaghetti rule) OR >= 1 lining (site) residue with pLDDT < 70
  ``rest``      all other sites
  -> DCC12 top-N+2 and DCA4 top-N+2 per stratum (cluster-bootstrap CIs; paired Δ vs ``--ref``).
Protein disorder bins (fraction of residues with pLDDT < 50): <10 %, 10-30 %, >30 %
  -> recall per bin and precision-type metrics: top-1 precision (rank-1 pocket hits a site, DCC <= 12),
     top-(N+2) pocket precision (fraction of each protein's top-(N+2) pockets hitting a site, pooled),
     pooled P@k with k = #proteins in the bin (cross-protein calibration at ~1 pocket per protein),
     hallucination rate of all / top-(N+2) pockets (pocketdetect.af_eval, unpruned pLDDT).

  python scripts/af_dev_strata.py --sites results/dev_af/af_dev_sites_all.tsv --methods a=dir b=dir --ref a --out X.csv
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.af_eval import hallucination_counts, pocket_plddt, residue_plddt  # noqa: E402
from pocketdetect.burrow.af_variants import CONFIGS, keep_mask  # noqa: E402
from pocketdetect.evaluate import _bootstrap_weights, load_prediction_dir  # noqa: E402
from pocketdetect.seeds import BOOTSTRAP_SEED  # noqa: E402
from pocketdetect.site_eval import SiteEvalResult, compare_site_methods, evaluate_sites  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402
from pocketdetect.structure import load_structure  # noqa: E402

BINS = [("dis<10%", 0.0, 0.10), ("dis10-30%", 0.10, 0.30), ("dis>30%", 0.30, 1.01)]
_SITES = {}


def _annot(args):
    sid, path = args
    s = load_structure(path, structure_id=sid, protein_only=True)
    s = s.select(s.protein_heavy_mask())
    pl = residue_plddt(s)
    rid = s.residue_ids
    keep = keep_mask(s, CONFIGS["prune"])
    frac50 = float(np.mean([pl[r] < 50 for r in rid]))
    pruned_xyz = s.coords[~keep[s.res_index]]
    t = cKDTree(pruned_xyz) if len(pruned_xyz) else None
    atom_rid = rid[s.res_index]
    out = {}
    for x in _SITES[sid]:
        m = np.isin(atom_rid, list(x.residues))
        near = bool(t is not None and m.any() and t.query(s.coords[m], k=1)[0].min() <= 8.0)
        low = any(pl.get(r, 100.0) < 70 for r in x.residues)
        out[x.site_id] = near or low
    return sid, out, frac50, pl, s


def boot_ratio(num, den, gi, G, W):
    a = np.bincount(gi, weights=num, minlength=G)
    b = np.bincount(gi, weights=den, minlength=G)
    v = a.sum() / b.sum() if b.sum() else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (W @ a) / (W @ b)
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan)
    return f"{100 * v:.1f} [{100 * lo:.1f},{100 * hi:.1f}]"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    st = read_sites(a.sites)
    by = sites_by_structure(st)
    _SITES.update(by)
    sids = list(by)
    paths = [tuple(x) for x in st.drop_duplicates("structure_id")[["structure_id", "structure_path"]].itertuples(index=False)]
    with ProcessPoolExecutor(a.jobs) as ex:
        ann = {r[0]: r[1:] for r in ex.map(_annot, paths, chunksize=4)}
    near = {(sid, k): v for sid, (o, _, _, _) in ann.items() for k, v in o.items()}
    pbin = {}
    for sid, (_, f, _, _) in ann.items():
        pbin[sid] = next(n for n, lo, hi in BINS if lo <= f < hi)
    groups = {sid: str(by[sid][0].meta.get("group")) for sid in sids}
    gnames = sorted(set(groups.values()))
    gidx = {g: i for i, g in enumerate(gnames)}
    G = len(gnames)
    W = _bootstrap_weights(G, a.n_boot, BOOTSTRAP_SEED)
    res = {}
    rows = []
    for name, d in (m.split("=", 1) for m in a.methods):
        r = evaluate_sites(by, d)
        res[name] = r
        S = r.sites.copy()
        S["near_low"] = [near[(x, y)] for x, y in zip(S.structure_id, S.site_id)]
        S["pbin"] = S.structure_id.map(pbin)
        gi_s = S.structure_id.map(groups).map(gidx).to_numpy()
        row = dict(method=name)
        for lab, msk in (("nearlow", S.near_low.to_numpy()), ("rest", ~S.near_low.to_numpy())):
            row[f"n_{lab}"] = int(msk.sum())
            for met, short in (("recall_topn2", "DCC12"), ("recall_ldca_topn2", "DCA4")):
                x = S[met].astype(float).to_numpy()
                ok = np.isfinite(x) & msk
                row[f"{short}_{lab}"] = boot_ratio(np.where(ok, x, 0), ok.astype(float), gi_s, G, W)
            if name != a.ref and a.ref in res:
                sub = lambda rr: SiteEvalResult(rr.sites[rr.sites.set_index(["structure_id", "site_id"]).index.isin(
                    S[msk].set_index(["structure_id", "site_id"]).index)], rr.structures, rr.pockets, rr.protocol)
                c = compare_site_methods(sub(r), sub(res[a.ref]), metrics=["recall_topn2", "recall_ldca_topn2"],
                                         n_boot=a.n_boot, include_p1k=False).set_index("metric")
                for met, short in (("recall_topn2", "DCC12"), ("recall_ldca_topn2", "DCA4")):
                    cc = c.loc[met]
                    row[f"d{short}_{lab}"] = f"{100 * cc['diff']:+.1f} [{100 * cc['ci_low']:+.1f},{100 * cc['ci_high']:+.1f}]"
        # per protein-disorder bin
        preds = load_prediction_dir(d, sids)
        K = r.pockets.copy()
        N = {sid: int(by[sid][0].n_sites) for sid in sids}
        K["in_topn2"] = K["rank"] <= K.structure_id.map(N) + 2
        for bn, _, _ in BINS:
            bs = [s for s in sids if pbin[s] == bn]
            if not bs:
                continue
            gi = np.array([gidx[groups[s]] for s in bs])
            Sb = S[S.pbin == bn]
            gsb = Sb.structure_id.map(groups).map(gidx).to_numpy()
            row[f"n_prot_{bn}"] = len(bs)
            row[f"DCC12_{bn}"] = boot_ratio(Sb.recall_topn2.astype(float).to_numpy(), np.ones(len(Sb)), gsb, G, W)
            Kb = K[K.structure_id.isin(bs)]
            t1 = Kb[Kb["rank"] == 1].set_index("structure_id")["tp"].reindex(bs, fill_value=False).astype(float).to_numpy()
            row[f"top1prec_{bn}"] = boot_ratio(t1, np.ones(len(bs)), gi, G, W)
            tn = Kb[Kb.in_topn2].groupby("structure_id").tp.agg(["sum", "count"]).reindex(bs, fill_value=0)
            row[f"topN2prec_{bn}"] = boot_ratio(tn["sum"].to_numpy(float), tn["count"].to_numpy(float), gi, G, W)
            k = len(bs)
            o = Kb.sort_values(["score", "structure_id", "rank"], ascending=[False, True, True])
            row[f"P@nprot_{bn}"] = f"{100 * o.tp.to_numpy()[:k].mean():.1f}" if len(o) else "nan"
            h = []
            for s in bs:
                pp = pocket_plddt(preds[s], ann[s][3], ann[s][2])
                h.append(hallucination_counts(preds[s], pp, by[s]))
            H = pd.DataFrame(h)
            row[f"halluc_{bn}"] = f"{100 * H.n_hallucinated.sum() / max(H.n_pockets.sum(), 1):.1f}"
            row[f"halluc_topn2_{bn}"] = f"{100 * H.n_hallucinated_topn2.sum() / max(H.n_pockets_topn2.sum(), 1):.1f}"
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 80):
        print(out.T.to_string())


if __name__ == "__main__":
    main()
