#!/usr/bin/env python
"""Reviewer gaming checks on the BioLiP2 dev set (ranker `all`, dev_sites_all.tsv).

1. pocket count / residues per pocket / centre burial (#heavy atoms within 8 A, distance to nearest atom) for all
   pockets and for the top-(N+2) pockets, per method;
2. top-N+2 recall vs DCC threshold (4..12 A) with the shared harness (closest-pocket rule) and the 1:1 variant;
3. DCC distribution of successful (top-N+2) matches.
Writes results/review_dev/gaming_*.csv
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.site_eval import SiteProtocol, evaluate_sites  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402

B = ROOT / "results/dev/biolip"
C = ROOT / "results/review_dev/controls"
METHODS = {"p2rank": B / "preds/p2rank", "fpocket": B / "preds/fpocket", "burrow_default_s0": B / "preds_cli/burrow_all_default_s0",
           "burrow_accurate_s0": B / "preds_cli/burrow_all_accurate_s0", "tile12_bur": C / "tile12_bur", "burnms10": C / "burnms10"}
OUT = ROOT / "results/review_dev"


def struct_stats(args):
    sid, N = args
    st = gemmi.read_structure(str(B / "inputs" / f"{sid}.cif"))
    st.remove_hydrogens()
    xyz = np.array([a.pos.tolist() for ch in st[0] for r in ch for a in r])
    t = cKDTree(xyz)
    rows = []
    for m, d in METHODS.items():
        f = Path(d) / f"{sid}.csv"
        if not f.exists():
            rows.append(dict(structure_id=sid, method=m, missing=True))
            continue
        p = pd.read_csv(f).sort_values("rank")
        if len(p) == 0:
            rows.append(dict(structure_id=sid, method=m, n_pockets=0))
            continue
        c = p[["center_x", "center_y", "center_z"]].to_numpy(float)
        dn, _ = t.query(c)
        n8 = np.array([len(x) for x in t.query_ball_point(c, 8.0)])
        nres = p.residues.fillna("").astype(str).map(lambda s: len(s.split())).to_numpy()
        top = np.arange(len(p)) < N + 2
        rows.append(dict(structure_id=sid, method=m, n_pockets=len(p), n_res_mean=nres.mean(), n_res_top=nres[top].mean(),
                         frac_empty_res=(nres == 0).mean(), frac_empty_res_top=(nres[top] == 0).mean(),
                         dnear_mean=dn.mean(), dnear_top=dn[top].mean(), frac_dnear_gt4=(dn > 4).mean(), frac_dnear_gt4_top=(dn[top] > 4).mean(),
                         n8_mean=n8.mean(), n8_top=n8[top].mean(), n_res_struct=len({(ch.name, r.seqid.num) for ch in st[0] for r in ch})))
    return rows


def main():
    st = read_sites(B / "dev_sites_all.tsv")
    by = sites_by_structure(st)
    jobs = [(sid, int(ss[0].n_sites)) for sid, ss in by.items()]
    with ProcessPoolExecutor(12) as ex:
        rows = [r for rr in ex.map(struct_stats, jobs, chunksize=4) for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gaming_per_structure.csv", index=False)
    agg = df.groupby("method").agg(n_pockets_mean=("n_pockets", "mean"), n_pockets_median=("n_pockets", "median"),
                                   frac_at_cap50=("n_pockets", lambda x: (x >= 50).mean()),
                                   res_per_pocket_all=("n_res_mean", "mean"), res_per_pocket_topN2=("n_res_top", "mean"),
                                   frac_empty_res_all=("frac_empty_res", "mean"), frac_empty_res_topN2=("frac_empty_res_top", "mean"),
                                   dist_nearest_atom_all=("dnear_mean", "mean"), dist_nearest_atom_topN2=("dnear_top", "mean"),
                                   frac_centre_gt4A_from_atoms_topN2=("frac_dnear_gt4_top", "mean"),
                                   atoms_within8_all=("n8_mean", "mean"), atoms_within8_topN2=("n8_top", "mean")).round(3)
    agg.to_csv(OUT / "gaming_pocket_stats.csv")
    print(agg.T.to_string())
    groups = {sid: ss[0].meta.get("group") for sid, ss in by.items()}
    th = []
    for m, d in METHODS.items():
        for t in (4, 6, 8, 10, 12, 14):
            r = evaluate_sites(by, str(d), protocol=SiteProtocol(dcc=float(t)), groups=groups)
            s = r.sites
            th.append(dict(method=m, dcc=t, recall_topn2=s.recall_topn2.mean(), recall_1to1_topn2=s.recall_1to1_topn2.mean(),
                           recall_anyp_topn2=s.recall_anyp_topn2.mean(), recall_topn=s.recall_topn.mean()))
            if t == 12:
                ok = s[s.recall_topn2]
                th[-1].update(hit_dcc_median=ok.dcc_closest.median(), hit_frac_dcc_gt8=(ok.dcc_closest > 8).mean())
    th = pd.DataFrame(th).round(4)
    th.to_csv(OUT / "gaming_dcc_threshold.csv", index=False)
    print(th.to_string())


if __name__ == "__main__":
    main()
