#!/usr/bin/env python
"""Evaluate prediction directories on the AF-dev site table (LIGYSIS-style site protocol).

  python scripts/af_dev_eval.py --methods naive=results/dev_af/preds/naive prune=... [--ref naive] \
      [--sites results/dev_af/af_dev_sites_all.tsv] [--out results/dev_af/eval_main.csv]

Per method: DCC12 top-N+2 (``recall_topn2``; closest pocket DCC <= 12 Å, rank <= N+2), DCA4 top-N+2
(``recall_ldca_topn2``), any-pocket DCA4 top-N+2, max recall, P1K (pooled), pockets/protein;
hallucination (pocket mean pLDDT < 50 on the *unpruned* model, matching no site at DCC <= 12 Å;
pocketdetect.af_eval) over all pockets and over the top-(N+2); cluster-bootstrap CIs (cluster30);
paired difference vs ``--ref``.  Strata (descriptive): sites of disorder-rich proteins (>= 10 % of
residues with pLDDT < 50) and sites near low confidence (ligand atom <= 8 Å of a pLDDT < 50 atom).
With ``--pdb`` the PDB-dev chain inputs are evaluated instead (no hallucination columns).
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
from pocketdetect.af_eval import hallucination_counts, pocket_plddt, residue_plddt, summarize_hallucination  # noqa: E402
from pocketdetect.evaluate import load_prediction_dir  # noqa: E402
from pocketdetect.site_eval import compare_site_methods, evaluate_sites, summarize_sites  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402
from pocketdetect.structure import load_structure  # noqa: E402

METRICS = ["recall_topn2", "recall_ldca_topn2", "recall_anyp_dca_topn2", "recall_max"]
_S = {}


def _load(args):
    sid, path = args
    s = load_structure(path, structure_id=sid, protein_only=True)
    s = s.select(s.protein_heavy_mask())
    return sid, s


def site_strata(by, structs):
    """{(sid, site_id): {'disorder_rich': bool, 'near_low': bool}}"""
    out = {}
    for sid, ss in by.items():
        s = structs[sid]
        pl = residue_plddt(s)
        rid = s.residue_ids
        low_res = np.array([pl[r] < 50 for r in rid])
        frac = low_res.mean() if len(low_res) else 0.0
        low_atoms = s.coords[low_res[s.res_index]]
        t = cKDTree(low_atoms) if len(low_atoms) else None
        for x in ss:
            pts = x.ligand_atoms if x.ligand_atoms is not None else x.center[None]
            near = bool(t is not None and t.query(pts, k=1)[0].min() <= 8.0)
            out[(sid, x.site_id)] = dict(disorder_rich=frac >= 0.10, near_low=near)
    return out


def fmt(v, lo, hi):
    return f"{100 * v:.1f} [{100 * lo:.1f},{100 * hi:.1f}]"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--sites", type=Path, default=ROOT / "results/dev_af/af_dev_sites_all.tsv")
    ap.add_argument("--pdb", action="store_true", help="PDB-chain evaluation (no pLDDT, no hallucination)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    st = read_sites(a.sites)
    by = sites_by_structure(st)
    sids = list(by)
    methods = [m.split("=", 1) for m in a.methods]
    structs = {}
    strata = {}
    if not a.pdb:
        paths = st.drop_duplicates("structure_id")[["structure_id", "structure_path"]].itertuples(index=False)
        with ProcessPoolExecutor(16) as ex:
            structs = dict(ex.map(_load, [tuple(x) for x in paths], chunksize=4))
        strata = site_strata(by, structs)
        plddt = {sid: residue_plddt(s) for sid, s in structs.items()}
    results = {}
    rows = []
    for name, d in methods:
        r = evaluate_sites(by, d)
        results[name] = r
        s = summarize_sites(r, n_boot=a.n_boot).set_index("metric")
        row = dict(method=name, n_sites=int(s.loc["n_scored_sites", "value"]),
                   pockets_mean=round(r.pockets.groupby("structure_id").size().reindex(sids, fill_value=0).mean(), 1),
                   n_missing=int(s.loc["n_pred_missing", "value"]))
        for m in METRICS + ["P1K"]:
            row[m] = fmt(*s.loc[m, ["value", "ci_low", "ci_high"]])
        # top-1 precision per protein (not pooled; P1K depth differs between sets: 1,000 pockets over
        # 278 AF-dev vs 2,844 AF-train models): fraction of structures whose rank-1 pocket hits a site
        K = r.pockets
        t1 = K[K["rank"] == 1].set_index("structure_id")["tp"].reindex(sids, fill_value=False).astype(float)
        row["prec_top1"] = f"{100 * t1.mean():.1f}"
        if strata:
            S = r.sites.copy()
            for k in ("disorder_rich", "near_low"):
                S[k] = [strata[(a_, b_)][k] for a_, b_ in zip(S.structure_id, S.site_id)]
                sub = S[S[k]]
                row[f"topn2_{k}"] = f"{100 * sub.recall_topn2.mean():.1f} (n={len(sub)})"
            preds = load_prediction_dir(d, sids)
            hrows = []
            for sid in sids:
                p = preds[sid]
                pp = pocket_plddt(p, structs[sid], plddt[sid])
                hrows.append(dict(structure_id=sid, **hallucination_counts(p, pp, by[sid])))
            H = pd.DataFrame(hrows)
            groups = {sid: by[sid][0].meta.get("group") for sid in sids}
            h = summarize_hallucination(H, groups, n_boot=a.n_boot).set_index("metric")
            row["halluc_rate"] = fmt(*h.loc["hallucination_rate", ["value", "ci_low", "ci_high"]])
            row["halluc_rate_topn2"] = fmt(*h.loc["hallucination_rate_topn2", ["value", "ci_low", "ci_high"]])
            row["n_halluc"] = int(H.n_hallucinated.sum())
            row["n_halluc_topn2"] = int(H.n_hallucinated_topn2.sum())
            row["low_plddt_pocket_rate"] = f"{100 * h.loc['low_plddt_pocket_rate', 'value']:.1f}"
        if a.ref and name != a.ref and a.ref in results:
            c = compare_site_methods(r, results[a.ref], metrics=METRICS[:3], n_boot=a.n_boot).set_index("metric")
            for m in METRICS[:3] + ["P1K"]:
                cc = c.loc[m]
                row[f"d_{m}"] = f"{100 * cc['diff']:+.1f} [{100 * cc['ci_low']:+.1f},{100 * cc['ci_high']:+.1f}]"
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
