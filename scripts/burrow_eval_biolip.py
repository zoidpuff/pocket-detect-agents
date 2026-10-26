#!/usr/bin/env python
"""BURROW dev evaluation on Track A's BioLiP2 LIGYSIS-like dev set (site protocol, pocketdetect.site_eval).

Cluster bootstrap over MMseqs2 30 % clusters (site-table ``group`` = cluster30); paired A − B vs a
reference method (default P2Rank 2.5.1) with P1K recomputed per resample.

  python scripts/burrow_eval_biolip.py --variant all --methods burrow=results/dev/biolip/preds/x [--ref p2rank] \
      --out results/dev/biolip/eval_all.csv
Metrics: recall_topn / recall_topn2 (LIGYSIS: closest pocket DCC ≤ 12 Å within top-N(+2)),
recall_ldca_topn2 (closest-by-DCA pocket ≤ 4 Å within top-N+2; C1-strict proxy),
recall_anyp_dca_topn2 (any top-N+2 pocket with DCA ≤ 4 Å), recall_max, P1K (pooled), #pockets.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.site_eval import compare_site_methods, evaluate_sites, summarize_sites  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402

METRICS = ["recall_topn", "recall_topn2", "recall_ldca_topn2", "recall_anyp_dca_topn2", "recall_max"]
BASE = {"p2rank": ROOT / "results/dev/biolip/preds/p2rank", "fpocket": ROOT / "results/dev/biolip/preds/fpocket"}


def evaluate_dir(by, groups, d):
    return evaluate_sites(by, str(d), groups=groups)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["all", "small"], default="all")
    ap.add_argument("--methods", nargs="*", default=[])
    ap.add_argument("--ref", default="p2rank")
    ap.add_argument("--baselines", action="store_true", help="also report p2rank and fpocket rows")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    st = read_sites(ROOT / f"results/dev/biolip/dev_sites_{a.variant}.tsv")
    by = sites_by_structure(st)
    groups = {sid: ss[0].meta.get("group") for sid, ss in by.items()}
    ref = evaluate_dir(by, groups, BASE[a.ref])
    methods = [m.split("=", 1) for m in a.methods]
    if a.baselines:
        methods = [[k, str(v)] for k, v in BASE.items()] + methods
    rows = []
    for name, d in methods:
        r = ref if (name == a.ref) else evaluate_dir(by, groups, Path(d))
        s = summarize_sites(r, n_boot=a.n_boot).set_index("metric")
        row = dict(method=name, variant=a.variant, n_sites=int(s.loc["n_scored_sites", "value"]),
                   pockets_mean=round(r.pockets.groupby("structure_id").size().reindex(list(by), fill_value=0).mean(), 1))
        for m in METRICS + ["P1K"]:
            row[m] = f"{100 * s.loc[m, 'value']:.1f} [{100 * s.loc[m, 'ci_low']:.1f},{100 * s.loc[m, 'ci_high']:.1f}]"
        if name != a.ref:
            c = compare_site_methods(r, ref, metrics=METRICS[:4], n_boot=a.n_boot).set_index("metric")
            for m in METRICS[:4] + ["P1K"]:
                if m in c.index:
                    cc = c.loc[m]
                    lo, hi = [k for k in c.columns if "lo" in k][0], [k for k in c.columns if "hi" in k][0]
                    row[f"d_{m}"] = f"{100 * cc['diff']:+.1f} [{100 * cc[lo]:+.1f},{100 * cc[hi]:+.1f}]"
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
