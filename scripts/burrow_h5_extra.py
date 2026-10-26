#!/usr/bin/env python
"""Supplementary H5 report (D12): max recall, pockets/structure, one-to-one P1K and any-pocket top-N+2
for v1 and H5 on the four v2 dev sets, from the prediction directories written by burrow_v2_eval.py."""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.site_eval import evaluate_sites, summarize_sites  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402

B = ROOT / "results/dev/biolip"
GT_APO = ROOT / "data/processed/gt/cryptobench_trainfolds"
csv.field_size_limit(1 << 30)


def main():
    uni = {}
    for f in ("sites.tsv", "sites_noncryptic.tsv"):
        for r in csv.DictReader(open(GT_APO / f), delimiter="\t"):
            uni[r["structure_id"]] = r["uniprot"].split()[0]
    rows = []
    for v in ("small", "all"):
        tabs = [(f"biolip_dev_{v}", B / f"dev_sites_{v}.tsv", "v2_preds", None),
                (f"apo_track_dev_{v}", B / f"apo_track_dev/sites_{v}.tsv", "v2_apotd_preds", None),
                ("af_dev_all", ROOT / "results/dev_af/af_dev_sites_all.tsv", "af_preds", None),
                ("apo_cryptic", GT_APO / "sites.tsv", "v2_apo_preds", uni),
                ("apo_noncryptic", GT_APO / "sites_noncryptic.tsv", "v2_apo_preds", uni)]
        for t, f, sub, groups in tabs:
            by = sites_by_structure(read_sites(f))
            g = groups or {s: ss[0].meta.get("group") for s, ss in by.items()}
            for nm in (f"v1_{v}_default_s0", f"H5_{v}_default_s0"):
                d = B / sub / nm
                if not d.exists():
                    continue
                r = evaluate_sites(by, str(d), groups=g)
                s = summarize_sites(r, n_boot=0).set_index("metric")["value"]
                rows.append(dict(ranker=v, set=t, model=nm.split("_")[0], topn2=100 * s["recall_topn2"],
                                 anyp_topn2=100 * s["recall_anyp_topn2"], max_recall=100 * s["recall_max"],
                                 P1K=100 * s["P1K"], P1K_1to1=100 * s["P1K_1to1"],
                                 pockets=r.structures.n_pred.mean()))
    df = pd.DataFrame(rows).round(1)
    df.to_csv(B / "h5_extra.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
