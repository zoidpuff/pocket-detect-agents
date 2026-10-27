#!/usr/bin/env python
"""v3b single dev confirmation: apply an output rule to the full-list (no-floor) predictions of the frozen
models and compare, paired, with the v2-floor predictions of the same models, on BioLiP dev (non-ion and all
sites), apo-dev (CryptoBench train folds, cryptic / non-cryptic), apo-track-dev and AF dev.

  python scripts/burrow_v3b_dev.py --rule c16_0.005
Writes results/dev/biolip/v3b/dev_<rule>.csv and the filtered prediction dirs under results/dev/biolip/v3b/preds/.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.output_rules import filter_prediction_files  # noqa: E402
from pocketdetect.site_eval import compare_site_methods, evaluate_sites, summarize_sites  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402

B = ROOT / "results/dev/biolip"
GT_APO = ROOT / "data/processed/gt/cryptobench_trainfolds"
csv.field_size_limit(1 << 30)


def dirs(preset, v):
    """(full-list dir, v2 dir) per set for a preset/ranker."""
    if preset == "default":
        full = {"biolip": B / f"preds_freeze/burrow_{v}_default_s0", "apo": B / f"v2_apo_preds/v1_{v}_default_s0",
                "apotd": B / f"v2_apotd_preds/v1_{v}_default_s0", "af": B / f"af_preds/v1_{v}_default_s0"}
        ref = f"H5_{v}_default_s0"
    else:
        full = {"biolip": B / f"v2_preds/fastfull_{v}_s0", "apo": B / f"v2_apo_preds/fastfull_{v}_s0",
                "apotd": B / f"v2_apotd_preds/fastfull_{v}_s0", "af": B / f"af_preds/fastfull_{v}_s0"}
        ref = f"fast_{v}_s0"
    v2 = {"biolip": B / f"v2_preds/{ref}", "apo": B / f"v2_apo_preds/{ref}", "apotd": B / f"v2_apotd_preds/{ref}",
          "af": B / f"af_preds/{ref}"}
    return full, v2


def apply(rule, src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in src.glob("*.csv"):
        if f.name == "timing.csv" or f.name.endswith(".points.csv") or \
                not f.read_text()[:30].startswith("rank,score,center_x"):
            continue
        shutil.copy(f, dst / f.name)
        filter_prediction_files(dst, f.stem, rule, 300.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--presets", default="default,fast")
    a = ap.parse_args()
    uni = {}
    for f in ("sites.tsv", "sites_noncryptic.tsv"):
        for r in csv.DictReader(open(GT_APO / f), delimiter="\t"):
            uni[r["structure_id"]] = r["uniprot"].split()[0]
    rows = []
    for preset in a.presets.split(","):
        for v in ("small", "all"):
            full, v2 = dirs(preset, v)
            new = {}
            for k, src in full.items():
                new[k] = B / "v3b/preds" / f"{preset}_{v}_{k}_{a.rule}"
                apply(a.rule, src, new[k])
            tables = [("biolip_nonion", "biolip", B / "dev_sites_small.tsv", None),
                      ("apo_cryptic", "apo", GT_APO / "sites.tsv", uni), ("apo_noncryptic", "apo", GT_APO / "sites_noncryptic.tsv", uni),
                      (f"apo_track_dev_{v}", "apotd", B / f"apo_track_dev/sites_{v}.tsv", None),
                      ("af_dev_all", "af", ROOT / "results/dev_af/af_dev_sites_all.tsv", None)]
            if v == "all":
                tables.insert(1, ("biolip_all", "biolip", B / "dev_sites_all.tsv", None))
            for name, k, tab, groups in tables:
                by = sites_by_structure(read_sites(tab))
                g = groups or {s: ss[0].meta.get("group") for s, ss in by.items()}
                rn, rr = evaluate_sites(by, str(new[k]), groups=g), evaluate_sites(by, str(v2[k]), groups=g)
                sn = summarize_sites(rn, n_boot=0).set_index("metric")["value"]
                sr = summarize_sites(rr, n_boot=0).set_index("metric")["value"]
                mets = [m for m in ("recall_topn2", "recall_ldca_topn2", "recall_max") if m in sn.index]
                c = compare_site_methods(rn, rr, metrics=mets, n_boot=1000).set_index("metric")
                row = dict(preset=preset, ranker=v, table=name, rule=a.rule,
                           pockets_new=round(rn.structures.n_pred.mean(), 1), pockets_v2=round(rr.structures.n_pred.mean(), 1))
                for m in mets + ["P1K", "P1K_1to1"]:
                    row[m] = round(100 * sn[m], 1)
                    row[f"v2_{m}"] = round(100 * sr[m], 1)
                    if m in c.index:
                        row[f"d_{m}"] = f"{100 * c.loc[m, 'diff']:+.1f} [{100 * c.loc[m, 'ci_low']:+.1f},{100 * c.loc[m, 'ci_high']:+.1f}]"
                rows.append(row)
                print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(B / f"v3b/dev_{a.rule}.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
