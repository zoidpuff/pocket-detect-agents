#!/usr/bin/env python
"""Evaluate BURROW variants and baselines on the interim dev sets with the project harness.

Subsets: chen11, joined, joined_novel (JOINED structures whose MMseqs2 30 % group contains no
CHEN11 structure).  Criteria DCA4 (P2Rank) and LDCC12 (LIGYSIS-style: closest pocket by DCC);
success at top-n and top-(n+2), ligand-averaged, 95 % bootstrap CI over structures.
Baseline prediction files ``<sid>.pockets.csv`` and BURROW ``<sid>.csv`` are both accepted.

  python scripts/burrow_eval_dev.py --methods p2rank=results/dev/baselines/p2rank burrow=results/dev/preds/x ...
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # shared per-user thread limit; LightGBM gets num_threads explicitly

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.devdata import load_gts, read_manifest  # noqa: E402
from pocketdetect.evaluate import empty_predictions, evaluate, read_predictions  # noqa: E402

METRICS = ["success_DCA4_topn", "success_DCA4_topn2", "success_LDCC12_topn", "success_LDCC12_topn2",
           "success_DCC4_topn"]


def load_dir(d: Path, sids):
    out = {}
    for s in sids:
        for f in (d / f"{s}.csv", d / f"{s}.pockets.csv"):
            if f.exists():
                out[s] = read_predictions(f, s)
                break
        else:
            out[s] = empty_predictions(s, missing=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "results/dev/manifests/all.csv")
    ap.add_argument("--groups", type=Path, default=ROOT / "results/dev/manifests/groups_all_mmseqs30.json")
    ap.add_argument("--methods", nargs="+", required=True, help="name=pred_dir")
    ap.add_argument("--subsets", default="chen11,joined,joined_novel")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    df = read_manifest(a.manifest)
    gts = load_gts(a.manifest)
    groups = json.loads(a.groups.read_text())
    sub = dict(zip(df.structure_id, df.subset))
    chen_groups = {groups[s] for s in df.structure_id if sub[s] == "chen11"}
    subsets = {
        "chen11": [s for s in df.structure_id if sub[s] == "chen11"],
        "joined": [s for s in df.structure_id if sub[s] != "chen11"],
        "joined_novel": [s for s in df.structure_id if sub[s] != "chen11" and groups[s] not in chen_groups],
    }
    rows = []
    for spec in a.methods:
        name, d = spec.split("=", 1)
        for sname in a.subsets.split(","):
            sids = subsets[sname]
            preds = load_dir(Path(d), sids)
            n_missing = sum(p.missing if hasattr(p, "missing") else 0 for p in preds.values())
            res = evaluate([gts[s] for s in sids], preds, criteria=["DCA4", "DCC4", "LDCC12"], n_boot=a.n_boot)
            sm = res.summary
            row = dict(method=name, subset=sname, n_struct=res.info["n_structures"] - res.info["n_structures_no_ligands"],
                       n_lig=res.info["n_ligands"], n_pred_missing=res.info["n_pred_missing"],
                       pockets_mean=round(sum(p.n for p in preds.values()) / len(preds), 1))
            for m in METRICS:
                r = sm[(sm.metric == m) & (sm.average == "ligand")].iloc[0]
                row[m] = f"{100 * r.value:.1f} [{100 * r.ci_low:.1f},{100 * r.ci_high:.1f}]"
            rows.append(row)
            print(row, flush=True)
    out = pd.DataFrame(rows)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
