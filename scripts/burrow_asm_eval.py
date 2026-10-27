#!/usr/bin/env python
"""v3 dev signal: score predictions on Track A assembly-1 inputs with the dimer-benchmark rules
(scripts/eval_dimer.py: j50 symmetric-copy grouping, label-free prediction collapse, a site is hit
if ANY kept prediction of rank <= k hits ANY copy; DCC 12 Å / DCA 4 Å).

Reports interface-site (homo + hetero) top-N / top-N+2 (DCC12, DCA4) and all-site top-N+2, with a
paired cluster bootstrap (Track A cluster30 of the parent entry; B = 1000) vs ``--ref``.

  python scripts/burrow_asm_eval.py --split train --methods v2=DIR p2rank=DIR --ref p2rank [--out CSV]
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_dimer import INTERFACE, evaluate_unit, fill_missing_residues, site_gt  # noqa: E402
from pocketdetect.burrow.biolip import base_id  # noqa: E402
from pocketdetect.evaluate import empty_predictions, read_predictions  # noqa: E402
from pocketdetect.seeds import BOOTSTRAP_SEED  # noqa: E402

GT = ROOT / "results/dev/biolip/asm/gt"
_G = {}


def _unit(args):
    f, d = args
    gt = json.loads(Path(f).read_text())
    g = site_gt(gt, "j50")
    p = Path(d) / f"{g.structure_id}.csv"
    pred = read_predictions(p, g.structure_id) if p.exists() else empty_predictions(g.structure_id, missing=True)
    pred = fill_missing_residues(pred, Path(gt["asm_path"]))
    rows, unit = evaluate_unit(g, pred)
    return rows, unit


def evaluate_dir(files, d, jobs):
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(_unit, [(f, d) for f in files], chunksize=8))
    S = pd.DataFrame([r for rr, _ in res for r in rr])
    U = pd.DataFrame([u for _, u in res])
    return S, U


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--only-with-interface", action="store_true", help="restrict to inputs with >= 1 interface site")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    files = sorted(str(p) for p in (GT / a.split).glob("*.json"))
    man = pd.concat([pd.read_csv(ROOT / f"data/processed/{s}/manifest.csv") for s in ("train", "dev")])
    grp = dict(zip(man.entry_id, man.cluster30))
    res = {}
    for spec in a.methods:
        name, d = spec.split("=", 1)
        S, U = evaluate_dir(files, d, a.jobs)
        S["group"] = S.structure_id.map(lambda s: grp.get(base_id(s), s))
        res[name] = (S, U)
    ref = a.ref or a.methods[0].split("=")[0]
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for name, (S, U) in res.items():
        Si = S[S.site_class.isin(INTERFACE)]
        if a.only_with_interface:
            keep = set(Si.structure_id)
            S = S[S.structure_id.isin(keep)]
        row = dict(method=name, split=a.split, n_inputs=U.shape[0], n_interface_sites=len(Si), n_sites=len(S),
                   pockets_kept=round(U.n_pred_kept.mean(), 1))
        for lab, sub in (("if", Si), ("all", S)):
            for c in ("success_DCC12_topn", "success_DCC12_topn2", "success_DCA4_topn2", "success_DCC12_all"):
                row[f"{lab}_{c.replace('success_', '')}"] = round(100 * sub[c].mean(), 1)
        if name != ref:
            R = res[ref][0]
            Ri = R[R.site_class.isin(INTERFACE)]
            for lab, A, Bf in (("if", Si, Ri), ("all", S, R)):
                key = ["structure_id", "site"]
                m = A[key + ["group", "success_DCC12_topn2", "success_DCA4_topn2"]].merge(
                    Bf[key + ["success_DCC12_topn2", "success_DCA4_topn2"]], on=key, suffixes=("_a", "_b"))
                for c in ("success_DCC12_topn2", "success_DCA4_topn2"):
                    dlt = m[f"{c}_a"].astype(float) - m[f"{c}_b"].astype(float)
                    g = m.group.astype(str).values
                    ug, gi = np.unique(g, return_inverse=True)
                    num = np.bincount(gi, weights=dlt.values, minlength=len(ug))
                    den = np.bincount(gi, minlength=len(ug)).astype(float)
                    W = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=1000)
                    bs = (W @ num) / (W @ den)
                    row[f"d_{lab}_{c.replace('success_', '')}_vs_{ref}"] = (
                        f"{100 * dlt.mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f},{100 * np.percentile(bs, 97.5):+.1f}]")
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    if a.out:
        if a.out.exists():
            df = pd.concat([pd.read_csv(a.out), df], ignore_index=True)
        df.to_csv(a.out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
