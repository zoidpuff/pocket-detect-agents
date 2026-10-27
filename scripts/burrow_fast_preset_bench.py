#!/usr/bin/env python
"""Timing of model JSONs with the fast backend, 1 core, warm, on the speed200 chains (the set used in
docs/PERF.md): total per structure EXCLUDING parsing (median / p90) and stage medians.

  python scripts/burrow_fast_preset_bench.py name=model.json [name=model.json ...] [--n 200]
Writes results/dev/biolip/fast_preset/bench.csv (appends).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results/dev/biolip/fast_preset/bench.csv"


def _run(args):
    name, path, n = args
    from pocketdetect.burrow.fast.pipeline import FastModel, predict_protein
    from pocketdetect.burrow.prep import load_protein
    files = [l for l in (ROOT / "baselines/ligysis_speed200.txt").read_text().splitlines()
             if l and not l.startswith("#")][:n]
    fm = FastModel.load(path)
    prots = []
    parse = []
    for f in files:
        t = time.perf_counter()
        prots.append(load_protein(f))
        parse.append((time.perf_counter() - t) * 1e3)
    for P in prots[:3]:
        predict_protein(P, fm)                       # warm-up (JIT, caches)
    rows = []
    for P in prots:
        pr = predict_protein(P, fm)
        rows.append(dict(pr.timings, n_pockets=len(pr.scores)))
    t = pd.DataFrame(rows)
    out = dict(model=name, n=len(t), total_ms_median=t.total.median(), total_ms_p90=t.total.quantile(0.9),
               parse_ms_median=float(np.median(parse)), pockets_median=t.n_pockets.median())
    for c in ("features", "point_score", "pockets", "rank"):
        if c in t:
            out[f"{c}_ms"] = t[c].median()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    specs = [m.split("=", 1) for m in a.models]
    with mp.get_context("spawn").Pool(len(specs)) as pool:
        rows = pool.map(_run, [(n, p, a.n) for n, p in specs])
    df = pd.DataFrame(rows).round(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    (pd.concat([pd.read_csv(OUT), df]) if OUT.exists() else df).to_csv(OUT, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
