#!/usr/bin/env python
"""Timing screen for a `fast` preset: fast-backend featurization only (1 thread, warm) on the speed200
chains for candidate feature configs (BurrowConfig fields only; no reference code change).

  python scripts/burrow_fast_preset_timing.py [--n 200] [--configs C0,C1,...]
Writes results/dev/biolip/fast_preset/timing_screen.csv (median / p90 ms, median #points).
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

L3 = (6.0, 10.0, 16.0)
CONFIGS = {  # name: BurrowConfig overrides
    "C0_default": dict(n_rays=30, lengths=L3, subsample=3),
    "C1_r14_s4": dict(n_rays=14, lengths=L3, subsample=4),
    "C2_r14_s4_d5_p6_nometal": dict(n_rays=14, lengths=L3, subsample=4, max_dist=5.0, psp_length=6.0, metal=False),
    "C3_h125_r14_s3_d55_p6": dict(spacing=1.25, n_rays=14, lengths=L3, subsample=3, max_dist=5.5, psp_length=6.0),
    "C4_h15_r14_s2_p6": dict(spacing=1.5, n_rays=14, lengths=L3, subsample=2, psp_length=6.0),
    "C5_h125_r12_s3_d55_p6_nometal": dict(spacing=1.25, n_rays=12, lengths=L3, subsample=3, max_dist=5.5,
                                         psp_length=6.0, metal=False),
    "C6_h15_r12_s2_d55_p6_nometal": dict(spacing=1.5, n_rays=12, lengths=L3, subsample=2, max_dist=5.5,
                                        psp_length=6.0, metal=False),
}


def files(n):
    return [l for l in (ROOT / "baselines/ligysis_speed200.txt").read_text().splitlines() if l and not l.startswith("#")][:n]


def _run(args):
    name, n = args
    from pocketdetect.burrow.features import BurrowConfig
    from pocketdetect.burrow.fast.pipeline import featurize
    from pocketdetect.burrow.prep import load_protein
    cfg = BurrowConfig(**CONFIGS[name])
    prots = [load_protein(f) for f in files(n)]
    featurize(prots[0], cfg)  # JIT warm-up
    ts, npts = [], []
    for P in prots:
        t = time.perf_counter()
        ps = featurize(P, cfg)
        ts.append((time.perf_counter() - t) * 1e3)
        npts.append(len(ps.points))
    return dict(config=name, n=len(ts), feat_ms_median=np.median(ts), feat_ms_p90=np.percentile(ts, 90),
                points_median=np.median(npts), **{k: str(v) for k, v in CONFIGS[name].items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    a = ap.parse_args()
    names = a.configs.split(",")
    with mp.get_context("spawn").Pool(len(names)) as pool:
        rows = pool.map(_run, [(n, a.n) for n in names])
    df = pd.DataFrame(rows).round(1)
    out = ROOT / "results/dev/biolip/fast_preset"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "timing_screen.csv", index=False)
    print(df[["config", "n", "feat_ms_median", "feat_ms_p90", "points_median"]].to_string(index=False))


if __name__ == "__main__":
    main()
