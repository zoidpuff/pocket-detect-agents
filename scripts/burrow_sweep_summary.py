#!/usr/bin/env python
"""Speed/accuracy sweep summary for BURROW presets (BioLiP2 dev, ranker `all`, seed 0).

Accuracy: harness site metrics (pocketdetect.site_eval) on results/dev/biolip/dev_sites_all.tsv.
Speed: *estimated* fast-backend ms/structure/core, scaled from the fast-backend engineer's
per-stage measurements of the reference design (60 rays x 24 Å, 2 Å sub-lattice, 250x63 point
model, all points scored):  grid+PSP 30, rays 80, other point features 15, point model 55,
post-feature (pockets+ranker+NMS) 24 ms.  Scaling:
  points-proportional stages x (mean #points of the setting / reference)
  rays x n_rays/60 x max_len/24;  point model x trees/250 x log2(leaves)/6 x scored fraction.
These are estimates for choosing a preset, not measurements (the fast backend measures).

  python scripts/burrow_sweep_summary.py
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.site_eval import evaluate_sites, summarize_sites  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402

B = ROOT / "results/dev/biolip"
REF_MS = dict(grid=30.0, rays=80.0, other=15.0, point=55.0, post=24.0)
SETTINGS = {  # name: (pred dir, n_rays, max_len, timing file key, trees, leaves, score_floor)
    "REF": ("burrow_all_s0", 60, 24, "REF", 250, 63, None),
    "B": ("burrow_all_s0_B", 60, 24, "REF", 100, 31, None),
    "C": ("burrow_all_s0_C", 60, 24, "REF", 100, 31, 0.7),
    "F1": ("burrow_all_s0_F1", 30, 24, "F1", 100, 31, None),
    "F2": ("burrow_all_s0_F2", 60, 16, "F2", 100, 31, None),
    "F3": ("burrow_all_s0_F3", 60, 24, "F3", 100, 31, None),
    "F4": ("burrow_all_s0_F4", 30, 16, "F4", 100, 31, 0.7),
    "F5/default s0": ("burrow_all_default_s0", 30, 16, "F4", 250, 63, 0.7),
    "REF/accurate s0": ("burrow_all_accurate_s0", 60, 24, "REF", 250, 63, None),
}


def scored_fraction(tkey, floor):
    """Fraction of a setting's points with b >= floor (or metal candidates): measured on dev
    samples — 0.31 with the 2 Å sub-lattice (b over 24 Å rays), 0.45 with the 3 Å sub-lattice +
    16 Å rays (the coarser lattice drops mostly exposed points)."""
    if floor is None:
        return 1.0
    return 0.45 if tkey in ("F3", "F4") else 0.31


def main():
    by = sites_by_structure(read_sites(B / "dev_sites_all.tsv"))
    groups = {sid: ss[0].meta.get("group") for sid, ss in by.items()}
    ref_t = pd.read_csv(B / "timing_feat_dev_REF.csv")
    # point-count reference: a sub-lattice-2 timing file of the same dev set as the variants
    # (F1: 30 rays, same candidate points as REF); the REF file predates the dev re-split
    same_era = B / "timing_feat_dev_F1.csv"
    ref_pts = (pd.read_csv(same_era) if same_era.exists() else ref_t).query("status == 'ok'").n_points.mean()
    rows = []
    for name, (pred, rays, L, tkey, trees, leaves, floor) in SETTINGS.items():
        d = B / "preds" / pred
        if not d.exists():
            continue
        tf = B / f"timing_feat_dev_{tkey}.csv"
        t = pd.read_csv(tf) if (tf.exists() and tkey != "REF") else pd.read_csv(same_era)
        pr = t[t.status == "ok"].n_points.mean() / ref_pts
        fs = scored_fraction(tkey, floor)
        ms = (REF_MS["grid"] * pr + REF_MS["rays"] * pr * rays / 60 * L / 24 + REF_MS["other"] * pr
              + REF_MS["point"] * pr * trees / 250 * math.log2(leaves) / 6 * fs + REF_MS["post"])
        r = evaluate_sites(by, str(d), groups=groups)
        s = summarize_sites(r, n_boot=0).set_index("metric")["value"]
        rows.append(dict(setting=name, rays=rays, max_len=L, subsample=3 if tkey in ("F3", "F4") else 2,
                         trees=trees, leaves=leaves, score_floor=floor, points_ratio=round(pr, 2),
                         est_ms=round(ms), topn=100 * s["recall_topn"], topn2=100 * s["recall_topn2"],
                         dca4_topn2=100 * s["recall_ldca_topn2"], P1K=100 * s["P1K"],
                         P1K_1to1=100 * s["P1K_1to1"]))
    df = pd.DataFrame(rows).round(1)
    df.to_csv(B / "speed_accuracy_sweep.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
