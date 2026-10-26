#!/usr/bin/env python
"""Re-emit predictions from a stored BURROW dev run with different post-ranking settings
(NMS / overlap / score threshold / max pockets) — no retraining.

  python scripts/burrow_emit.py burrow_chen11_s0 --out-name x --set nms_overlap=0.5 min_score=0.1
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # shared per-user thread limit; LightGBM gets num_threads explicitly

import argparse
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_train_dev import emit  # noqa: E402
from pocketdetect.burrow.model import FormationConfig  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--noranker", action="store_true", help="rank by Σ vol·point score instead of the ranker")
    ap.add_argument("--out-root", type=Path, default=ROOT / "results/dev/preds")
    a = ap.parse_args(argv)
    store = pickle.loads((ROOT / "results/dev/cache/runs" / f"{a.run}.pkl").read_bytes())
    fc = FormationConfig()
    kv = {}
    for s in a.set:
        k, v = s.split("=")
        kv[k] = None if v == "None" else type(getattr(fc, k) if getattr(fc, k) is not None else 0.0)(v)
    fc = replace(fc, **kv)
    out = a.out_root / a.out_name
    for sid, (light, rs, ss, y) in store.items():
        emit(out, sid, light, ss if a.noranker else rs, fc)
    print(out, fc)


if __name__ == "__main__":
    main()
