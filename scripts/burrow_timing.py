#!/usr/bin/env python
"""Per-stage BURROW CPU timing, 1 core (NUMBA_NUM_THREADS=1), warm (JIT compiled beforehand).

  python scripts/burrow_timing.py --n 60 --out results/dev/timing_cpu1.csv
Structures: a fixed random sample (seed SEEDS[0]) of the dev manifest.  Parse time is reported
separately (gemmi read + polymer selection + chemistry tables).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.devdata import read_manifest  # noqa: E402
from pocketdetect.burrow.model import BurrowModel  # noqa: E402
from pocketdetect.burrow.pipeline import predict_protein  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "results/dev/manifests/all.csv")
    ap.add_argument("--model", type=Path, default=ROOT / "results/dev/models/burrow_chen11_s0.json")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", type=Path, default=ROOT / "results/dev/timing_cpu1.csv")
    a = ap.parse_args(argv)
    df = read_manifest(a.manifest).sample(a.n, random_state=SEEDS[0])
    model = BurrowModel.load(a.model)
    P = load_protein(df.structure_path.iloc[0])
    predict_protein(P, model)  # warm-up / JIT
    rows = []
    for sid, path in zip(df.structure_id, df.structure_path):
        t0 = time.perf_counter()
        P = load_protein(path, structure_id=sid)
        parse = (time.perf_counter() - t0) * 1e3
        for mode in ("learned", "geometry"):
            pr = predict_protein(P, model, ranker=mode)
            rows.append(dict(structure_id=sid, mode=mode, n_atoms=len(P.coords), parse_ms=parse, **pr.timings))
    t = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)
    cols = [c for c in t.columns if c.endswith("_ms") or c.startswith("feat_") or c in
            ("features", "point_score", "pockets", "rank", "total")]
    print(t.groupby("mode")[["n_atoms"] + cols].median().T.round(1).to_string())
    print("mean:\n", t.groupby("mode")[["total", "parse_ms"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
