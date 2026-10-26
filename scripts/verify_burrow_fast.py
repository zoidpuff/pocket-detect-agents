#!/usr/bin/env python
"""Equivalence of the fast BURROW backend vs the reference implementation on dev structures.

  python scripts/verify_burrow_fast.py --n 60 [--device cpu|cuda:0] [--ranker learned|geometry]
Criterion (PI): identical top-10 pocket centres (<= 0.1 Å) and ranks.  Also reports exact equality
of all outputs (scores, centres, residue sets).  Structures: fixed random sample of the dev manifest.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.devdata import read_manifest  # noqa: E402
from pocketdetect.burrow.model import BurrowModel  # noqa: E402
from pocketdetect.burrow.pipeline import predict_protein as ref_predict  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402
from pocketdetect.burrow.fast import pipeline as FP  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402


def compare(ref, fast, k=10, tol=0.1):
    kk = min(k, len(ref.scores))
    top_ok = (len(fast.scores) >= kk and np.all(np.abs(ref.centers[:kk] - fast.centers[:kk]).max(1, initial=0) <= tol)
              if kk else len(fast.scores) == 0)
    exact = (len(ref.scores) == len(fast.scores) and np.array_equal(ref.scores, fast.scores)
             and np.array_equal(ref.centers, fast.centers) and list(ref.residues) == list(fast.residues))
    return bool(top_ok), bool(exact)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "results/dev/manifests/all.csv")
    ap.add_argument("--model", type=Path, default=ROOT / "results/dev/models/burrow_chen11_s0.json")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--ranker", default="learned")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    df = read_manifest(a.manifest).sample(a.n, random_state=SEEDS[1])
    model = BurrowModel.load(a.model)
    fm = FP.FastModel.from_model(model)
    Ps = [load_protein(p, structure_id=s) for s, p in zip(df.structure_id, df.structure_path)]
    if a.device == "cpu":
        fast = [FP.predict_protein(P, fm, ranker=a.ranker) for P in Ps]
    else:
        from pocketdetect.burrow.fast.gpu import predict_batch
        fast = predict_batch(Ps, fm, device=a.device, ranker=a.ranker)
    rows, n_top, n_exact = [], 0, 0
    for P, f in zip(Ps, fast):
        r = ref_predict(P, model, ranker=a.ranker)
        top, ex = compare(r, f)
        n_top += top
        n_exact += ex
        rows.append(dict(id=P.id, n_ref=len(r.scores), n_fast=len(f.scores), top10_ok=top, exact=ex))
        if not top:
            print("TOP10 MISMATCH", P.id, r.scores[:5], f.scores[:5], flush=True)
    summ = dict(n=len(Ps), device=a.device, ranker=a.ranker, top10_identical=n_top, exact=n_exact,
                date=time.strftime("%Y-%m-%d %H:%M"))
    print(json.dumps(summ))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))
    return 0 if n_top == len(Ps) else 1


if __name__ == "__main__":
    sys.exit(main())
