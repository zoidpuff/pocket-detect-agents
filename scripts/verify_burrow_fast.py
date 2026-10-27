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
    ap.add_argument("--manifest", type=Path, default=ROOT / "results/dev/biolip/inputs_audited.csv")
    ap.add_argument("--model", type=Path, default=ROOT / "results/dev/freeze/models/burrow_all_default_s0.json")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--ranker", default="learned")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--path-col", default="structure_path")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--af-mode", action="store_true",
                    help="AF mode (pruning + soft occlusion weights) on AF models; manifest column model_path")
    a = ap.parse_args(argv)
    df = read_manifest(a.manifest).sample(a.n, random_state=SEEDS[1])
    model = BurrowModel.load(a.model)
    fm = FP.FastModel.from_model(model)
    res = lambda p: str(p if Path(p).is_absolute() else ROOT / p)
    Ws = [None] * len(df)
    n_exact70 = 0
    if a.af_mode:
        from pocketdetect.structure import load_structure
        from pocketdetect.burrow.af import af_protein
        Ps, Ws = [], []
        for s, p in zip(df.structure_id, df[a.path_col]):
            P, w, _ = af_protein(load_structure(res(p), structure_id=s))
            Ps.append(P)
            Ws.append(w)
            n_exact70 += bool(w is not None and np.any((w == 0.75) | (w == 0.5)))
    else:
        Ps = [load_protein(res(p), structure_id=s) for s, p in zip(df.structure_id, df.structure_path)]
    if a.device == "cpu":
        fast = [FP.predict_protein(P, fm, ranker=a.ranker, weights=w) for P, w in zip(Ps, Ws)]
    else:
        from pocketdetect.burrow.fast.gpu import predict_batch
        fast = predict_batch(Ps, fm, device=a.device, ranker=a.ranker, weights=Ws)
    rows, n_top, n_exact = [], 0, 0
    for P, f, w in zip(Ps, fast, Ws):
        r = ref_predict(P, model, ranker=a.ranker, weights=w)
        top, ex = compare(r, f)
        n_top += top
        n_exact += ex
        rows.append(dict(id=P.id, n_ref=len(r.scores), n_fast=len(f.scores), top10_ok=top, exact=ex))
        if not top:
            print("TOP10 MISMATCH", P.id, r.scores[:5], f.scores[:5], flush=True)
    summ = dict(n=len(Ps), model=a.model.name, device=a.device, ranker=a.ranker, af_mode=a.af_mode,
                n_with_exact_w075_or_w05_atoms=n_exact70, top10_identical=n_top, exact=n_exact,
                date=time.strftime("%Y-%m-%d %H:%M"))
    print(json.dumps(summ))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))
    return 0 if n_top == len(Ps) else 1


if __name__ == "__main__":
    sys.exit(main())
