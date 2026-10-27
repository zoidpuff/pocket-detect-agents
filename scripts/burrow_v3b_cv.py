#!/usr/bin/env python
"""v3b (track T2): relative / size-scaled / per-region output floors, selected on TRAIN cluster-CV only.

All rules are post-filters on the score-ordered, NMS'd full pocket list (v1 list, no floor) — exactly how
the frozen v2 floor acts (keep the first max(min_keep, #score >= tau) pockets):
  v2         keep rank < 3 or score >= 0.03
  a<rho>     keep rank < 3 or score >= min(0.03, rho * max score of the protein)          (relative floor)
  b<k>       keep rank < max(3, k * ceil(n_res / 150)) or score >= 0.03                    (size-scaled min_keep)
  c<R>_<lo>  v2, plus any pocket with score >= lo whose centre is > R Å from every kept pocket (per region)
  combos     a + c
Metrics (held-out chain inputs, closest-pocket protocol as site_eval): top-N, top-N+2, DCA4 top-N+2,
one-to-one P1K, max recall, pockets/structure.  OOF ranker scores per fold (as burrow_tau_cv.py).

  python scripts/burrow_v3b_cv.py --run p3_small_default_s0 --variant small
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_ranker_cv import VARIANTS, build_X, train_ranker  # noqa: E402
from burrow_tau_cv import metrics  # noqa: E402
from pocketdetect.burrow.biolip import ASM, entries, sites  # noqa: E402
from pocketdetect.burrow.model import FormationConfig  # noqa: E402
from pocketdetect.burrow.output_rules import apply_rule  # noqa: E402
from pocketdetect.burrow.pipeline import finalize  # noqa: E402
from pocketdetect.datasets.trainset import load_entry  # noqa: E402

RULES = (["v1", "v2"] + [f"a{r}" for r in (0.1, 0.2, 0.3)] + [f"b{k}" for k in (1, 2)]
         + [f"c{R}_{lo}" for R in (12, 16, 20) for lo in (0.005, 0.01)]
         + [f"a{r}+c{R}_{lo}" for r in (0.2,) for R in (16,) for lo in (0.005, 0.01)])
_G = {}


class _P:
    def __init__(self, c, r, idx):
        self.center, self.residues, self.idx = c, r, idx


def _job(args):
    sid, rule = args
    light, rs, ss, nres = _G["light"][sid], _G["scores"][sid], _G["sites"][sid], _G["nres"][sid]
    keep = finalize([_P(*l) for l in light], rs, _G["fc_v1"]) if len(light) else []
    C = np.array([light[i][0] for i in keep]).reshape(-1, 3)
    sc = np.asarray(rs)[keep]
    k2 = apply_rule(rule, sc, C, nres)
    C, sc = C[k2], sc[k2]
    N = len(ss)
    out = []
    D = np.zeros((len(ss), len(C)))
    for k, s in enumerate(ss):
        r = {}
        if len(C):
            dcc = np.round(np.linalg.norm(C - s["center"], axis=1), 2)
            D[k] = dcc
            j = int(np.lexsort((np.arange(len(C)), dcc))[0])
            dca = np.round(cKDTree(s["atoms"]).query(C, k=1)[0], 2)
            jd = int(np.lexsort((np.arange(len(C)), dca))[0])
            r.update(topn=dcc[j] <= 12 and j + 1 <= N, topn2=dcc[j] <= 12 and j + 1 <= N + 2,
                     dca4=dca[jd] <= 4 and jd + 1 <= N + 2, maxr=dcc[j] <= 12)
        else:
            r.update(topn=False, topn2=False, dca4=False, maxr=False)
        out.append(r)
    pockets = [(float(sc[j]), sid, j, [k for k in np.argsort(D[:, j]) if D[k, j] <= 12] if len(ss) else [])
               for j in range(len(C))]
    return out, pockets, len(C)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", choices=["all", "small"], required=True)
    ap.add_argument("--eval-variant", default=None, help="site definition for scoring (default = --variant)")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args(argv)
    ev = a.eval_variant or a.variant
    t0 = time.time()
    pdir = ROOT / "results/dev/cache/pockets" / a.run
    meta = pickle.loads((pdir / "meta.pkl").read_bytes())
    data = pickle.loads((pdir / "train.pkl").read_bytes())
    fn, fold = meta["fn"], meta["fold"]
    cache = ROOT / f"results/dev/cache/oof_rank_scores_{a.run}.pkl"
    sids = [s for s in data if len(data[s][0])]
    if cache.exists():
        scores = pickle.loads(cache.read_bytes())
    else:
        Xs, _ = build_X(data, sids, fn, None)
        scores = {}
        for k in range(5):
            tr = [i for i, s in enumerate(sids) if fold[s] != k]
            R = train_ranker(np.concatenate([Xs[i] for i in tr]), np.concatenate([data[sids[i]][1] for i in tr]),
                             [len(Xs[i]) for i in tr], VARIANTS["V0_base"], 0)
            for i, s in enumerate(sids):
                if fold[s] == k and not s.endswith(ASM):
                    scores[s] = R.predict(Xs[i], num_threads=16)
            print(f"[{time.time() - t0:.0f}s] fold {k}", flush=True)
        cache.write_bytes(pickle.dumps(scores))
    npz = dict(zip(entries("train").entry_id, entries("train").npz))
    st = {}
    for s in scores:
        ss = sites(load_entry(npz[s]), ev, False)
        if ss:
            st[s] = ss
    scores = {s: scores[s] for s in st}
    ia = fn.index("n_atoms_prot")
    nres = {s: float(data[s][0][0, ia]) / 8.0 for s in scores}   # heavy atoms / 8 ~ residues
    fc_v1 = replace(FormationConfig(), min_score=None)
    _G.update(light={s: data[s][2] for s in scores}, scores=scores, sites=st, nres=nres, fc_v1=fc_v1)
    rows = []
    for rule in RULES:
        with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
            res = list(ex.map(_job, [(s, rule) for s in scores], chunksize=16))
        rows.append(dict(rule=rule, **metrics(res)))
        print(f"[{time.time() - t0:.0f}s] {rows[-1]}", flush=True)
    df = pd.DataFrame(rows)
    ref = df[df.rule == "v2"].iloc[0]
    for c in ("topn", "topn2", "dca4", "P1K_1to1", "max_recall"):
        df[f"d_{c}"] = df[c] - ref[c]
    out = ROOT / f"results/dev/biolip/v3b_cv_{a.run}_{ev}.csv"
    df.round(3).to_csv(out, index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
