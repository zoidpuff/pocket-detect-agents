#!/usr/bin/env python
"""v2/H5 (PI decision D12): choose the output score floor tau (optionally + max-pockets cap) on TRAIN
cluster-CV only.

Out-of-fold scores of the v1 (V0, graded) ranker for all 5 folds of a cached run (candidates built
from out-of-fold point scores); for every setting, NMS (FormationConfig with min_score=tau,
min_keep, max_out) and the site_eval definitions on held-out *chain* inputs:
closest-pocket top-N / top-N+2 (DCC 12 Å), closest-by-DCA top-N+2 (4 Å), one-to-one P1K (pooled top
1,000; a pocket is TP only if it hits a site not yet claimed by a higher-scored pocket), raw P1K,
max recall, pockets/structure.
Selection rule (D12): maximise mean(top-N+2, DCA4 top-N+2, P1K_1to1) subject to no loss > 0.5 pt vs
tau = none (v1) in any of those three nor in top-N.

  python scripts/burrow_tau_cv.py --run p3_small_default_s0 --variant small
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
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
from pocketdetect.burrow.biolip import ASM, entries, sites  # noqa: E402
from pocketdetect.burrow.model import FormationConfig  # noqa: E402
from pocketdetect.burrow.pipeline import finalize  # noqa: E402
from pocketdetect.datasets.trainset import load_entry  # noqa: E402

GRID = ([dict(min_score=None, min_keep=0, max_out=50)]
        + [dict(min_score=t, min_keep=k, max_out=50) for t in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2)
           for k in (1, 3)]
        + [dict(min_score=None, min_keep=0, max_out=m) for m in (5, 10, 20)])
_G = {}


class _P:
    def __init__(self, c, r, idx):
        self.center, self.residues, self.idx = c, r, idx


def _job(args):
    sid, gi = args
    light, rs, ss = _G["light"][sid], _G["scores"][sid], _G["sites"][sid]
    fc = _G["fcs"][gi]
    keep = finalize([_P(*l) for l in light], rs, fc) if len(light) else []
    C = np.array([light[i][0] for i in keep]).reshape(-1, 3)
    sc = np.asarray(rs)[keep]
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
    pockets = []
    for j in range(len(C)):
        hits = [k for k in np.argsort(D[:, j]) if D[k, j] <= 12] if len(ss) else []
        pockets.append((float(sc[j]), sid, j, hits))
    return out, pockets, len(C)


def metrics(res):
    S = pd.DataFrame([r for rr, _, _ in res for r in rr])
    P = sorted([p for _, pp, _ in res for p in pp], key=lambda t: (-t[0], t[1], t[2]))[:1000]
    raw = np.mean([len(p[3]) > 0 for p in P])
    claimed, tp11 = set(), 0
    for sc, sid, j, hits in P:
        free = [h for h in hits if (sid, h) not in claimed]
        if free:
            claimed.add((sid, free[0]))
            tp11 += 1
    return dict(topn=100 * S.topn.mean(), topn2=100 * S.topn2.mean(), dca4=100 * S.dca4.mean(),
                P1K_1to1=100 * tp11 / max(len(P), 1), P1K=100 * raw, max_recall=100 * S.maxr.mean(),
                pockets=np.mean([n for _, _, n in res]), n_sites=len(S))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", choices=["all", "small"], required=True)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args(argv)
    t0 = time.time()
    pdir = ROOT / "results/dev/cache/pockets" / a.run
    meta = pickle.loads((pdir / "meta.pkl").read_bytes())
    data = pickle.loads((pdir / "train.pkl").read_bytes())
    fn, fold = meta["fn"], meta["fold"]
    m = entries("train")
    npz = dict(zip(m.entry_id, m.npz))
    sids = [s for s in data if len(data[s][0])]
    Xs, _ = build_X(data, sids, fn, None)
    v = VARIANTS["V0_base"]
    scores = {}
    for k in range(5):
        tr = [i for i, s in enumerate(sids) if fold[s] != k]
        R = train_ranker(np.concatenate([Xs[i] for i in tr]), np.concatenate([data[sids[i]][1] for i in tr]),
                         [len(Xs[i]) for i in tr], v, 0)
        for i, s in enumerate(sids):
            if fold[s] == k and not s.endswith(ASM):
                scores[s] = R.predict(Xs[i], num_threads=16)
        print(f"[{time.time() - t0:.0f}s] fold {k} ranker done", flush=True)
    st = {}
    for s in scores:
        ss = sites(load_entry(npz[s]), a.variant, False)
        if ss:
            st[s] = ss
    scores = {s: scores[s] for s in st}
    fcs = [replace(FormationConfig(), **g) for g in GRID]
    _G.update(light={s: data[s][2] for s in scores}, scores=scores, sites=st, fcs=fcs)
    rows = []
    for gi, g in enumerate(GRID):
        with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
            res = list(ex.map(_job, [(s, gi) for s in scores], chunksize=16))
        rows.append(dict(**g, **metrics(res)))
        print(f"[{time.time() - t0:.0f}s] {rows[-1]}", flush=True)
    df = pd.DataFrame(rows)
    ref = df.iloc[0]
    df["criterion"] = df[["topn2", "dca4", "P1K_1to1"]].mean(axis=1)
    ok = np.ones(len(df), bool)
    for c in ("topn2", "dca4", "P1K_1to1", "topn"):
        ok &= df[c] >= ref[c] - 0.5
    df["admissible"] = ok
    out = ROOT / f"results/dev/biolip/tau_cv_{a.run}.csv"
    df.round(3).to_csv(out, index=False)
    print(df.round(2).to_string(index=False))
    best = df[df.admissible].sort_values("criterion", ascending=False).iloc[0]
    print("SELECTED:", best.to_dict())


if __name__ == "__main__":
    main()
