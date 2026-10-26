#!/usr/bin/env python
"""Phase-3 ranker selection by train-set cluster cross-validation (no dev).

Uses candidate pockets cached by `burrow_train_biolip.py --save-pockets full` (features built from
OUT-OF-FOLD point scores; folds = MMseqs2 30 % clusters).  For each ranker variant: train on 4/5 of
the train inputs, predict the held-out fold, NMS (FormationConfig), and score held-out *chain*
inputs against their pooled sites (assembly inputs are training-only).  Held-out folds: 0 and 1.

Metrics (same definitions as pocketdetect.site_eval): top-N / top-N+2 = closest pocket by DCC to the
site centre within 12 Å and rank <= N(+2); DCA4 top-N+2 = closest-by-DCA pocket within 4 Å and rank
<= N+2; P1K = pooled precision of the top 1,000 pockets (TP: DCC <= 12 to any site).

  python scripts/burrow_ranker_cv.py --run p3_small_default_s0 --variant small
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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_train_dev import rank_params, rank_target  # noqa: E402
from pocketdetect.burrow.biolip import ASM, entries, sites  # noqa: E402
from pocketdetect.burrow.model import RANK_PARAMS, FormationConfig, context_features, train_lgb  # noqa: E402
from pocketdetect.burrow.pipeline import finalize  # noqa: E402
from pocketdetect.datasets.trainset import load_entry  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402

VARIANTS = {
    "V0_base": dict(ctx=None, obj="graded", ens=1, weight=None),
    "V1_rel": dict(ctx="rel", obj="graded", ens=1, weight=None),
    "V2_rel_pair": dict(ctx="rel+pair", obj="graded", ens=1, weight=None),
    "V3_rel_pair_lambdarank": dict(ctx="rel+pair", obj="lambdarank", ens=1, weight=None),
    "V4_rel_pair_ens3": dict(ctx="rel+pair", obj="graded", ens=3, weight=None),
}
_G = {}


class _P:
    def __init__(self, c, r, idx):
        self.center, self.residues, self.idx = c, r, idx


def _metrics_job(sid):
    light, rs = _G["light"][sid], _G["scores"][sid]
    ss = _G["sites"][sid]
    keep = finalize([_P(*l) for l in light], rs, _G["fc"]) if len(light) else []
    C = np.array([light[i][0] for i in keep]).reshape(-1, 3)
    sc = np.asarray(rs)[keep]
    N = len(ss)
    rows = []
    for s in ss:
        r = dict(sid=sid)
        if len(C):
            dcc = np.round(np.linalg.norm(C - s["center"], axis=1), 2)
            j = int(np.lexsort((np.arange(len(C)), dcc))[0])
            dca = np.round(cKDTree(s["atoms"]).query(C, k=1)[0], 2)
            jd = int(np.lexsort((np.arange(len(C)), dca))[0])
            r.update(topn=dcc[j] <= 12 and j + 1 <= N, topn2=dcc[j] <= 12 and j + 1 <= N + 2,
                     dca4=dca[jd] <= 4 and jd + 1 <= N + 2)
        else:
            r.update(topn=False, topn2=False, dca4=False)
        rows.append(r)
    tp = np.zeros(len(C), bool)
    for s in ss:
        if len(C):
            tp |= np.linalg.norm(C - s["center"], axis=1) <= 12
    return rows, list(zip(sc.tolist(), tp.tolist(), [sid] * len(C), range(len(C))))


def evaluate(light, scores, sites_, fc, jobs):
    _G.update(light=light, scores=scores, sites=sites_, fc=fc)
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(_metrics_job, list(scores), chunksize=8))
    S = pd.DataFrame([r for rr, _ in res for r in rr])
    pk = sorted([p for _, pp in res for p in pp], key=lambda t: (-t[0], t[2], t[3]))
    p1k = np.mean([t[1] for t in pk[:1000]]) if pk else np.nan
    return dict(n_sites=len(S), topn=100 * S.topn.mean(), topn2=100 * S.topn2.mean(), dca4=100 * S.dca4.mean(),
                P1K=100 * p1k, pockets=np.mean([len(v) for v in scores.values()]))


def build_X(data, sids, fn, ctx):
    Xs, extra_names = [], []
    for s in sids:
        Fp, q, light = data[s]
        if ctx and len(Fp):
            Fx, extra_names = context_features(Fp, fn, np.array([l[0] for l in light]), ctx)
            Fp = np.concatenate([Fp, Fx], 1)
        Xs.append(Fp)
    return Xs, extra_names


def lambdarank_labels(q):
    return np.where(q <= 12, np.clip(np.round(4 * (1 - q / 12)), 0, 4), 0).astype(int)


def train_ranker(X, q, groups_sizes, v, seed_idx):
    if v["obj"] == "lambdarank":
        import lightgbm as lgb
        p = dict(RANK_PARAMS, objective="lambdarank", lambdarank_truncation_level=20, label_gain=[0, 1, 3, 7, 15])
        p.update(seed=SEEDS[seed_idx], bagging_seed=SEEDS[seed_idx], feature_fraction_seed=SEEDS[seed_idx],
                 deterministic=True)
        ds = lgb.Dataset(X, label=lambdarank_labels(q), group=groups_sizes)
        return lgb.train(p, ds, num_boost_round=300)
    return train_lgb(X, rank_target(q, v["obj"]), rank_params(v["obj"]), 300, seed_idx)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", choices=["all", "small"], required=True)
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args(argv)
    t0 = time.time()
    pdir = ROOT / "results/dev/cache/pockets" / a.run
    meta = pickle.loads((pdir / "meta.pkl").read_bytes())
    data = pickle.loads((pdir / "train.pkl").read_bytes())
    fn, fold = meta["fn"], meta["fold"]
    npz = dict(zip(entries("train").entry_id, entries("train").npz))
    sids = [s for s in data if len(data[s][0])]
    print(f"[{time.time() - t0:.0f}s] loaded {len(sids)} train inputs", flush=True)
    sites_ = {}
    for s in sids:
        if s.endswith(ASM):
            continue
        ss = sites(load_entry(npz[s]), a.variant, False)
        if ss:
            sites_[s] = ss
    fc = FormationConfig()
    rows = []
    for vname, v in VARIANTS.items():
        if a.only and vname not in a.only.split(","):
            continue
        Xs, _ = build_X(data, sids, fn, v["ctx"])
        for k in [int(x) for x in a.folds.split(",")]:
            tr = [i for i, s in enumerate(sids) if fold[s] != k]
            te = [i for i, s in enumerate(sids) if fold[s] == k and s in sites_]
            X = np.concatenate([Xs[i] for i in tr])
            q = np.concatenate([data[sids[i]][1] for i in tr])
            gs = [len(Xs[i]) for i in tr]
            preds = {sids[i]: 0.0 for i in te}
            for e in range(v["ens"]):
                R = train_ranker(X, q, gs, v, e)
                for i in te:
                    preds[sids[i]] = preds[sids[i]] + R.predict(Xs[i], num_threads=16) / v["ens"]
            m = evaluate({sids[i]: data[sids[i]][2] for i in te}, preds, {sids[i]: sites_[sids[i]] for i in te},
                         fc, a.jobs)
            rows.append(dict(variant=vname, fold=k, **m))
            print(f"[{time.time() - t0:.0f}s] {rows[-1]}", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("variant", sort=False)[["topn", "topn2", "dca4", "P1K", "pockets", "n_sites"]].mean().round(2)
    out = ROOT / f"results/dev/biolip/ranker_cv_{a.run}.csv"
    df.to_csv(out, index=False)
    print(agg.to_string())


if __name__ == "__main__":
    main()
