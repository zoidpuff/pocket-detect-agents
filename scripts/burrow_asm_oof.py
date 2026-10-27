#!/usr/bin/env python
"""v3 dev signal (i): out-of-fold BURROW predictions on the TRAIN assembly-1 inputs.

Candidates of the cached run (built from out-of-fold point scores, 5 folds of cluster30); the
pocket ranker (v1 graded ranker recipe) is retrained per fold on all train inputs of the other
folds and applied to the held-out fold's assembly inputs.  Output rules are re-emitted from the
same scores: v1 (no floor), v2 (floor 0.03, min_keep 3), and variants given by --rules.

  python scripts/burrow_asm_oof.py --run p3_all_default_s0 --rules v1,v2,a3n
Writes results/dev/biolip/asm/preds/oof_<rule>/<structure_id>.csv
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_ranker_cv import VARIANTS, build_X, train_ranker  # noqa: E402
from burrow_train_dev import emit  # noqa: E402
from pocketdetect.burrow.biolip import ASM  # noqa: E402
from pocketdetect.burrow.model import FormationConfig  # noqa: E402

B = ROOT / "results/dev/biolip/asm"


def _if_job(args):
    sid, fd, asm_path, light = args
    from pocketdetect.burrow.model import interface_pocket_features
    from pocketdetect.burrow.prep import load_protein
    if asm_path is None:
        return sid, np.zeros((len(light), 2), np.float32)
    P = load_protein(asm_path, structure_id=sid)
    pts = np.load(Path(fd) / f"{sid}.npz")["points"]
    return sid, interface_pocket_features([l[2] for l in light], np.array([l[0] for l in light]).reshape(-1, 3),
                                          pts, P.coords, P.chain)


def interface_features_all(data, sids, meta):
    from concurrent.futures import ProcessPoolExecutor
    from burrow_featurize_biolip import feats_dir, make_cfg
    args = meta["args"]
    fd = str(feats_dir(make_cfg(args["n_rays"], args["max_len"], args["subsample"])))
    asm = {}
    for f in (B / "gt/train").glob("*.json"):
        asm[f.stem] = json.loads(f.read_text())["asm_path"]
    tasks = [(s, fd, asm.get(s) if s.endswith(ASM) else None, data[s][2]) for s in sids]
    with ProcessPoolExecutor(12) as ex:
        return dict(ex.map(_if_job, tasks, chunksize=8))


def rule_fc(rule, n_chains):
    fc = FormationConfig()
    if rule == "v1":
        return replace(fc, min_score=None)
    if rule == "v2":
        return replace(fc, min_score=0.03, min_keep=3)
    if rule.startswith("a") and rule.endswith("n"):  # (a): min_keep = k x n_chains, floor 0.03
        return replace(fc, min_score=0.03, min_keep=int(rule[1:-1]) * n_chains)
    raise ValueError(rule)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="p3_all_default_s0")
    ap.add_argument("--rules", default="v1,v2,a3n")
    ap.add_argument("--asm-weight", type=float, default=1.0, help="(c) ranker sample weight of assembly inputs")
    ap.add_argument("--if-feats", action="store_true", help="(b) add interface-aware pocket features")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    cache = ROOT / f"results/dev/cache/asm_oof_scores_{a.run}{a.tag}.pkl"
    pdir = ROOT / "results/dev/cache/pockets" / a.run
    data = pickle.loads((pdir / "train.pkl").read_bytes())
    meta = pickle.loads((pdir / "meta.pkl").read_bytes())
    if cache.exists():
        scores = pickle.loads(cache.read_bytes())
    else:
        fn, fold = meta["fn"], meta["fold"]
        sids = [s for s in data if len(data[s][0])]
        Xs, _ = build_X(data, sids, fn, None)
        if a.if_feats:
            ifx = interface_features_all(data, sids, meta)
            Xs = [np.concatenate([x, ifx[s]], 1) for x, s in zip(Xs, sids)]
        scores = {}
        for k in range(5):
            tr = [i for i, s in enumerate(sids) if fold[s] != k]
            if a.asm_weight != 1.0:
                import lightgbm as lgb
                from burrow_train_dev import rank_params, rank_target
                from pocketdetect.seeds import SEEDS
                X = np.concatenate([Xs[i] for i in tr])
                y = rank_target(np.concatenate([data[sids[i]][1] for i in tr]), "graded")
                w = np.concatenate([np.full(len(Xs[i]), a.asm_weight if sids[i].endswith(ASM) else 1.0) for i in tr])
                p = dict(rank_params("graded"), seed=SEEDS[0], bagging_seed=SEEDS[0], feature_fraction_seed=SEEDS[0],
                         data_random_seed=SEEDS[0], deterministic=True)
                R = lgb.train(p, lgb.Dataset(X.astype(np.float32), label=y, weight=w), num_boost_round=300)
            else:
                R = train_ranker(np.concatenate([Xs[i] for i in tr]), np.concatenate([data[sids[i]][1] for i in tr]),
                                 [len(Xs[i]) for i in tr], VARIANTS["V0_base"], 0)
            for i, s in enumerate(sids):
                if fold[s] == k and s.endswith(ASM):
                    scores[s] = R.predict(Xs[i], num_threads=16)
            print(f"[{time.time() - t0:.0f}s] fold {k}", flush=True)
        cache.write_bytes(pickle.dumps(scores))
    nch = {}
    for f in (B / "gt/train").glob("*.json"):
        nch[f.stem] = len(json.loads(f.read_text())["chains"])
    for rule in a.rules.split(","):
        out = B / "preds" / f"oof{a.tag}_{rule}"
        for s, rs in scores.items():
            if s not in nch:
                continue
            light = [(c, r, 0.0, np.zeros((0, 3)), idx) for c, r, idx in data[s][2]]
            emit(out, s, light, rs, rule_fc(rule, nch[s]))
        print(f"[{time.time() - t0:.0f}s] emitted {rule} -> {out}", flush=True)


if __name__ == "__main__":
    main()
