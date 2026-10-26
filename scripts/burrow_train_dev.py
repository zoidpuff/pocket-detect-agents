#!/usr/bin/env python
"""Train / cross-fit BURROW on the interim dev data and write harness-format predictions.

Schemes (--scheme):
  cv5          5-fold CV over CHEN11+JOINED, folds = MMseqs2 30 % groups.  Point scores are
               out-of-fold (cross-fitted); the pocket ranker is trained on OOF-score pockets of
               the other folds.  Predictions for all 788 structures.
  chen11       train on CHEN11 only (point scorer cross-fitted inside CHEN11 for the ranker's
               training features; final point scorer on all CHEN11), predict JOINED.  Same
               training set as P2Rank's default model -> fair comparison on JOINED.
Also writes the geometry-only baseline (no learning; rank = Σ vol·buriedness) for all structures
and a no-ranker ablation (rank = Σ vol·point score).

  python scripts/burrow_train_dev.py --feats results/dev/cache/feats/<hash> --scheme cv5 --seed-idx 0
Outputs: results/dev/preds/<method>/<sid>.csv (+ .points.csv), models in results/dev/models/.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # shared per-user thread limit; LightGBM gets num_threads explicitly

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.devdata import load_gts, point_labels, read_manifest  # noqa: E402
from pocketdetect.burrow.model import (POINT_PARAMS, RANK_PARAMS, BurrowModel, FormationConfig,  # noqa: E402
                                       candidates, geo_score, pocket_features, train_lgb)
from pocketdetect.burrow.pipeline import Prediction, finalize, write_prediction  # noqa: E402
from pocketdetect.burrow.pockets import assign_residues  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402

N_POINT_ROUNDS = 250
N_RANK_ROUNDS = 300
NEG_FRAC = 0.1

_G = {}


def load_feats(sid):
    z = np.load(Path(_G["feats"]) / f"{sid}.npz")
    return {k: z[k] for k in ("points", "X", "vol", "feats", "d_all", "d_ion")} | {"names": list(z["names"])}


def sample_points(sids, rng):
    Xs, ys = [], []
    for s in sids:
        d = load_feats(s)
        y = point_labels(d["d_all"], d["d_ion"])
        m = (y == 1) | (rng.random(len(y)) < NEG_FRAC)
        Xs.append(d["feats"][m])
        ys.append(y[m])
    return np.concatenate(Xs), np.concatenate(ys)


def hit_label(centers, gt):
    """Per pocket: min DCC over ligands whose DCA <= 4 Å (inf if none)."""
    q = np.full(len(centers), np.inf)
    if len(centers) == 0:
        return q
    for lig in gt.ligands:
        dcc = np.linalg.norm(centers - lig.center, axis=1)
        dca = cKDTree(lig.coords).query(centers, k=1)[0]
        q = np.minimum(q, np.where(dca <= 4, dcc, np.inf))
    return q


def rank_target(q, kind):
    """Ranker target from q = min DCC over sites with DCA <= 4: hit (q <= 12), tight (q <= 8),
    graded (1 - q/12 clipped, cross-entropy objective)."""
    if kind == "hit":
        return (q <= 12).astype(np.float32)
    if kind == "tight":
        return (q <= 8).astype(np.float32)
    return np.clip(1.0 - q / 12.0, 0.0, 1.0).astype(np.float32)


def rank_params(kind):
    return dict(RANK_PARAMS, objective="cross_entropy") if kind == "graded" else RANK_PARAMS


def _pocket_job(sid):
    """Candidates + pocket features + labels for one structure, given its point scores."""
    d = load_feats(sid)
    score = _G["scores"][sid]
    P = load_protein(_G["paths"][sid], structure_id=sid)
    fc = _G["fc"]
    pk = candidates(d["X"], d["points"], d["vol"], d["feats"], d["names"], score, fc)
    assign_residues(pk, d["points"], cKDTree(P.coords), P.res_index, P.residue_ids)
    Fp, fn = pocket_features(pk, d["X"], d["vol"], d["feats"], d["names"], score, len(P.coords))
    C = np.array([p.center for p in pk]).reshape(-1, 3)
    y = hit_label(C, _G["gts"][sid])
    light = [(p.center, p.residues, p.strength, d["points"][p.idx], p.idx) for p in pk]
    return sid, Fp, fn, y, light


def _geo_job(sid):
    d = load_feats(sid)
    P = load_protein(_G["paths"][sid], structure_id=sid)
    fc = _G["fc"]
    pk = candidates(d["X"], d["points"], d["vol"], d["feats"], d["names"], None, fc)
    assign_residues(pk, d["points"], cKDTree(P.coords), P.res_index, P.residue_ids)
    rs = geo_score(pk, d["vol"], d["feats"], d["names"], fc)
    keep = finalize(pk, rs, fc)
    return Prediction(sid, np.array([pk[i].center for i in keep]).reshape(-1, 3), rs[keep],
                      [pk[i].residues for i in keep], [d["points"][pk[i].idx] for i in keep], {})


class _P:  # minimal pocket stand-in for NMS
    def __init__(self, c, r, idx):
        self.center, self.residues, self.idx = c, r, idx


def emit(out_dir, sid, light, rs, fc):
    pk = [_P(c, r, idx) for c, r, _, _, idx in light]
    keep = finalize(pk, rs, fc)
    write_prediction(out_dir, Prediction(sid, np.array([light[i][0] for i in keep]).reshape(-1, 3),
                                         np.asarray(rs)[keep], [light[i][1] for i in keep],
                                         [light[i][3] for i in keep], {}))


def save_run(name, store):
    import pickle
    d = Path(_G["cache"]) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.pkl").write_bytes(pickle.dumps(store))


def make_folds(sids, groups, k=5, seed=SEEDS[0]):
    g = np.array([groups[s] for s in sids])
    ug = np.unique(g)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ug)
    fold_of_group = {gg: i % k for i, gg in enumerate(perm)}
    return np.array([fold_of_group[x] for x in g])


def predict_scores(model, sids):
    out = {}
    for s in sids:
        d = load_feats(s)
        out[s] = model.predict(d["feats"], num_threads=16) if len(d["feats"]) else np.zeros(0)
    return out


def pockets_for(sids, jobs):
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        return {r[0]: r[1:] for r in ex.map(_pocket_job, sids, chunksize=2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "results/dev/manifests/all.csv")
    ap.add_argument("--groups", type=Path, default=ROOT / "results/dev/manifests/groups_all_mmseqs30.json")
    ap.add_argument("--feats", type=Path, required=True)
    ap.add_argument("--scheme", choices=["cv5", "chen11"], default="cv5")
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", type=Path, default=ROOT / "results/dev")
    ap.add_argument("--geo", action="store_true", help="also write the geometry-only baseline")
    ap.add_argument("--rank-label", choices=["hit", "tight", "graded"], default="hit")
    a = ap.parse_args(argv)
    t0 = time.time()
    df = read_manifest(a.manifest)
    groups = json.loads(a.groups.read_text())
    gts = load_gts(a.manifest)
    fc = FormationConfig()
    _G.update(cache=str(ROOT / "results/dev/cache"), feats=str(a.feats), paths=dict(zip(df.structure_id, df.structure_path)), gts=gts, fc=fc)
    sids = list(df.structure_id)
    names = load_feats(sids[0])["names"]
    rng = np.random.default_rng(SEEDS[a.seed_idx])
    tag = f"s{a.seed_idx}" + ("" if a.rank_label == "hit" else f"_{a.rank_label}")
    if a.geo:
        with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
            for pr in ex.map(_geo_job, sids, chunksize=2):
                write_prediction(a.out / "preds/burrow_geo", pr)
        print(f"geometry-only baseline written ({time.time() - t0:.0f}s)")

    if a.scheme == "cv5":
        train_sids, test_sids = sids, sids
        folds = make_folds(sids, groups)
        fold_of = dict(zip(sids, folds))
        oof = {}
        for k in range(5):
            tr = [s for s in sids if fold_of[s] != k]
            te = [s for s in sids if fold_of[s] == k]
            Xp, yp = sample_points(tr, rng)
            m = train_lgb(Xp, yp, POINT_PARAMS, N_POINT_ROUNDS, a.seed_idx)
            oof.update(predict_scores(m, te))
            print(f"fold {k}: point model on {len(Xp)} pts ({yp.mean():.3f} pos) ({time.time() - t0:.0f}s)")
        _G["scores"] = oof
        pk = pockets_for(sids, a.jobs)
        fn = next(v[1] for v in pk.values() if v[1])
        print(f"pockets: {sum(len(v[2]) for v in pk.values())} candidates ({time.time() - t0:.0f}s)")
        for meth in (f"burrow_cv5_{tag}", f"burrow_cv5_noranker_{tag}"):
            (a.out / "preds" / meth).mkdir(parents=True, exist_ok=True)
        store = {}
        for k in range(5):
            tr = [s for s in sids if fold_of[s] != k and len(pk[s][2])]
            te = [s for s in sids if fold_of[s] == k]
            R = train_lgb(np.concatenate([pk[s][0] for s in tr]),
                          rank_target(np.concatenate([pk[s][2] for s in tr]), a.rank_label),
                          rank_params(a.rank_label), N_RANK_ROUNDS, a.seed_idx)
            for s in te:
                Fp, _, y, light = pk[s]
                rs = R.predict(Fp, num_threads=16) if len(light) else np.zeros(0)
                emit(a.out / "preds" / f"burrow_cv5_{tag}", s, light, rs, fc)
                ss = Fp[:, fn.index("score_sum")] if len(light) else np.zeros(0)
                emit(a.out / "preds" / f"burrow_cv5_noranker_{tag}", s, light, ss, fc)
                store[s] = (light, rs, ss, (y <= 12).astype(np.int8))
        save_run(f"burrow_cv5_{tag}", store)
        print(f"cv5 done ({time.time() - t0:.0f}s)")
    else:
        sub = dict(zip(df.structure_id, df.subset))
        tr_all = [s for s in sids if sub[s] == "chen11"]
        te = [s for s in sids if sub[s] != "chen11"]
        folds = make_folds(tr_all, groups)
        fold_of = dict(zip(tr_all, folds))
        oof = {}
        for k in range(5):
            tr = [s for s in tr_all if fold_of[s] != k]
            Xp, yp = sample_points(tr, rng)
            m = train_lgb(Xp, yp, POINT_PARAMS, N_POINT_ROUNDS, a.seed_idx)
            oof.update(predict_scores(m, [s for s in tr_all if fold_of[s] == k]))
        Xp, yp = sample_points(tr_all, rng)
        pm = train_lgb(Xp, yp, POINT_PARAMS, N_POINT_ROUNDS, a.seed_idx)
        oof.update(predict_scores(pm, te))
        _G["scores"] = oof
        pk = pockets_for(tr_all + te, a.jobs)
        fn = next(v[1] for v in pk.values() if v[1])
        tr = [s for s in tr_all if len(pk[s][2])]
        R = train_lgb(np.concatenate([pk[s][0] for s in tr]),
                      rank_target(np.concatenate([pk[s][2] for s in tr]), a.rank_label),
                      rank_params(a.rank_label), N_RANK_ROUNDS, a.seed_idx)
        meth = f"burrow_chen11_{tag}"
        store = {}
        for s in te:
            Fp, _, y, light = pk[s]
            rs = R.predict(Fp, num_threads=16) if len(light) else np.zeros(0)
            emit(a.out / "preds" / meth, s, light, rs, fc)
            store[s] = (light, rs, Fp[:, fn.index("score_sum")] if len(light) else np.zeros(0), (y <= 12).astype(np.int8))
        save_run(meth, store)
        (a.out / "models").mkdir(parents=True, exist_ok=True)
        bm = BurrowModel(pm.model_to_string(), names, R.model_to_string(), fn,
                         meta=dict(trained_on="CHEN11 (P2Rank datasets, p2rank rules)", seed=SEEDS[a.seed_idx],
                                   point_rounds=N_POINT_ROUNDS, rank_rounds=N_RANK_ROUNDS, neg_frac=NEG_FRAC,
                                   feats_cache=str(a.feats)))
        bm.save(a.out / "models" / f"burrow_chen11_{tag}.json")
        imp = pd.DataFrame({"feature": names, "gain": pm.feature_importance("gain")}).sort_values("gain", ascending=False)
        imp.to_csv(a.out / "models" / f"point_importance_chen11_{tag}.csv", index=False)
        imp = pd.DataFrame({"feature": fn, "gain": R.feature_importance("gain")}).sort_values("gain", ascending=False)
        imp.to_csv(a.out / "models" / f"rank_importance_chen11_{tag}.csv", index=False)
        print(f"chen11 -> joined done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
