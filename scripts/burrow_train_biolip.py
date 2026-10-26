#!/usr/bin/env python
"""Train BURROW (`all` or `small` ranker) on Track A's BioLiP2 train set; predict the dev set.

  python scripts/burrow_train_biolip.py --variant all --seed-idx 0

* Inputs: train chain inputs + stored assembly-1 inputs (`<entry>__asm1`), dev chain inputs.
* Point labels: all  = <= 4 Å of a non-ion pooled ligand atom or <= 3 Å of an ion;
                small = <= 4 Å of a non-ion ligand atom.
* Point scorer cross-fitted over 5 folds of MMseqs2 30 % clusters (``cluster30``; assembly inputs
  share their entry's cluster) -> out-of-fold scores for every train structure; final point model
  on all train.
* Pocket ranker trained on candidates built from OOF scores; label = centre DCC <= 12 Å to a
  target site centre AND DCA <= 4 Å to that site's ligand atoms (targets per variant).
* Dev: final point model -> candidates -> ranker -> NMS (FormationConfig) -> predictions.
Outputs: results/dev/biolip/preds/<name>/, results/dev/biolip/models/<name>.json,
results/dev/cache/runs/<name>.pkl (dev candidates + ranker scores for re-emission).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
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
from burrow_featurize_biolip import cfg_args, feats_dir, make_cfg  # noqa: E402
from burrow_train_dev import emit, rank_params, rank_target  # noqa: E402
from pocketdetect.burrow.biolip import ASM, entries, sites, structure_ids  # noqa: E402
from pocketdetect.burrow.model import (POINT_PARAMS, RANK_PARAMS, BurrowModel, FormationConfig,  # noqa: E402
                                       candidates, geo_value, pocket_features, score_points, train_lgb)
from pocketdetect.burrow.pockets import assign_residues  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402
from pocketdetect.datasets.trainset import load_entry  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402

DEV_INPUTS = ROOT / "results/dev/biolip/inputs"
OUT = ROOT / "results/dev/biolip"
_G = {}


def load_feats(sid):
    z = np.load(Path(_G["feats"]) / f"{sid}.npz")
    return {k: z[k] for k in ("points", "X", "vol", "feats", "d_lig", "d_ion")} | {"names": list(z["names"])}


def point_label(d, variant):
    y = d["d_lig"] <= 4.0
    if variant == "all":
        y |= d["d_ion"] <= 3.0
    return y.astype(np.int8)


def _sample_job(sid):
    d = load_feats(sid)
    y = point_label(d, _G["variant"])
    h = int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([SEEDS[_G["seed_idx"]], h])
    r = rng.random(len(y))
    m = np.where(y == 1, r < _G["pos_frac"], r < _G["neg_frac"])
    if _G["score_floor"] is not None:  # train only on points that will be scored
        F = d["feats"]
        m &= (geo_value(F, d["names"]) >= _G["score_floor"]) | (F[:, d["names"].index("metal_cand")] > 0)
    return sid, d["feats"][m], y[m]


def _protein(sid, split):
    eid = sid.replace(ASM, "")
    if split == "dev":
        return load_protein(DEV_INPUTS / f"{eid}.cif", structure_id=sid)
    from pocketdetect.burrow.biolip import load_protein_entry
    return load_protein_entry(load_entry(_G["npz"][eid]), assembly=sid.endswith(ASM))


def _pocket_job(args):
    sid, split, keep_light = args
    d = load_feats(sid)
    score = _G["scores"][sid]
    eid = sid.replace(ASM, "")
    e = load_entry(_G["npz"][eid])
    P = _protein(sid, split)
    fc = _G["fc"]
    pk = candidates(d["X"], d["points"], d["vol"], d["feats"], d["names"], score, fc)
    assign_residues(pk, d["points"], cKDTree(P.coords), P.res_index, P.residue_ids)
    Fp, fn = pocket_features(pk, d["X"], d["vol"], d["feats"], d["names"], score, len(P.coords))
    C = np.array([p.center for p in pk]).reshape(-1, 3)
    y = np.full(len(pk), np.inf)  # min DCC over target sites with DCA <= 4 Å
    for s in sites(e, _G["variant"], sid.endswith(ASM)):
        if len(C):
            dcc = np.linalg.norm(C - s["center"], axis=1)
            dca = cKDTree(s["atoms"]).query(C, k=1)[0]
            y = np.minimum(y, np.where(dca <= 4, dcc, np.inf))
    light = [(p.center, p.residues, p.strength, d["points"][p.idx], p.idx) for p in pk] if keep_light else None
    return sid, Fp, fn, y, light


def predict_scores(model, sids):
    out = {}
    for s in sids:
        d = load_feats(s)
        out[s] = score_points(model, d["feats"], d["names"], d["names"], _G["score_floor"], num_threads=16)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["all", "small"], default="all")
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--pos-frac", type=float, default=0.3)
    ap.add_argument("--neg-frac", type=float, default=0.03)
    ap.add_argument("--point-rounds", type=int, default=250)
    ap.add_argument("--rank-rounds", type=int, default=300)
    ap.add_argument("--point-leaves", type=int, default=63)
    ap.add_argument("--score-floor", type=float, default=None, help="score only points with b >= floor")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-assembly", action="store_true")
    cfg_args(ap)
    a = ap.parse_args(argv)
    from dataclasses import asdict
    fcfg = make_cfg(a.n_rays, a.max_len, a.subsample)
    name = a.name or f"burrow_{a.variant}_s{a.seed_idx}"
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    mt, md = entries("train"), entries("dev")
    fdir = feats_dir(fcfg)
    tr = [(s, asm) for s, _, asm in structure_ids(mt, not a.no_assembly) if (fdir / f"{s}.npz").exists()]
    dv = [s for s, _, _ in structure_ids(md, False) if (fdir / f"{s}.npz").exists()]
    grp_of = dict(zip(mt.entry_id, mt.cluster30))
    npz = dict(zip(pd.concat([mt.entry_id, md.entry_id]), pd.concat([mt.npz, md.npz])))
    fc = FormationConfig()
    _G.update(feats=str(fdir), variant=a.variant, seed_idx=a.seed_idx, pos_frac=a.pos_frac, neg_frac=a.neg_frac,
              npz=npz, fc=fc, score_floor=a.score_floor)
    point_params = dict(POINT_PARAMS, num_leaves=a.point_leaves)
    tr_ids = [s for s, _ in tr]
    log(f"train inputs {len(tr_ids)} (assemblies {sum(x for _, x in tr)}), dev {len(dv)}")
    # folds by cluster30
    ug = sorted({grp_of[s.replace(ASM, '')] for s in tr_ids})
    perm = np.random.default_rng(SEEDS[0]).permutation(len(ug))
    fold_g = {g: int(perm[i] % 5) for i, g in enumerate(ug)}
    fold = {s: fold_g[grp_of[s.replace(ASM, '')]] for s in tr_ids}
    # point samples
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(a.jobs, mp_context=ctx) as ex:
        samp = {s: (Xs, ys) for s, Xs, ys in ex.map(_sample_job, tr_ids, chunksize=8)}
    names = load_feats(tr_ids[0])["names"]
    log(f"point samples: {sum(len(v[1]) for v in samp.values())} rows, "
        f"pos {np.mean(np.concatenate([v[1] for v in samp.values()])):.3f}")
    oof = {}
    for k in range(5):
        trk = [s for s in tr_ids if fold[s] != k]
        m = train_lgb(np.concatenate([samp[s][0] for s in trk]), np.concatenate([samp[s][1] for s in trk]),
                      point_params, a.point_rounds, a.seed_idx)
        oof.update(predict_scores(m, [s for s in tr_ids if fold[s] == k]))
        log(f"fold {k} point model done")
    pm = train_lgb(np.concatenate([v[0] for v in samp.values()]), np.concatenate([v[1] for v in samp.values()]),
                   point_params, a.point_rounds, a.seed_idx)
    del samp
    oof.update(predict_scores(pm, dv))
    _G["scores"] = oof
    log("final point model + dev scores done")
    with ProcessPoolExecutor(a.jobs, mp_context=ctx) as ex:
        pk_tr = {r[0]: r[1:] for r in ex.map(_pocket_job, [(s, "train", False) for s in tr_ids], chunksize=4)}
        pk_dv = {r[0]: r[1:] for r in ex.map(_pocket_job, [(s, "dev", True) for s in dv], chunksize=2)}
    fn = next(v[1] for v in pk_tr.values() if v[1])
    Xr = np.concatenate([v[0] for v in pk_tr.values() if len(v[2])])
    yr = np.concatenate([v[2] for v in pk_tr.values() if len(v[2])])
    log(f"ranker data: {len(yr)} candidates, hit rate {(yr <= 12).mean():.3f}")
    rankers = {}
    for lab in ("graded", "hit"):
        rankers[lab] = train_lgb(Xr, rank_target(yr, lab), rank_params(lab), a.rank_rounds, a.seed_idx)
    stores = {}
    for lab, R in rankers.items():
        nm = name if lab == "graded" else f"{name}_hit"
        store = {}
        for s in dv:
            Fp, _, y, light = pk_dv[s]
            rs = R.predict(Fp, num_threads=16) if len(light) else np.zeros(0)
            emit(OUT / "preds" / nm, s, light, rs, fc)
            store[s] = (light, rs, Fp[:, fn.index("score_sum")] if len(light) else np.zeros(0),
                        (y <= 12).astype(np.int8))
        stores[nm] = store
    R = rankers["graded"]
    rd = ROOT / "results/dev/cache/runs"
    rd.mkdir(parents=True, exist_ok=True)
    for nm, store in stores.items():
        (rd / f"{nm}.pkl").write_bytes(pickle.dumps(store))
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    BurrowModel(pm.model_to_string(), names, R.model_to_string(), fn,
                meta=dict(trained_on="BioLiP2 train (Track A, docs/TRAINSET.md)", variant=a.variant, rank_label="graded", score_floor=a.score_floor,
                          point_leaves=a.point_leaves, feature_cfg=asdict(fcfg),
                          seed=SEEDS[a.seed_idx], n_train_inputs=len(tr_ids), point_rounds=a.point_rounds,
                          rank_rounds=a.rank_rounds, pos_frac=a.pos_frac, neg_frac=a.neg_frac,
                          feats_cache=str(fdir))).save(OUT / "models" / f"{name}.json")
    pd.DataFrame({"feature": names, "gain": pm.feature_importance("gain")}).sort_values("gain", ascending=False) \
        .to_csv(OUT / "models" / f"{name}_point_importance.csv", index=False)
    pd.DataFrame({"feature": fn, "gain": R.feature_importance("gain")}).sort_values("gain", ascending=False) \
        .to_csv(OUT / "models" / f"{name}_rank_importance.csv", index=False)
    log(f"done -> {OUT / 'preds' / name}")


if __name__ == "__main__":
    main()
