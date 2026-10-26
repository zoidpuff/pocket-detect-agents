#!/usr/bin/env python
"""Train the AF-aware pocket ranker (pocketdetect.burrow.af_rank) on AF models of Track A *train*
entries (Track C2 transferred labels), optionally mixed with the PDB train inputs.

  BURROW_TRAINSET=<snapshot> python scripts/af_dev_train_ranker.py --model M.json --config prune_soft \
      --tag v1 [--jobs 24]

1. Cross-fitting: 5 point models on the PDB train point samples, folds by ``cluster30`` exactly as
   scripts/burrow_train_biolip.py (same sampling, params from the model's meta); cached under
   results/dev_af/cache/folds/<model-hash>/.  An AF train model is scored with the fold model that
   did not see its entry's cluster (out-of-fold point scores, as for the standard ranker).
2. AF train table: AF model -> AF-mode input (``--config``) -> point features -> OOF scores ->
   candidates -> standard + AF pocket features; target q = min DCC over transferred sites whose
   ligand atoms are within DCA <= 4 Å (graded target clip(1 - q/12, 0, 1), as the standard ranker).
3. PDB train table (``--mix``): the standard ranker's inputs (chain + assembly-1) with OOF scores;
   AF features computed with uniform pLDDT = 100 (confident input).
4. Rankers (300 rounds, RANK_PARAMS, cross-entropy, SEEDS[seed_idx]) for feature sets
   std / std+pl / std+pl+dom and data af / af+pdb -> results/dev_af/models/afr_<tag>_<data>_<feats>.json
Tables are cached (results/dev_af/cache/table_<tag>_{af,pdb}.npz) so rankers can be re-fit quickly.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import multiprocessing as mp
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
sys.path.insert(0, str(ROOT / "scripts" / "data"))
import burrow_train_biolip as BT  # noqa: E402
from build_af_train_io import load_af_entry, read_af_manifest  # noqa: E402
from pocketdetect.burrow.af_rank import AF_FEATURES, AFRanker, af_pocket_features, candidate_table  # noqa: E402
from pocketdetect.burrow.af_variants import CONFIGS, af_input  # noqa: E402
from pocketdetect.burrow.biolip import ASM, entries, sites, structure_ids  # noqa: E402
from pocketdetect.burrow.model import (POINT_PARAMS, RANK_PARAMS, BurrowModel, FormationConfig, candidates,  # noqa: E402
                                       pocket_features, score_points, train_lgb)
from pocketdetect.burrow.pockets import assign_residues  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402
from pocketdetect.structure import load_structure  # noqa: E402

OUT = ROOT / "results/dev_af"
_G = {}
DOM = ["af_dom_span", "af_dom_pack"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def graded(q):
    return np.clip(1.0 - q / 12.0, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------- fold point models
def fold_models(model: BurrowModel, model_path: Path, jobs: int):
    meta = model.meta
    h = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:12] + "_" + \
        hashlib.sha256(os.environ.get("BURROW_TRAINSET", "").encode()).hexdigest()[:6]
    d = OUT / "cache/folds" / h
    fdir = Path(meta["feats_cache"])
    mt = entries("train")
    tr_ids = [s for s, _, _ in structure_ids(mt, True) if (fdir / f"{s}.npz").exists()]
    grp_of = dict(zip(mt.entry_id, mt.cluster30))
    ug = sorted({grp_of[s.replace(ASM, '')] for s in tr_ids})
    perm = np.random.default_rng(SEEDS[0]).permutation(len(ug))
    fold_g = {g: int(perm[i] % 5) for i, g in enumerate(ug)}
    import lightgbm as lgb
    if (d / "fold4.txt").exists():
        log(f"fold models cached in {d}")
        return [lgb.Booster(model_file=str(d / f"fold{k}.txt")) for k in range(5)], fold_g, tr_ids, mt
    d.mkdir(parents=True, exist_ok=True)
    BT._G.update(feats=str(fdir), variant=meta.get("variant", "all"), seed_idx=0, pos_frac=meta.get("pos_frac", 0.3),
                 neg_frac=meta.get("neg_frac", 0.03), score_floor=meta.get("score_floor"))
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        samp = {s: (X, y) for s, X, y in ex.map(BT._sample_job, tr_ids, chunksize=8)}
    pp = dict(POINT_PARAMS, num_leaves=meta.get("point_leaves", 63), num_threads=jobs)
    out = []
    for k in range(5):
        trk = [s for s in tr_ids if fold_g[grp_of[s.replace(ASM, '')]] != k]
        m = train_lgb(np.concatenate([samp[s][0] for s in trk]), np.concatenate([samp[s][1] for s in trk]), pp,
                      meta.get("point_rounds", 250), 0)
        m.save_model(str(d / f"fold{k}.txt"))
        out.append(m)
        log(f"fold {k} point model trained ({len(trk)} inputs)")
    (d / "info.json").write_text(json.dumps(dict(model=str(model_path), trainset=os.environ.get("BURROW_TRAINSET", ""),
                                                 n_inputs=len(tr_ids), feats_cache=str(fdir))))
    return out, fold_g, tr_ids, mt


# ----------------------------------------------------------------------------- AF train table
def _af_job(args):
    eid, npz, fold = args
    try:
        e = load_af_entry(npz)
        s = load_structure(e.meta["model_path"], structure_id=eid, protein_only=True)
        P, w, _ = af_input(s, _G["cfg"])
        model = _G["model"]
        fb = _G["folds"][fold]
        floor = model.meta.get("score_floor")
        pk, ps, score, F, names = candidate_table(
            P, model, w, score_override=lambda ps: score_points(fb, ps.feats, ps.names, model.point_features, floor))
        C = np.array([p.center for p in pk]).reshape(-1, 3)
        xyz, site = e.ligand_atoms(ions=True)
        cen = e.arrays["site_center"]
        q = np.full(len(pk), np.inf)
        for i in range(e.n_sites):
            at = xyz[site == i]
            if len(C) and len(at):
                dcc = np.linalg.norm(C - cen[i], axis=1)
                dca = cKDTree(at).query(C, k=1)[0]
                q = np.minimum(q, np.where(dca <= 4, dcc, np.inf))
        return eid, F, names, q
    except Exception as ex:
        print(f"AF job {eid}: {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)
        return eid, np.zeros((0, 0), np.float32), None, np.zeros(0)


# ----------------------------------------------------------------------------- PDB train table
def _pdb_job(sid):
    try:
        d = BT.load_feats(sid)
        score = _G["scores"][sid]
        eid = sid.replace(ASM, "")
        from pocketdetect.datasets.trainset import load_entry
        e = load_entry(_G["npz"][eid])
        from pocketdetect.burrow.biolip import load_protein_entry
        P = load_protein_entry(e, assembly=sid.endswith(ASM))
        P.plddt = np.full(len(P.coords), 100.0, np.float32)
        fc = _G["fc"]
        pk = candidates(d["X"], d["points"], d["vol"], d["feats"], d["names"], score, fc)
        assign_residues(pk, d["points"], cKDTree(P.coords), P.res_index, P.residue_ids)
        Fp, fn = pocket_features(pk, d["X"], d["vol"], d["feats"], d["names"], score, len(P.coords))
        C = np.array([p.center for p in pk]).reshape(-1, 3)
        q = np.full(len(pk), np.inf)
        for st in sites(e, _G["variant"], sid.endswith(ASM)):
            if len(C):
                dcc = np.linalg.norm(C - st["center"], axis=1)
                dca = cKDTree(st["atoms"]).query(C, k=1)[0]
                q = np.minimum(q, np.where(dca <= 4, dcc, np.inf))
        if not len(pk):
            return sid, np.zeros((0, 0), np.float32), None, q
        Fa = af_pocket_features(pk, d["points"], P)
        return sid, np.concatenate([Fp, Fa], 1), list(fn) + AF_FEATURES, q
    except Exception as ex:
        print(f"PDB job {sid}: {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)
        return sid, np.zeros((0, 0), np.float32), None, np.zeros(0)


def stack(res):
    names = next(r[2] for r in res if r[2])
    F = np.concatenate([r[1] for r in res if r[2]])
    q = np.concatenate([r[3] for r in res if r[2]])
    sid = np.concatenate([[r[0]] * len(r[3]) for r in res if r[2]])
    return F, q, names, sid


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "results/dev/biolip/models/burrow_all_s0.json")
    ap.add_argument("--config", default="prune_soft", choices=sorted(CONFIGS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--rank-rounds", type=int, default=300)
    ap.add_argument("--no-mix", action="store_true")
    ap.add_argument("--fsets", nargs="+", default=["std", "pl", "pldom", "h50"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--folds-only", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    model = BurrowModel.load(a.model)
    if a.folds_only:
        fold_models(model, a.model, a.jobs)
        return
    # fold models are trained in a child process: no OpenMP state in this parent before forking workers
    import subprocess
    subprocess.run([sys.executable, __file__, "--folds-only", "--model", str(a.model), "--tag", a.tag,
                    "--jobs", str(a.jobs)], check=True)
    model.boosters()
    folds, fold_g, tr_ids, mt = fold_models(model, a.model, a.jobs)
    ctx = mp.get_context("fork")
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    # --- AF train table
    taf = cache / f"table_{a.tag}_af.npz"
    if taf.exists():
        z = np.load(taf)
        Fa, qa, na, sa = z["F"], z["q"], list(z["names"]), z["sid"]
    else:
        am = read_af_manifest("train")
        tasks = [(e, n, fold_g[g]) for e, n, g in zip(am.entry_id, am.npz, am.cluster30) if g in fold_g]
        log(f"AF train entries {len(am)}, with a fold {len(tasks)}")
        if a.limit:
            tasks = tasks[:a.limit]
        _G.update(cfg=CONFIGS[a.config], model=model, folds=folds)
        with ProcessPoolExecutor(a.jobs, mp_context=ctx) as ex:
            res = list(ex.map(_af_job, tasks, chunksize=2))
        Fa, qa, na, sa = stack(res)
        np.savez(taf, F=Fa, q=qa, names=np.array(na), sid=sa)
    log(f"AF table: {len(qa)} candidates from {len(set(sa))} models, hit rate {(qa <= 12).mean():.3f}")
    # --- PDB train table
    if not a.no_mix:
        tpdb = cache / f"table_{a.tag}_pdb.npz"
        if tpdb.exists():
            z = np.load(tpdb)
            Fb, qb, nb, sb = z["F"], z["q"], list(z["names"]), z["sid"]
        else:
            ids = tr_ids[:a.limit] if a.limit else tr_ids
            grp_of = dict(zip(mt.entry_id, mt.cluster30))
            meta = model.meta
            BT._G.update(feats=meta["feats_cache"], variant=meta.get("variant", "all"), score_floor=meta.get("score_floor"))
            scores = {}
            for s in ids:
                d = BT.load_feats(s)
                fb = folds[fold_g[grp_of[s.replace(ASM, '')]]]
                scores[s] = score_points(fb, d["feats"], d["names"], d["names"], meta.get("score_floor"), num_threads=a.jobs)
            log("PDB OOF scores done")
            _G.update(scores=scores, npz=dict(zip(mt.entry_id, mt.npz)), fc=FormationConfig(**model.formation),
                      variant=meta.get("variant", "all"))
            with ProcessPoolExecutor(a.jobs, mp_context=ctx) as ex:
                res = list(ex.map(_pdb_job, ids, chunksize=4))
            Fb, qb, nb, sb = stack(res)
            np.savez(tpdb, F=Fb, q=qb, names=np.array(nb), sid=sb)
        assert nb == na
        log(f"PDB table: {len(qb)} candidates, hit rate {(qb <= 12).mean():.3f}")
    # --- rankers
    std = [n for n in na if n not in AF_FEATURES]
    fsets = {"std": std, "pl": std + [f for f in AF_FEATURES if f not in DOM], "pldom": std + AF_FEATURES,
             "h50": std + ["af_res_lt50", "af_pts_lt50"]}  # hallucination-only cues (pLDDT < 50)
    params = dict(RANK_PARAMS, objective="cross_entropy", num_threads=a.jobs)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    datas = {"af": (Fa, qa)}
    if not a.no_mix:
        datas["afpdb"] = (np.concatenate([Fa, Fb]), np.concatenate([qa, qb]))
    for dn, (F, q) in datas.items():
        for fsn, fs in fsets.items():
            if fsn not in a.fsets:
                continue
            j = [na.index(n) for n in fs]
            R = train_lgb(F[:, j], graded(q), params, a.rank_rounds, a.seed_idx)
            name = f"afr_{a.tag}_{dn}_{fsn}"
            AFRanker(R.model_to_string(), fs, meta=dict(
                base_model=str(a.model), af_config=a.config, data=dn, features=fsn, n_rows=int(len(q)),
                seed=SEEDS[a.seed_idx], rank_rounds=a.rank_rounds, target="graded clip(1-q/12)",
                trainset=os.environ.get("BURROW_TRAINSET", ""))).save(OUT / "models" / f"{name}.json")
            pd.DataFrame({"feature": fs, "gain": R.feature_importance("gain")}).sort_values("gain", ascending=False) \
                .to_csv(OUT / "models" / f"{name}_importance.csv", index=False)
            log(f"ranker {name} trained on {len(q)} rows")
    log("RANKERS_DONE")


if __name__ == "__main__":
    main()
