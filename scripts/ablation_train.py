#!/usr/bin/env python
"""Pre-registered BURROW ablations (DESIGN §2.5, §5): train ablated models with the FREEZE v1 recipe.

Training logic = ``scripts/burrow_train_biolip.py`` at the freeze commit (copied here, theirs is not
edited): BioLiP2 train set (audited D6 snapshot), MMseqs2-30 % 5-fold cross-fitted point scores,
LightGBM point model (250 x 63, score floor 0.7, pos 0.3 / neg 0.03 sampling) + graded pocket ranker
(300 rounds), default preset features (30 rays, 6/10/16 Å, 3 Å sub-lattice), seed SEEDS[k].
Run with the frozen code: ``--code-root`` = git worktree at 5e13267.

Ablations (``--ablation``):
  none                  re-train v1 (sanity: must reproduce the frozen model)
  fold_disjoint (A1)    drop training entries with Foldseek max(qTM, tTM) >= 0.5 to any test chain
                        (data/processed/train/fold_to_test.tsv, column fold_disjoint)
  chen11 (A2)           train on CHEN11 only (P2Rank's training set, P2Rank ligand rules, no ions;
                        features computed here with the default preset)
  shuffled (A3)         negative control: point labels and pocket targets permuted within protein
  drop:<group> (A4)     feature-group drop-out (point model + ranker): rays, hitchem, bbbur, metal,
                        rescomp, atomclass (see GROUPS); candidate generation is unchanged

    python scripts/ablation_train.py --code-root /tmp/claude-548918/wt/abl --variant small --ablation drop:rays
Output: results/ablations/models/<name>.json (+ importance CSVs, train log).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import multiprocessing as mp
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(__file__).resolve().parents[1]
SNAP = MAIN / "results/dev/cache/trainset_snapshot/audited_d6"
FEATS = MAIN / "results/dev/cache/feats_biolip/69ddda668a"       # default-preset feature cache (v1)
CHEN_FEATS = MAIN / "data/processed/ablations/feats_chen11_default"
OUT = MAIN / "results/ablations"

GROUPS = {   # point-feature regex, pocket-descriptor regex
    "rays": (r"^(bur(6|10|16|24)|depth)$", r"^(bur(10|16|24)_(mean|q90|q10)|depth_mean|geo_strength)$"),
    "hitchem": (r"^(hit_\w+|escape)$", r"^(hit_\w+|escape)$"),
    "bbbur": (r"^burbb\d+$", r"^burbb\d+_mean$"),
    "metal": (r"^(metal_cand|n_coord28|coord_spread|(n4|f8|f12)_coord)$", r"^(metal_n|n_coord_max|f8_coord)$"),
    "rescomp": (r"^(n4|f8|f12)_rg_\w+$", r"^(n4|f8)_rg_\w+$"),
    "atomclass": (r"^(n4|f8|f12)_(C|N|O|S|aromatic|hydrophobic_sc_C|donor|acceptor|backbone|sidechain|positive|negative)$",
                  r"^f8_(C|N|O|S|aromatic|hydrophobic_sc_C|donor|acceptor|backbone|sidechain|positive|negative)$"),
}
_G = {}


def setup_paths(code_root: Path):
    sys.path.insert(0, str(code_root / "scripts"))
    sys.path.insert(0, str(code_root / "src"))
    os.environ["BURROW_TRAINSET"] = str(SNAP)


# ----------------------------------------------------------------------------- data access
def load_feats(sid):
    z = np.load(Path(_G["feats"]) / f"{sid}.npz")
    return {k: z[k] for k in ("points", "X", "vol", "feats", "d_lig", "d_ion")} | {"names": list(z["names"])}


def point_label(d, variant):
    y = d["d_lig"] <= 4.0
    if variant == "all":
        y |= d["d_ion"] <= 3.0
    return y.astype(np.int8)


def _rng(sid, salt=None):
    """Point-sampling RNG exactly as the freeze trainer ([seed, hash(sid)]); salted streams for A3."""
    h = int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)
    from pocketdetect.seeds import SEEDS
    return np.random.default_rng([SEEDS[_G["seed_idx"]], h] + ([] if salt is None else [salt]))


def _sample_job(sid):
    from pocketdetect.burrow.model import geo_value
    d = load_feats(sid)
    y = point_label(d, _G["variant"])
    if _G["shuffle"]:
        y = _rng(sid, 7).permutation(y)               # A3: labels permuted within protein
    r = _rng(sid).random(len(y))
    m = np.where(y == 1, r < _G["pos_frac"], r < _G["neg_frac"])
    if _G["score_floor"] is not None:
        F = d["feats"]
        m &= (geo_value(F, d["names"]) >= _G["score_floor"]) | (F[:, d["names"].index("metal_cand")] > 0)
    return sid, d["feats"][m][:, _G["keep_idx"]], y[m]


def _sites(sid):
    """Target sites [(centre, ligand atoms)] of a training structure."""
    if _G["dataset"] == "chen11":
        return _G["chen_sites"][sid]
    from pocketdetect.burrow.biolip import ASM, sites
    from pocketdetect.datasets.trainset import load_entry
    e = load_entry(_G["npz"][sid.replace(ASM, "")])
    return [(s["center"], s["atoms"]) for s in sites(e, _G["variant"], sid.endswith(ASM))]


def _protein(sid):
    if _G["dataset"] == "chen11":
        from pocketdetect.burrow.prep import load_protein
        return load_protein(_G["chen_paths"][sid], structure_id=sid)
    from pocketdetect.burrow.biolip import ASM, load_protein_entry
    from pocketdetect.datasets.trainset import load_entry
    return load_protein_entry(load_entry(_G["npz"][sid.replace(ASM, "")]), assembly=sid.endswith(ASM))


def _pocket_job(sid):
    from scipy.spatial import cKDTree
    from pocketdetect.burrow.model import candidates, pocket_features
    from pocketdetect.burrow.pockets import assign_residues
    d = load_feats(sid)
    score = _G["scores"][sid]
    P = _protein(sid)
    pk = candidates(d["X"], d["points"], d["vol"], d["feats"], d["names"], score, _G["fc"])
    assign_residues(pk, d["points"], cKDTree(P.coords), P.res_index, P.residue_ids)
    Fp, fn = pocket_features(pk, d["X"], d["vol"], d["feats"], d["names"], score, len(P.coords))
    C = np.array([p.center for p in pk]).reshape(-1, 3)
    y = np.full(len(pk), np.inf)
    for c, atoms in _sites(sid):
        if len(C):
            dcc = np.linalg.norm(C - c, axis=1)
            dca = cKDTree(atoms).query(C, k=1)[0]
            y = np.minimum(y, np.where(dca <= 4, dcc, np.inf))
    if _G["shuffle"] and len(y):
        y = _rng(sid, 11).permutation(y)               # A3: pocket targets permuted within protein
    return sid, Fp, fn, y


def predict_scores(model, sids):
    """OOF point scores (== model.score_points with the v1 score floor, on the kept feature columns)."""
    return {s: _floor_scores(model, load_feats(s)) for s in sids}


def _floor_scores(model, d):
    from pocketdetect.burrow.model import geo_value
    F = d["feats"]
    m = (geo_value(F, d["names"]) >= _G["score_floor"]) | (F[:, d["names"].index("metal_cand")] > 0)
    out = np.zeros(len(F), np.float32)
    if m.any():
        out[m] = model.predict(F[m][:, _G["keep_idx"]], num_threads=_G["threads"])
    return out


# ----------------------------------------------------------------------------- CHEN11 features
def _chen_job(args):
    sid, path = args
    from pocketdetect.burrow.devdata import ligand_distance
    from pocketdetect.burrow.features import point_features
    from pocketdetect.burrow.prep import load_protein
    out = CHEN_FEATS / f"{sid}.npz"
    if out.exists():
        return sid, "cached"
    P = load_protein(path, structure_id=sid)
    ps = point_features(P, _G["cfg"])
    gt = _G["gts"][sid]
    d_all, d_ion = ligand_distance(ps.points, gt)
    d_lig = np.full(len(ps.points), np.inf)        # non-ion ligands (P2Rank rules: in practice no ions)
    from pocketdetect.burrow.devdata import ION_MAX_ATOMS
    from scipy.spatial import cKDTree
    for lig in gt.ligands:
        if lig.n_atoms > ION_MAX_ATOMS:
            d_lig = np.minimum(d_lig, cKDTree(lig.coords).query(ps.points, k=1)[0])
    np.savez(out, points=ps.points, X=ps.X, vol=ps.vol, is_metal=ps.is_metal, feats=ps.feats,
             names=np.array(ps.names), d_lig=d_lig.astype(np.float32), d_ion=d_ion.astype(np.float32))
    return sid, "ok"


def chen11_data(cfg, jobs):
    from pocketdetect.burrow.devdata import load_gts, read_manifest
    man = MAIN / "results/dev/manifests/chen11.csv"
    df = read_manifest(man)
    gts = load_gts(man, "p2rank", jobs=jobs)
    CHEN_FEATS.mkdir(parents=True, exist_ok=True)
    _G.update(cfg=cfg, gts=gts)
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(_chen_job, list(zip(df.structure_id, df.structure_path)), chunksize=2))
    sites = {s: [(lig.coords.mean(0), lig.coords) for lig in gts[s].ligands] for s in df.structure_id}
    import json
    groups = json.loads((MAIN / "results/dev/manifests/groups_all_mmseqs30.json").read_text())
    return list(df.structure_id), dict(zip(df.structure_id, df.structure_path)), sites, \
        {s: groups.get(s, s) for s in df.structure_id}, sum(r[1] == "ok" for r in res)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-root", type=Path, required=True, help="worktree at the freeze commit (5e13267)")
    ap.add_argument("--variant", choices=["all", "small"], default="small")
    ap.add_argument("--ablation", default="none")
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--tag", default="", help="suffix for the model name")
    a = ap.parse_args(argv)
    setup_paths(a.code_root)
    from burrow_featurize_biolip import make_cfg
    from burrow_train_dev import rank_params, rank_target
    from pocketdetect.burrow.biolip import ASM, entries, structure_ids
    from pocketdetect.burrow.model import POINT_PARAMS, BurrowModel, FormationConfig, train_lgb
    from pocketdetect.seeds import SEEDS
    # FREEZE v1 default preset hyper-parameters
    fcfg = make_cfg(30, 16.0, 3)
    hp = dict(pos_frac=0.3, neg_frac=0.03, point_rounds=250, rank_rounds=300, point_leaves=63, score_floor=0.7)
    abl = a.ablation
    name = f"abl_{abl.replace(':', '-')}_{a.variant}_s{a.seed_idx}{a.tag}"
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    _G.update(variant=a.variant, seed_idx=a.seed_idx, pos_frac=hp["pos_frac"], neg_frac=hp["neg_frac"],
              score_floor=hp["score_floor"], fc=FormationConfig(), shuffle=abl == "shuffled",
              dataset="chen11" if abl == "chen11" else "biolip", feats=str(FEATS), threads=a.jobs)
    if abl == "chen11":
        tr_ids, paths, sites, grp_of, n_new = chen11_data(fcfg, a.jobs)
        _G.update(feats=str(CHEN_FEATS), chen_paths=paths, chen_sites=sites)
        log(f"CHEN11 features: {n_new} computed, {len(tr_ids)} structures")
    else:
        mt = entries("train")
        tr = [(s, asm) for s, _, asm in structure_ids(mt, True) if (FEATS / f"{s}.npz").exists()]
        if abl == "fold_disjoint":
            f2t = pd.read_csv(MAIN / "data/processed/train/fold_to_test.tsv", sep="\t")
            keep = set(f2t.entry_id[f2t.fold_disjoint.astype(str) == "True"])
            n0 = len(tr)
            tr = [(s, x) for s, x in tr if s.replace(ASM, "") in keep]
            log(f"fold-disjoint: kept {len(tr)}/{n0} train inputs ({len(keep)} disjoint entries in fold_to_test)")
        tr_ids = [s for s, _ in tr]
        grp_of = {s: g for s, g in zip(mt.entry_id, mt.cluster30)}
        grp_of = {s: grp_of[s.replace(ASM, "")] for s in tr_ids}
        _G["npz"] = dict(zip(mt.entry_id, mt.npz))
    names = load_feats(tr_ids[0])["names"]
    drop_pt, drop_pk = set(), None
    if abl.startswith("drop:"):
        g = abl.split(":", 1)[1]
        rp, rk = GROUPS[g]
        drop_pt = {n for n in names if re.match(rp, n)}
        drop_pk = rk
        log(f"drop group {g}: {len(drop_pt)} point features {sorted(drop_pt)}")
    keep_names = [n for n in names if n not in drop_pt]
    _G.update(keep_idx=[names.index(n) for n in keep_names], keep_names=keep_names)
    # folds by cluster
    ug = sorted(set(grp_of.values()), key=str)
    perm = np.random.default_rng(SEEDS[0]).permutation(len(ug))
    fold_g = {g: int(perm[i] % 5) for i, g in enumerate(ug)}
    fold = {s: fold_g[grp_of[s]] for s in tr_ids}
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(a.jobs, mp_context=ctx) as ex:
        samp = {s: (Xs, ys) for s, Xs, ys in ex.map(_sample_job, tr_ids, chunksize=8)}
    yall = np.concatenate([v[1] for v in samp.values()])
    log(f"train inputs {len(tr_ids)}; point samples {len(yall)} rows, pos {yall.mean():.3f}")
    point_params = dict(POINT_PARAMS, num_leaves=hp["point_leaves"], num_threads=a.jobs)
    oof = {}
    for k in range(5):
        trk = [s for s in tr_ids if fold[s] != k]
        m = train_lgb(np.concatenate([samp[s][0] for s in trk]), np.concatenate([samp[s][1] for s in trk]),
                      point_params, hp["point_rounds"], a.seed_idx)
        oof.update(predict_scores(m, [s for s in tr_ids if fold[s] == k]))
        log(f"fold {k} point model done")
    pm = train_lgb(np.concatenate([v[0] for v in samp.values()]), yall, point_params, hp["point_rounds"], a.seed_idx)
    del samp
    _G["scores"] = oof
    with ProcessPoolExecutor(a.jobs, mp_context=ctx) as ex:
        pk = [r for r in ex.map(_pocket_job, tr_ids, chunksize=4)]
    fn = next(r[2] for r in pk if r[2])
    keep_fn = [n for n in fn if not (drop_pk and re.match(drop_pk, n))]
    kj = [fn.index(n) for n in keep_fn]
    Xr = np.concatenate([r[1][:, kj] for r in pk if len(r[3])])
    yr = np.concatenate([r[3] for r in pk if len(r[3])])
    log(f"ranker data: {len(yr)} candidates, hit rate {(yr <= 12).mean():.3f}; ranker features {len(keep_fn)}/{len(fn)}")
    R = train_lgb(Xr, rank_target(yr, "graded"), dict(rank_params("graded"), num_threads=a.jobs), hp["rank_rounds"],
                  a.seed_idx)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    BurrowModel(pm.model_to_string(), keep_names, R.model_to_string(), keep_fn,
                meta=dict(ablation=abl, variant=a.variant, rank_label="graded", score_floor=hp["score_floor"],
                          point_leaves=hp["point_leaves"], feature_cfg=asdict(fcfg), seed=SEEDS[a.seed_idx],
                          seed_idx=a.seed_idx, n_train_inputs=len(tr_ids), point_rounds=hp["point_rounds"],
                          rank_rounds=hp["rank_rounds"], pos_frac=hp["pos_frac"], neg_frac=hp["neg_frac"],
                          dropped_point_features=sorted(drop_pt), preset="default",
                          trained_on="CHEN11 (P2Rank datasets, P2Rank ligand rules)" if abl == "chen11"
                          else "BioLiP2 train (audited D6 snapshot)",
                          lgb_threads=a.jobs, code="scripts/ablation_train.py with frozen code (worktree 5e13267)")
                ).save(OUT / "models" / f"{name}.json")
    pd.DataFrame({"feature": keep_names, "gain": pm.feature_importance("gain")}).sort_values("gain", ascending=False) \
        .to_csv(OUT / "models" / f"{name}_point_importance.csv", index=False)
    pd.DataFrame({"feature": keep_fn, "gain": R.feature_importance("gain")}).sort_values("gain", ascending=False) \
        .to_csv(OUT / "models" / f"{name}_rank_importance.csv", index=False)
    log(f"done -> {OUT / 'models' / name}.json")


if __name__ == "__main__":
    main()
