#!/usr/bin/env python
"""Ablation (DESIGN §2.2, mandatory): fpocket candidates + BURROW ranker, on the BioLiP2 dev set.

Each fpocket 4.2.3 pocket (centre = mean alpha-sphere centre, as in the wrapper) becomes a BURROW
candidate: its point set = BURROW grid points within 2 Å of any of its alpha-sphere centres
(fallback: the 5 grid points nearest to its centre); residues = fpocket's residues.  BURROW
pocket features (with the final point model's scores) are computed on these point sets and ranked
by the BURROW ranker trained on BURROW candidates (no retraining on fpocket candidates — a
distribution-shift caveat), then the same NMS.  Centres stay fpocket's, so the ablation isolates
candidate generation.  Also writes "fpocket candidates, fpocket order + BURROW NMS" for reference.

  python scripts/burrow_ablation_fpocket.py --model results/dev/biolip/models/burrow_all_s0.json --name abl_fpocket_all_s0
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_featurize_biolip import feats_dir  # noqa: E402
from burrow_train_dev import emit  # noqa: E402
from pocketdetect.burrow.biolip import entries  # noqa: E402
from pocketdetect.burrow.model import BurrowModel, FormationConfig, pocket_features  # noqa: E402
from pocketdetect.burrow.pockets import Pocket  # noqa: E402
from pocketdetect.evaluate import read_predictions  # noqa: E402

FP = ROOT / "results/dev/biolip/baselines/fpocket"
_G = {}


def _job(sid):
    z = np.load(Path(_G["feats"]) / f"{sid}.npz")
    names = list(z["names"])
    F, X, pts, vol = z["feats"], z["X"], z["points"], z["vol"]
    m = _G["model"]
    pb, rb = m.boosters()
    score = pb.predict(F[:, [names.index(n) for n in m.point_features]], num_threads=1)
    f = FP / f"{sid}.pockets.csv"
    if not f.exists():
        return sid, None
    pr = read_predictions(f, sid)
    if pr.n == 0:
        return sid, ([], np.zeros(0), np.zeros(0))
    ptab = pd.read_csv(FP / f"{sid}.points.csv")
    tree = cKDTree(pts)
    pk = []
    for r in range(1, pr.n + 1):
        sph = ptab.loc[ptab["rank"] == r, ["x", "y", "z"]].to_numpy(float)
        idx = np.unique(np.concatenate([np.asarray(l, np.int64) for l in tree.query_ball_point(sph, 2.0)])) \
            if len(sph) else np.zeros(0, np.int64)
        if len(idx) == 0:
            idx = np.sort(tree.query(pr.centers[r - 1], k=min(5, len(pts)))[1].reshape(-1))
        pk.append(Pocket(idx=idx, strength=0.0, center=pr.centers[r - 1], residues=frozenset(pr.residues[r - 1]),
                         level=0))
    Fp, fn = pocket_features(pk, X, vol, F, names, score, _G["n_atoms"].get(sid, 3000))
    rs = rb.predict(Fp[:, [fn.index(n) for n in m.rank_features]], num_threads=1)
    light = [(p.center, p.residues, 0.0, pts[p.idx], p.idx) for p in pk]
    return sid, (light, rs, -np.arange(pr.n, dtype=float))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args(argv)
    md = entries("dev")
    from pocketdetect.burrow.prep import load_protein
    _G.update(feats=str(feats_dir()), model=BurrowModel.load(a.model), n_atoms={})
    # protein size feature: number of heavy atoms of the input
    for eid in md.entry_id:
        z = ROOT / "results/dev/biolip/inputs" / f"{eid}.cif"
        _G["n_atoms"][eid] = len(load_protein(z).coords)
    fc = FormationConfig()
    out = ROOT / "results/dev/biolip/preds"
    n_missing = 0
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
        for sid, res in ex.map(_job, list(md.entry_id), chunksize=4):
            if res is None:
                n_missing += 1
                continue
            light, rs, fporder = res
            emit(out / a.name, sid, light, rs, fc)
            emit(out / (a.name + "_fporder"), sid, light, fporder, fc)
    print(f"done; fpocket outputs missing for {n_missing} structures (count as zero pockets)")


if __name__ == "__main__":
    main()
