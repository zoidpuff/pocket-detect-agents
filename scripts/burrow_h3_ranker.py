#!/usr/bin/env python
"""v2/H3: retrain the pocket ranker with apo-oriented pocket aggregates (model.APO_NAMES) on a run's
cached train candidates (out-of-fold point scores), keep the run's point model; write a model JSON.

  python scripts/burrow_h3_ranker.py --run p3_small_default_s0 \
      --base-model results/dev/freeze/models/burrow_small_default_s0.json --out results/dev/biolip/models_v2/H3.json
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_featurize_biolip import feats_dir, make_cfg  # noqa: E402
from burrow_train_dev import rank_params, rank_target  # noqa: E402
from pocketdetect.burrow.model import APO_NAMES, BurrowModel, apo_pocket_features, train_lgb  # noqa: E402

_G = {}


def _job(sid):
    Fp, q, light = _G["data"][sid]
    z = np.load(Path(_G["fd"]) / f"{sid}.npz")
    Fx = apo_pocket_features([l[2] for l in light], z["vol"], z["feats"], list(z["names"]))
    return sid, Fx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args(argv)
    pdir = ROOT / "results/dev/cache/pockets" / a.run
    meta = pickle.loads((pdir / "meta.pkl").read_bytes())
    args = meta["args"]
    data = pickle.loads((pdir / "train.pkl").read_bytes())
    sids = [s for s in data if len(data[s][0])]
    _G.update(data=data, fd=str(feats_dir(make_cfg(args["n_rays"], args["max_len"], args["subsample"]))))
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
        extra = dict(ex.map(_job, sids, chunksize=16))
    X = np.concatenate([np.concatenate([data[s][0], extra[s]], 1) for s in sids])
    q = np.concatenate([data[s][1] for s in sids])
    R = train_lgb(X, rank_target(q, "graded"), rank_params("graded"), 300, a.seed_idx)
    m = BurrowModel.load(a.base_model)
    m.rank_model = R.model_to_string()
    m.rank_features = list(meta["fn"]) + APO_NAMES
    m.meta = dict(m.meta, apo_feats=True, v2_variant="H3", v2_source_run=a.run)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    m.save(a.out)
    print("saved", a.out, X.shape)


if __name__ == "__main__":
    main()
