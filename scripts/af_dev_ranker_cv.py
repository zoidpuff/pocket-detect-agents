#!/usr/bin/env python
"""Cluster-cross-validated AF rankers (supplementary power for the ranker comparison).

Uses the cached tables of scripts/af_dev_train_ranker.py (``--tag``).  For each fold k of the
cross-fitting split (cluster30, identical to the point-model folds) a ranker is trained on all rows
of entries outside fold k (AF rows and/or PDB rows) and saved as
results/dev_af/models/cv/afrcv_<tag>_<data>_<feats>_k<k>.json.  Predicting the AF *train* models of
fold k with fold-k point model + fold-k ranker (scripts/af_dev_predict.py --oof --af-ranker '...k{k}.json')
gives fully out-of-fold predictions on 2,844 AF models.  ``pdb_std`` (PDB rows, standard features)
is the cross-validated analogue of the standard ranker, the fair reference.

  BURROW_TRAINSET=<snapshot> python scripts/af_dev_ranker_cv.py --model M.json --tag v1 [--variants pdb_std afpdb_pl ...]
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from af_dev_train_ranker import DOM, graded  # noqa: E402
from pocketdetect.burrow.af_rank import AF_FEATURES, AFRanker  # noqa: E402
from pocketdetect.burrow.biolip import ASM, entries, structure_ids  # noqa: E402
from pocketdetect.burrow.model import RANK_PARAMS, train_lgb  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402

OUT = ROOT / "results/dev_af"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variants", nargs="+", default=["pdb_std", "afpdb_std", "afpdb_pl", "af_pl"])
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--rank-rounds", type=int, default=300)
    a = ap.parse_args(argv)
    za = np.load(OUT / "cache" / f"table_{a.tag}_af.npz")
    zb = np.load(OUT / "cache" / f"table_{a.tag}_pdb.npz")
    names = list(za["names"])
    # fold of each entry: same rule as af_dev_train_ranker.fold_models (no booster loading needed)
    feats = None
    import json
    m = json.loads(Path(a.model).read_text())["meta"]
    fdir = Path(m["feats_cache"])
    mt = entries("train")
    tr_ids = [s for s, _, _ in structure_ids(mt, True) if (fdir / f"{s}.npz").exists()]
    grp_of = dict(zip(mt.entry_id, mt.cluster30))
    ug = sorted({grp_of[s.replace(ASM, '')] for s in tr_ids})
    perm = np.random.default_rng(SEEDS[0]).permutation(len(ug))
    fold_g = {g: int(perm[i] % 5) for i, g in enumerate(ug)}
    fa = np.array([fold_g[grp_of[s]] for s in za["sid"]])
    fb = np.array([fold_g[grp_of[s.replace(ASM, '')]] for s in zb["sid"]])
    std = [n for n in names if n not in AF_FEATURES]
    fsets = {"std": std, "pl": std + [f for f in AF_FEATURES if f not in DOM], "pldom": std + AF_FEATURES,
             "h50": std + ["af_res_lt50", "af_pts_lt50"]}  # hallucination-only cues (pLDDT < 50)
    params = dict(RANK_PARAMS, objective="cross_entropy", num_threads=a.jobs)
    d = OUT / "models" / "cv"
    d.mkdir(parents=True, exist_ok=True)
    for v in a.variants:
        data, fs = v.split("_")
        j = [names.index(n) for n in fsets[fs]]
        for k in range(5):
            parts = []
            if "af" in data:
                parts.append((za["F"][fa != k][:, j], za["q"][fa != k]))
            if "pdb" in data:
                parts.append((zb["F"][fb != k][:, j], zb["q"][fb != k]))
            X = np.concatenate([p[0] for p in parts])
            q = np.concatenate([p[1] for p in parts])
            R = train_lgb(X, graded(q), params, a.rank_rounds, 0)
            AFRanker(R.model_to_string(), fsets[fs], meta=dict(cv_fold=k, data=data, features=fs, n_rows=int(len(q)),
                                                               tag=a.tag)).save(d / f"afrcv_{a.tag}_{v}_k{k}.json")
            print(f"[{time.strftime('%H:%M:%S')}] {v} fold {k}: {len(q)} rows", flush=True)
    print("CV_DONE")


if __name__ == "__main__":
    main()
