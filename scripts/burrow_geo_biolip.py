#!/usr/bin/env python
"""Geometry-only BURROW baseline (no learning) on the BioLiP2 dev set -> results/dev/biolip/preds/burrow_geo."""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_featurize_biolip import feats_dir  # noqa: E402
from pocketdetect.burrow.biolip import entries  # noqa: E402
from pocketdetect.burrow.model import FormationConfig, candidates, geo_score  # noqa: E402
from pocketdetect.burrow.pipeline import Prediction, finalize, write_prediction  # noqa: E402
from pocketdetect.burrow.pockets import assign_residues  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402

OUT = ROOT / "results/dev/biolip/preds/burrow_geo"


def job(sid):
    z = np.load(feats_dir() / f"{sid}.npz")
    names = list(z["names"])
    P = load_protein(ROOT / "results/dev/biolip/inputs" / f"{sid}.cif", structure_id=sid)
    fc = FormationConfig()
    pk = candidates(z["X"], z["points"], z["vol"], z["feats"], names, None, fc)
    assign_residues(pk, z["points"], cKDTree(P.coords), P.res_index, P.residue_ids)
    rs = geo_score(pk, z["vol"], z["feats"], names, fc)
    keep = finalize(pk, rs, fc)
    write_prediction(OUT, Prediction(sid, np.array([pk[i].center for i in keep]).reshape(-1, 3), rs[keep],
                                     [pk[i].residues for i in keep], [z["points"][pk[i].idx] for i in keep], {}))
    return sid


if __name__ == "__main__":
    with ProcessPoolExecutor(8, mp_context=mp.get_context("fork")) as ex:
        n = len(list(ex.map(job, list(entries("dev").entry_id), chunksize=4)))
    print("written", n, OUT)
