"""Prediction with experimental ranker features that are NOT part of any frozen model.

v2 hypothesis H3 (``meta.apo_feats``: apo-oriented pocket aggregates) and v3 hypothesis (b)
(``meta.if_feats``: interface-aware pocket features) were both negative (results/dev/biolip/V2_CYCLE.md,
V3_CYCLE.md).  Their hooks live here instead of ``pipeline.predict_protein`` so that the production
path stays identical to the version validated for the fast backend (``burrow/fast/guard.py``).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

from .features import BurrowConfig, point_features
from .model import (BurrowModel, FormationConfig, candidates, context_features, geo_score, pocket_features,
                    rank_predict, score_points)
from .pipeline import Prediction, finalize, output_residues
from .pockets import assign_residues
from .prep import Protein

EXPERIMENTAL_FLAGS = ("apo_feats", "if_feats")


def is_experimental(model: BurrowModel | None) -> bool:
    return model is not None and any(model.meta.get(f) for f in EXPERIMENTAL_FLAGS)


def predict_protein_experimental(P: Protein, model: BurrowModel | None, cfg: BurrowConfig = BurrowConfig(),
                    weights=None, ranker: str = "learned") -> Prediction:
    """predict_protein + the negative-experiment ranker hooks (v2/H3 ``meta.apo_feats``, v3/(b) ``meta.if_feats``)."""
    tm = {}
    t0 = time.perf_counter()
    if model is not None and model.meta.get("feature_cfg"):  # the model's own feature design
        fd = dict(model.meta["feature_cfg"])
        fd["lengths"] = tuple(fd["lengths"])
        fd["nb_radii"] = tuple(fd["nb_radii"])
        cfg = BurrowConfig(**fd)
    ps = point_features(P, cfg, weights)
    tm.update({f"feat_{k}": v for k, v in ps.timings.items()})
    t1 = time.perf_counter()
    fc = FormationConfig(**model.formation) if model is not None else FormationConfig()
    geo = model is None or ranker == "geometry"
    if geo:
        score = None
    else:
        pb, rb = model.boosters()
        score = score_points(pb, ps.feats, ps.names, model.point_features, model.meta.get("score_floor"))
    t2 = time.perf_counter()
    pk = candidates(ps.X, ps.points, ps.vol, ps.feats, ps.names, score, fc)
    tree = cKDTree(P.coords)
    assign_residues(pk, ps.points, tree, P.res_index, P.residue_ids)
    t3 = time.perf_counter()
    if geo:
        rs = geo_score(pk, ps.vol, ps.feats, ps.names, fc)
    else:
        Fp, fn = pocket_features(pk, ps.X, ps.vol, ps.feats, ps.names, score, len(P.coords))
        if len(pk):
            ctx = model.meta.get("context")
            if ctx:  # within-protein / pocket-pair context features (phase 3)
                Fx, nx = context_features(Fp, fn, np.array([p.center for p in pk]), ctx)
                Fp, fn = np.concatenate([Fp, Fx], 1), fn + nx
            if model.meta.get("apo_feats"):  # v2/H3 apo-oriented pocket aggregates
                from .model import APO_NAMES, apo_pocket_features
                Fx = apo_pocket_features([p.idx for p in pk], ps.vol, ps.feats, ps.names)
                Fp, fn = np.concatenate([Fp, Fx], 1), fn + APO_NAMES
            if model.meta.get("if_feats"):  # v3/(b) interface-aware pocket features
                from .model import IF_NAMES, interface_pocket_features
                Fx = interface_pocket_features([p.idx for p in pk], np.array([p.center for p in pk]), ps.points,
                                               P.coords, P.chain)
                Fp, fn = np.concatenate([Fp, Fx], 1), fn + IF_NAMES
            if model.meta.get("propensity"):  # atom-type ligand-contact propensity (phase 3)
                from .propensity import PROP_NAMES, atom_logodds, pocket_propensity
                lo = atom_logodds(P, model.meta["propensity"])
                Fx = pocket_propensity([p.idx for p in pk], ps.points, P, lo, tree)
                Fp, fn = np.concatenate([Fp, Fx], 1), fn + PROP_NAMES
            j = [fn.index(n) for n in model.rank_features]
            rs = rank_predict(model, Fp[:, j])
        else:
            rs = np.zeros(0)
    keep = finalize(pk, rs, fc)
    out_res = [output_residues(pk[i], tree, P) for i in keep]
    t4 = time.perf_counter()
    tm.update(features=(t1 - t0) * 1e3, point_score=(t2 - t1) * 1e3, pockets=(t3 - t2) * 1e3,
              rank=(t4 - t3) * 1e3, total=(t4 - t0) * 1e3)
    return Prediction(P.id, np.array([pk[i].center for i in keep]).reshape(-1, 3), np.asarray(rs)[keep],
                      out_res, [ps.points[pk[i].idx] for i in keep], tm)


