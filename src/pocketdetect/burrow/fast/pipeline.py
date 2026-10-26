"""Fast BURROW prediction for one structure (CPU): same outputs as ``burrow.pipeline.predict_protein``.

Stages: point features (``fast.features`` if available for the current reference feature set,
else the reference featurizer) -> point scores (FlatForest, bit-identical to LightGBM) ->
candidate pockets + residues (``fast.formation``) -> pocket descriptors (``fast.pocketfeat``) ->
ranker (FlatForest) -> NMS (``fast.formation.nms``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..features import BurrowConfig
from ..model import BurrowModel, FormationConfig
from ..pipeline import Prediction
from . import formation as FF
from .pocketfeat import pocket_features, residue_chains
from .trees import FlatForest


@dataclass
class FastModel:
    """A :class:`BurrowModel` compiled to flat forests (+ lazily created GPU forests)."""
    model: BurrowModel | None
    point: FlatForest | None
    rank: FlatForest | None
    fc: FormationConfig

    @classmethod
    def from_model(cls, model: BurrowModel | None) -> "FastModel":
        if model is None:
            return cls(None, None, None, FormationConfig())
        pb, rb = model.boosters()
        return cls(model, FlatForest.from_booster(pb), FlatForest.from_booster(rb) if rb is not None else None,
                   FormationConfig(**model.formation))

    @classmethod
    def load(cls, path) -> "FastModel":
        return cls.from_model(BurrowModel.load(path))

    def point_index(self, names):
        return [names.index(n) for n in self.model.point_features]

    def rank_index(self, names):
        return [names.index(n) for n in self.model.rank_features]


def score_mask(fm: FastModel, F, names):
    """Points that the model scores (``model.meta['score_floor']``: b >= floor or metal candidate)."""
    floor = fm.model.meta.get("score_floor") if fm.model is not None else None
    if floor is None:
        return None
    gv = (F[:, names.index("bur10")] + F[:, names.index("bur16")] + F[:, names.index("bur24")]) / 3.0
    return (gv >= floor) | (F[:, names.index("metal_cand")] > 0)


def score_points(fm: FastModel, F, names):
    """Same as ``model.score_points`` (float32 scores; optional buriedness floor)."""
    if len(F) == 0:
        return np.zeros(0, np.float32)
    idx = fm.point_index(names)
    m = score_mask(fm, F, names)
    if m is None:
        return fm.point.predict(F[:, idx]).astype(np.float32)
    out = np.zeros(len(F), np.float32)
    if m.any():
        out[m] = fm.point.predict(F[m][:, idx])
    return out


def geo_scores(c, vol, gv, fc: FormationConfig):
    """Geometry-only pocket score, identical to ``model.geo_score`` (float32 pairwise sums)."""
    w = vol * np.clip((gv - fc.geo_floor) / (1.0 - fc.geo_floor), 0.0, 1.0) ** fc.geo_power
    return np.array([float(w[c.members(i)].sum()) for i in range(len(c))])


def featurize(P, cfg: BurrowConfig = BurrowConfig(), weights=None):
    """Point features: fast implementation when it matches the reference feature set."""
    from . import features as FFE
    if FFE.supports(cfg):
        return FFE.point_features(P, cfg, weights)
    from ..features import point_features
    return point_features(P, cfg, weights)


def _reference_post(P, ps, fm: FastModel, ranker, tm):
    """Reference implementation of everything after the features (drift fallback)."""
    from scipy.spatial import cKDTree
    from ..model import candidates, geo_score, pocket_features as ref_pf, score_points as ref_score
    from ..pipeline import finalize
    from ..pockets import assign_residues
    t1 = time.perf_counter()
    geo = fm.model is None or ranker == "geometry"
    score = None
    if not geo:
        pb, rb = fm.model.boosters()
        score = ref_score(pb, ps.feats, ps.names, fm.model.point_features, fm.model.meta.get("score_floor"))
    t2 = time.perf_counter()
    pk = candidates(ps.X, ps.points, ps.vol, ps.feats, ps.names, score, fm.fc)
    assign_residues(pk, ps.points, cKDTree(P.coords), P.res_index, P.residue_ids)
    t3 = time.perf_counter()
    if geo:
        rs = geo_score(pk, ps.vol, ps.feats, ps.names, fm.fc)
    else:
        Fp, fn = ref_pf(pk, ps.X, ps.vol, ps.feats, ps.names, score, len(P.coords))
        rs = rb.predict(Fp[:, [fn.index(n) for n in fm.model.rank_features]], num_threads=1) if len(pk) else np.zeros(0)
    keep = finalize(pk, rs, fm.fc)
    t4 = time.perf_counter()
    tm.update(point_score=(t2 - t1) * 1e3, pockets=(t3 - t2) * 1e3, rank=(t4 - t3) * 1e3)
    return Prediction(P.id, np.array([pk[i].center for i in keep]).reshape(-1, 3), np.asarray(rs)[keep],
                      [pk[i].residues for i in keep], [ps.points[pk[i].idx] for i in keep], tm)


def predict_from_points(P, ps, fm: FastModel, ranker: str = "learned", score=None, timings=None) -> Prediction:
    """Everything after the point features (CPU).  ``score`` may be precomputed (e.g. on GPU)."""
    from .guard import stage_ok
    tm = {} if timings is None else timings
    if not all(stage_ok(s) for s in ("formation", "pocket_features", "scoring")):
        return _reference_post(P, ps, fm, ranker, tm)
    t1 = time.perf_counter()
    fc = fm.fc
    geo = fm.model is None or ranker == "geometry"
    if not geo and score is None:
        score = score_points(fm, ps.feats, ps.names)
    t2 = time.perf_counter()
    names = ps.names
    gv = (ps.feats[:, names.index("bur10")] + ps.feats[:, names.index("bur16")] + ps.feats[:, names.index("bur24")]) / 3.0
    is_metal = ps.feats[:, names.index("metal_cand")] > 0
    c = FF.candidates(ps.X, ps.points, ps.vol, gv, None if geo else score, is_metal, fc)
    FF.assign_residues(c, ps.points, P.coords, P.res_index, len(P.residue_ids))
    t3 = time.perf_counter()
    if geo:
        rs = geo_scores(c, ps.vol, gv, fc)
    elif len(c):
        rch, nch = residue_chains(P.residue_ids)
        Fp, fn = pocket_features(c, ps.X, ps.vol, ps.feats, names, score, len(P.coords), rch, nch)
        rs = fm.rank.predict(Fp[:, [fn.index(n) for n in fm.model.rank_features]])
    else:
        rs = np.zeros(0)
    keep = FF.nms(c, rs, fc.nms_dist, fc.nms_jacc, fc.max_out, fc.nms_overlap, fc.min_score, fc.min_keep)
    t4 = time.perf_counter()
    tm.update(point_score=(t2 - t1) * 1e3, pockets=(t3 - t2) * 1e3, rank=(t4 - t3) * 1e3)
    rid = P.residue_ids
    return Prediction(P.id, c.center[keep].reshape(-1, 3), np.asarray(rs)[keep],
                      [frozenset(rid[c.residues(i)].tolist()) for i in keep],
                      [ps.points[c.members(i)] for i in keep], tm)


def predict_protein(P, fm: FastModel, cfg: BurrowConfig = BurrowConfig(), weights=None,
                    ranker: str = "learned") -> Prediction:
    """Drop-in fast replacement for ``burrow.pipeline.predict_protein`` (takes a :class:`FastModel`)."""
    t0 = time.perf_counter()
    ps = featurize(P, cfg, weights)
    t1 = time.perf_counter()
    tm = {f"feat_{k}": v for k, v in ps.timings.items()}
    tm["features"] = (t1 - t0) * 1e3
    pred = predict_from_points(P, ps, fm, ranker, timings=tm)
    tm["total"] = (time.perf_counter() - t0) * 1e3
    return pred
