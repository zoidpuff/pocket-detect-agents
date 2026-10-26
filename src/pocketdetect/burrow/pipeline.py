"""BURROW end-to-end prediction for one structure (CPU) and output writers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .features import BurrowConfig, point_features
from .model import BurrowModel, FormationConfig, candidates, geo_score, geo_value, pocket_features, score_points
from .pockets import assign_residues, nms
from .prep import Protein


@dataclass
class Prediction:
    structure_id: str
    centers: np.ndarray      # (K,3)
    scores: np.ndarray       # (K,)
    residues: list           # K frozensets
    points: list             # K (n,3) arrays
    timings: dict


def finalize(pockets, scores, fc: FormationConfig):
    keep = nms(pockets, scores, fc.nms_dist, fc.nms_jacc, fc.max_out, fc.nms_overlap, fc.min_score, fc.min_keep)
    return keep


def predict_protein(P: Protein, model: BurrowModel | None, cfg: BurrowConfig = BurrowConfig(),
                    weights=None, ranker: str = "learned") -> Prediction:
    """``model=None`` or ``ranker='geometry'`` -> geometry-only baseline (buried volume, see geo_score)."""
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
            j = [fn.index(n) for n in model.rank_features]
            rs = rb.predict(Fp[:, j], num_threads=1)
        else:
            rs = np.zeros(0)
    keep = finalize(pk, rs, fc)
    t4 = time.perf_counter()
    tm.update(features=(t1 - t0) * 1e3, point_score=(t2 - t1) * 1e3, pockets=(t3 - t2) * 1e3,
              rank=(t4 - t3) * 1e3, total=(t4 - t0) * 1e3)
    return Prediction(P.id, np.array([pk[i].center for i in keep]).reshape(-1, 3), np.asarray(rs)[keep],
                      [pk[i].residues for i in keep], [ps.points[pk[i].idx] for i in keep], tm)


def write_prediction(out_dir, pred: Prediction, points: bool = True):
    """``<out>/<id>.csv`` (common format) and ``<out>/<id>.points.csv`` (rank,x,y,z)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["rank,score,center_x,center_y,center_z,residues"]
    for r, (c, s, res) in enumerate(zip(pred.centers, pred.scores, pred.residues), 1):
        rs = " ".join(sorted(res, key=_res_key))
        lines.append(f"{r},{s:.6g},{c[0]:.3f},{c[1]:.3f},{c[2]:.3f},{rs}")
    (out / f"{pred.structure_id}.csv").write_text("\n".join(lines) + "\n")
    if points:
        pl = ["rank,x,y,z"]
        for r, P in enumerate(pred.points, 1):
            pl += [f"{r},{x:.2f},{y:.2f},{z:.2f}" for x, y, z in P]
        (out / f"{pred.structure_id}.points.csv").write_text("\n".join(pl) + "\n")


def _res_key(r):
    ch, _, num = r.rpartition(":")
    digits = "".join(c for c in num if c.isdigit() or c == "-")
    try:
        n = int(digits)
    except ValueError:
        n = 0
    return (ch, n, num)


__all__ = ["Prediction", "predict_protein", "write_prediction", "finalize", "geo_value"]
