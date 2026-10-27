"""Export every pocket representation that BURROW computes for one structure (for downstream work such as
pocket comparison).  This module only re-uses the reference pipeline functions; it does not change any frozen
code path.  The returned pockets are identical to ``pipeline.predict_protein`` (tested in
``tests/test_burrow_export.py``).

    from pocketdetect.burrow.export import pocket_representations
    rep = pocket_representations("1abc.cif.gz", ranker="all", preset="default")
    rep["pockets"]            # one row per output pocket: rank, score, centre, residues, n_points, 76 descriptors
    rep["points"]             # one row per candidate point: coordinates, volume, point score, 84/86 point features
    rep["pocket_points"]      # rank -> indices into rep["points"]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .features import BurrowConfig, point_features
from .model import BurrowModel, FormationConfig, candidates, pocket_features, rank_predict, score_points
from .pipeline import finalize, output_residues
from .pockets import assign_residues

ROOT = Path(__file__).resolve().parents[3]


def _model_path(ranker: str, preset: str, model) -> Path:
    if model is not None:
        return Path(model)
    from .cli import DEFAULT_MODELS
    return ROOT / DEFAULT_MODELS[(ranker, preset)]


def _feature_cfg(m: BurrowModel) -> BurrowConfig:
    fd = dict(m.meta["feature_cfg"])
    fd["lengths"] = tuple(fd["lengths"])
    fd["nb_radii"] = tuple(fd["nb_radii"])
    return BurrowConfig(**fd)


def pocket_representations(structure_path, ranker: str = "all", preset: str = "default", model=None,
                           min_score="model", output_rule: str = "v2", af_mode: bool = False,
                           structure_id: str | None = None) -> dict:
    """Run BURROW on one structure and return all intermediate representations.

    Parameters
    ----------
    structure_path : PDB/mmCIF file (``.gz`` allowed).  Only protein residues are used.
    ranker, preset : as ``burrow predict`` (``all``/``small``; ``default``/``accurate``/``fast``).
    model          : path to a model JSON (overrides ranker/preset).
    min_score      : ``"model"`` (the model's output floor; 0.03 for FREEZE v2), ``None`` (full list) or a float.
    output_rule    : ``"v2"`` or ``"region"`` (as ``--output-rule``; ``region`` needs ``min_score="model"``).
    af_mode        : pLDDT pruning + soft occlusion (as ``--af-mode``; standard ranker).

    Returns a dict with keys
      ``points``        DataFrame, one row per candidate point (input frame x, y, z; canonical-frame X, Y, Z;
                        ``vol`` = number of 1-grid-voxel volume elements the point stands for; ``is_metal``;
                        ``point_score``; then every point feature by name),
      ``pockets``       DataFrame, one row per output pocket in rank order (``rank``, ``score``, ``center_x/y/z``,
                        ``residues`` space-separated ``chain:resnum[icode]``, ``n_points``, ``volume_A3``,
                        ``plddt_mean``/``plddt_min`` of the lining residues if ``af_mode``, then the 76 ranker
                        descriptors by name),
      ``pocket_points`` dict rank -> np.ndarray of row indices into ``points``,
      ``point_feature_names``, ``descriptor_names``, ``model_path``, ``feature_cfg``, ``formation``,
      ``n_candidates`` (pockets before suppression), ``residue_ids`` (all protein residues).
    """
    from ..structure import load_structure
    from .prep import protein_from_structure

    mp = _model_path(ranker, preset, model)
    m = BurrowModel.load(mp)
    fc_dict = dict(m.formation)
    if min_score != "model":
        fc_dict["min_score"] = min_score
    if output_rule == "region":
        if min_score != "model":
            raise ValueError("output_rule='region' needs min_score='model'")
        fc_dict["min_score"] = None                       # full list, then the post-filter below
    fc = FormationConfig(**fc_dict)
    cfg = _feature_cfg(m)

    sid = structure_id or Path(structure_path).name.split(".")[0]
    s = load_structure(str(structure_path), structure_id=sid)
    weights = None
    if af_mode:
        from .af import af_protein
        P, weights, _ = af_protein(s)
    else:
        P = protein_from_structure(s)

    ps = point_features(P, cfg, weights)
    pb, rb = m.boosters()
    score = score_points(pb, ps.feats, ps.names, m.point_features, m.meta.get("score_floor"))
    pk = candidates(ps.X, ps.points, ps.vol, ps.feats, ps.names, score, fc)
    tree = cKDTree(P.coords)
    assign_residues(pk, ps.points, tree, P.res_index, P.residue_ids)
    Fp, fn = pocket_features(pk, ps.X, ps.vol, ps.feats, ps.names, score, len(P.coords))
    rs = rank_predict(m, Fp[:, [fn.index(n) for n in m.rank_features]]) if len(pk) else np.zeros(0)
    keep = finalize(pk, rs, fc)
    if output_rule == "region":
        from .cli import REGION_RULE
        from .output_rules import apply_rule
        sub = apply_rule(REGION_RULE, np.asarray(rs)[keep], np.array([pk[i].center for i in keep]).reshape(-1, 3))
        keep = [keep[i] for i in sub]

    points = pd.DataFrame({"x": ps.points[:, 0], "y": ps.points[:, 1], "z": ps.points[:, 2],
                           "X": ps.X[:, 0], "Y": ps.X[:, 1], "Z": ps.X[:, 2],
                           "vol": ps.vol, "is_metal": ps.is_metal, "point_score": score})
    points = pd.concat([points, pd.DataFrame(ps.feats, columns=ps.names)], axis=1)

    plddt_res = None
    if af_mode:
        plddt_res = pd.Series(P.plddt).groupby(pd.Series(P.residue_ids[P.res_index])).mean()
    rows, members = [], {}
    h3 = cfg.spacing ** 3
    for r, i in enumerate(keep, 1):
        p = pk[i]
        res = output_residues(p, tree, P)
        row = {"rank": r, "score": float(rs[i]), "center_x": p.center[0], "center_y": p.center[1],
               "center_z": p.center[2], "residues": " ".join(sorted(res)), "n_points": len(p.idx),
               "volume_A3": float(ps.vol[p.idx].sum() * h3)}
        if plddt_res is not None:
            v = plddt_res.reindex(sorted(res)).to_numpy(float)
            row["plddt_mean"], row["plddt_min"] = float(np.nanmean(v)), float(np.nanmin(v))
        row.update({n: float(Fp[i, j]) for j, n in enumerate(fn)})
        rows.append(row)
        members[r] = np.asarray(p.idx, np.int64)
    pockets = pd.DataFrame(rows)
    return {"points": points, "pockets": pockets, "pocket_points": members,
            "point_feature_names": list(ps.names), "descriptor_names": list(fn), "model_path": str(mp),
            "feature_cfg": json.loads(json.dumps(m.meta["feature_cfg"])), "formation": fc_dict,
            "n_candidates": len(pk), "residue_ids": list(P.residue_ids)}
