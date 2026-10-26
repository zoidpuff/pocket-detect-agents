"""BURROW point scorer + pocket ranker (DESIGN §2.4), LightGBM.

Candidate pockets = geometric recall pass (multi-threshold buriedness components + sub-pockets)
  ∪ score pass (multi-threshold point-score components + sub-pockets).
Every candidate is described by aggregates of its points (cross-fitted point scores at training
time), ranked by the pocket ranker, then NMS'ed with the LIGYSIS redundancy rule.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from ..seeds import SEEDS
from .pockets import Pocket, _extent, generate

POINT_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=100,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                    verbose=-1, num_threads=16)
RANK_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=30,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   verbose=-1, num_threads=16)


@dataclass(frozen=True)
class FormationConfig:
    geo_thresholds: tuple = (0.4, 0.55, 0.7, 0.85)
    score_thresholds: tuple = (0.15, 0.3, 0.5)
    link: float = 2.05
    max_extent: float = 20.0
    split_size: float = 12.0
    sub_size: float = 6.0
    min_vol: float = 10.0
    nms_dist: float = 10.0             # LIGYSIS redundancy uses 5 Å; 10 Å selected on CHEN11 cv5-OOF (dev)
    nms_jacc: float = 0.75
    nms_overlap: float | None = 0.3    # nested-pocket suppression (point overlap coefficient)
    geo_floor: float = 0.7             # geometry-only ranker: Σ vol·clip((b - floor)/(1 - floor))^power
    geo_power: float = 2.0
    min_score: float | None = None     # drop pockets below this ranker score ...
    min_keep: int = 3                  # ... but always keep at least this many
    max_out: int = 50


def geo_value(F, names):
    c = lambda n: F[:, names.index(n)]
    return (c("bur10") + c("bur16") + c("bur24")) / 3.0


def geo_score(pockets, vol, F, names, fc: FormationConfig = FormationConfig()):
    """Geometry-only pocket score (no learning): buried volume above a buriedness floor."""
    gv = geo_value(F, names)
    w = vol * np.clip((gv - fc.geo_floor) / (1.0 - fc.geo_floor), 0.0, 1.0) ** fc.geo_power
    return np.array([float(w[p.idx].sum()) for p in pockets])


def candidates(X, pts, vol, F, names, score, fc: FormationConfig = FormationConfig()):
    """Union of geometric and score-based candidate pockets (each tagged with its source)."""
    gv = geo_value(F, names)
    out = []
    for p in generate(X, pts, vol, gv, fc.geo_thresholds, fc.link, fc.max_extent, fc.split_size, fc.min_vol,
                      sub_size=fc.sub_size):
        p.level = p.level  # geometric: 0..3 (+100 for sub-pockets)
        out.append(p)
    if score is not None:
        is_metal = F[:, names.index("metal_cand")] > 0
        for p in generate(X, pts, vol, score, fc.score_thresholds, fc.link, fc.max_extent, fc.split_size,
                          fc.min_vol, extra_value=is_metal, sub_size=fc.sub_size):
            p.level = p.level + 10
            out.append(p)
    return out


POCKET_FEATURES = None  # filled lazily (names)


def pocket_features(pockets, X, vol, F, names, score, n_atoms, n_chains_of=None):
    """(n_pockets, n_feat) matrix of pocket descriptors and their names."""
    col = {n: F[:, i] for i, n in enumerate(names)}
    gv = geo_value(F, names)
    rows = []
    strengths = np.array([p.strength for p in pockets]) if pockets else np.zeros(0)
    sc_sum = np.array([float((vol[p.idx] * score[p.idx]).sum()) for p in pockets]) if pockets else np.zeros(0)
    rank_s = np.argsort(np.argsort(-sc_sum, kind="stable"), kind="stable")
    max_s = sc_sum.max() if len(sc_sum) else 1.0
    fnames = None
    for k, p in enumerate(pockets):
        i = p.idx
        v = vol[i]
        s = score[i]
        vs = v.sum()
        f = {}
        f["n_pts"] = len(i)
        f["vol"] = vs
        f["sqrt_vol"] = np.sqrt(vs)
        f["sat_vol"] = vs / (vs + 100.0)
        f["score_sum"] = sc_sum[k]
        f["score_sqrt"] = sc_sum[k] / np.sqrt(vs + 1.0)
        f["score_mean"] = float((s * v).sum() / max(vs, 1e-6))
        f["score_max"] = float(s.max())
        srt = np.sort(s)[::-1]
        f["score_top10"] = float(srt[:10].mean())
        f["score_top30"] = float(srt[:30].mean())
        f["score_q50"] = float(np.median(s))
        f["score_rank"] = float(rank_s[k])
        f["score_rel"] = sc_sum[k] / max(max_s, 1e-6)
        f["geo_strength"] = float((v * gv[i]).sum())
        for nm in ("bur10", "bur16", "bur24", "burbb10", "burbb16", "depth", "psp", "nearest"):
            x = col[nm][i]
            f[f"{nm}_mean"] = float((x * v).sum() / max(vs, 1e-6)) if vs > 0 else float(x.mean())
        for nm in ("bur16", "bur24"):
            f[f"{nm}_q90"] = float(np.quantile(col[nm][i], 0.9))
            f[f"{nm}_q10"] = float(np.quantile(col[nm][i], 0.1))
        for nm in ("hit_apolar", "hit_polar", "hit_charged", "hit_aromatic", "escape"):
            f[nm] = float(col[nm][i].mean())
        for nm in names:
            if nm.startswith("f8_") or nm.startswith("n4_rg") or nm in ("n_all8", "n_all12", "prot_frac"):
                f[nm] = float(col[nm][i].mean())
        f["metal_n"] = float(col["metal_cand"][i].sum())
        f["n_coord_max"] = float(col["n_coord28"][i].max())
        if len(i) >= 3:
            c = X[i] - X[i].mean(0)
            ev = np.sort(np.linalg.eigvalsh(c.T @ c / len(i)))[::-1]
            ev = np.sqrt(np.maximum(ev, 0))
        else:
            ev = np.zeros(3)
        f["pca1"], f["pca2"], f["pca3"] = ev
        f["aniso"] = ev[2] / (ev[0] + 1e-6)
        f["extent"] = _extent(X[i]) if len(i) > 2 else 0.0
        f["n_res"] = len(p.residues)
        f["n_chains"] = len({r.rpartition(":")[0] for r in p.residues})
        f["level"] = p.level % 100
        f["is_sub"] = float(p.level >= 100)
        f["is_score_src"] = float((p.level % 100) >= 10)
        f["n_atoms_prot"] = float(n_atoms)
        f["log_atoms"] = float(np.log(n_atoms + 1))
        if fnames is None:
            fnames = list(f.keys())
        rows.append([f[n] for n in fnames])
    if fnames is None:
        return np.zeros((0, 0), np.float32), []
    return np.asarray(rows, np.float32), fnames


@dataclass
class BurrowModel:
    point_model: str            # LightGBM model string
    point_features: list
    rank_model: str | None      # LightGBM model string (None = geometry-only ranking)
    rank_features: list
    formation: dict = field(default_factory=lambda: asdict(FormationConfig()))
    meta: dict = field(default_factory=dict)
    _pb = None
    _rb = None

    def boosters(self):
        import lightgbm as lgb
        if self._pb is None:
            self._pb = lgb.Booster(model_str=self.point_model)
            self._rb = lgb.Booster(model_str=self.rank_model) if self.rank_model else None
        return self._pb, self._rb

    def save(self, path):
        d = dict(point_model=self.point_model, point_features=self.point_features, rank_model=self.rank_model,
                 rank_features=self.rank_features, formation=self.formation, meta=self.meta)
        with open(path, "w") as fh:
            json.dump(d, fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        return cls(**d)


def train_lgb(X, y, params, n_rounds, seed_idx=0):
    import lightgbm as lgb
    p = dict(params)
    s = SEEDS[seed_idx]
    p.update(seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s, deterministic=True)
    ds = lgb.Dataset(np.asarray(X, np.float32), label=np.asarray(y, np.float32), free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=n_rounds)
