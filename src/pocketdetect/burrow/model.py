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


def longest_bur(names):
    """Name of the longest-range full-atom buriedness column (bur24 by default, bur16 if L<=16)."""
    b = [n for n in names if n.startswith("bur") and not n.startswith("burbb")]
    return max(b, key=lambda n: int(n[3:]))


def geo_value(F, names):
    """Geometric point value b = mean(blocked@10, blocked@16, blocked@longest)."""
    c = lambda n: F[:, names.index(n)]
    return (c("bur10") + c("bur16") + c(longest_bur(names))) / 3.0


def geo_score(pockets, vol, F, names, fc: FormationConfig = FormationConfig()):
    """Geometry-only pocket score (no learning): buried volume above a buriedness floor."""
    gv = geo_value(F, names)
    w = vol * np.clip((gv - fc.geo_floor) / (1.0 - fc.geo_floor), 0.0, 1.0) ** fc.geo_power
    return np.array([float(w[p.idx].sum()) for p in pockets])


def candidates(X, pts, vol, F, names, score, fc: FormationConfig = FormationConfig()):
    """Union of geometric and score-based candidate pockets (each tagged with its source level:
    geometric 0..3, score-based 10..12, +100 for sub-pockets); identical point sets deduplicated."""
    from .pockets import PointGraph
    gv = geo_value(F, names)
    graph = PointGraph(X, fc.link)
    out = generate(X, pts, vol, gv, fc.geo_thresholds, fc.link, fc.max_extent, fc.split_size, fc.min_vol,
                   sub_size=fc.sub_size, graph=graph)
    if score is not None:
        is_metal = F[:, names.index("metal_cand")] > 0
        for p in generate(X, pts, vol, score, fc.score_thresholds, fc.link, fc.max_extent, fc.split_size,
                          fc.min_vol, extra_value=is_metal, sub_size=fc.sub_size, graph=graph):
            p.level = p.level + 10
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        key = np.sort(p.idx).tobytes()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


POCKET_FEATURES = None  # filled lazily (names)


def _seg_sorted(vals, pid, starts, counts):
    """Values sorted ascending within each pocket segment (segments contiguous in pid order)."""
    o = np.lexsort((vals, pid))
    return vals[o]


def _seg_quantile(sorted_vals, starts, counts, q):
    """'lower' quantile per segment of pre-sorted values."""
    return sorted_vals[starts + np.floor(q * (counts - 1)).astype(np.int64)]


def _seg_topk_mean(sorted_vals, starts, counts, k):
    cs = np.concatenate([[0.0], np.cumsum(sorted_vals, dtype=np.float64)])
    ends = starts + counts
    kk = np.minimum(k, counts)
    return (cs[ends] - cs[ends - kk]) / kk


def pocket_features(pockets, X, vol, F, names, score, n_atoms, n_chains_of=None):
    """(n_pockets, n_feat) matrix of pocket descriptors and their names (vectorised segment
    reductions over the concatenated point lists of all pockets)."""
    n_p = len(pockets)
    if n_p == 0:
        return np.zeros((0, 0), np.float32), []
    col = {n: F[:, i].astype(np.float64) for i, n in enumerate(names)}
    gv = geo_value(F, names).astype(np.float64)
    counts = np.array([len(p.idx) for p in pockets], np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    I = np.concatenate([p.idx for p in pockets])
    pid = np.repeat(np.arange(n_p), counts)
    v = vol[I].astype(np.float64)
    sI = score[I].astype(np.float64)
    bc = lambda w: np.bincount(pid, weights=w, minlength=n_p)
    vs = bc(v)
    vs_safe = np.maximum(vs, 1e-6)
    f = {}
    f["n_pts"] = counts.astype(np.float64)
    f["vol"] = vs
    f["sqrt_vol"] = np.sqrt(vs)
    f["sat_vol"] = vs / (vs + 100.0)
    sc_sum = bc(sI * v)
    f["score_sum"] = sc_sum
    f["score_sqrt"] = sc_sum / np.sqrt(vs + 1.0)
    f["score_mean"] = np.where(vs > 0, sc_sum / vs_safe, bc(sI) / counts)
    ss = _seg_sorted(sI, pid, starts, counts)
    f["score_max"] = ss[starts + counts - 1]
    f["score_top10"] = _seg_topk_mean(ss, starts, counts, 10)
    f["score_top30"] = _seg_topk_mean(ss, starts, counts, 30)
    f["score_q50"] = _seg_quantile(ss, starts, counts, 0.5)
    f["score_rank"] = np.argsort(np.argsort(-sc_sum, kind="stable"), kind="stable").astype(np.float64)
    f["score_rel"] = sc_sum / max(sc_sum.max(), 1e-6)
    f["geo_strength"] = bc(v * gv[I])
    lb = longest_bur(names)
    for nm in ("bur10", "bur16", "bur24", "burbb10", "burbb16", "depth", "psp", "nearest"):
        if nm not in col:
            nm_src = lb if nm == "bur24" else nm
            x = col[nm_src][I]
            f[f"{nm}_mean"] = np.where(vs > 0, bc(x * v) / vs_safe, bc(x) / counts)
            continue
        x = col[nm][I]
        f[f"{nm}_mean"] = np.where(vs > 0, bc(x * v) / vs_safe, bc(x) / counts)
    for nm in ("bur16", "bur24"):
        sv = _seg_sorted(col[nm if nm in col else lb][I], pid, starts, counts)
        f[f"{nm}_q90"] = _seg_quantile(sv, starts, counts, 0.9)
        f[f"{nm}_q10"] = _seg_quantile(sv, starts, counts, 0.1)
    for nm in ("hit_apolar", "hit_polar", "hit_charged", "hit_aromatic", "escape"):
        f[nm] = bc(col[nm][I]) / counts
    for nm in names:
        if nm.startswith("f8_") or nm.startswith("n4_rg") or nm in ("n_all8", "n_all12", "prot_frac"):
            f[nm] = bc(col[nm][I]) / counts
    f["metal_n"] = bc(col["metal_cand"][I])
    f["n_coord_max"] = np.maximum.reduceat(col["n_coord28"][I], starts)
    # shape: per-pocket covariance -> eigenvalues / principal axis extent
    XI = X[I].astype(np.float64)
    mu = np.stack([bc(XI[:, k]) for k in range(3)], 1) / counts[:, None]
    D = XI - mu[pid]
    C = np.stack([np.stack([bc(D[:, a] * D[:, b]) for b in range(3)], 1) for a in range(3)], 1) / counts[:, None, None]
    ev, evec = np.linalg.eigh(C)
    sq = np.sqrt(np.maximum(ev[:, ::-1], 0))
    small = counts < 3
    sq[small] = 0.0
    f["pca1"], f["pca2"], f["pca3"] = sq[:, 0], sq[:, 1], sq[:, 2]
    f["aniso"] = sq[:, 2] / (sq[:, 0] + 1e-6)
    proj = (D * evec[pid, :, 2]).sum(1)
    ext = np.maximum.reduceat(proj, starts) - np.minimum.reduceat(proj, starts)
    f["extent"] = np.where(counts > 2, ext, 0.0)
    f["n_res"] = np.array([len(p.residues) for p in pockets], np.float64)
    f["n_chains"] = np.array([len({r.rpartition(":")[0] for r in p.residues}) for p in pockets], np.float64)
    lev = np.array([p.level for p in pockets])
    f["level"] = (lev % 100).astype(np.float64)
    f["is_sub"] = (lev >= 100).astype(np.float64)
    f["is_score_src"] = ((lev % 100) >= 10).astype(np.float64)
    f["n_atoms_prot"] = np.full(n_p, float(n_atoms))
    f["log_atoms"] = np.full(n_p, float(np.log(n_atoms + 1)))
    fnames = list(f.keys())
    return np.stack([f[n] for n in fnames], 1).astype(np.float32), fnames


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


def score_points(booster, F, names, feature_names, score_floor: float | None = None, num_threads: int = 1):
    """Point scores; with ``score_floor`` only points with geometric buriedness b >= floor (or metal
    candidates) are scored by the model, the rest get 0 (speed: fewer tree traversals)."""
    if len(F) == 0:
        return np.zeros(0, np.float32)
    idx = [names.index(n) for n in feature_names]
    if score_floor is None:
        return booster.predict(F[:, idx], num_threads=num_threads).astype(np.float32)
    m = (geo_value(F, names) >= score_floor) | (F[:, names.index("metal_cand")] > 0)
    out = np.zeros(len(F), np.float32)
    if m.any():
        out[m] = booster.predict(F[m][:, idx], num_threads=num_threads)
    return out


def train_lgb(X, y, params, n_rounds, seed_idx=0):
    import lightgbm as lgb
    p = dict(params)
    s = SEEDS[seed_idx]
    p.update(seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s, deterministic=True)
    ds = lgb.Dataset(np.asarray(X, np.float32), label=np.asarray(y, np.float32), free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=n_rounds)
