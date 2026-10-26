"""Fast pocket descriptors: numerically identical to ``burrow.model.pocket_features``.

All per-pocket sums are sequential float64 accumulations over the concatenated membership list in
the reference's order (== ``np.bincount(..., weights=...)``), segment sorts / quantiles / top-k
means reproduce the reference's global-cumsum arithmetic, and the 3x3 shape eigenproblems are
solved by the same batched ``np.linalg.eigh`` call on bit-identical matrices.  The equivalence
test (tests/test_burrow_fast_equivalence.py) compares the full matrix against the reference, so a
change of the reference descriptors is detected immediately.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from ..model import longest_bur

MEAN_COLS = ("bur10", "bur16", "bur24", "burbb10", "burbb16", "depth", "psp", "nearest")
Q_COLS = ("bur16", "bur24")
FRAC_COLS = ("hit_apolar", "hit_polar", "hit_charged", "hit_aromatic", "escape")


@njit(cache=True)
def _seg_sums(ptr, idx, w, cols):
    """sums[p, j] = sum_{i in pocket p} w[i] * cols[i, j]  (sequential, membership order)."""
    k = ptr.shape[0] - 1
    m = cols.shape[1]
    out = np.zeros((k, m))
    for p in range(k):
        for t in range(ptr[p], ptr[p + 1]):
            i = idx[t]
            for j in range(m):
                out[p, j] += np.float64(cols[i, j]) * w[i]
    return out


@njit(cache=True)
def _seg_plain(ptr, idx, cols):
    k = ptr.shape[0] - 1
    m = cols.shape[1]
    out = np.zeros((k, m))
    for p in range(k):
        for t in range(ptr[p], ptr[p + 1]):
            i = idx[t]
            for j in range(m):
                out[p, j] += np.float64(cols[i, j])
    return out


@njit(cache=True)
def _seg_sorted(ptr, idx, vals):
    out = np.empty(idx.shape[0])
    for p in range(ptr.shape[0] - 1):
        a, b = ptr[p], ptr[p + 1]
        for t in range(a, b):
            out[t] = vals[idx[t]]
        out[a:b] = np.sort(out[a:b])
    return out


@njit(cache=True)
def _topk_mean(sorted_vals, ptr, k):
    n = sorted_vals.shape[0]
    cs = np.empty(n + 1)
    cs[0] = 0.0
    acc = 0.0
    for i in range(n):
        acc += sorted_vals[i]
        cs[i + 1] = acc
    kp = ptr.shape[0] - 1
    out = np.empty(kp)
    for p in range(kp):
        cnt = ptr[p + 1] - ptr[p]
        kk = min(k, cnt)
        e = ptr[p + 1]
        out[p] = (cs[e] - cs[e - kk]) / kk
    return out


@njit(cache=True)
def _quantile(sorted_vals, ptr, q):
    kp = ptr.shape[0] - 1
    out = np.empty(kp)
    for p in range(kp):
        cnt = ptr[p + 1] - ptr[p]
        out[p] = sorted_vals[ptr[p] + np.int64(math.floor(q * (cnt - 1)))]
    return out


@njit(cache=True)
def _shape(ptr, idx, X):
    """Per-pocket mean (bincount / counts) and covariance exactly as the reference."""
    kp = ptr.shape[0] - 1
    mu = np.zeros((kp, 3))
    C = np.zeros((kp, 3, 3))
    for p in range(kp):
        cnt = ptr[p + 1] - ptr[p]
        for t in range(ptr[p], ptr[p + 1]):
            for d in range(3):
                mu[p, d] += np.float64(X[idx[t], d])
        for d in range(3):
            mu[p, d] = mu[p, d] / cnt
    for a in range(3):
        for b in range(3):
            for p in range(kp):
                cnt = ptr[p + 1] - ptr[p]
                s = 0.0
                for t in range(ptr[p], ptr[p + 1]):
                    i = idx[t]
                    s += (np.float64(X[i, a]) - mu[p, a]) * (np.float64(X[i, b]) - mu[p, b])
                C[p, a, b] = s / cnt
    return mu, C


@njit(cache=True)
def _extent_proj(ptr, idx, X, mu, evec):
    kp = ptr.shape[0] - 1
    out = np.zeros(kp)
    for p in range(kp):
        lo = np.inf
        hi = -np.inf
        for t in range(ptr[p], ptr[p + 1]):
            i = idx[t]
            v = ((np.float64(X[i, 0]) - mu[p, 0]) * evec[p, 0, 2] + (np.float64(X[i, 1]) - mu[p, 1]) * evec[p, 1, 2]) \
                + (np.float64(X[i, 2]) - mu[p, 2]) * evec[p, 2, 2]
            lo = min(lo, v)
            hi = max(hi, v)
        out[p] = hi - lo
    return out


@njit(cache=True)
def _seg_max(ptr, idx, vals):
    kp = ptr.shape[0] - 1
    out = np.empty(kp)
    for p in range(kp):
        m = -np.inf
        for t in range(ptr[p], ptr[p + 1]):
            v = np.float64(vals[idx[t]])
            if v > m:
                m = v
        out[p] = m
    return out


@njit(cache=True)
def _n_chains(res_ptr, res, res_chain, n_chain):
    kp = res_ptr.shape[0] - 1
    stamp = np.full(max(n_chain, 1), -1, np.int64)
    out = np.zeros(kp)
    for p in range(kp):
        c = 0
        for t in range(res_ptr[p], res_ptr[p + 1]):
            ch = res_chain[res[t]]
            if stamp[ch] != p:
                stamp[ch] = p
                c += 1
        out[p] = c
    return out


def residue_chains(residue_ids):
    """Chain index per residue (chain = text before the last ':' of the residue id)."""
    ch = np.array([r.rpartition(":")[0] for r in residue_ids.tolist()], dtype=object)
    _, inv = np.unique(ch.astype(str), return_inverse=True)
    return inv.astype(np.int64), int(inv.max()) + 1 if len(inv) else 0


def pocket_features(c, X, vol, F, names, score, n_atoms, res_chain, n_chain):
    """(K, n_feat) float32 matrix and names, identical to the reference ``pocket_features``."""
    n_p = len(c)
    if n_p == 0:
        return np.zeros((0, 0), np.float32), []
    ptr, I = c.ptr, c.idx
    counts = np.diff(ptr).astype(np.int64)
    starts = ptr[:-1]
    ci = {n: i for i, n in enumerate(names)}
    lb = longest_bur(names)
    src = lambda n: ci[n] if n in ci else ci[lb]          # reference: bur24 -> longest column if absent
    v64 = vol.astype(np.float64)
    s64 = np.asarray(score, np.float64)
    gv = ((F[:, ci["bur10"]] + F[:, ci["bur16"]] + F[:, ci[lb]]) / 3.0).astype(np.float64)
    one = np.ones(len(vol))
    # weighted sums (weight vol): vol, score*vol, gv*vol, mean columns*vol
    wcols = np.stack([one, s64, gv] + [F[:, src(n)].astype(np.float64) for n in MEAN_COLS], 1)
    WS = _seg_sums(ptr, I, v64, wcols)
    frac_names = [n for n in names if n.startswith("f8_") or n.startswith("n4_rg") or n in ("n_all8", "n_all12", "prot_frac")]
    pcols = np.stack([s64] + [F[:, src(n)].astype(np.float64) for n in MEAN_COLS]
                     + [F[:, ci[n]].astype(np.float64) for n in FRAC_COLS]
                     + [F[:, ci[n]].astype(np.float64) for n in frac_names]
                     + [F[:, ci["metal_cand"]].astype(np.float64)], 1)
    PS = _seg_plain(ptr, I, pcols)
    vs = WS[:, 0]
    vs_safe = np.maximum(vs, 1e-6)
    f = {}
    f["n_pts"] = counts.astype(np.float64)
    f["vol"] = vs
    f["sqrt_vol"] = np.sqrt(vs)
    f["sat_vol"] = vs / (vs + 100.0)
    sc_sum = WS[:, 1]
    f["score_sum"] = sc_sum
    f["score_sqrt"] = sc_sum / np.sqrt(vs + 1.0)
    f["score_mean"] = np.where(vs > 0, sc_sum / vs_safe, PS[:, 0] / counts)
    ss = _seg_sorted(ptr, I, s64)
    f["score_max"] = ss[starts + counts - 1]
    f["score_top10"] = _topk_mean(ss, ptr, 10)
    f["score_top30"] = _topk_mean(ss, ptr, 30)
    f["score_q50"] = _quantile(ss, ptr, 0.5)
    f["score_rank"] = np.argsort(np.argsort(-sc_sum, kind="stable"), kind="stable").astype(np.float64)
    f["score_rel"] = sc_sum / max(sc_sum.max(), 1e-6)
    f["geo_strength"] = WS[:, 2]
    for j, nm in enumerate(MEAN_COLS):
        f[f"{nm}_mean"] = np.where(vs > 0, WS[:, 3 + j] / vs_safe, PS[:, 1 + j] / counts)
    for nm in Q_COLS:
        sv = _seg_sorted(ptr, I, F[:, src(nm)].astype(np.float64))
        f[f"{nm}_q90"] = _quantile(sv, ptr, 0.9)
        f[f"{nm}_q10"] = _quantile(sv, ptr, 0.1)
    o = 1 + len(MEAN_COLS)
    for j, nm in enumerate(FRAC_COLS):
        f[nm] = PS[:, o + j] / counts
    o += len(FRAC_COLS)
    for j, nm in enumerate(frac_names):
        f[nm] = PS[:, o + j] / counts
    f["metal_n"] = PS[:, -1]
    f["n_coord_max"] = _seg_max(ptr, I, F[:, ci["n_coord28"]])
    mu, C = _shape(ptr, I, X)
    ev, evec = np.linalg.eigh(C)
    sq = np.sqrt(np.maximum(ev[:, ::-1], 0))
    sq[counts < 3] = 0.0
    f["pca1"], f["pca2"], f["pca3"] = sq[:, 0], sq[:, 1], sq[:, 2]
    f["aniso"] = sq[:, 2] / (sq[:, 0] + 1e-6)
    ext = _extent_proj(ptr, I, X, mu, np.ascontiguousarray(evec))
    f["extent"] = np.where(counts > 2, ext, 0.0)
    f["n_res"] = np.diff(c.res_ptr).astype(np.float64)
    f["n_chains"] = _n_chains(c.res_ptr, c.res, res_chain, n_chain)
    lev = c.level
    f["level"] = (lev % 100).astype(np.float64)
    f["is_sub"] = (lev >= 100).astype(np.float64)
    f["is_score_src"] = ((lev % 100) >= 10).astype(np.float64)
    f["n_atoms_prot"] = np.full(n_p, float(n_atoms))
    f["log_atoms"] = np.full(n_p, float(np.log(n_atoms + 1)))
    fnames = list(f.keys())
    return np.stack([f[n] for n in fnames], 1).astype(np.float32), fnames
