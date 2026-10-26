"""Fast pocket formation (Numba): same candidates, residues and NMS as ``burrow.pockets`` /
``burrow.model.candidates``.

Pockets are kept in CSR form (``ptr``, ``idx``; point indices ascending inside a pocket, as in
the reference) with per-pocket ``level``, ``center`` (input frame) and ``strength``.

Exactness notes (vs the NumPy reference):
* components, their order (lowest member first) and member order are identical;
* weighted k-means (chaining control / multi-scale sub-pockets) replicates NumPy's arithmetic
  (pairwise summation of 1-D sums, sequential axis-0 sums, float32/float64 dtypes) -> identical
  labels;
* the principal-axis extent is computed in float64 (reference: float32 BLAS/LAPACK); the two
  differ by ~1e-6 Å, so a component whose extent is within ~1e-5 Å of a split threshold could be
  treated differently (not observed on dev data, see tests).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, types
from numba.typed import List

# --------------------------------------------------------------------------- NumPy-exact sums


@njit(cache=True)
def _pw_block(a, s, n):
    if n < 8:
        res = np.zeros(1, a.dtype)[0]          # same-dtype accumulator (a[s] * 0 would be float64)
        for i in range(n):
            res += a[s + i]
        return res
    r0 = a[s]; r1 = a[s + 1]; r2 = a[s + 2]; r3 = a[s + 3]
    r4 = a[s + 4]; r5 = a[s + 5]; r6 = a[s + 6]; r7 = a[s + 7]
    i = 8
    lim = n - (n % 8)
    while i < lim:
        r0 += a[s + i]; r1 += a[s + i + 1]; r2 += a[s + i + 2]; r3 += a[s + i + 3]
        r4 += a[s + i + 4]; r5 += a[s + i + 5]; r6 += a[s + i + 6]; r7 += a[s + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[s + i]
        i += 1
    return res


@njit(cache=True)
def _pw(a, s, n):
    """NumPy's pairwise summation of a[s:s+n] with same-dtype accumulators.

    Iterative (explicit stack) on purpose: recursive njit functions break numba's disk cache."""
    if n <= 128:
        return _pw_block(a, s, n)
    fs = np.empty(64, np.int64)
    fn = np.empty(64, np.int64)
    st = np.zeros(64, np.int64)
    vals = np.empty(64, a.dtype)
    nf = 0
    nv = 0
    fs[0] = s
    fn[0] = n
    st[0] = 0
    nf = 1
    while nf > 0:
        t = nf - 1
        m = fn[t]
        if m <= 128:
            vals[nv] = _pw_block(a, fs[t], m)
            nv += 1
            nf -= 1
            continue
        n2 = m // 2
        n2 -= n2 % 8
        if st[t] == 0:
            st[t] = 1
            fs[nf] = fs[t]
            fn[nf] = n2
            st[nf] = 0
            nf += 1
        elif st[t] == 1:
            st[t] = 2
            fs[nf] = fs[t] + n2
            fn[nf] = m - n2
            st[nf] = 0
            nf += 1
        else:
            r = vals[nv - 1]
            l = vals[nv - 2]
            nv -= 2
            vals[nv] = l + r
            nv += 1
            nf -= 1
    return vals[0]


# --------------------------------------------------------------------------- geometry helpers
@njit(cache=True)
def _extent(X, idx):
    """Principal-axis extent of points X[idx] (float64)."""
    n = idx.shape[0]
    if n < 2:
        return 0.0
    m = np.zeros(3)
    for i in range(n):
        for d in range(3):
            m[d] += X[idx[i], d]
    m /= n
    C = np.zeros((3, 3))
    for i in range(n):
        for a in range(3):
            da = X[idx[i], a] - m[a]
            for b in range(3):
                C[a, b] += da * (X[idx[i], b] - m[b])
    ev, evec = np.linalg.eigh(C)
    lo = np.inf
    hi = -np.inf
    for i in range(n):
        p = 0.0
        for d in range(3):
            p += (X[idx[i], d] - m[d]) * evec[d, 2]
        lo = min(lo, p)
        hi = max(hi, p)
    return hi - lo


@njit(cache=True)
def _kmeans_split(Xs, w, k, eps, n_iter):
    """Exact port of ``pockets._kmeans_split`` (Xs float64 (n,3); w float32/64; eps = w.dtype(1e-9))."""
    n = Xs.shape[0]
    c = np.empty(k, np.int64)
    c[0] = np.argmax(w)
    d = np.empty(n)
    for i in range(n):
        dx = Xs[i, 0] - Xs[c[0], 0]
        dy = Xs[i, 1] - Xs[c[0], 1]
        dz = Xs[i, 2] - Xs[c[0], 2]
        d[i] = dx * dx + dy * dy + dz * dz
    for j in range(1, k):
        c[j] = np.argmax(d)
        for i in range(n):
            dx = Xs[i, 0] - Xs[c[j], 0]
            dy = Xs[i, 1] - Xs[c[j], 1]
            dz = Xs[i, 2] - Xs[c[j], 2]
            v = dx * dx + dy * dy + dz * dz
            if v < d[i]:
                d[i] = v
    C = np.empty((k, 3))
    for j in range(k):
        C[j] = Xs[c[j]]
    lab = np.zeros(n, np.int64)
    ww = np.empty(n, w.dtype)
    for _ in range(n_iter):
        for i in range(n):
            best = np.inf
            bj = 0
            for j in range(k):
                dx = Xs[i, 0] - C[j, 0]
                dy = Xs[i, 1] - C[j, 1]
                dz = Xs[i, 2] - C[j, 2]
                v = dx * dx + dy * dy + dz * dz
                if v < best:
                    best = v
                    bj = j
            lab[i] = bj
        for j in range(k):
            m = 0
            sx = 0.0
            sy = 0.0
            sz = 0.0
            for i in range(n):
                if lab[i] == j:
                    wi = w[i] + eps
                    ww[m] = wi
                    m += 1
                    sx += Xs[i, 0] * wi
                    sy += Xs[i, 1] * wi
                    sz += Xs[i, 2] * wi
            if m:
                tot = _pw(ww, 0, m)
                C[j, 0] = sx / tot
                C[j, 1] = sy / tot
                C[j, 2] = sz / tot
    return lab


@njit(cache=True)
def _make_center(idx, pts, w):
    """(strength, centre) exactly as ``pockets.make_pocket`` (w = vol * value, reference dtype)."""
    n = idx.shape[0]
    ww = np.empty(n, w.dtype)
    for i in range(n):
        ww[i] = w[idx[i]]
    s = _pw(ww, 0, n)
    c = np.zeros(3)
    if s <= 0:
        for i in range(n):
            for d in range(3):
                c[d] += pts[idx[i], d]
        return float(s), c / n
    for i in range(n):
        wi = ww[i]
        for d in range(3):
            c[d] += pts[idx[i], d] * wi
    return float(s), c / s


# --------------------------------------------------------------------------- candidate generation
@njit(cache=True)
def _components(adj_ptr, adj, mask):
    """Connected components of the masked graph; components ordered by lowest member, members
    ascending (== reference single linkage + canonical_labels + stable grouping)."""
    n = mask.shape[0]
    comp = np.full(n, -1, np.int64)
    order = np.empty(n, np.int64)
    ptr = np.zeros(n + 1, np.int64)
    stack = np.empty(n, np.int64)
    pos = 0
    nc = 0
    for s in range(n):
        if not mask[s] or comp[s] >= 0:
            continue
        top = 0
        stack[0] = s
        comp[s] = nc
        start = pos
        while top >= 0:
            u = stack[top]
            top -= 1
            order[pos] = u
            pos += 1
            for e in range(adj_ptr[u], adj_ptr[u + 1]):
                v = adj[e]
                if mask[v] and comp[v] < 0:
                    comp[v] = nc
                    top += 1
                    stack[top] = v
        order[start:pos] = np.sort(order[start:pos])
        nc += 1
        ptr[nc] = pos
    return ptr[:nc + 1], order[:pos]


@njit(cache=True)
def _vol_sum(vol, idx):
    s = np.zeros(1, vol.dtype)[0]
    for i in range(idx.shape[0]):
        s += vol[idx[i]]
    return s


@njit(cache=True)
def _any(extra, idx):
    for i in range(idx.shape[0]):
        if extra[idx[i]]:
            return True
    return False


@njit(cache=True)
def _generate_level(X64, pts64, vol, value, w, eps, th, lev, adj_ptr, adj, extra, use_extra,
                    max_extent, split_size, min_vol, sub_size):
    """One threshold of ``pockets.generate`` (components -> chaining split -> min-volume ->
    pocket + multi-scale sub-pockets).  Returns CSR (ptr, idx) + level, strength, centre arrays.
    (No typed containers cross the function boundary: numba's on-disk cache segfaults on them.)"""
    out_idx = List.empty_list(types.int64[::1])
    out_lev = List.empty_list(types.int64)
    out_s = List.empty_list(types.float64)
    out_ctr = List.empty_list(types.float64[::1])
    mask = value >= th
    ptr, members = _components(adj_ptr, adj, mask)
    for ci in range(ptr.shape[0] - 1):
        idx = members[ptr[ci]:ptr[ci + 1]]
        ext = _extent(X64, idx) if idx.shape[0] > 10 else 0.0
        parts = List.empty_list(types.int64[::1])
        if ext > max_extent:
            k = int(math.ceil(ext / split_size))
            sub = _kmeans_split(X64[idx], w[idx], k, eps, 10)
            for j in range(k):
                pj = idx[sub == j]
                if pj.shape[0]:
                    parts.append(pj)
        else:
            parts.append(idx.copy())
        for p in parts:
            if _vol_sum(vol, p) < min_vol and not (use_extra and _any(extra, p)):
                continue
            s, c = _make_center(p, pts64, w)
            out_idx.append(p)
            out_lev.append(lev)
            out_s.append(s)
            out_ctr.append(c)
            if sub_size > 0:
                e2 = _extent(X64, p) if p.shape[0] > 10 else 0.0
                if e2 > 10.0:
                    k2 = int(math.ceil(e2 / sub_size))
                    sub = _kmeans_split(X64[p], w[p], k2, eps, 10)
                    for j in range(k2):
                        q = p[sub == j]
                        if q.shape[0] and _vol_sum(vol, q) >= min_vol:
                            s2, c2 = _make_center(q, pts64, w)
                            out_idx.append(q)
                            out_lev.append(lev + 100)
                            out_s.append(s2)
                            out_ctr.append(c2)
    n = len(out_idx)
    ptr = np.zeros(n + 1, np.int64)
    for i in range(n):
        ptr[i + 1] = ptr[i] + out_idx[i].shape[0]
    idx = np.empty(ptr[n], np.int64)
    levs = np.empty(n, np.int64)
    ss = np.empty(n, np.float64)
    ctr = np.empty((n, 3), np.float64)
    for i in range(n):
        idx[ptr[i]:ptr[i + 1]] = out_idx[i]
        levs[i] = out_lev[i]
        ss[i] = out_s[i]
        ctr[i] = out_ctr[i]
    return ptr, idx, levs, ss, ctr


def _generate(out, X64, pts64, vol, value, thresholds, adj_ptr, adj, fc, extra=None, lev_offset=0):
    w = vol * value                       # dtype as in the reference (float32 geo / float64 score)
    eps = w.dtype.type(1e-9)
    ex = np.zeros(1, np.bool_) if extra is None else np.ascontiguousarray(extra, np.bool_)
    for lev, th in enumerate(thresholds):
        # the reference compares ``value >= th`` with th a Python float (weak scalar -> value dtype)
        out.append(_generate_level(X64, pts64, vol, value, w, eps, value.dtype.type(th), lev + lev_offset,
                                   adj_ptr, adj, ex, extra is not None, float(fc.max_extent), float(fc.split_size),
                                   float(fc.min_vol), float(fc.sub_size or 0.0)))


@njit(cache=True)
def _dedup(ptr0, idx0):
    """Keep the first of every group of pockets with identical (sorted) point sets; CSR."""
    n = ptr0.shape[0] - 1
    h = np.empty(n, np.int64)
    for i in range(n):
        p = idx0[ptr0[i]:ptr0[i + 1]]
        v = np.int64(p.shape[0]) * np.int64(1000003)
        for j in range(p.shape[0]):
            v = v * np.int64(31) + p[j]
        h[i] = v
    order = np.argsort(h, kind="mergesort")      # stable: equal hashes keep index order
    dup = np.zeros(n, np.bool_)
    for a in range(n):
        i = order[a]
        if dup[i]:
            continue
        b = a + 1
        while b < n and h[order[b]] == h[i]:
            j = order[b]
            pj = idx0[ptr0[j]:ptr0[j + 1]]
            pi = idx0[ptr0[i]:ptr0[i + 1]]
            if not dup[j] and pj.shape[0] == pi.shape[0] and np.all(pj == pi):
                dup[j] = True                     # j > i (stable order)
            b += 1
    keep = np.flatnonzero(~dup)
    ptr = np.zeros(keep.shape[0] + 1, np.int64)
    for t in range(keep.shape[0]):
        ptr[t + 1] = ptr[t] + ptr0[keep[t] + 1] - ptr0[keep[t]]
    idx = np.empty(ptr[-1], np.int64)
    for t in range(keep.shape[0]):
        idx[ptr[t]:ptr[t + 1]] = idx0[ptr0[keep[t]]:ptr0[keep[t] + 1]]
    return ptr, idx, keep


@dataclass
class Candidates:
    ptr: np.ndarray       # (K+1,)
    idx: np.ndarray       # concatenated point indices
    level: np.ndarray     # (K,)
    center: np.ndarray    # (K,3) input frame
    strength: np.ndarray  # (K,)
    res_ptr: np.ndarray = None   # residue CSR (sorted residue indices)
    res: np.ndarray = None

    def __len__(self):
        return len(self.level)

    def members(self, i):
        return self.idx[self.ptr[i]:self.ptr[i + 1]]

    def residues(self, i):
        return self.res[self.res_ptr[i]:self.res_ptr[i + 1]]


@njit(cache=True)
def _radius_graph(X, r, origin, dims, start, order):
    """Two-pass (count, fill) CSR of neighbours within r (excluding self), ascending per row."""
    n = X.shape[0]
    r2 = r * r
    ptr = np.zeros(n + 1, np.int64)
    out = np.empty(0, np.int64)
    for pas in range(2):
        for i in range(n):
            x, y, z = X[i, 0], X[i, 1], X[i, 2]
            x0 = max(0, int(math.floor((x - r - origin[0]) / r)))
            x1 = min(dims[0], int(math.floor((x + r - origin[0]) / r)) + 1)
            y0 = max(0, int(math.floor((y - r - origin[1]) / r)))
            y1 = min(dims[1], int(math.floor((y + r - origin[1]) / r)) + 1)
            z0 = max(0, int(math.floor((z - r - origin[2]) / r)))
            z1 = min(dims[2], int(math.floor((z + r - origin[2]) / r)) + 1)
            k = 0
            for cx in range(x0, x1):
                for cy in range(y0, y1):
                    for cz in range(z0, z1):
                        c = (cx * dims[1] + cy) * dims[2] + cz
                        for t in range(start[c], start[c + 1]):
                            j = order[t]
                            if j == i:
                                continue
                            dx = X[j, 0] - x
                            dy = X[j, 1] - y
                            dz = X[j, 2] - z
                            if dx * dx + dy * dy + dz * dz <= r2:
                                if pas == 1:
                                    out[ptr[i] + k] = j
                                k += 1
            if pas == 0:
                ptr[i + 1] = ptr[i] + k
        if pas == 0:
            out = np.empty(ptr[n], np.int64)
    return ptr, out


def point_graph(X, link):
    """Symmetric CSR adjacency of point pairs with distance <= ``link`` (as cKDTree.query_pairs)."""
    from ...geometry._cells import build_cell_list
    X = np.ascontiguousarray(X, np.float64)
    if len(X) == 0:
        return np.zeros(1, np.int64), np.zeros(0, np.int64)
    origin, dims, start, order = build_cell_list(X, float(link))
    return _radius_graph(X, float(link), origin, dims, start, order)


def candidates(X, pts, vol, gv, score, is_metal, fc) -> Candidates:
    """Fast equivalent of ``burrow.model.candidates`` (geometric pass ∪ score pass, deduplicated)."""
    X64 = np.ascontiguousarray(X, np.float64)
    pts64 = np.ascontiguousarray(pts, np.float64)
    adj_ptr, adj = point_graph(X, fc.link)
    out = []
    _generate(out, X64, pts64, vol, gv, fc.geo_thresholds, adj_ptr, adj, fc)
    if score is not None:
        _generate(out, X64, pts64, vol, np.asarray(score), fc.score_thresholds, adj_ptr, adj, fc,
                  extra=is_metal, lev_offset=10)
    cnt = np.concatenate([np.diff(o[0]) for o in out])
    if len(cnt) == 0:
        return Candidates(np.zeros(1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64),
                          np.zeros((0, 3)), np.zeros(0))
    ptr0 = np.concatenate([[0], np.cumsum(cnt)])
    idx0 = np.concatenate([o[1] for o in out])
    ptr, idx, keep = _dedup(ptr0, idx0)
    lev = np.concatenate([o[2] for o in out])
    s = np.concatenate([o[3] for o in out])
    ctr = np.concatenate([o[4] for o in out])
    return Candidates(ptr, idx, lev[keep], ctr[keep], s[keep])


# --------------------------------------------------------------------------- residues
@njit(cache=True)
def _point_residues(pts, atoms, atom_res, r, cell, origin, dims, start, order, used, n_res):
    """Per used point: unique residue indices of atoms within r (CSR, two passes, stamping)."""
    n_used = used.shape[0]
    r2 = r * r
    stamp = np.full(n_res, -1, np.int64)
    ptr = np.zeros(n_used + 1, np.int64)
    out = np.empty(0, np.int64)
    for pas in range(2):
        stamp[:] = -1
        for ui in range(n_used):
            p = used[ui]
            x, y, z = pts[p, 0], pts[p, 1], pts[p, 2]
            x0 = max(0, int(math.floor((x - r - origin[0]) / cell)))
            x1 = min(dims[0], int(math.floor((x + r - origin[0]) / cell)) + 1)
            y0 = max(0, int(math.floor((y - r - origin[1]) / cell)))
            y1 = min(dims[1], int(math.floor((y + r - origin[1]) / cell)) + 1)
            z0 = max(0, int(math.floor((z - r - origin[2]) / cell)))
            z1 = min(dims[2], int(math.floor((z + r - origin[2]) / cell)) + 1)
            k = 0
            for cx in range(x0, x1):
                for cy in range(y0, y1):
                    for cz in range(z0, z1):
                        c = (cx * dims[1] + cy) * dims[2] + cz
                        for t in range(start[c], start[c + 1]):
                            a = order[t]
                            dx = atoms[a, 0] - x
                            dy = atoms[a, 1] - y
                            dz = atoms[a, 2] - z
                            if dx * dx + dy * dy + dz * dz <= r2:
                                rr = atom_res[a]
                                if stamp[rr] != ui:
                                    stamp[rr] = ui
                                    if pas == 1:
                                        out[ptr[ui] + k] = rr
                                    k += 1
            if pas == 0:
                ptr[ui + 1] = ptr[ui] + k
        if pas == 0:
            out = np.empty(ptr[n_used], np.int64)
    return ptr, out


@njit(cache=True)
def _pocket_residues(ptr, idx, slot, pr_ptr, pr, n_res):
    """Union of point residue lists per pocket -> sorted CSR (two passes, stamping)."""
    k = ptr.shape[0] - 1
    stamp = np.full(n_res, -1, np.int64)
    rptr = np.zeros(k + 1, np.int64)
    out = np.empty(0, np.int64)
    for pas in range(2):
        stamp[:] = -1
        for i in range(k):
            m = 0
            base = rptr[i]
            for t in range(ptr[i], ptr[i + 1]):
                u = slot[idx[t]]
                for e in range(pr_ptr[u], pr_ptr[u + 1]):
                    rr = pr[e]
                    if stamp[rr] != i:
                        stamp[rr] = i
                        if pas == 1:
                            out[base + m] = rr
                        m += 1
            if pas == 0:
                rptr[i + 1] = rptr[i] + m
            else:
                out[base:base + m] = np.sort(out[base:base + m])
        if pas == 0:
            out = np.empty(rptr[k], np.int64)
    return rptr, out


def assign_residues(c: Candidates, pts, atom_coords, atom_res, n_res, r: float = 4.0):
    """Residues with any heavy atom <= r of any pocket point (input frame)."""
    from ...geometry._cells import build_cell_list
    if len(c) == 0:
        c.res_ptr, c.res = np.zeros(1, np.int64), np.zeros(0, np.int64)
        return c
    used = np.unique(c.idx)
    slot = np.full(len(pts), -1, np.int64)
    slot[used] = np.arange(len(used))
    A = np.ascontiguousarray(atom_coords, np.float64)
    origin, dims, start, order = build_cell_list(A, r)
    pr_ptr, pr = _point_residues(np.ascontiguousarray(pts, np.float64), A, np.asarray(atom_res, np.int64), r, r,
                                 origin, dims, start, order, used, int(n_res))
    c.res_ptr, c.res = _pocket_residues(c.ptr, c.idx, slot, pr_ptr, pr, int(n_res))
    return c


# --------------------------------------------------------------------------- NMS
@njit(cache=True)
def _inter_sorted(a, s0, e0, b, s1, e1):
    i, j, n = s0, s1, 0
    while i < e0 and j < e1:
        if a[i] == b[j]:
            n += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return n


@njit(cache=True)
def _nms(order, scores, centers, res_ptr, res, ptr, idx, dist, jacc, use_jacc, overlap, use_overlap,
         max_n, min_score, use_min_score, min_keep):
    kept = np.empty(order.shape[0], np.int64)
    nk = 0
    for oi in range(order.shape[0]):
        i = order[oi]
        if use_min_score and nk >= min_keep and scores[i] < min_score:
            break
        ok = True
        for kk in range(nk):
            j = kept[kk]
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dz = centers[i, 2] - centers[j, 2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) < dist:
                ok = False
                break
            ni = res_ptr[i + 1] - res_ptr[i]
            nj = res_ptr[j + 1] - res_ptr[j]
            if use_jacc and ni > 0 and nj > 0:
                inter = _inter_sorted(res, res_ptr[i], res_ptr[i + 1], res, res_ptr[j], res_ptr[j + 1])
                if inter / (ni + nj - inter) >= jacc:
                    ok = False
                    break
            if use_overlap:
                pi = ptr[i + 1] - ptr[i]
                pj = ptr[j + 1] - ptr[j]
                if pi > 0 and pj > 0:
                    inter = _inter_sorted(idx, ptr[i], ptr[i + 1], idx, ptr[j], ptr[j + 1])
                    if inter / min(pi, pj) >= overlap:
                        ok = False
                        break
        if ok:
            kept[nk] = i
            nk += 1
            if max_n > 0 and nk >= max_n:
                break
    return kept[:nk]


def nms(c: Candidates, scores, dist=5.0, jacc=0.75, max_n=None, overlap=None, min_score=None, min_keep=0):
    """Same semantics as ``burrow.pockets.nms`` on CSR candidates; returns kept indices."""
    scores = np.asarray(scores, np.float64)
    if len(c) == 0:
        return np.zeros(0, np.int64)
    order = np.argsort(-scores, kind="stable")
    return _nms(order, scores, c.center, c.res_ptr, c.res, c.ptr, c.idx, float(dist),
                float(jacc if jacc is not None else 0.0), jacc is not None,
                float(overlap if overlap is not None else 0.0), overlap is not None,
                int(max_n or 0), float(min_score if min_score is not None else 0.0), min_score is not None,
                int(min_keep))
