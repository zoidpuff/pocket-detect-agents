"""Tree-ensemble inference for LightGBM models without LightGBM (CPU Numba + GPU torch).

``FlatForest.from_booster(booster)`` flattens ``booster.dump_model()`` into node arrays:

* ``feat`` (int32), ``thr`` (float64), ``left``/``right`` (int32 child code: >= 0 internal node,
  < 0 leaf ``~code``), ``default_left`` (bool), ``missing`` (0 None, 1 Zero, 2 NaN)
* ``leaf`` (float64) leaf values (shrinkage already folded in by LightGBM), ``roots`` (child code)

Decision rule = LightGBM predictor + ``Tree::NumericalDecision``::

    if |x| <= kZeroThreshold: x = 0        (the predictor drops such values as zeros)
    if isnan(x) and missing != NaN: x = 0
    if (missing == Zero and |x| <= 1e-35) or (missing == NaN and isnan(x)): go default
    else: go left iff x <= thr

CPU: float64 comparison and float64 accumulation in tree order -> bit-identical to
``booster.predict(..., raw_score=True)``.  GPU: features are float32, so ``x <= thr`` (double) is
evaluated exactly as ``x <= thr32`` with ``thr32`` = largest float32 <= thr; leaf values are summed
in float64 (order may differ from LightGBM: |Δraw| ~ 1e-15).

Only numerical splits are supported (categorical splits / linear trees raise).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

K_ZERO = float(np.float32(1e-35))   # LightGBM kZeroThreshold (1e-35f as double)
_MISS = {"None": 0, "Zero": 1, "NaN": 2}


@dataclass
class FlatForest:
    feat: np.ndarray
    thr: np.ndarray
    thr32: np.ndarray
    left: np.ndarray
    right: np.ndarray
    default_left: np.ndarray
    missing: np.ndarray
    leaf: np.ndarray
    roots: np.ndarray
    n_features: int
    sigmoid: float = 1.0          # binary objective scale (0 -> objective has no sigmoid)
    average_output: bool = False
    max_depth: int = 0
    feature_names: list | None = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_booster(cls, booster) -> "FlatForest":
        return cls.from_dump(booster.dump_model())

    @classmethod
    def from_model_string(cls, s: str) -> "FlatForest":
        import lightgbm as lgb
        return cls.from_booster(lgb.Booster(model_str=s))

    @classmethod
    def from_dump(cls, d: dict) -> "FlatForest":
        if d.get("num_tree_per_iteration", 1) != 1:
            raise NotImplementedError("multiclass models are not supported")
        feat, thr, left, right, dl, miss, leaf, roots = [], [], [], [], [], [], [], []
        max_depth = 0

        def rec(n, depth):
            nonlocal max_depth
            if "leaf_value" in n and "split_feature" not in n:
                leaf.append(float(n["leaf_value"]))
                max_depth = max(max_depth, depth)
                return ~(len(leaf) - 1)
            if n["decision_type"] != "<=":
                raise NotImplementedError("categorical splits are not supported")
            i = len(feat)
            feat.append(int(n["split_feature"]))
            thr.append(float(n["threshold"]))
            dl.append(bool(n["default_left"]))
            miss.append(_MISS[n["missing_type"]])
            left.append(0)
            right.append(0)
            left[i] = rec(n["left_child"], depth + 1)
            right[i] = rec(n["right_child"], depth + 1)
            return i

        for t in d["tree_info"]:
            if t.get("is_linear"):
                raise NotImplementedError("linear trees are not supported")
            roots.append(rec(t["tree_structure"], 0))
        obj = str(d.get("objective", ""))
        sig = 0.0
        if obj.startswith("binary") or obj.startswith("cross_entropy"):
            sig = 1.0
            for tok in obj.split():
                if tok.startswith("sigmoid:"):
                    sig = float(tok.split(":")[1])
        thr_a = np.array(thr, np.float64)
        return cls(feat=np.array(feat, np.int32), thr=thr_a, thr32=_floor_f32(thr_a),
                   left=np.array(left, np.int32), right=np.array(right, np.int32),
                   default_left=np.array(dl, np.bool_), missing=np.array(miss, np.int8),
                   leaf=np.array(leaf, np.float64), roots=np.array(roots, np.int32),
                   n_features=int(d.get("max_feature_idx", -1)) + 1, sigmoid=sig,
                   average_output=bool(d.get("average_output", False)), max_depth=max_depth,
                   feature_names=list(d.get("feature_names") or []) or None)

    @property
    def n_trees(self) -> int:
        return len(self.roots)

    def packed(self):
        """Branch-free layout: leaves become self-looping nodes (thr=+inf), children interleaved
        ``child[2n] = left, child[2n+1] = right``; per-tree depth; value per node (cached)."""
        if getattr(self, "_packed", None) is not None:
            return self._packed
        n_int, n_leaf = len(self.feat), len(self.leaf)
        n = n_int + n_leaf
        feat = np.zeros(n, np.int32)
        thr = np.full(n, np.inf, np.float64)
        child = np.empty(2 * n, np.int32)
        val = np.zeros(n, np.float64)
        feat[:n_int] = self.feat
        thr[:n_int] = self.thr
        enc = lambda c: np.where(c >= 0, c, n_int + ~c).astype(np.int32)
        child[0:2 * n_int:2] = enc(self.left)
        child[1:2 * n_int:2] = enc(self.right)
        li = np.arange(n_int, n, dtype=np.int32)
        child[2 * n_int::2] = li
        child[2 * n_int + 1::2] = li
        val[n_int:] = self.leaf
        roots = enc(self.roots)
        depth = _tree_depths(self.roots, self.left, self.right)
        self._packed = (feat, thr, child, val, roots, depth)
        return self._packed

    # ------------------------------------------------------------------ inference
    def _link(self, raw, raw_score):
        if raw_score or self.sigmoid == 0.0:
            return raw
        return _sigmoid(raw, self.sigmoid)    # scalar libm exp, as LightGBM (numpy's SIMD exp differs by 1 ulp)

    def predict(self, X, raw_score: bool = False) -> np.ndarray:
        """CPU (Numba, parallel over rows) prediction; bit-identical to LightGBM."""
        X = np.ascontiguousarray(X, np.float64)
        if X.ndim != 2 or X.shape[1] < self.n_features:
            raise ValueError(f"expected (n, >= {self.n_features}) features, got {X.shape}")
        if not (self.missing != 0).any():
            # missing=None everywhere: NaN -> 0 and |x| <= kZero -> 0 up front, then branch-free
            X = np.where(np.isnan(X) | (np.abs(X) <= K_ZERO), 0.0, X)
            feat, thr, child, val, roots, depth = self.packed()
            raw = _predict_packed(X, feat, thr, child, val, roots, depth, len(self.feat))
        else:
            raw = _predict_kernel(X, self.feat, self.thr, self.left, self.right, self.default_left,
                                  self.missing, self.leaf, self.roots)
        if self.average_output:
            raw = raw / self.n_trees
        return self._link(raw, raw_score)

    def to_torch(self, device) -> "TorchForest":
        return TorchForest(self, device)


def _floor_f32(t: np.ndarray) -> np.ndarray:
    """Largest float32 <= t (so that float32 x <= t  <=>  x <= result)."""
    f = t.astype(np.float32)
    over = f.astype(np.float64) > t
    f[over] = np.nextafter(f[over], np.float32(-np.inf))
    return f


@njit(cache=True)
def _sigmoid(raw, a):
    out = np.empty_like(raw)
    for i in range(raw.shape[0]):
        out[i] = 1.0 / (1.0 + math.exp(-a * raw[i]))
    return out


@njit(cache=True, inline="always")
def _tree_value(x, root, feat, thr, left, right, dl, miss, leaf):
    node = root
    while node >= 0:
        v = x[feat[node]]
        m = miss[node]
        if -K_ZERO <= v <= K_ZERO:
            v = 0.0
        if math.isnan(v) and m != 2:
            v = 0.0
        if (m == 1 and -K_ZERO <= v <= K_ZERO) or (m == 2 and math.isnan(v)):
            node = left[node] if dl[node] else right[node]
        elif v <= thr[node]:
            node = left[node]
        else:
            node = right[node]
    return leaf[~node]


@njit(parallel=True, cache=True)
def _predict_kernel(X, feat, thr, left, right, dl, miss, leaf, roots):
    n = X.shape[0]
    out = np.empty(n, np.float64)
    for i in prange(n):
        x = X[i]
        s = 0.0
        for t in range(roots.shape[0]):
            s += _tree_value(x, roots[t], feat, thr, left, right, dl, miss, leaf)
        out[i] = s
    return out


def _tree_depths(roots, left, right):
    out = np.zeros(len(roots), np.int32)
    for t, r in enumerate(roots):
        stack, d = [(int(r), 0)], 0
        while stack:
            nd, dep = stack.pop()
            if nd < 0:
                d = max(d, dep)
            else:
                stack.append((int(left[nd]), dep + 1))
                stack.append((int(right[nd]), dep + 1))
        out[t] = d
    return out


@njit(parallel=True, cache=True)
def _predict_packed(X, feat, thr, child, val, roots, depth, n_int):
    """Rows in blocks of 16 traverse each tree in lock-step (independent loads -> ILP, no
    unpredictable branches: leaves loop onto themselves).  Accumulation per row in tree order."""
    n = X.shape[0]
    B = 16
    n_blk = (n + B - 1) // B
    out = np.zeros(n, np.float64)
    for b in prange(n_blk):
        r0 = b * B
        m = min(B, n - r0)
        acc = np.zeros(B, np.float64)
        nodes = np.empty(B, np.int32)
        for t in range(roots.shape[0]):
            for r in range(m):
                nodes[r] = roots[t]
            for _ in range(depth[t]):
                done = True
                for r in range(m):
                    nd = nodes[r]
                    nn = child[2 * nd + (X[r0 + r, feat[nd]] > thr[nd])]
                    nodes[r] = nn
                    done &= nn >= n_int
                if done:
                    break
            for r in range(m):
                acc[r] += val[nodes[r]]
        for r in range(m):
            out[r0 + r] = acc[r]
    return out


class TorchForest:
    """GPU traversal: all (row, tree) pairs advance one level per iteration (gather-based)."""

    def __init__(self, f: FlatForest, device):
        import torch
        dev = torch.device(device)
        self.f = f
        self.dev = dev
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=dev)
        n_nodes = max(len(f.feat), 1)
        pad = lambda a, v: a if len(a) else np.array([v], a.dtype)
        self.feat = t(pad(f.feat, 0), torch.long)
        self.thr = t(pad(f.thr32, 0), torch.float32)
        self.left = t(pad(f.left, -1), torch.long)
        self.right = t(pad(f.right, -1), torch.long)
        self.dl = t(pad(f.default_left, True), torch.bool)
        self.miss = t(pad(f.missing, 0), torch.int8)
        self.leaf = t(f.leaf, torch.float64)
        self.roots = t(f.roots, torch.long)
        self.any_missing = bool((f.missing != 0).any())
        self.n_nodes = n_nodes
        self._packed = None

    def predict(self, X, raw_score: bool = False, max_elems: int = 1 << 25, use_triton: bool | None = None):
        import torch
        if use_triton is None:
            use_triton = _HAVE_TRITON and self.dev.type == "cuda" and not self.any_missing
        if use_triton:
            return predict_triton(self, X, raw_score)
        X = torch.as_tensor(X, device=self.dev)
        if X.dtype != torch.float32:
            X = X.float()
        n, T = X.shape[0], len(self.roots)
        out = torch.empty(n, dtype=torch.float64, device=self.dev)
        chunk = max(1, max_elems // max(T, 1))
        for s in range(0, n, chunk):
            Xc = X[s:s + chunk]
            m = Xc.shape[0]
            node = self.roots[None, :].expand(m, T).clone()
            rows = torch.arange(m, device=self.dev)[:, None].expand(m, T)
            Xc = torch.where(Xc.abs() <= K_ZERO, torch.zeros((), device=self.dev), Xc)
            for _ in range(self.f.max_depth):
                internal = node >= 0
                nd = node.clamp(min=0)
                v = Xc[rows, self.feat[nd]]
                go_left = v <= self.thr[nd]
                if self.any_missing:
                    ms = self.miss[nd]
                    nan = torch.isnan(v)
                    v0 = torch.where(nan & (ms != 2), torch.zeros_like(v), v)
                    use_def = ((ms == 1) & (v0.abs() <= K_ZERO)) | ((ms == 2) & nan)
                    go_left = torch.where(use_def, self.dl[nd], v0 <= self.thr[nd])
                else:
                    go_left = go_left | torch.isnan(v) & (0.0 <= self.thr[nd])   # NaN -> 0 (missing None)
                nxt = torch.where(go_left, self.left[nd], self.right[nd])
                node = torch.where(internal, nxt, node)
            raw = self.leaf[~node].sum(1)
            out[s:s + chunk] = raw
        if self.f.average_output:
            out = out / T
        if raw_score or self.f.sigmoid == 0.0:
            return out
        return torch.sigmoid(self.f.sigmoid * out)


# --------------------------------------------------------------------------- Triton fast path
try:  # pragma: no cover
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False

if _HAVE_TRITON:
    @triton.jit
    def _forest_kernel(X_ptr, F, feat_ptr, thr_ptr, child_ptr, val_ptr, roots_ptr, depth_ptr, out_ptr,
                       N, T, BR: tl.constexpr):
        pid = tl.program_id(0)
        r = pid * BR + tl.arange(0, BR)
        rm = r < N
        acc = tl.zeros([BR], dtype=tl.float64)
        row = r.to(tl.int64) * F
        for t in range(T):
            nd = tl.zeros([BR], dtype=tl.int32) + tl.load(roots_ptr + t)
            dep = tl.load(depth_ptr + t)
            for _ in range(dep):
                f = tl.load(feat_ptr + nd)
                th = tl.load(thr_ptr + nd)
                x = tl.load(X_ptr + row + f, mask=rm, other=0.0)
                nd = tl.load(child_ptr + 2 * nd + (x > th).to(tl.int32))
            acc += tl.load(val_ptr + nd)
        tl.store(out_ptr + r, acc, mask=rm)


def _torch_packed(forest: FlatForest, device):
    import torch
    feat, thr, child, val, roots, depth = forest.packed()
    n_int = len(forest.feat)
    t32 = _floor_f32(thr)
    t32[n_int:] = np.inf
    dev = torch.device(device)
    return dict(feat=torch.as_tensor(feat, device=dev), thr=torch.as_tensor(t32, device=dev),
                child=torch.as_tensor(child, device=dev), val=torch.as_tensor(val, device=dev),
                roots=torch.as_tensor(roots, device=dev), depth=torch.as_tensor(depth, device=dev),
                depth_np=depth)


def predict_triton(tf: "TorchForest", X, raw_score: bool = False, block_rows: int = 128):
    """Fused GPU traversal (missing=None models): one lane per row, trees in sequence."""
    import torch
    if tf._packed is None:
        tf._packed = _torch_packed(tf.f, tf.dev)
    p = tf._packed
    X = torch.as_tensor(X, device=tf.dev).float()
    X = torch.where(torch.isnan(X) | (X.abs() <= K_ZERO), torch.zeros((), device=tf.dev), X).contiguous()
    n = X.shape[0]
    out = torch.empty(n, dtype=torch.float64, device=tf.dev)
    if n:
        with torch.cuda.device(tf.dev):
            _forest_kernel[(triton.cdiv(n, block_rows),)](X, X.shape[1], p["feat"], p["thr"], p["child"], p["val"],
                                                         p["roots"], p["depth"], out, n, len(p["depth_np"]),
                                                         BR=block_rows, num_warps=4)
    if tf.f.average_output:
        out = out / tf.f.n_trees
    if raw_score or tf.f.sigmoid == 0.0:
        return out
    return torch.sigmoid(tf.f.sigmoid * out)
