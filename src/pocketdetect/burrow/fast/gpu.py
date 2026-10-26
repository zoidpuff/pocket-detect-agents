"""Batched GPU BURROW featurizer + point scoring; pocket formation/ranking stay on CPU.

``predict_batch(proteins, fm, device)``:

    host prep (canonical frame, metal candidates; CPU, parallelisable)
    -> GPU, whole batch at once: solvent-shell grid points, dual absorption grid (+ first-hit labels),
       PSP prefilter + cascade, dual-grid 60-ray march, neighbour sums, metal geometry, feature
       assembly, point-model scores (Triton forest)
    -> host: formation / descriptors / ranker / NMS per structure (``pipeline.predict_from_points``)

The kernels replicate the CPU reference arithmetic (float64 distances, positions and absorption
sums; float32 casts in the same places; NumPy/OpenBLAS float32 semantics for the metal geometry),
because the tree models split exactly at feature values (e.g. 17/60) and a 1-ulp difference changes
point scores.  Agreement is checked by ``scripts/verify_burrow_fast.py --device cuda:N``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ...geometry.frames import canonical_frame, from_frame, to_frame
from ...geometry.sphere import fibonacci_sphere
from ..features import BurrowConfig, PointSet
from ..prep import HIT_CLASSES
from . import features as FFE
from .pipeline import FastModel, predict_from_points


@dataclass
class Prepped:
    P: object
    R: np.ndarray
    c: np.ndarray
    X: np.ndarray
    weights: np.ndarray | None
    mp: np.ndarray          # metal candidates (canonical frame)
    md: np.ndarray          # their nearest-atom distance
    order_x: np.ndarray     # atoms sorted by x (stable): label tie-break order of the reference


def prep(P, cfg: BurrowConfig = BurrowConfig(), weights=None) -> Prepped:
    """Host-side per-structure preparation (picklable; run in worker processes)."""
    if cfg.canonical:
        R, c = canonical_frame(P.coords)
        X = to_frame(P.coords, R, c)
    else:
        R, c, X = np.eye(3), np.zeros(3), np.asarray(P.coords, np.float32)
    mp = FFE.metal_candidates(X, P, cfg)[0] if cfg.metal else np.zeros((0, 3), np.float32)
    md = (FFE._Tree(X, 4.0).nearest(np.ascontiguousarray(mp, np.float64)).astype(np.float32)
          if len(mp) else np.zeros(0, np.float32))
    return Prepped(P, R, c, X, weights, mp, md, np.argsort(X[:, 0].astype(np.float64), kind="stable"))


class _Frames:
    """Per-structure absolute-aligned grids (host-computed, float64 exact like the CPU code)."""

    def __init__(self, Xs, pads, spacing, dev):
        import torch
        from ...geometry._grid import grid_frame
        fr = [grid_frame(X.astype(np.float64).min(0), X.astype(np.float64).max(0), spacing, pad)
              if len(X) else (np.zeros(3), np.ones(3, np.int64)) for X, pad in zip(Xs, pads)]
        self.origin = torch.tensor(np.stack([f[0] for f in fr]), dtype=torch.float64, device=dev)
        self.dims = torch.tensor(np.stack([f[1] for f in fr]), dtype=torch.long, device=dev)
        nv = self.dims.prod(1)
        self.base = torch.cumsum(nv, 0) - nv
        self.total = int(nv.sum().item())
        self.spacing = float(spacing)

    def atom_c0(self, X64, b):
        import torch
        return torch.floor((X64 - self.origin[b]) / self.spacing).to(torch.int32).contiguous()


def _grid_points(X64, radii64, batch, Xs, cfg, dev):
    """Exact GPU equivalent of geometry.candidates.grid_candidates (points, nearest_dist, batch)."""
    import torch
    from ...geometry._grid import stamp_offsets_torch
    from .gpu_kernels import _grid64_kernel, _launch
    fr = _Frames(Xs, [cfg.max_dist] * len(Xs), cfg.spacing, dev)
    INIT = torch.iinfo(torch.int64).max
    key = torch.full((fr.total,), INIT, dtype=torch.int64, device=dev)
    clash = torch.zeros(fr.total, dtype=torch.int32, device=dev)
    off = stamp_offsets_torch(cfg.max_dist, cfg.spacing, dev)
    n = len(X64)
    if n:
        # tie-break index = atom index within its structure (as the CPU kernel)
        first = torch.zeros(len(Xs), dtype=torch.long, device=dev)
        first[1:] = torch.cumsum(torch.bincount(batch, minlength=len(Xs)), 0)[:-1]
        gidx = (torch.arange(n, device=dev) - first[batch]).to(torch.int32)
        _launch(_grid64_kernel, (n,), dev, X64.contiguous(), fr.origin[batch].contiguous(), fr.atom_c0(X64, batch),
                fr.base[batch].contiguous(), fr.dims[batch].to(torch.int32).contiguous(), off, off.shape[0],
                ((radii64 + cfg.probe) ** 2).contiguous(), gidx, float(cfg.spacing), float(cfg.max_dist) ** 2,
                key, clash, BO=256, num_warps=4)
    flat = torch.nonzero((key != INIT) & (clash == 0), as_tuple=True)[0]
    vb = torch.searchsorted(fr.base, flat, right=True) - 1
    loc = flat - fr.base[vb]
    dv = fr.dims[vb]
    ijk = torch.stack([loc // (dv[:, 2] * dv[:, 1]), (loc // dv[:, 2]) % dv[:, 1], loc % dv[:, 2]], 1).double()
    pts = (fr.origin[vb] + (ijk + 0.5) * cfg.spacing).float()
    d2 = (key[flat] >> 32).to(torch.int32).view(torch.float32)
    return pts, torch.sqrt(d2), vb


class _DualGrid:
    """Exact GPU equivalent of ``burrow.raycast.dual_grid`` for a batch (flat grids)."""

    def __init__(self, X64, rb64, hit_class, bb, mu_atom, batch, order_rank, Xs, rbs, spacing, dev):
        import torch
        from ...geometry._grid import stamp_offsets_torch
        from .gpu_kernels import _dual64_kernel, _launch
        fr = _Frames(Xs, [float(r.max()) if len(r) else 0.0 for r in rbs], spacing, dev)
        self.origin, self.dims, self.base, self.spacing = fr.origin, fr.dims, fr.base, spacing
        self.origin32 = fr.origin.float()
        self.ref_length = 2.0
        self.mu = torch.zeros(fr.total, dtype=torch.float32, device=dev)
        self.mub = torch.zeros(fr.total, dtype=torch.float32, device=dev)
        INIT = torch.iinfo(torch.int64).max
        key = torch.full((fr.total,), INIT, dtype=torch.int64, device=dev)
        rmax = float(rb64.max().item()) if len(rb64) else 0.0
        off = stamp_offsets_torch(rmax, spacing, dev)
        n = len(X64)
        if n:
            _launch(_dual64_kernel, (n,), dev, X64.contiguous(), fr.origin[batch].contiguous(), fr.atom_c0(X64, batch),
                    fr.base[batch].contiguous(), fr.dims[batch].to(torch.int32).contiguous(), off, off.shape[0],
                    rb64.contiguous(), mu_atom.contiguous(), bb.to(torch.int8).contiguous(),
                    order_rank.to(torch.int32).contiguous(), float(spacing), self.mu, self.mub, key, BO=256, num_warps=4)
        has = key != INIT
        atom_of_rank = torch.empty_like(order_rank)
        atom_of_rank[order_rank] = torch.arange(len(order_rank), device=dev)
        lab = torch.full((fr.total,), -1, dtype=torch.int8, device=dev)
        lab[has] = hit_class[atom_of_rank[(key[has] & 0xFFFFFFFF)]].to(torch.int8)
        self.label = lab


def _psp(pts, pb, grid, cfg, dev):
    """Exact GPU equivalent of geometry.candidates.psp_counts on the dual grid's ``mu``."""
    import torch
    from ...geometry.candidates import _psp_setup
    from .gpu_kernels import _launch, _psp64_kernel
    P = len(pts)
    out = torch.zeros(P, dtype=torch.int32, device=dev)
    if P == 0:
        return out
    dirs, n_steps, min_mu = _psp_setup(cfg.psp_length, grid.spacing, None, 0.5, grid.ref_length)
    dirs = torch.as_tensor(np.ascontiguousarray(dirs[:cfg.psp_dirs]), dtype=torch.float64, device=dev)
    bp = 64
    _launch(_psp64_kernel, ((P + bp - 1) // bp,), dev, pts.contiguous(), grid.origin[pb].contiguous(),
            grid.dims[pb].to(torch.int32).contiguous(), grid.base[pb].contiguous(), dirs, grid.mu, out, P,
            dirs.shape[0], int(n_steps), float(grid.spacing), float(min_mu), BP=bp, DB=8, num_warps=4)
    return out


def _march(pts, pb, grid, cfg, dev):
    import torch
    from .gpu_kernels import _launch, _march64_kernel
    P, K = len(pts), cfg.n_rays
    step = 1.0
    ck = [int(round(L / step)) for L in cfg.lengths]
    NL, NC = len(ck), len(HIT_CLASSES)
    blocked = torch.zeros((P, NL), dtype=torch.float32, device=dev)
    blocked_bb = torch.zeros((P, NL), dtype=torch.float32, device=dev)
    depth = torch.zeros(P, dtype=torch.float32, device=dev)
    hit = torch.zeros((P, NC), dtype=torch.float32, device=dev)
    if P == 0:
        return blocked, blocked_bb, depth, hit
    dirs = torch.as_tensor(fibonacci_sphere(K).astype(np.float64), device=dev)
    rb = 64
    bp = 16
    _launch(_march64_kernel, ((P + bp - 1) // bp,), dev, pts.contiguous(), grid.origin[pb].contiguous(),
            grid.dims[pb].to(torch.int32).contiguous(), grid.base[pb].contiguous(), dirs.contiguous(), grid.mu,
            grid.mub, grid.label, torch.as_tensor(ck, dtype=torch.int32, device=dev), blocked, blocked_bb, depth, hit,
            P, K, float(step), float(ck[-1] * step), 1.0 / grid.spacing, NL=NL, NC=NC, BP=bp, RB=rb, num_warps=4)
    return blocked, blocked_bb, depth, hit


def _neighbor_sums(pts, pb, X, batch, props, wcn_col, radii, B, dev):
    """S (P, R, F) identical to the CPU reference: integer-valued props via the fp64-distance Triton
    kernel (fp32 dot is exact for integers), the continuous ``wcn`` column via float64 pair sums
    (a float64 sum of these float32 values is exact in any order)."""
    import torch
    from ...geometry._torch_cells import TorchCellList
    from ...geometry.torch_utils import batch_bounds
    from .gpu_kernels import _launch, _nf64_kernel
    P, F = len(pts), props.shape[1]
    R = len(radii)
    S = torch.zeros((P, R, F), dtype=torch.float32, device=dev)
    if P == 0 or len(X) == 0:
        return S
    rcut = max(radii)
    # ---- integer props (wcn column zeroed) with the blocked-cell Triton kernel
    pint = props.clone()
    pint[:, wcn_col] = 0.0
    FP = max(16, 1 << int(np.ceil(np.log2(max(F, 1)))))
    cs = rcut / 2.0
    lo_a, hi_a = batch_bounds(X, batch, B)
    lo_p, hi_p = batch_bounds(pts, pb, B)
    has_p = torch.bincount(pb, minlength=B) > 0
    lo = torch.where(has_p[:, None], torch.minimum(lo_a, lo_p), lo_a)
    hi = torch.where(has_p[:, None], torch.maximum(hi_a, hi_p), hi_a)
    dims = torch.floor((hi - lo) / cs).long() + 1
    ncell = dims.prod(1)
    base = torch.cumsum(ncell, 0) - ncell
    total = int(ncell.sum().item())

    def lin_of(x, b):
        c = torch.floor((x - lo[b]) / cs).long()
        c = torch.minimum(torch.maximum(c, torch.zeros_like(c)), dims[b] - 1)
        return base[b] + (c[:, 0] * dims[b, 1] + c[:, 1]) * dims[b, 2] + c[:, 2], c

    alin, _ = lin_of(X, batch)
    aord = torch.argsort(alin)
    cstart = torch.cat([alin.new_zeros(1), torch.cumsum(torch.bincount(alin, minlength=total), 0)])
    props_s = torch.zeros((len(X), FP), dtype=torch.float32, device=dev)
    props_s[:, :F] = pint[aord]
    plin, _ = lin_of(pts, pb)
    pord = torch.argsort(plin)
    pcnt = torch.bincount(plin, minlength=total)
    pstart = torch.cumsum(pcnt, 0) - pcnt
    cells = torch.nonzero(pcnt, as_tuple=True)[0]
    bp = 16
    nblk = (pcnt[cells] + bp - 1) // bp
    bcell = torch.repeat_interleave(cells, nblk)
    first = torch.cumsum(nblk, 0) - nblk
    j = torch.arange(bcell.shape[0], device=dev) - torch.repeat_interleave(first, nblk)
    bb = torch.searchsorted(base, bcell, right=True) - 1
    loc = bcell - base[bb]
    blk = torch.stack([pstart[bcell] + j * bp, torch.clamp(pcnt[bcell] - j * bp, max=bp), loc // (dims[bb, 1] * dims[bb, 2]),
                       (loc // dims[bb, 2]) % dims[bb, 1], loc % dims[bb, 2], bb], 1).to(torch.int32).contiguous()
    th = torch.tensor([r * r for r in radii] + [-1.0] * (4 - R), dtype=torch.float64, device=dev)
    out = torch.zeros((P, R, FP), dtype=torch.float32, device=dev)
    _launch(_nf64_kernel, (blk.shape[0],), dev, pts[pord].contiguous(), X[aord].contiguous(), props_s, cstart, blk,
            dims.to(torch.int32).contiguous(), base.contiguous(), th, out, KR=2, BP=bp, BA=16, NJ=R, FP=FP,
            num_warps=4, num_stages=1)
    S[pord] = out[:, :, :F]
    # ---- wcn column: float64 pair sums, nested radii
    X64 = X.double()
    P64 = pts.double()
    acc = torch.zeros((P, R), dtype=torch.float64, device=dev)
    w64 = props[:, wcn_col].double()
    r2 = torch.tensor([r * r for r in radii], dtype=torch.float64, device=dev)
    for qi, aj, d2 in TorchCellList(X64, batch, rcut / 2.0, B).pairs(P64, pb, rcut):
        diff = X64[aj] - P64[qi]
        d2 = (diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]) + diff[:, 2] * diff[:, 2]
        for ri in range(R):
            m = d2 <= r2[ri]
            acc[:, ri].index_add_(0, qi[m], w64[aj[m]])
    S[:, :, wcn_col] = acc.float()
    return S


def _metal_geometry(pts, pb, Xco, bco, B, dev, r=2.8, max_nb=16):
    """n_coord28 / coord_spread with the reference's float32 arithmetic (see features.metal_geometry)."""
    import torch
    from ...geometry._torch_cells import TorchCellList
    P = len(pts)
    n_co = torch.zeros(P, dtype=torch.float32, device=dev)
    spread = torch.zeros(P, dtype=torch.float32, device=dev)
    if len(Xco) == 0 or P == 0:
        return n_co, spread
    qs, as_ = [], []
    P64, C64 = pts.double(), Xco.double()
    for qi, aj, _ in TorchCellList(C64, bco, r, B).pairs(P64, pb, r):
        diff = C64[aj] - P64[qi]
        d2 = (diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]) + diff[:, 2] * diff[:, 2]
        m = d2 <= r * r
        qs.append(qi[m])
        as_.append(aj[m])
    qi, aj = torch.cat(qs), torch.cat(as_)
    if len(qi) == 0:
        return n_co, spread
    o = torch.argsort(qi * (len(Xco) + 1) + aj)          # per point, coordinating atoms ascending
    qi, aj = qi[o], aj[o]
    v = Xco[aj] - pts[qi]                                # float32
    nrm = torch.sqrt((v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]) + v[:, 2] * v[:, 2])
    v = v / (nrm + 1e-9)[:, None]
    cnt = torch.bincount(qi, minlength=P)
    has = torch.nonzero(cnt, as_tuple=True)[0]
    start = torch.cumsum(cnt, 0) - cnt
    k = torch.arange(len(qi), device=dev) - start[qi]
    pad = torch.zeros((P, int(cnt.max().item()), 3), dtype=torch.float32, device=dev)
    pad[qi, k] = v
    s = torch.zeros((P, 3), dtype=torch.float32, device=dev)
    for t in range(pad.shape[1]):                       # sequential float32 sum (== mean's add.reduce)
        s = s + pad[:, t]
    mean = s[has] / cnt[has].float()[:, None]
    p2 = (mean * mean).double()                         # OpenBLAS sdot: float32 products, float64 sum
    nrm1 = torch.sqrt(((p2[:, 0] + p2[:, 1]) + p2[:, 2]).float())
    n_co[has] = cnt[has].float()
    spread[has] = 1.0 - nrm1
    return n_co, spread


def featurize_batch(items, cfg: BurrowConfig, device, fm: FastModel | None = None):
    """GPU features (+ point scores if ``fm``) for a list of :class:`Prepped`.

    Returns a list of ``(PointSet, score or None)`` with host numpy arrays."""
    import torch
    from ...geometry.buriedness import absorption_from_weights
    from ...geometry.candidates import sublattice_mask_torch

    dev = torch.device(device)
    tm = {}
    t0 = time.perf_counter()
    B = len(items)
    nat = np.array([len(it.X) for it in items])
    bnp = np.repeat(np.arange(B), nat)
    T = lambda a, dt=None: torch.as_tensor(np.ascontiguousarray(a), device=dev, dtype=dt)
    Xs = [it.X for it in items]
    X = T(np.concatenate(Xs), torch.float32)
    X64 = X.double()
    batch = T(bnp, torch.long)
    radii_np = [np.asarray(it.P.radii, np.float32) for it in items]
    radii64 = T(np.concatenate(radii_np).astype(np.float64), torch.float64)
    w_np = [np.ones(n, np.float32) if it.weights is None else np.asarray(it.weights, np.float32)
            for it, n in zip(items, nat)]
    w_all = np.concatenate(w_np)
    mu_atom = T(absorption_from_weights(w_all, 2.0), torch.float32)
    hit_class = T(np.concatenate([it.P.hit_class for it in items]), torch.long)
    bbm = T(np.concatenate([it.P.bb_mask for it in items]), torch.bool)
    props = T(np.concatenate([it.P.props for it in items]), torch.float32)
    off = np.concatenate([[0], np.cumsum(nat)[:-1]])
    rank = np.empty(len(bnp), np.int64)
    for it, o in zip(items, off):
        rank[o + it.order_x] = o + np.arange(len(it.X))
    rank_t = T(rank, torch.long)
    # ---- solvent-shell grid points (AF mode: only atoms with w >= 0.5 exclude points)
    if any(it.weights is not None for it in items):
        cms = [w >= 0.5 for w in w_np]
        cm = T(np.concatenate(cms), torch.bool)
        gp, gnd, gb = _grid_points(X64[cm], radii64[cm], batch[cm], [x[m] for x, m in zip(Xs, cms)], cfg, dev)
    else:
        gp, gnd, gb = _grid_points(X64, radii64, batch, Xs, cfg, dev)
    t1 = time.perf_counter()
    rb_np = [r.astype(np.float64) * 1.0 + cfg.radius_pad for r in radii_np]
    grid = _DualGrid(X64, T(np.concatenate(rb_np), torch.float64), hit_class, bbm, mu_atom, batch, rank_t, Xs, rb_np,
                     cfg.spacing, dev)
    psp = _psp(gp, gb, grid, cfg, dev)
    dense = psp >= cfg.min_psp
    keep = dense.clone()
    if cfg.subsample and cfg.subsample > 1:
        keep |= sublattice_mask_torch(gp, cfg.spacing, cfg.subsample)
    gp, gb = gp[keep], gb[keep]
    gvol = torch.where(dense[keep], 1.0, float(cfg.subsample ** 3)).float()
    gpsp = psp[keep].float()
    gnd = gnd[keep].float()
    nmp = np.array([len(it.mp) for it in items])
    if nmp.sum():
        mp = T(np.concatenate([it.mp for it in items if len(it.mp)]), torch.float32)
        mb = T(np.repeat(np.arange(B), nmp), torch.long)
        mpsp = _psp(mp, mb, grid, cfg, dev).float()
        md = T(np.concatenate([it.md for it in items if len(it.md)]), torch.float32)
        pts = torch.cat([gp, mp])
        pb = torch.cat([gb, mb])
        vol = torch.cat([gvol, torch.zeros(len(mp), device=dev)])
        pspk = torch.cat([gpsp, mpsp])
        ndist = torch.cat([gnd, md])
        is_metal = torch.cat([torch.zeros(len(gp), dtype=torch.bool, device=dev),
                              torch.ones(len(mp), dtype=torch.bool, device=dev)])
        o = torch.argsort(pb * 2 + is_metal.long(), stable=True)
        pts, pb, vol, pspk, ndist, is_metal = pts[o], pb[o], vol[o], pspk[o], ndist[o], is_metal[o]
    else:
        pts, pb, vol, pspk, ndist = gp, gb, gvol, gpsp, gnd
        is_metal = torch.zeros(len(gp), dtype=torch.bool, device=dev)
    t2 = time.perf_counter()
    blocked, blocked_bb, depth, hit = _march(pts, pb, grid, cfg, dev)
    t3 = time.perf_counter()
    pn = items[0].P.prop_names
    S = _neighbor_sums(pts, pb, X, batch, props, pn.index("wcn"), tuple(float(r) for r in cfg.nb_radii), B, dev)
    comask = T(np.concatenate([it.P.coord_mask for it in items]), torch.bool)
    n_co, spread = _metal_geometry(pts, pb, X[comask], batch[comask], B, dev)
    # ---- assembly (same names / order / float32 arithmetic as the reference)
    names, cols = [], []

    def add(n, v):
        names.append(n)
        cols.append(v.float())

    for i, L in enumerate(cfg.lengths):
        add(f"bur{int(L)}", blocked[:, i])
    for i, L in enumerate(cfg.lengths):
        add(f"burbb{int(L)}", blocked_bb[:, i])
    add("depth", depth)
    for i, h in enumerate(HIT_CLASSES):
        add(f"hit_{h}", hit[:, i])
    hs = hit[:, 0]
    for i in range(1, hit.shape[1]):
        hs = hs + hit[:, i]
    add("escape", 1.0 - hs)
    add("psp", pspk)
    add("nearest", ndist)
    add("metal_cand", is_metal.float())
    add("n_coord28", n_co)
    add("coord_spread", spread)
    ia = pn.index("all")
    for ri, r in enumerate(cfg.nb_radii):
        add(f"n_all{int(r)}", S[:, ri, ia])
    r4 = list(cfg.nb_radii).index(4.0)
    for fi, f in enumerate(pn):
        if f in ("all", "wcn"):
            continue
        add(f"n4_{f}", S[:, r4, fi])
    for r in (8.0, 12.0):
        ri = list(cfg.nb_radii).index(r)
        den = torch.clamp(S[:, ri, ia], min=1.0)
        for fi, f in enumerate(pn):
            if f == "all":
                continue
            add(f"f{int(r)}_{f}", S[:, ri, fi] / den)
    natom_pt = torch.as_tensor(np.maximum(nat, 1), device=dev, dtype=torch.float32)[pb]
    add("prot_frac", S[:, list(cfg.nb_radii).index(10.0), ia] / natom_pt)
    feats = torch.stack(cols, 1)
    score = None
    if fm is not None and fm.point is not None:
        tf = _torch_forest(fm, "point", dev)
        idx = torch.as_tensor(fm.point_index(names), device=dev)
        floor = fm.model.meta.get("score_floor")
        if floor is None:
            score = tf.predict(feats[:, idx]).float()
        else:
            gv = (feats[:, names.index("bur10")] + feats[:, names.index("bur16")] + feats[:, names.index("bur24")]) / 3.0
            msk = (gv >= floor) | is_metal
            score = torch.zeros(len(feats), device=dev)
            if bool(msk.any()):
                score[msk] = tf.predict(feats[msk][:, idx]).float()
    torch.cuda.synchronize(dev)
    t4 = time.perf_counter()
    cnt = torch.bincount(pb, minlength=B).cpu().numpy()
    H = lambda t: t.cpu().numpy()
    pts_h, vol_h, feats_h, met_h = H(pts), H(vol), H(feats), H(is_metal)
    sc_h = H(score) if score is not None else None
    t5 = time.perf_counter()
    tm.update(gpu_points=(t1 - t0) * 1e3, gpu_prefilter=(t2 - t1) * 1e3, gpu_rays=(t3 - t2) * 1e3,
              gpu_rest=(t4 - t3) * 1e3, gpu_d2h=(t5 - t4) * 1e3, batch=B)
    out = []
    s0 = 0
    for it, n in zip(items, cnt):
        sl = slice(s0, s0 + n)
        s0 += n
        Xp = pts_h[sl]
        ps = PointSet(from_frame(Xp, it.R, it.c) if cfg.canonical else Xp.copy(), Xp, vol_h[sl], met_h[sl],
                      feats_h[sl], list(names), dict(tm))
        out.append((ps, None if sc_h is None else sc_h[sl]))
    return out


_FORESTS = {}


def _torch_forest(fm: FastModel, which: str, dev):
    key = (id(fm), which, str(dev))
    if key not in _FORESTS:
        _FORESTS[key] = getattr(fm, which).to_torch(dev)
    return _FORESTS[key]


def predict_batch(proteins, fm: FastModel, device="cuda:0", cfg: BurrowConfig = BurrowConfig(), ranker="learned",
                  weights=None, batch_size: int = 32):
    """Predict many structures: GPU featurization + scoring in batches, CPU formation/ranking."""
    out = []
    weights = weights or [None] * len(proteins)
    for s in range(0, len(proteins), batch_size):
        t0 = time.perf_counter()
        items = [prep(P, cfg, w) for P, w in zip(proteins[s:s + batch_size], weights[s:s + batch_size])]
        t1 = time.perf_counter()
        feats = featurize_batch(items, cfg, device, None if ranker == "geometry" else fm)
        for it, (ps, sc) in zip(items, feats):
            tm = dict(ps.timings)
            tm["host_prep_batch"] = (t1 - t0) * 1e3
            out.append(predict_from_points(it.P, ps, fm, ranker, score=sc, timings=tm))
    return out
