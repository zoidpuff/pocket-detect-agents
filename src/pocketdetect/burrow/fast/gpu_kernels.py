"""Triton kernels for the BURROW GPU featurizer: dual absorption grid + dual-grid ray march.

Semantics follow ``burrow.raycast`` (float32 arithmetic on the GPU, so results agree with the
CPU reference up to voxel-boundary / rounding flips; see tests/test_burrow_fast_gpu.py).
"""
from __future__ import annotations

import math

import triton
import triton.language as tl

LABEL_SHIFT = 32
LABEL_OFFSET = 16.0          # sc = d - r in [-r_max, 0]; (sc + 16) * 2^24 is positive and < 2^31
LABEL_SCALE = float(1 << 24)


@triton.jit
def _dual_stamp_kernel(frac_ptr, c0_ptr, lin0_ptr, dims_ptr, off_ptr, n_off, r_ptr, mu_ptr, bb_ptr, rank_ptr,
                       sp, grid_mu, grid_mub, grid_key, BO: tl.constexpr):
    a = tl.program_id(0)
    fx = tl.load(frac_ptr + a * 3 + 0)
    fy = tl.load(frac_ptr + a * 3 + 1)
    fz = tl.load(frac_ptr + a * 3 + 2)
    cx = tl.load(c0_ptr + a * 3 + 0)
    cy = tl.load(c0_ptr + a * 3 + 1)
    cz = tl.load(c0_ptr + a * 3 + 2)
    gx = tl.load(dims_ptr + a * 3 + 0)
    gy = tl.load(dims_ptr + a * 3 + 1)
    gz = tl.load(dims_ptr + a * 3 + 2)
    base = tl.load(lin0_ptr + a)
    r = tl.load(r_ptr + a)
    mu = tl.load(mu_ptr + a)
    bb = tl.load(bb_ptr + a)
    rank = tl.load(rank_ptr + a).to(tl.int64)
    r2 = r * r
    for t in range(0, n_off, BO):
        o = t + tl.arange(0, BO)
        om = o < n_off
        ox = tl.load(off_ptr + o * 3 + 0, mask=om, other=0)
        oy = tl.load(off_ptr + o * 3 + 1, mask=om, other=0)
        oz = tl.load(off_ptr + o * 3 + 2, mask=om, other=0)
        dx = (ox.to(tl.float32) + 0.5 - fx) * sp
        dy = (oy.to(tl.float32) + 0.5 - fy) * sp
        dz = (oz.to(tl.float32) + 0.5 - fz) * sp
        d2 = dx * dx + dy * dy + dz * dz
        vx = cx + ox
        vy = cy + oy
        vz = cz + oz
        ok = om & (vx >= 0) & (vx < gx) & (vy >= 0) & (vy < gy) & (vz >= 0) & (vz < gz) & (d2 <= r2) & (mu > 0.0)
        lin = base + ((vx * gy + vy) * gz + vz).to(tl.int64)
        tl.atomic_max(grid_mu + lin, mu, mask=ok)
        tl.atomic_max(grid_mub + lin, mu, mask=ok & (bb != 0))
        sc = tl.sqrt(d2) - r
        q = ((sc + 16.0) * 16777216.0).to(tl.int64)
        key = (q << 32) | rank
        tl.atomic_min(grid_key + lin, key, mask=ok)


@triton.jit
def _dual_march_kernel(rel_ptr, dims_ptr, base_ptr, dirs_ptr, mu_ptr, mub_ptr, lab_ptr, ck_ptr,
                       blk_ptr, blkb_ptr, depth_ptr, hit_ptr, P, K, step, lmax,
                       NL: tl.constexpr, NC: tl.constexpr, BP: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    p = pid * BP + tl.arange(0, BP)
    k = tl.arange(0, RB)
    pm = p < P
    km = k < K
    m2 = pm[:, None] & km[None, :]
    px = tl.load(rel_ptr + p * 3 + 0, mask=pm, other=0.0)[:, None]
    py = tl.load(rel_ptr + p * 3 + 1, mask=pm, other=0.0)[:, None]
    pz = tl.load(rel_ptr + p * 3 + 2, mask=pm, other=0.0)[:, None]
    nx = tl.load(dims_ptr + p * 3 + 0, mask=pm, other=0)[:, None]
    ny = tl.load(dims_ptr + p * 3 + 1, mask=pm, other=0)[:, None]
    nz = tl.load(dims_ptr + p * 3 + 2, mask=pm, other=0)[:, None]
    base = tl.load(base_ptr + p, mask=pm, other=0)[:, None]
    dx = tl.load(dirs_ptr + k * 3 + 0, mask=km, other=0.0)[None, :]
    dy = tl.load(dirs_ptr + k * 3 + 1, mask=km, other=0.0)[None, :]
    dz = tl.load(dirs_ptr + k * 3 + 2, mask=km, other=0.0)[None, :]
    tau = tl.zeros([BP, RB], dtype=tl.float32)
    taub = tl.zeros([BP, RB], dtype=tl.float32)
    hit_t = tl.zeros([BP, RB], dtype=tl.float32) + lmax
    hcls = tl.zeros([BP, RB], dtype=tl.int32) - 1
    prev = 0
    for li in tl.static_range(NL):
        e = tl.load(ck_ptr + li)
        for s in range(prev, e):
            t = (s + 0.5) * step
            fx = px + dx * t
            fy = py + dy * t
            fz = pz + dz * t
            inb = (fx >= 0.0) & (fy >= 0.0) & (fz >= 0.0) & (fx < nx) & (fy < ny) & (fz < nz)
            lin = base + ((fx.to(tl.int32) * ny + fy.to(tl.int32)) * nz + fz.to(tl.int32)).to(tl.int64)
            act = m2 & inb & ((tau <= 30.0) | (taub <= 30.0))
            v = tl.load(mu_ptr + lin, mask=act, other=0.0)
            vb = tl.load(mub_ptr + lin, mask=act, other=0.0)
            tau_new = tau + v * step
            first = (tau_new > 0.6931471805599453) & (tau <= 0.6931471805599453)
            lab = tl.load(lab_ptr + lin, mask=act & first, other=-1).to(tl.int32)
            hit_t = tl.where(first, t, hit_t)
            hcls = tl.where(first, lab, hcls)
            tau = tau_new
            taub = taub + vb * step
        prev = e
        b = tl.sum(tl.where(m2, 1.0 - tl.exp(-tau), 0.0), axis=1) / K
        bb = tl.sum(tl.where(m2, 1.0 - tl.exp(-taub), 0.0), axis=1) / K
        tl.store(blk_ptr + p * NL + li, b, mask=pm)
        tl.store(blkb_ptr + p * NL + li, bb, mask=pm)
    tl.store(depth_ptr + p, tl.sum(tl.where(m2, hit_t, 0.0), axis=1) / K, mask=pm)
    for c in tl.static_range(NC):
        h = tl.sum(tl.where(m2 & (hcls == c), 1.0, 0.0), axis=1) / K
        tl.store(hit_ptr + p * NC + c, h, mask=pm)


def dual_march(rel, dims, base, dirs_vox, mu, mub, lab, ck, step, lmax, n_cls=4, block_points=16):
    import torch
    P, K, NL = rel.shape[0], dirs_vox.shape[0], ck.shape[0]
    dev = rel.device
    blocked = torch.empty((P, NL), dtype=torch.float32, device=dev)
    blocked_bb = torch.empty((P, NL), dtype=torch.float32, device=dev)
    depth = torch.empty(P, dtype=torch.float32, device=dev)
    hit = torch.empty((P, n_cls), dtype=torch.float32, device=dev)
    if P == 0:
        return blocked, blocked_bb, depth, hit
    rb = 1 << max(0, math.ceil(math.log2(K)))
    bp = max(1, block_points * 64 // rb)
    with torch.cuda.device(dev):
        _dual_march_kernel[(triton.cdiv(P, bp),)](rel, dims, base, dirs_vox, mu, mub, lab, ck, blocked, blocked_bb,
                                                  depth, hit, P, K, float(step), float(lmax), NL=NL, NC=n_cls,
                                                  BP=bp, RB=rb, num_warps=4)
    return blocked, blocked_bb, depth, hit


def dual_stamp(frac, c0, lin0, dims, off, r, mu, bb, rank, spacing, grid_mu, grid_mub, grid_key, bo=256):
    import torch
    n = frac.shape[0]
    if n == 0:
        return
    with torch.cuda.device(frac.device):
        _dual_stamp_kernel[(n,)](frac, c0, lin0, dims, off, off.shape[0], r, mu, bb, rank, float(spacing),
                                 grid_mu, grid_mub, grid_key, BO=bo, num_warps=4)


# =========================================================================== exact (fp64) variants
# These replicate the CPU reference arithmetic (float64 distances / positions / accumulations,
# float32 casts at the same places) so that GPU features are bit-identical in practice.

@triton.jit
def _grid64_kernel(ax_ptr, org_ptr, c0_ptr, lin0_ptr, dims_ptr, off_ptr, n_off, rc2_ptr, gidx_ptr, h, dmax2,
                   key_ptr, clash_ptr, BO: tl.constexpr):
    a = tl.program_id(0)
    xa = tl.load(ax_ptr + a * 3 + 0)
    ya = tl.load(ax_ptr + a * 3 + 1)
    za = tl.load(ax_ptr + a * 3 + 2)
    o0 = tl.load(org_ptr + a * 3 + 0)
    o1 = tl.load(org_ptr + a * 3 + 1)
    o2 = tl.load(org_ptr + a * 3 + 2)
    cx = tl.load(c0_ptr + a * 3 + 0)
    cy = tl.load(c0_ptr + a * 3 + 1)
    cz = tl.load(c0_ptr + a * 3 + 2)
    gx = tl.load(dims_ptr + a * 3 + 0)
    gy = tl.load(dims_ptr + a * 3 + 1)
    gz = tl.load(dims_ptr + a * 3 + 2)
    base = tl.load(lin0_ptr + a)
    rc2 = tl.load(rc2_ptr + a)
    gi = tl.load(gidx_ptr + a).to(tl.int64)
    for t in range(0, n_off, BO):
        o = t + tl.arange(0, BO)
        om = o < n_off
        vx = cx + tl.load(off_ptr + o * 3 + 0, mask=om, other=0)
        vy = cy + tl.load(off_ptr + o * 3 + 1, mask=om, other=0)
        vz = cz + tl.load(off_ptr + o * 3 + 2, mask=om, other=0)
        x = o0 + (vx.to(tl.float64) + 0.5) * h
        y = o1 + (vy.to(tl.float64) + 0.5) * h
        z = o2 + (vz.to(tl.float64) + 0.5) * h
        dx2 = (x - xa) * (x - xa)
        dy2 = (y - ya) * (y - ya)
        dz = z - za
        d2 = (dx2 + dy2) + dz * dz
        ok = om & (vx >= 0) & (vx < gx) & (vy >= 0) & (vy < gy) & (vz >= 0) & (vz < gz) & (d2 <= dmax2)
        lin = base + ((vx * gy + vy) * gz + vz).to(tl.int64)
        f = d2.to(tl.float32)
        key = (f.to(tl.int32, bitcast=True).to(tl.int64) << 32) | gi
        tl.atomic_min(key_ptr + lin, key, mask=ok)
        tl.atomic_max(clash_ptr + lin, 1, mask=ok & (d2 < rc2))


@triton.jit
def _dual64_kernel(ax_ptr, org_ptr, c0_ptr, lin0_ptr, dims_ptr, off_ptr, n_off, r_ptr, mu_ptr, bb_ptr, rank_ptr, h,
                   grid_mu, grid_mub, grid_key, BO: tl.constexpr):
    a = tl.program_id(0)
    xa = tl.load(ax_ptr + a * 3 + 0)
    ya = tl.load(ax_ptr + a * 3 + 1)
    za = tl.load(ax_ptr + a * 3 + 2)
    o0 = tl.load(org_ptr + a * 3 + 0)
    o1 = tl.load(org_ptr + a * 3 + 1)
    o2 = tl.load(org_ptr + a * 3 + 2)
    cx = tl.load(c0_ptr + a * 3 + 0)
    cy = tl.load(c0_ptr + a * 3 + 1)
    cz = tl.load(c0_ptr + a * 3 + 2)
    gx = tl.load(dims_ptr + a * 3 + 0)
    gy = tl.load(dims_ptr + a * 3 + 1)
    gz = tl.load(dims_ptr + a * 3 + 2)
    base = tl.load(lin0_ptr + a)
    r = tl.load(r_ptr + a)
    mu = tl.load(mu_ptr + a)
    bb = tl.load(bb_ptr + a)
    rank = tl.load(rank_ptr + a).to(tl.int64)
    r2 = r * r
    for t in range(0, n_off, BO):
        o = t + tl.arange(0, BO)
        om = o < n_off
        vx = cx + tl.load(off_ptr + o * 3 + 0, mask=om, other=0)
        vy = cy + tl.load(off_ptr + o * 3 + 1, mask=om, other=0)
        vz = cz + tl.load(off_ptr + o * 3 + 2, mask=om, other=0)
        x = o0 + (vx.to(tl.float64) + 0.5) * h
        y = o1 + (vy.to(tl.float64) + 0.5) * h
        z = o2 + (vz.to(tl.float64) + 0.5) * h
        d2 = ((x - xa) * (x - xa) + (y - ya) * (y - ya)) + (z - za) * (z - za)
        ok = om & (vx >= 0) & (vx < gx) & (vy >= 0) & (vy < gy) & (vz >= 0) & (vz < gz) & (d2 <= r2) & (mu > 0.0)
        lin = base + ((vx * gy + vy) * gz + vz).to(tl.int64)
        tl.atomic_max(grid_mu + lin, mu, mask=ok)
        tl.atomic_max(grid_mub + lin, mu, mask=ok & (bb != 0))
        sc = (tl.sqrt(d2) - r).to(tl.float32)            # <= 0 inside the sphere
        u = (0.0 - sc) + 0.0                             # +0.0 for sc == 0
        ub = u.to(tl.int32, bitcast=True).to(tl.int64)
        key = ((2147483647 - ub) << 32) | rank           # min key = most negative sc, then lowest x-rank
        tl.atomic_min(grid_key + lin, key, mask=ok)


@triton.jit
def _psp64_kernel(pts_ptr, org_ptr, dims_ptr, base_ptr, dirs_ptr, mu_ptr, out_ptr, P, ND, n_steps, h, min_mu_ptr,
                  BP: tl.constexpr, DB: tl.constexpr):
    min_mu = tl.load(min_mu_ptr)              # fp64
    pid = tl.program_id(0)
    p = pid * BP + tl.arange(0, BP)
    k = tl.arange(0, DB)
    pm = p < P
    km = k < ND
    m2 = pm[:, None] & km[None, :]
    px = ((tl.load(pts_ptr + p * 3 + 0, mask=pm, other=0.0).to(tl.float64)
           - tl.load(org_ptr + p * 3 + 0, mask=pm, other=0.0)) / h)[:, None]
    py = ((tl.load(pts_ptr + p * 3 + 1, mask=pm, other=0.0).to(tl.float64)
           - tl.load(org_ptr + p * 3 + 1, mask=pm, other=0.0)) / h)[:, None]
    pz = ((tl.load(pts_ptr + p * 3 + 2, mask=pm, other=0.0).to(tl.float64)
           - tl.load(org_ptr + p * 3 + 2, mask=pm, other=0.0)) / h)[:, None]
    nx = tl.load(dims_ptr + p * 3 + 0, mask=pm, other=0)[:, None]
    ny = tl.load(dims_ptr + p * 3 + 1, mask=pm, other=0)[:, None]
    nz = tl.load(dims_ptr + p * 3 + 2, mask=pm, other=0)[:, None]
    base = tl.load(base_ptr + p, mask=pm, other=0)[:, None]
    dx = tl.load(dirs_ptr + k * 3 + 0, mask=km, other=0.0)[None, :]
    dy = tl.load(dirs_ptr + k * 3 + 1, mask=km, other=0.0)[None, :]
    dz = tl.load(dirs_ptr + k * 3 + 2, mask=km, other=0.0)[None, :]
    hp = tl.zeros([BP, DB], dtype=tl.int32)
    hm = tl.zeros([BP, DB], dtype=tl.int32)
    for s in range(1, n_steps + 1):
        sf = s * 1.0
        fx = px + dx * sf
        fy = py + dy * sf
        fz = pz + dz * sf
        inb = (fx >= 0.0) & (fy >= 0.0) & (fz >= 0.0) & (fx < nx) & (fy < ny) & (fz < nz)
        lin = base + ((fx.to(tl.int64) * ny + fy.to(tl.int64)) * nz + fz.to(tl.int64))
        v = tl.load(mu_ptr + lin, mask=m2 & inb & (hp == 0), other=0.0)
        hp = tl.where(v.to(tl.float64) > min_mu, 1, hp)
        fx = px - dx * sf
        fy = py - dy * sf
        fz = pz - dz * sf
        inb = (fx >= 0.0) & (fy >= 0.0) & (fz >= 0.0) & (fx < nx) & (fy < ny) & (fz < nz)
        lin = base + ((fx.to(tl.int64) * ny + fy.to(tl.int64)) * nz + fz.to(tl.int64))
        v = tl.load(mu_ptr + lin, mask=m2 & inb & (hm == 0), other=0.0)
        hm = tl.where(v.to(tl.float64) > min_mu, 1, hm)
    tl.store(out_ptr + p, tl.sum(hp * hm, axis=1), mask=pm)


@triton.jit
def _march64_kernel(pts_ptr, org_ptr, dims_ptr, base_ptr, dirs_ptr, mu_ptr, mub_ptr, lab_ptr, ck_ptr,
                    blk_ptr, blkb_ptr, depth_ptr, hit_ptr, P, K, step, lmax, inv_h_ptr, ln2_ptr,
                    NL: tl.constexpr, NC: tl.constexpr, BP: tl.constexpr, RB: tl.constexpr):
    ln2 = tl.load(ln2_ptr)                    # fp64 (Python float kernel args are passed as fp32!)
    inv_h = tl.load(inv_h_ptr)                # fp64 1/spacing (1/1.5 Å is not fp32-exact; --preset fast)
    pid = tl.program_id(0)
    p = pid * BP + tl.arange(0, BP)
    k = tl.arange(0, RB)
    pm = p < P
    km = k < K
    m2 = pm[:, None] & km[None, :]
    px = ((tl.load(pts_ptr + p * 3 + 0, mask=pm, other=0.0).to(tl.float64)
           - tl.load(org_ptr + p * 3 + 0, mask=pm, other=0.0)) * inv_h)[:, None]
    py = ((tl.load(pts_ptr + p * 3 + 1, mask=pm, other=0.0).to(tl.float64)
           - tl.load(org_ptr + p * 3 + 1, mask=pm, other=0.0)) * inv_h)[:, None]
    pz = ((tl.load(pts_ptr + p * 3 + 2, mask=pm, other=0.0).to(tl.float64)
           - tl.load(org_ptr + p * 3 + 2, mask=pm, other=0.0)) * inv_h)[:, None]
    nx = tl.load(dims_ptr + p * 3 + 0, mask=pm, other=0)[:, None]
    ny = tl.load(dims_ptr + p * 3 + 1, mask=pm, other=0)[:, None]
    nz = tl.load(dims_ptr + p * 3 + 2, mask=pm, other=0)[:, None]
    base = tl.load(base_ptr + p, mask=pm, other=0)[:, None]
    dx = (tl.load(dirs_ptr + k * 3 + 0, mask=km, other=0.0) * inv_h)[None, :]
    dy = (tl.load(dirs_ptr + k * 3 + 1, mask=km, other=0.0) * inv_h)[None, :]
    dz = (tl.load(dirs_ptr + k * 3 + 2, mask=km, other=0.0) * inv_h)[None, :]
    tau = tl.zeros([BP, RB], dtype=tl.float64)
    taub = tl.zeros([BP, RB], dtype=tl.float64)
    hit_t = tl.zeros([BP, RB], dtype=tl.float64) + lmax
    hcls = tl.zeros([BP, RB], dtype=tl.int32) - 1
    prev = 0
    for li in tl.static_range(NL):
        e = tl.load(ck_ptr + li)
        for s in range(prev, e):
            t = (s + 0.5) * step
            fx = px + dx * t
            fy = py + dy * t
            fz = pz + dz * t
            inb = (fx >= 0.0) & (fy >= 0.0) & (fz >= 0.0) & (fx < nx) & (fy < ny) & (fz < nz)
            lin = base + ((fx.to(tl.int64) * ny + fy.to(tl.int64)) * nz + fz.to(tl.int64))
            act = m2 & inb & ((tau <= 30.0) | (taub <= 30.0))
            v = tl.load(mu_ptr + lin, mask=act, other=0.0).to(tl.float64)
            vb = tl.load(mub_ptr + lin, mask=act, other=0.0).to(tl.float64)
            tau_new = tl.where(v > 0.0, tau + v * step, tau)
            first = (tau_new > ln2) & (tau <= ln2)   # fp64 arg: a literal would be fp32
            lab = tl.load(lab_ptr + lin, mask=act & first, other=-1).to(tl.int32)
            hit_t = tl.where(first, t, hit_t)
            hcls = tl.where(first, lab, hcls)
            tau = tau_new
            taub = tl.where(vb > 0.0, taub + vb * step, taub)
        prev = e
        b = tl.sum(tl.where(m2, 1.0 - tl.exp(-tau), 0.0), axis=1) / K
        bb = tl.sum(tl.where(m2, 1.0 - tl.exp(-taub), 0.0), axis=1) / K
        tl.store(blk_ptr + p * NL + li, b.to(tl.float32), mask=pm)
        tl.store(blkb_ptr + p * NL + li, bb.to(tl.float32), mask=pm)
    tl.store(depth_ptr + p, (tl.sum(tl.where(m2, hit_t, 0.0), axis=1) / K).to(tl.float32), mask=pm)
    for c in tl.static_range(NC):
        hc = tl.sum(tl.where(m2 & (hcls == c), 1.0, 0.0).to(tl.float64), axis=1) / K
        tl.store(hit_ptr + p * NC + c, hc.to(tl.float32), mask=pm)


@triton.jit
def _nf64_kernel(pts_ptr, atoms_ptr, props_ptr, cstart_ptr, blk_ptr, dims_ptr, base_ptr, th_ptr, out_ptr,
                 wcn_ptr, outw_ptr,
                 KR: tl.constexpr, BP: tl.constexpr, BA: tl.constexpr, NJ: tl.constexpr, FP: tl.constexpr):
    """Radius sums of integer-valued props (fp32 dot is exact) with float64 distances, plus float64
    sums of one continuous column (``wcn``; a float64 sum of these float32 values is exact in any
    order, so the result equals the reference after the float32 cast)."""
    blk = tl.program_id(0)
    pstart = tl.load(blk_ptr + blk * 6 + 0)
    pcount = tl.load(blk_ptr + blk * 6 + 1)
    cx = tl.load(blk_ptr + blk * 6 + 2)
    cy = tl.load(blk_ptr + blk * 6 + 3)
    cz = tl.load(blk_ptr + blk * 6 + 4)
    b = tl.load(blk_ptr + blk * 6 + 5)
    gx = tl.load(dims_ptr + b * 3 + 0)
    gy = tl.load(dims_ptr + b * 3 + 1)
    gz = tl.load(dims_ptr + b * 3 + 2)
    base = tl.load(base_ptr + b)
    pi = tl.arange(0, BP)
    pm = pi < pcount
    prow = (pstart + pi).to(tl.int64)
    px = tl.load(pts_ptr + prow * 3 + 0, mask=pm, other=0.0).to(tl.float64)
    py = tl.load(pts_ptr + prow * 3 + 1, mask=pm, other=0.0).to(tl.float64)
    pz = tl.load(pts_ptr + prow * 3 + 2, mask=pm, other=0.0).to(tl.float64)
    fi = tl.arange(0, FP)
    a0 = tl.zeros([BP, FP], dtype=tl.float32)
    a1 = tl.zeros([BP, FP], dtype=tl.float32)
    a2 = tl.zeros([BP, FP], dtype=tl.float32)
    a3 = tl.zeros([BP, FP], dtype=tl.float32)
    t0 = tl.load(th_ptr + 0)
    t1 = tl.load(th_ptr + 1)
    t2 = tl.load(th_ptr + 2)
    t3 = tl.load(th_ptr + 3)
    w0 = tl.zeros([BP], dtype=tl.float64)
    w1 = tl.zeros([BP], dtype=tl.float64)
    w2 = tl.zeros([BP], dtype=tl.float64)
    w3 = tl.zeros([BP], dtype=tl.float64)
    W: tl.constexpr = 2 * KR + 1
    for oi in range(W * W * W):
        nx = cx + oi // (W * W) - KR
        ny = cy + (oi // W) % W - KR
        nz = cz + oi % W - KR
        inside = (nx >= 0) & (nx < gx) & (ny >= 0) & (ny < gy) & (nz >= 0) & (nz < gz)
        cid = tl.where(inside, base + ((nx * gy + ny) * gz + nz).to(tl.int64), 0)
        s = tl.load(cstart_ptr + cid)
        e = tl.where(inside, tl.load(cstart_ptr + cid + 1), s)
        for t in range(s, e, BA):
            ai = t + tl.arange(0, BA)
            am = ai < e
            ax = tl.load(atoms_ptr + ai * 3 + 0, mask=am).to(tl.float64)
            ay = tl.load(atoms_ptr + ai * 3 + 1, mask=am).to(tl.float64)
            az = tl.load(atoms_ptr + ai * 3 + 2, mask=am).to(tl.float64)
            ddx = ax[None, :] - px[:, None]
            ddy = ay[None, :] - py[:, None]
            ddz = az[None, :] - pz[:, None]
            d2 = ddx * ddx + ddy * ddy + ddz * ddz
            d2 = tl.where((t + tl.arange(0, BA))[None, :] < e, d2, 1.0e300)
            ai2 = t + tl.arange(0, BA)[:, None]
            pr = tl.load(props_ptr + ai2 * FP + fi[None, :], mask=ai2 < e, other=0.0)
            wv = tl.load(wcn_ptr + ai, mask=am, other=0.0)[None, :]
            if NJ > 0:
                a0 += tl.dot(tl.where(d2 <= t0, 1.0, 0.0).to(tl.float32), pr, input_precision="ieee")
                w0 += tl.sum(tl.where(d2 <= t0, wv, 0.0), axis=1)
            if NJ > 1:
                a1 += tl.dot(tl.where(d2 <= t1, 1.0, 0.0).to(tl.float32), pr, input_precision="ieee")
                w1 += tl.sum(tl.where(d2 <= t1, wv, 0.0), axis=1)
            if NJ > 2:
                a2 += tl.dot(tl.where(d2 <= t2, 1.0, 0.0).to(tl.float32), pr, input_precision="ieee")
                w2 += tl.sum(tl.where(d2 <= t2, wv, 0.0), axis=1)
            if NJ > 3:
                a3 += tl.dot(tl.where(d2 <= t3, 1.0, 0.0).to(tl.float32), pr, input_precision="ieee")
                w3 += tl.sum(tl.where(d2 <= t3, wv, 0.0), axis=1)
    if NJ > 0:
        tl.store(out_ptr + (prow[:, None] * NJ + 0) * FP + fi[None, :], a0, mask=pm[:, None])
    if NJ > 1:
        tl.store(out_ptr + (prow[:, None] * NJ + 1) * FP + fi[None, :], a1, mask=pm[:, None])
    if NJ > 2:
        tl.store(out_ptr + (prow[:, None] * NJ + 2) * FP + fi[None, :], a2, mask=pm[:, None])
    if NJ > 3:
        tl.store(out_ptr + (prow[:, None] * NJ + 3) * FP + fi[None, :], a3, mask=pm[:, None])
    if NJ > 0:
        tl.store(outw_ptr + prow * NJ + 0, w0, mask=pm)
    if NJ > 1:
        tl.store(outw_ptr + prow * NJ + 1, w1, mask=pm)
    if NJ > 2:
        tl.store(outw_ptr + prow * NJ + 2, w2, mask=pm)
    if NJ > 3:
        tl.store(outw_ptr + prow * NJ + 3, w3, mask=pm)


def _launch(kernel, grid, dev, *args, **kw):
    import torch
    with torch.cuda.device(dev):
        kernel[grid](*args, **kw)
