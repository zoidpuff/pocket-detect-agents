"""Optional Triton kernels (fast path of the GPU backend).

Used automatically when ``triton`` imports and tensors are on CUDA; the pure
PyTorch implementations remain the reference/fallback and are tested to
agree with these kernels.
"""
from __future__ import annotations

import math

try:  # pragma: no cover - availability depends on the environment
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False


def triton_available(device) -> bool:
    return HAVE_TRITON and getattr(device, "type", str(device)) == "cuda"


if HAVE_TRITON:

    @triton.jit
    def _ray_march_kernel(rel_ptr, dims_ptr, base_ptr, dirs_ptr, mu_ptr, ck_ptr, out_ptr, depth_ptr,
                          P, K, step, lmax,
                          NL: tl.constexpr, BP: tl.constexpr, RB: tl.constexpr):
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
        hit_t = tl.zeros([BP, RB], dtype=tl.float32) + lmax
        prev = 0
        for li in tl.static_range(NL):
            e = tl.load(ck_ptr + li)
            for s in range(prev, e):
                t = (s + 0.5) * step
                fx = px + dx * t
                fy = py + dy * t
                fz = pz + dz * t
                inb = (fx >= 0.0) & (fy >= 0.0) & (fz >= 0.0) & (fx < nx) & (fy < ny) & (fz < nz)
                ix = fx.to(tl.int32)
                iy = fy.to(tl.int32)
                iz = fz.to(tl.int32)
                lin = base + ((ix * ny + iy) * nz + iz).to(tl.int64)
                v = tl.load(mu_ptr + lin, mask=m2 & inb & (tau <= 30.0), other=0.0)
                tau_new = tau + v * step
                hit_t = tl.where((tau_new > 0.6931471805599453) & (tau <= 0.6931471805599453), t, hit_t)
                tau = tau_new
            prev = e
            b = tl.sum(tl.where(m2, 1.0 - tl.exp(-tau), 0.0), axis=1) / K
            tl.store(out_ptr + p * NL + li, b, mask=pm)
        d = tl.sum(tl.where(m2, hit_t, 0.0), axis=1) / K
        tl.store(depth_ptr + p, d, mask=pm)


def ray_march_triton(rel, dims, base, dirs_vox, mu, ck, step, lmax, block_points: int = 16):
    """Launch the ray-march kernel.

    rel (P,3) f32 point position in voxel units relative to its grid origin;
    dims (P,3) i32 grid dims of the point's structure; base (P,) i64 offset of
    that grid in ``mu``; dirs_vox (K,3) f32 ray directions divided by spacing;
    ck (NL,) i32 cumulative sample counts per length.
    """
    import torch
    p_n = rel.shape[0]
    k_n = dirs_vox.shape[0]
    nl = ck.shape[0]
    out = torch.empty((p_n, nl), dtype=torch.float32, device=rel.device)
    depth = torch.empty(p_n, dtype=torch.float32, device=rel.device)
    if p_n == 0:
        return out, depth
    rb = 1 << max(0, math.ceil(math.log2(k_n)))
    bp = max(1, block_points * 64 // rb)
    grid = (triton.cdiv(p_n, bp),)
    with torch.cuda.device(rel.device):      # Triton launches on the *current* device
        _ray_march_kernel[grid](rel, dims, base, dirs_vox, mu, ck, out, depth, p_n, k_n, float(step),
                                float(lmax), NL=nl, BP=bp, RB=rb, num_warps=4)
    return out, depth


if HAVE_TRITON:

    @triton.jit
    def _wtile(d2, th_ptr, co_ptr, J: tl.constexpr, NR: tl.constexpr):
        th = tl.load(th_ptr + J)
        if J < NR:
            return tl.where(d2 <= th, 1.0, 0.0)
        else:
            co = tl.load(co_ptr + J)
            return tl.where(d2 <= th, tl.exp(-d2 * co), 0.0)

    @triton.jit
    def _nf_kernel(pts_ptr, atoms_ptr, props_ptr, cstart_ptr, blk_ptr, dims_ptr, base_ptr,
                   th_ptr, co_ptr, out_ptr,
                   KR: tl.constexpr, BP: tl.constexpr, BA: tl.constexpr, NJ: tl.constexpr,
                   NR: tl.constexpr, FP: tl.constexpr):
        # NJ <= 8 kernels: the first NR are hard radius cut-offs, the rest truncated Gaussians
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
        px = tl.load(pts_ptr + prow * 3 + 0, mask=pm, other=0.0)
        py = tl.load(pts_ptr + prow * 3 + 1, mask=pm, other=0.0)
        pz = tl.load(pts_ptr + prow * 3 + 2, mask=pm, other=0.0)
        fi = tl.arange(0, FP)
        a0 = tl.zeros([BP, FP], dtype=tl.float32)
        a1 = tl.zeros([BP, FP], dtype=tl.float32)
        a2 = tl.zeros([BP, FP], dtype=tl.float32)
        a3 = tl.zeros([BP, FP], dtype=tl.float32)
        a4 = tl.zeros([BP, FP], dtype=tl.float32)
        a5 = tl.zeros([BP, FP], dtype=tl.float32)
        a6 = tl.zeros([BP, FP], dtype=tl.float32)
        a7 = tl.zeros([BP, FP], dtype=tl.float32)
        W: tl.constexpr = 2 * KR + 1
        for oi in range(W * W * W):
            ox = oi // (W * W) - KR
            oy = (oi // W) % W - KR
            oz = oi % W - KR
            nx = cx + ox
            ny = cy + oy
            nz = cz + oz
            inside = (nx >= 0) & (nx < gx) & (ny >= 0) & (ny < gy) & (nz >= 0) & (nz < gz)
            cid = base + ((nx * gy + ny) * gz + nz).to(tl.int64)
            cid = tl.where(inside, cid, 0)
            s = tl.load(cstart_ptr + cid)
            e = tl.load(cstart_ptr + cid + 1)
            e = tl.where(inside, e, s)
            for t in range(s, e, BA):
                ai = t + tl.arange(0, BA)
                am = ai < e
                ax = tl.load(atoms_ptr + ai * 3 + 0, mask=am)
                ay = tl.load(atoms_ptr + ai * 3 + 1, mask=am)
                az = tl.load(atoms_ptr + ai * 3 + 2, mask=am)
                ddx = px[:, None] - ax[None, :]
                ddy = py[:, None] - ay[None, :]
                ddz = pz[:, None] - az[None, :]
                d2 = ddx * ddx + ddy * ddy + ddz * ddz                     # (BP,BA)
                # (separately built 2-D mask: works around a Triton 3.2 layout bug)
                d2 = tl.where((t + tl.arange(0, BA))[None, :] < e, d2, 1.0e18)
                ai2 = t + tl.arange(0, BA)[:, None]
                pr = tl.load(props_ptr + ai2 * FP + fi[None, :], mask=ai2 < e, other=0.0)
                if NJ > 0:
                    a0 += tl.dot(_wtile(d2, th_ptr, co_ptr, 0, NR), pr, input_precision="ieee")
                if NJ > 1:
                    a1 += tl.dot(_wtile(d2, th_ptr, co_ptr, 1, NR), pr, input_precision="ieee")
                if NJ > 2:
                    a2 += tl.dot(_wtile(d2, th_ptr, co_ptr, 2, NR), pr, input_precision="ieee")
                if NJ > 3:
                    a3 += tl.dot(_wtile(d2, th_ptr, co_ptr, 3, NR), pr, input_precision="ieee")
                if NJ > 4:
                    a4 += tl.dot(_wtile(d2, th_ptr, co_ptr, 4, NR), pr, input_precision="ieee")
                if NJ > 5:
                    a5 += tl.dot(_wtile(d2, th_ptr, co_ptr, 5, NR), pr, input_precision="ieee")
                if NJ > 6:
                    a6 += tl.dot(_wtile(d2, th_ptr, co_ptr, 6, NR), pr, input_precision="ieee")
                if NJ > 7:
                    a7 += tl.dot(_wtile(d2, th_ptr, co_ptr, 7, NR), pr, input_precision="ieee")
        if NJ > 0:
            tl.store(out_ptr + (prow[:, None] * NJ + 0) * FP + fi[None, :], a0, mask=pm[:, None])
        if NJ > 1:
            tl.store(out_ptr + (prow[:, None] * NJ + 1) * FP + fi[None, :], a1, mask=pm[:, None])
        if NJ > 2:
            tl.store(out_ptr + (prow[:, None] * NJ + 2) * FP + fi[None, :], a2, mask=pm[:, None])
        if NJ > 3:
            tl.store(out_ptr + (prow[:, None] * NJ + 3) * FP + fi[None, :], a3, mask=pm[:, None])
        if NJ > 4:
            tl.store(out_ptr + (prow[:, None] * NJ + 4) * FP + fi[None, :], a4, mask=pm[:, None])
        if NJ > 5:
            tl.store(out_ptr + (prow[:, None] * NJ + 5) * FP + fi[None, :], a5, mask=pm[:, None])
        if NJ > 6:
            tl.store(out_ptr + (prow[:, None] * NJ + 6) * FP + fi[None, :], a6, mask=pm[:, None])
        if NJ > 7:
            tl.store(out_ptr + (prow[:, None] * NJ + 7) * FP + fi[None, :], a7, mask=pm[:, None])


def neighbor_sums_triton(points, point_batch, coords, atom_batch, props, thresholds2, coefs, rcut, n_batch,
                         n_hard: int, bp: int = 16, ba: int = 16):
    """Fused per-point accumulation ``out[p, j, :] = sum_a [d2 <= th_j] exp(-d2 coef_j) props[a]``.

    Points are grouped by cell (cell size rcut/2, same per-structure frame as
    the atoms); each program owns <= ``bp`` points of one cell and sweeps the
    5^3 neighbouring cells in ``ba``-atom tiles, accumulating all kernels at
    once with one fp32 ``tl.dot`` per kernel and tile.  Returns (P, n_kernels, F).
    """
    import torch
    from .torch_utils import batch_bounds
    dev = points.device
    p_n, f_n = points.shape[0], props.shape[1]
    nj = len(thresholds2)
    if nj > 8:
        raise ValueError("at most 8 radii + sigmas supported by the Triton kernel")
    NB = 8
    FP = max(16, 1 << math.ceil(math.log2(max(f_n, 1))))
    cs = rcut / 2.0
    # per-structure frame covering atoms and points
    lo_a, hi_a = batch_bounds(coords, atom_batch, n_batch)
    lo_p, hi_p = batch_bounds(points, point_batch, n_batch)
    has_p = torch.bincount(point_batch, minlength=n_batch) > 0
    lo = torch.where(has_p[:, None], torch.minimum(lo_a, lo_p), lo_a)
    hi = torch.where(has_p[:, None], torch.maximum(hi_a, hi_p), hi_a)
    dims = (torch.floor((hi - lo) / cs).long() + 1)
    ncell = dims.prod(1)
    base = torch.cumsum(ncell, 0) - ncell
    total = int(ncell.sum().item())

    def lin_of(x, b):
        c = torch.floor((x - lo[b]) / cs).long()
        c = torch.minimum(torch.maximum(c, torch.zeros_like(c)), dims[b] - 1)
        return base[b] + (c[:, 0] * dims[b, 1] + c[:, 1]) * dims[b, 2] + c[:, 2]

    alin = lin_of(coords, atom_batch)
    aord = torch.argsort(alin)
    acnt = torch.bincount(alin, minlength=total)
    cstart = torch.cat([acnt.new_zeros(1), torch.cumsum(acnt, 0)])
    atoms_s = coords[aord].contiguous()
    props_s = torch.zeros((coords.shape[0], FP), dtype=torch.float32, device=dev)
    props_s[:, :f_n] = props[aord]

    plin = lin_of(points, point_batch)
    pord = torch.argsort(plin)
    pts_s = points[pord].contiguous()
    pcnt = torch.bincount(plin, minlength=total)
    pstart = torch.cumsum(pcnt, 0) - pcnt
    cells = torch.nonzero(pcnt, as_tuple=True)[0]
    nblk = (pcnt[cells] + bp - 1) // bp
    bcell = torch.repeat_interleave(cells, nblk)
    first = torch.cumsum(nblk, 0) - nblk
    j = torch.arange(bcell.shape[0], device=dev) - torch.repeat_interleave(first, nblk)
    bstart = pstart[bcell] + j * bp
    bcount = torch.clamp(pcnt[bcell] - j * bp, max=bp)
    bb = torch.searchsorted(base, bcell, right=True) - 1
    loc = bcell - base[bb]
    dz = dims[bb, 2]
    dy = dims[bb, 1]
    blk = torch.stack([bstart, bcount, loc // (dy * dz), (loc // dz) % dy, loc % dz, bb], 1).to(torch.int32).contiguous()

    th = torch.full((NB,), -1.0, dtype=torch.float32, device=dev)
    co = torch.zeros(NB, dtype=torch.float32, device=dev)
    th[:nj] = torch.as_tensor(thresholds2, dtype=torch.float32, device=dev)
    co[:nj] = torch.as_tensor(coefs, dtype=torch.float32, device=dev)
    out = torch.empty((p_n, nj, FP), dtype=torch.float32, device=dev)
    if blk.shape[0] > 0:
        with torch.cuda.device(dev):
            _nf_kernel[(blk.shape[0],)](pts_s, atoms_s, props_s, cstart, blk, dims.to(torch.int32).contiguous(),
                                        base.contiguous(), th, co, out, KR=2, BP=bp, BA=ba, NJ=nj, NR=n_hard,
                                        FP=FP, num_warps=4, num_stages=1)  # stages>1: wrong a0 (Triton 3.2 pipeliner bug)
    res = torch.empty_like(out)
    res[pord] = out
    return res[:, :nj, :f_n]


if HAVE_TRITON:

    @triton.jit
    def _sas_kernel(pts_ptr, atoms_ptr, rp_ptr, idx_ptr, cell_ptr, cstart_ptr, dims_ptr, base_ptr, batch_ptr,
                    keep_ptr, K, BK: tl.constexpr, BA: tl.constexpr):
        i = tl.program_id(0)                       # atom (original index)
        b = tl.load(batch_ptr + i)
        cx = tl.load(cell_ptr + i * 3 + 0)
        cy = tl.load(cell_ptr + i * 3 + 1)
        cz = tl.load(cell_ptr + i * 3 + 2)
        gx = tl.load(dims_ptr + b * 3 + 0)
        gy = tl.load(dims_ptr + b * 3 + 1)
        gz = tl.load(dims_ptr + b * 3 + 2)
        base = tl.load(base_ptr + b)
        k = tl.arange(0, BK)
        km = k < K
        row = i.to(tl.int64) * K + k
        px = tl.load(pts_ptr + row * 3 + 0, mask=km, other=0.0)
        py = tl.load(pts_ptr + row * 3 + 1, mask=km, other=0.0)
        pz = tl.load(pts_ptr + row * 3 + 2, mask=km, other=0.0)
        buried = tl.zeros([BK], dtype=tl.int32)
        for oi in range(27):
            nx = cx + oi // 9 - 1
            ny = cy + (oi // 3) % 3 - 1
            nz = cz + oi % 3 - 1
            inside = (nx >= 0) & (nx < gx) & (ny >= 0) & (ny < gy) & (nz >= 0) & (nz < gz)
            cid = tl.where(inside, base + ((nx * gy + ny) * gz + nz).to(tl.int64), 0)
            s = tl.load(cstart_ptr + cid)
            e = tl.where(inside, tl.load(cstart_ptr + cid + 1), s)
            for t in range(s, e, BA):
                aj = t + tl.arange(0, BA)
                am = aj < e
                ax = tl.load(atoms_ptr + aj * 3 + 0, mask=am)
                ay = tl.load(atoms_ptr + aj * 3 + 1, mask=am)
                az = tl.load(atoms_ptr + aj * 3 + 2, mask=am)
                rj = tl.load(rp_ptr + aj, mask=am)
                oj = tl.load(idx_ptr + aj, mask=am)
                dx = ax[None, :] - px[:, None]
                dy = ay[None, :] - py[:, None]
                dz = az[None, :] - pz[:, None]
                d2 = dx * dx + dy * dy + dz * dz
                ok = ((t + tl.arange(0, BA))[None, :] < e) & (oj[None, :] != i)
                hit = ok & (d2 < (rj * rj)[None, :])
                buried += tl.sum(hit.to(tl.int32), axis=1)
        tl.store(keep_ptr + row, (buried == 0).to(tl.int8), mask=km)


def sas_keep_triton(coords, rp, pts, batch, n_batch, ba: int = 16):
    """(N, K) keep mask of SAS points: not strictly inside any other expanded sphere."""
    import torch
    from ._torch_cells import TorchCellList
    n, k = pts.shape[0], pts.shape[1]
    keep = torch.ones((n, k), dtype=torch.int8, device=coords.device)
    if n == 0:
        return keep.bool()
    cs = 2.0 * float(rp.max().item())
    cl = TorchCellList(coords, batch, cs, n_batch)
    cell = cl._cell_index(coords, batch).to(torch.int32).contiguous()
    order = cl.order
    cstart = torch.cat([cl.start, cl.start.new_tensor([coords.shape[0]])]).contiguous()
    bk = max(16, 1 << math.ceil(math.log2(k)))
    with torch.cuda.device(coords.device):
        _sas_kernel[(n,)](pts.contiguous(), coords[order].contiguous(), rp[order].contiguous(),
                          order.to(torch.int32).contiguous(), cell, cstart,
                          cl.dims.to(torch.int32).contiguous(), cl.base.contiguous(),
                          batch.to(torch.int32).contiguous(), keep, k, BK=bk, BA=ba, num_warps=2, num_stages=1)
    return keep.bool()


if HAVE_TRITON:

    @triton.jit
    def _stamp_kernel(frac_ptr, c0_ptr, lin0_ptr, dims_ptr, off_ptr, n_off, r2_ptr, rc2_ptr, val_ptr,
                      key_ptr, clash_ptr, mu_ptr, MODE: tl.constexpr, BO: tl.constexpr):
        """Per-atom voxel stamping.  MODE 0: nearest-atom key (atomic min) + clash flag;
        MODE 1: absorption grid (atomic max of the atom's mu)."""
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
        r2 = tl.load(r2_ptr + a)
        for t in range(0, n_off, BO):
            o = t + tl.arange(0, BO)
            om = o < n_off
            ox = tl.load(off_ptr + o * 3 + 0, mask=om, other=0)
            oy = tl.load(off_ptr + o * 3 + 1, mask=om, other=0)
            oz = tl.load(off_ptr + o * 3 + 2, mask=om, other=0)
            dx = (ox.to(tl.float32) + 0.5 - fx)
            dy = (oy.to(tl.float32) + 0.5 - fy)
            dz = (oz.to(tl.float32) + 0.5 - fz)
            vx = cx + ox
            vy = cy + oy
            vz = cz + oz
            ok = om & (vx >= 0) & (vx < gx) & (vy >= 0) & (vy < gy) & (vz >= 0) & (vz < gz)
            lin = base + ((vx * gy + vy) * gz + vz).to(tl.int64)
            if MODE == 0:
                sp = tl.load(val_ptr)                          # spacing
                d2 = (dx * sp) * (dx * sp) + (dy * sp) * (dy * sp) + (dz * sp) * (dz * sp)
                ok = ok & (d2 <= r2)
                key = (tl.floor(d2 * 1048576.0).to(tl.int64) << 31) | a.to(tl.int64)
                tl.atomic_min(key_ptr + lin, key, mask=ok)
                rc2 = tl.load(rc2_ptr + a)
                tl.atomic_max(clash_ptr + lin, 1, mask=ok & (d2 < rc2))
            else:
                sp = tl.load(val_ptr)
                d2 = (dx * sp) * (dx * sp) + (dy * sp) * (dy * sp) + (dz * sp) * (dz * sp)
                mu = tl.load(mu_ptr + a)
                tl.atomic_max(key_ptr + lin, mu, mask=ok & (d2 <= r2))


def stamp_triton(frac, c0, lin0, dims, off, r2, rc2, spacing, key=None, clash=None, mu=None, grid=None,
                 mode: int = 0, bo: int = 256):
    """Launch :func:`_stamp_kernel` (one program per atom; see callers for argument layout)."""
    import torch
    n = frac.shape[0]
    if n == 0:
        return
    sp = torch.tensor([spacing], dtype=torch.float32, device=frac.device)
    with torch.cuda.device(frac.device):
        if mode == 0:
            _stamp_kernel[(n,)](frac, c0, lin0, dims, off, off.shape[0], r2, rc2, sp, key, clash, r2,
                                MODE=0, BO=bo, num_warps=4)
        else:
            _stamp_kernel[(n,)](frac, c0, lin0, dims, off, off.shape[0], r2, r2, sp, grid, grid, mu,
                                MODE=1, BO=bo, num_warps=4)


if HAVE_TRITON:

    @triton.jit
    def _psp_kernel(rel_ptr, dims_ptr, base_ptr, dirs_ptr, mu_ptr, out_ptr, P, ND, n_steps, min_mu,
                    BP: tl.constexpr, DB: tl.constexpr):
        pid = tl.program_id(0)
        p = pid * BP + tl.arange(0, BP)
        k = tl.arange(0, DB)
        pm = p < P
        km = k < ND
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
        hp = tl.zeros([BP, DB], dtype=tl.int32)
        hm = tl.zeros([BP, DB], dtype=tl.int32)
        for s in range(1, n_steps + 1):
            t = s * 1.0
            fx = px + dx * t
            fy = py + dy * t
            fz = pz + dz * t
            inb = (fx >= 0.0) & (fy >= 0.0) & (fz >= 0.0) & (fx < nx) & (fy < ny) & (fz < nz)
            lin = base + ((fx.to(tl.int32) * ny + fy.to(tl.int32)) * nz + fz.to(tl.int32)).to(tl.int64)
            v = tl.load(mu_ptr + lin, mask=m2 & inb & (hp == 0), other=0.0)
            hp = tl.where(v > min_mu, 1, hp)
            fx = px - dx * t
            fy = py - dy * t
            fz = pz - dz * t
            inb = (fx >= 0.0) & (fy >= 0.0) & (fz >= 0.0) & (fx < nx) & (fy < ny) & (fz < nz)
            lin = base + ((fx.to(tl.int32) * ny + fy.to(tl.int32)) * nz + fz.to(tl.int32)).to(tl.int64)
            v = tl.load(mu_ptr + lin, mask=m2 & inb & (hm == 0), other=0.0)
            hm = tl.where(v > min_mu, 1, hm)
        cnt = tl.sum(hp * hm, axis=1)
        tl.store(out_ptr + p, cnt, mask=pm)


def psp_triton(rel, dims, base, dirs_step_vox, mu, n_steps, min_mu, bp: int = 64):
    """LIGSITE-style buriedness counts (number of directions enclosed on both sides)."""
    import torch
    p_n = rel.shape[0]
    out = torch.zeros(p_n, dtype=torch.int32, device=rel.device)
    if p_n == 0:
        return out
    with torch.cuda.device(rel.device):
        _psp_kernel[(triton.cdiv(p_n, bp),)](rel, dims, base, dirs_step_vox, mu, out, p_n,
                                             dirs_step_vox.shape[0], int(n_steps), float(min_mu),
                                             BP=bp, DB=16, num_warps=4)
    return out
