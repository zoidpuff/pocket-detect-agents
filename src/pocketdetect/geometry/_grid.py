"""Shared helpers for absolute-aligned voxel grids (CPU frame + GPU stamping)."""
from __future__ import annotations

import math

import numpy as np


def grid_frame(lo, hi, spacing, pad):
    """Origin (aligned to multiples of ``spacing``) and dims of a grid covering [lo-pad, hi+pad].

    Voxel ``i`` covers ``[origin + i h, origin + (i+1) h)``; its centre is at
    ``origin + (i + 1/2) h``.  Computed in float64 so CPU and GPU agree exactly.
    """
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    origin = np.floor((lo - pad) / spacing) * spacing - spacing
    dims = np.ceil((hi + pad - origin) / spacing).astype(np.int64) + 1
    return origin, dims


def stamp_offsets_torch(rmax: float, spacing: float, device):
    """Integer voxel offsets (O,3) that can hold a voxel centre within ``rmax`` of an
    atom lying anywhere inside the offset-0 voxel."""
    import torch
    m = int(math.ceil(rmax / spacing)) + 1
    rng = torch.arange(-m, m + 1, device=device)
    off = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    gap = (off.abs().double() - 0.5).clamp(min=0) * spacing
    return off[(gap * gap).sum(1) <= rmax * rmax + 1e-9].to(torch.int32).contiguous()


class TorchGridFrames:
    """Per-structure absolute-aligned grids of a batch, concatenated in one flat index space."""

    def __init__(self, coords, batch, n_batch: int, spacing: float, pad: float):
        import torch
        from .torch_utils import batch_bounds
        dev = coords.device
        lo, hi = batch_bounds(coords, batch, n_batch)
        lo64, hi64 = lo.double().cpu().numpy(), hi.double().cpu().numpy()
        frames = [grid_frame(lo64[b], hi64[b], spacing, pad) for b in range(n_batch)]
        self.spacing = float(spacing)
        self.origin = torch.tensor(np.stack([f[0] for f in frames]), dtype=torch.float64, device=dev)
        self.origin32 = self.origin.float()
        self.dims = torch.tensor(np.stack([f[1] for f in frames]), dtype=torch.long, device=dev)
        nvox = self.dims.prod(1)
        self.base = torch.cumsum(nvox, 0) - nvox
        self.total = int(nvox.sum().item())

    def atom_voxels(self, coords, batch):
        """float32 fractional position inside, and int32 index of, each atom's voxel."""
        import torch
        rel = (coords.double() - self.origin[batch]) / self.spacing
        c0 = torch.floor(rel)
        return (rel - c0).float().contiguous(), c0.to(torch.int32).contiguous()

    def decode(self, flat):
        """Flat voxel index -> (batch, voxel centre float64)."""
        import torch
        vb = torch.searchsorted(self.base, flat, right=True) - 1
        loc = flat - self.base[vb]
        dv = self.dims[vb]
        iz = loc % dv[:, 2]
        iy = (loc // dv[:, 2]) % dv[:, 1]
        ix = loc // (dv[:, 2] * dv[:, 1])
        ijk = torch.stack([ix, iy, iz], 1).double()
        return vb, self.origin[vb] + (ijk + 0.5) * self.spacing
