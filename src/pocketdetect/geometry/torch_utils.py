"""Small PyTorch helpers shared by the GPU implementations."""
from __future__ import annotations

import numpy as np
import torch


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_tensor(x, dtype, device=None) -> torch.Tensor:
    """Convert array-like to a tensor of ``dtype`` on ``device``.

    Tensors keep their device unless ``device`` is given; numpy arrays go to
    ``device`` or the default device (CUDA if available).
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device if device is not None else x.device)
    dev = device if device is not None else default_device()
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=dev)


def batch_bounds(coords: torch.Tensor, batch: torch.Tensor, n_batch: int):
    """Per-structure (n_batch, 3) bounding-box min and max."""
    big = torch.finfo(coords.dtype).max
    lo = torch.full((n_batch, 3), big, device=coords.device, dtype=coords.dtype)
    hi = torch.full((n_batch, 3), -big, device=coords.device, dtype=coords.dtype)
    bi = batch[:, None].expand(-1, 3)
    lo.scatter_reduce_(0, bi, coords, "amin")
    hi.scatter_reduce_(0, bi, coords, "amax")
    empty = lo[:, 0] > hi[:, 0]
    lo[empty] = 0
    hi[empty] = 0
    return lo, hi


def segment_offsets(batch: torch.Tensor, n_batch: int) -> torch.Tensor:
    """(n_batch+1,) CSR offsets of a sorted batch vector."""
    cnt = torch.bincount(batch, minlength=n_batch)
    return torch.cat([cnt.new_zeros(1), torch.cumsum(cnt, 0)])
