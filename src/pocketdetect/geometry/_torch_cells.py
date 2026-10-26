"""Batched GPU (PyTorch) cell list and radius-pair search.

All structures of a batch share one flat cell table: structure ``b`` owns cells
``base[b] .. base[b] + prod(dims[b])`` of its own grid (origin = its atom
bounding-box minimum).  Atoms never interact across structures because pairs
are only formed inside a structure's own cell range.

The pair search is fully vectorised: for each query we enumerate the
``(2k+1)^3`` neighbour cells (k = ceil(r / cell_size)), read their CSR ranges,
and expand the ranges with ``repeat_interleave``.  No padding to a maximum cell
occupancy is needed, so dense and sparse regions cost the same per atom.
"""
from __future__ import annotations

import math

import torch

from .torch_utils import batch_bounds


def _offsets(k: int, cell_size: float, r: float, device) -> torch.Tensor:
    """Integer cell offsets whose cell can contain a point within ``r``."""
    rng = torch.arange(-k, k + 1, device=device)
    off = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    # min distance between query cell and offset cell (query anywhere in its cell)
    gap = (off.abs() - 1).clamp(min=0).to(torch.float32) * cell_size
    keep = (gap * gap).sum(1) <= r * r + 1e-6
    return off[keep]


class TorchCellList:
    """Cell list over (possibly batched) atom coordinates on a torch device.

    Parameters
    ----------
    coords : (N, 3) float tensor
    batch : (N,) long tensor of structure ids in ``[0, n_batch)`` or None
    cell_size : float, edge length in Å
    n_batch : number of structures (inferred if None)
    """

    def __init__(self, coords: torch.Tensor, batch: torch.Tensor | None, cell_size: float,
                 n_batch: int | None = None):
        dev = coords.device
        n = coords.shape[0]
        if batch is None:
            batch = torch.zeros(n, dtype=torch.long, device=dev)
        if n_batch is None:
            n_batch = int(batch.max().item()) + 1 if n > 0 else 1
        self.coords = coords
        self.batch = batch
        self.cs = float(cell_size)
        self.n_batch = n_batch
        lo, hi = batch_bounds(coords, batch, n_batch)
        self.origin = lo
        self.dims = (torch.floor((hi - lo) / self.cs).long() + 1)
        ncell = self.dims.prod(1)
        self.base = torch.cumsum(ncell, 0) - ncell
        self.total = int(ncell.sum().item())
        cell = self._cell_index(coords, batch)
        lin = self._linear(cell, batch)
        self.order = torch.argsort(lin)
        counts = torch.bincount(lin, minlength=self.total)
        self.start = torch.cumsum(counts, 0) - counts
        self.counts = counts

    def _cell_index(self, x, b):
        return torch.floor((x - self.origin[b]) / self.cs).long()

    def _linear(self, cell, b):
        d = self.dims[b]
        return self.base[b] + (cell[:, 0] * d[:, 1] + cell[:, 1]) * d[:, 2] + cell[:, 2]

    def pairs(self, q: torch.Tensor, q_batch: torch.Tensor | None, r: float,
              max_candidates: int = 1 << 26):
        """All (query, atom) pairs with ``|q_i - a_j| <= r`` within one structure.

        Yields ``(qi, aj, d2)`` chunks (query indices are global indices into
        ``q``).  Chunked over queries to bound memory by ``max_candidates``.
        """
        dev = q.device
        if q_batch is None:
            q_batch = torch.zeros(q.shape[0], dtype=torch.long, device=dev)
        k = int(math.ceil(r / self.cs))
        off = _offsets(k, self.cs, r, dev)
        n_off = off.shape[0]
        mean_occ = max(1.0, self.coords.shape[0] / max(self.total, 1))
        # occupancy is far from uniform for proteins; assume 4x mean density
        chunk = max(256, int(max_candidates / (n_off * 4 * mean_occ + 1)))
        r2 = r * r
        for s in range(0, q.shape[0], chunk):
            qc = q[s:s + chunk]
            bc = q_batch[s:s + chunk]
            yield self._pairs_chunk(qc, bc, off, r2, s)

    def _pairs_chunk(self, qc, bc, off, r2, q_offset):
        cell = self._cell_index(qc, bc)                       # (P,3)
        nc = cell[:, None, :] + off[None, :, :]               # (P,O,3)
        d = self.dims[bc][:, None, :]
        valid = ((nc >= 0) & (nc < d)).all(-1)
        lin = self.base[bc][:, None] + (nc[..., 0] * d[..., 1] + nc[..., 1]) * d[..., 2] + nc[..., 2]
        lin = torch.where(valid, lin, torch.zeros_like(lin))
        cnt = torch.where(valid, self.counts[lin], torch.zeros_like(lin)).reshape(-1)
        st = self.start[lin].reshape(-1)
        n_off = off.shape[0]
        seg = torch.repeat_interleave(torch.arange(cnt.shape[0], device=qc.device), cnt)
        if seg.numel() == 0:
            e = torch.empty(0, dtype=torch.long, device=qc.device)
            return e, e, torch.empty(0, dtype=qc.dtype, device=qc.device)
        seg_start = torch.cumsum(cnt, 0) - cnt
        within = torch.arange(seg.shape[0], device=qc.device) - seg_start[seg]
        aj = self.order[st[seg] + within]
        qi = seg // n_off
        diff = qc[qi] - self.coords[aj]
        d2 = (diff * diff).sum(1)
        m = d2 <= r2
        return qi[m] + q_offset, aj[m], d2[m]


def radius_pairs_torch(q, a, r, q_batch=None, a_batch=None, n_batch=None, n_sub: int = 2):
    """Concatenated (qi, aj, d2) pairs with ``|q_i - a_j| <= r`` (same batch)."""
    cl = TorchCellList(a, a_batch, r / n_sub, n_batch)
    out = list(cl.pairs(q, q_batch, r))
    return tuple(torch.cat([o[i] for o in out]) for i in range(3))
