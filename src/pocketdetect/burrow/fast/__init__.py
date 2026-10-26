"""BURROW fast backend (CPU Numba + batched GPU), output-identical to the reference implementation.

Modules: ``trees`` (LightGBM forests without LightGBM), ``formation`` (candidate pockets, residues,
NMS), ``pocketfeat`` (pocket descriptors), ``features`` (CPU featurizer), ``gpu`` / ``gpu_kernels``
(batched GPU featurizer), ``pipeline`` (per-structure prediction), ``run`` (CLI / multi-GPU driver),
``guard`` (falls back to the reference when its code changes).
"""
