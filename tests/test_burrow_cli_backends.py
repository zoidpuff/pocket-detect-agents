"""Unified CLI: identical outputs from the reference implementation, the fast CPU backend and the fast
GPU backend (``burrow predict --backend reference|fast --device cpu|cuda:N``) on 10 structures."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from pocketdetect.burrow.cli import main

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [ROOT / "results/dev/biolip/inputs", ROOT / "data/raw/p2rank_datasets/current/chen11"]


def _inputs(n=10):
    for d in CANDIDATES:
        if d.exists():
            fs = sorted(p for p in d.iterdir() if p.suffix in (".cif", ".pdb"))[:n]
            if len(fs) == n:
                return fs
    pytest.skip("no structure set available")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _run(tmp_path, name, files, *extra):
    lst = tmp_path / "in.list"
    lst.write_text("\n".join(map(str, files)) + "\n")
    out = tmp_path / name
    assert main(["predict", f"@{lst}", "--out", str(out), "--no-points", "--workers", "2", *extra]) == 0
    return out


def _same(a, b, files):
    from pocketdetect.burrow.cli import _stem
    for f in files:
        s = _stem(str(f))
        assert (a / f"{s}.csv").read_text() == (b / f"{s}.csv").read_text(), s


@pytest.mark.slow
@pytest.mark.parametrize("extra", [[], ["--min-score", "none"], ["--af-mode"], ["--ranker", "small"]])
def test_reference_equals_fast_cpu(tmp_path, extra):
    files = _inputs()
    ref = _run(tmp_path, "ref", files, "--backend", "reference", *extra)
    fast = _run(tmp_path, "fast", files, "--backend", "fast", "--device", "cpu", *extra)
    _same(ref, fast, files)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not _cuda(), reason="CUDA not available")
def test_reference_equals_fast_gpu(tmp_path):
    files = _inputs()
    dev = os.environ.get("BURROW_TEST_GPU", "cuda:0")
    ref = _run(tmp_path, "ref", files, "--backend", "reference")
    gpu = _run(tmp_path, "gpu", files, "--backend", "fast", "--device", dev)
    _same(ref, gpu, files)


def test_auto_falls_back_to_reference(tmp_path, monkeypatch):
    """If the fast backend raises, `--backend auto` still produces the reference output."""
    files = _inputs(2)
    import pocketdetect.burrow.cli as cli

    def boom(*a, **k):
        raise RuntimeError("simulated fast-backend failure")

    monkeypatch.setattr(cli, "_run_fast", boom)
    auto = _run(tmp_path, "auto", files)
    ref = _run(tmp_path, "ref2", files, "--backend", "reference")
    _same(auto, ref, files)
