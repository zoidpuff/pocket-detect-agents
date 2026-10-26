#!/usr/bin/env python
"""Compute and cache BURROW point sets/features (+ ligand distances for labels) for a manifest.

  python scripts/burrow_featurize.py results/dev/manifests/all.csv --jobs 32 [--rules p2rank]

Writes results/dev/cache/feats/<cfg-hash>/<structure_id>.npz (git-ignored) and timing.csv.
One process per structure, NUMBA_NUM_THREADS=1 (so timings are 1-core timings).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # shared per-user thread limit; LightGBM gets num_threads explicitly

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketdetect.burrow.devdata import CACHE, ligand_distance, load_gts, read_manifest  # noqa: E402
from pocketdetect.burrow.features import BurrowConfig, point_features  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402


def cfg_hash(cfg: BurrowConfig) -> str:
    import hashlib
    src = b"".join((ROOT / "src/pocketdetect/burrow" / f).read_bytes() for f in ("features.py", "raycast.py", "prep.py"))
    return hashlib.sha256(repr(cfg).encode() + src).hexdigest()[:10]


_G = {}


def _job(args):
    sid, path, outdir = args
    out = Path(outdir) / f"{sid}.npz"
    gt = _G["gts"].get(sid)
    try:
        t0 = time.perf_counter()
        P = load_protein(path, structure_id=sid)
        t1 = time.perf_counter()
        ps = point_features(P, _G["cfg"])
        t2 = time.perf_counter()
        d_all, d_ion = ligand_distance(ps.points, gt) if gt is not None else (np.full(len(ps.points), np.inf),) * 2
        np.savez(out, points=ps.points, X=ps.X, vol=ps.vol, is_metal=ps.is_metal, feats=ps.feats,
                 names=np.array(ps.names), d_all=d_all.astype(np.float32), d_ion=d_ion.astype(np.float32))
        return dict(structure_id=sid, status="ok", n_atoms=len(P.coords), n_points=len(ps.points),
                    load_ms=(t1 - t0) * 1e3, feat_ms=(t2 - t1) * 1e3, **{f"t_{k}": v for k, v in ps.timings.items()})
    except Exception as e:
        return dict(structure_id=sid, status=f"error: {type(e).__name__}: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--rules", default="p2rank")
    a = ap.parse_args(argv)
    cfg = BurrowConfig()
    h = cfg_hash(cfg)
    outdir = CACHE / "feats" / h
    outdir.mkdir(parents=True, exist_ok=True)
    df = read_manifest(a.manifest)
    gts = load_gts(a.manifest, a.rules, jobs=a.jobs)
    _G["gts"], _G["cfg"] = gts, cfg
    tasks = [(s, p, str(outdir)) for s, p in zip(df.structure_id, df.structure_path)
             if not (outdir / f"{s}.npz").exists()]
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
        rows = list(ex.map(_job, tasks, chunksize=1))
    tf = outdir / "timing.csv"
    new = pd.DataFrame(rows)
    if tf.exists() and len(new):
        new = pd.concat([pd.read_csv(tf), new], ignore_index=True)
    if len(new):
        new.to_csv(tf, index=False)
    bad = new[new.status != "ok"] if len(new) else new
    print(f"cfg {h}: {len(tasks)} computed, errors {len(bad)}; dir {outdir}")
    if len(bad):
        print(bad.head())
    print(outdir)


if __name__ == "__main__":
    main()
