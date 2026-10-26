#!/usr/bin/env python
"""Cache BURROW point features + label distances for Track A's BioLiP2 train/dev sets.

  python scripts/burrow_featurize_biolip.py --split train --assembly --jobs 24
  python scripts/burrow_featurize_biolip.py --split dev --jobs 24

Train inputs are read from the representative's mmCIF (chain) or assembly-1 file; dev chain
inputs from results/dev/biolip/inputs/<entry>.cif (the files every method gets).
Output: results/dev/cache/feats_biolip/<cfg-hash>/<structure_id>.npz with points, X, vol,
is_metal, feats, names, d_lig (min dist to non-ion pooled ligand atoms), d_ion (to ion atoms);
plus timing.csv.  Skips existing files (restart-safe).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_featurize import cfg_hash  # noqa: E402
from pocketdetect.burrow.biolip import ASM, entries, ligand_atoms, load_protein_entry, structure_ids  # noqa: E402
from pocketdetect.burrow.devdata import CACHE  # noqa: E402
from pocketdetect.burrow.features import BurrowConfig, point_features  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402
from pocketdetect.datasets.trainset import load_entry  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402

DEV_INPUTS = ROOT / "results/dev/biolip/inputs"
_CFG = [BurrowConfig()]


def _dist(pts, xyz):
    if len(xyz) == 0 or len(pts) == 0:
        return np.full(len(pts), np.inf, np.float32)
    return cKDTree(xyz).query(pts, k=1, distance_upper_bound=20.0)[0].astype(np.float32)


def _job(args):
    sid, npz, is_asm, split, outdir = args
    out = Path(outdir) / f"{sid}.npz"
    try:
        t0 = time.perf_counter()
        from pocketdetect.burrow.biolip import AF, SCD, load_af, load_protein_af, load_protein_scd
        if sid.endswith(AF):
            e = load_af(npz)
            P = load_protein_af(e)
        elif sid.endswith(SCD):
            e = load_entry(npz)
            P = load_protein_scd(e, SEEDS[0])
        else:
            e = load_entry(npz)
            if split == "dev" and not is_asm and "__apo_" not in sid:
                P = load_protein(DEV_INPUTS / f"{e.entry_id}.cif", structure_id=sid)
            else:
                P = load_protein_entry(e, assembly=is_asm)
        t1 = time.perf_counter()
        ps = point_features(P, _CFG[0])
        t2 = time.perf_counter()
        xyz, site, ion = ligand_atoms(e, is_asm)
        np.savez(out, points=ps.points, X=ps.X, vol=ps.vol, is_metal=ps.is_metal, feats=ps.feats,
                 names=np.array(ps.names), d_lig=_dist(ps.points, xyz[~ion]), d_ion=_dist(ps.points, xyz[ion]))
        return dict(structure_id=sid, status="ok", n_atoms=len(P.coords), n_points=len(ps.points),
                    load_ms=(t1 - t0) * 1e3, feat_ms=(t2 - t1) * 1e3, **{f"t_{k}": v for k, v in ps.timings.items()})
    except Exception as ex:
        return dict(structure_id=sid, status=f"error: {type(ex).__name__}: {ex}")


def make_cfg(n_rays=60, max_len=24.0, subsample=2) -> BurrowConfig:
    lengths = (6.0, 10.0, 16.0, 24.0) if max_len >= 24 else (6.0, 10.0, 16.0)
    return BurrowConfig(n_rays=n_rays, lengths=lengths, subsample=subsample)


def cfg_args(ap):
    ap.add_argument("--n-rays", type=int, default=60)
    ap.add_argument("--max-len", type=float, default=24.0)
    ap.add_argument("--subsample", type=int, default=2)


def feats_dir(cfg: BurrowConfig | None = None):
    return CACHE / "feats_biolip" / cfg_hash(cfg or BurrowConfig())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    ap.add_argument("--assembly", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--extra", default="", help="train only: extra inputs 'af' (AF models) and/or 'scd' (side-chain dropout)")
    cfg_args(ap)
    a = ap.parse_args(argv)
    _CFG[0] = make_cfg(a.n_rays, a.max_len, a.subsample)
    outdir = feats_dir(_CFG[0])
    print("config", _CFG[0], outdir, flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    m = entries(a.split)
    ids = structure_ids(m, a.assembly)
    from pocketdetect.burrow.biolip import AF, SCD, af_manifest
    ex = [x for x in a.extra.split(",") if x]
    if "scd" in ex:
        ids += [(e + SCD, n, False) for e, n in zip(m.entry_id, m.npz)]
    if "apo" in ex:
        from pocketdetect.datasets.apotrain import read_apo_manifest
        pm = read_apo_manifest(a.split)
        ids += [(e, n, False) for e, n, par in zip(pm.entry_id, pm.npz, pm.parent_entry_id) if par in set(m.entry_id)]
    if "af" in ex:
        am = af_manifest(a.split)
        ids += [(e + AF, n, False) for e, n in zip(am.entry_id, am.npz) if e in set(m.entry_id)]
    tasks = [(s, n, asm, a.split, str(outdir)) for s, n, asm in ids if not (outdir / f"{s}.npz").exists()]
    # big ones first for load balance
    from pocketdetect.burrow.biolip import base_id
    size = dict(zip(m.entry_id, m.n_res))
    tasks.sort(key=lambda t: -(size.get(base_id(t[0]), 300) * (4 if t[2] else 1)))
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
        for i, r in enumerate(ex.map(_job, tasks, chunksize=1)):
            rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    tf = outdir / f"timing_{a.split}.csv"
    new = pd.DataFrame(rows)
    if tf.exists() and len(new):
        new = pd.concat([pd.read_csv(tf), new], ignore_index=True)
    if len(new):
        new.to_csv(tf, index=False)
    bad = new[new.status != "ok"] if len(new) else new
    print(f"{a.split}: {len(tasks)} computed, errors {len(bad)} -> {outdir} ({time.time() - t0:.0f}s)")
    if len(bad):
        print(bad.head(10).to_string())


if __name__ == "__main__":
    main()
