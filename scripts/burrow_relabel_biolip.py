#!/usr/bin/env python
"""After a Track A rebuild: (1) snapshot manifests + entries into results/dev/cache/trainset_snapshot/
(so later rebuilds cannot break running jobs; select with env BURROW_TRAINSET=<snapshot dir>);
(2) update the label distances (d_lig, d_ion) of cached BURROW feature files where pooled ligands
changed (features themselves depend only on the input structure); report counts.

  python scripts/burrow_relabel_biolip.py [--feats-dir DIR]
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_featurize_biolip import _dist, feats_dir  # noqa: E402
from pocketdetect.burrow.biolip import entries, ligand_atoms, structure_ids  # noqa: E402
from pocketdetect.datasets.trainset import load_entry  # noqa: E402

SNAP = ROOT / "results/dev/cache/trainset_snapshot"


def snapshot() -> Path:
    h = hashlib.sha256()
    for sp in ("train", "dev"):
        h.update((ROOT / f"data/processed/{sp}/manifest.csv").read_bytes())
    d = SNAP / h.hexdigest()[:12]
    if not d.exists():
        for sp in ("train", "dev"):
            shutil.copytree(ROOT / f"data/processed/{sp}/entries", d / sp / "entries")
            shutil.copy(ROOT / f"data/processed/{sp}/manifest.csv", d / sp / "manifest.csv")
    return d


def _job(args):
    sid, npz, asm, fdir = args
    f = Path(fdir) / f"{sid}.npz"
    if not f.exists():
        return "missing"
    z = dict(np.load(f))
    xyz, site, ion = ligand_atoms(load_entry(npz), asm)
    dl, di = _dist(z["points"], xyz[~ion]), _dist(z["points"], xyz[ion])
    if np.allclose(np.minimum(dl, 99), np.minimum(z["d_lig"], 99), atol=1e-3) and \
            np.allclose(np.minimum(di, 99), np.minimum(z["d_ion"], 99), atol=1e-3):
        return "same"
    z["d_lig"], z["d_ion"] = dl, di
    np.savez(f, **z)
    return "updated"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats-dir", type=Path, default=None)
    a = ap.parse_args()
    d = snapshot()
    os.environ["BURROW_TRAINSET"] = str(d)
    print("snapshot", d, flush=True)
    fdir = a.feats_dir or feats_dir()
    from collections import Counter
    c = Counter()
    for sp in ("train", "dev"):
        m = entries(sp)
        tasks = [(s, n, asm, str(fdir)) for s, n, asm in structure_ids(m, sp == "train")]
        with ProcessPoolExecutor(16) as ex:
            r = list(ex.map(_job, tasks, chunksize=16))
        c.update(f"{sp}:{x}" for x in r)
    print(dict(c))


if __name__ == "__main__":
    main()
