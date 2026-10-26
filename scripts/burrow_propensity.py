#!/usr/bin/env python
"""Atom-type ligand-contact propensity (phase-3 ranker feature V5) for a saved BURROW run.

1. per train input: exposed / contact counts per atom type (non-ion pooled ligands, 4 Å);
2. tables: one per CV fold (built from the other folds = out-of-fold for that fold's inputs) and
   one from all train inputs (used for dev / inference) -> results/dev/biolip/propensity_<run>.json;
3. pocket features (mean / sum / side-chain mean log-odds of lining atoms) for every cached train
   candidate (its fold's OOF table) and dev candidate (full table)
   -> results/dev/cache/pockets/<run>/prop.pkl {"train": {sid: (n,3)}, "dev": {...}}.

  python scripts/burrow_propensity.py --run p3_small_default_s0
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_featurize_biolip import feats_dir, make_cfg  # noqa: E402
from pocketdetect.burrow.biolip import ASM, entries, ligand_atoms, load_protein_entry  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402
from pocketdetect.burrow.propensity import atom_logodds, build_table, pocket_propensity, type_counts  # noqa: E402
from pocketdetect.datasets.trainset import load_entry  # noqa: E402

_G = {}


def _protein(sid, split):
    if split == "dev":
        return load_protein(ROOT / "results/dev/biolip/inputs" / f"{sid}.cif", structure_id=sid)
    return load_protein_entry(load_entry(_G["npz"][sid.replace(ASM, "")]), assembly=sid.endswith(ASM))


def _count_job(sid):
    P = _protein(sid, "train")
    pts = np.load(Path(_G["fd"]) / f"{sid}.npz")["points"]
    xyz, _, ion = ligand_atoms(load_entry(_G["npz"][sid.replace(ASM, "")]), sid.endswith(ASM))
    return sid, type_counts(P, pts, xyz[~ion])


def _feat_job(args):
    sid, split = args
    P = _protein(sid, split)
    pts = np.load(Path(_G["fd"]) / f"{sid}.npz")["points"]
    tab = _G["tables"]["full"] if split == "dev" else _G["tables"][str(_G["fold"][sid])]
    light = _G[split][sid][2]
    idxs = [l[2] if len(l) == 3 else l[4] for l in light]
    return sid, pocket_propensity(idxs, pts, P, atom_logodds(P, tab))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args(argv)
    t0 = time.time()
    pdir = ROOT / "results/dev/cache/pockets" / a.run
    meta = pickle.loads((pdir / "meta.pkl").read_bytes())
    args = meta["args"]
    fd = feats_dir(make_cfg(args["n_rays"], args["max_len"], args["subsample"]))
    m = entries("train")
    _G.update(npz=dict(zip(m.entry_id, m.npz)), fd=str(fd), fold=meta["fold"])
    sids = list(meta["fold"])
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(a.jobs, mp_context=ctx) as ex:
        counts = dict(ex.map(_count_job, sids, chunksize=8))
    tables = {str(k): build_table([counts[s] for s in sids if meta["fold"][s] != k]) for k in range(5)}
    tables["full"] = build_table(counts.values())
    out = ROOT / f"results/dev/biolip/propensity_{a.run}.json"
    out.write_text(json.dumps(tables))
    top = sorted(tables["full"].items(), key=lambda kv: -kv[1])[:8]
    print(f"[{time.time() - t0:.0f}s] tables ({len(tables['full'])} types) -> {out}; top: {top}", flush=True)
    _G["tables"] = tables
    _G["train"] = pickle.loads((pdir / "train.pkl").read_bytes())
    _G["dev"] = pickle.loads((pdir / "dev.pkl").read_bytes())
    res = {}
    with ProcessPoolExecutor(a.jobs, mp_context=ctx) as ex:
        for split in ("train", "dev"):
            res[split] = dict(ex.map(_feat_job, [(s, split) for s in _G[split]], chunksize=8))
    (pdir / "prop.pkl").write_bytes(pickle.dumps(res))
    print(f"[{time.time() - t0:.0f}s] pocket propensity features -> {pdir / 'prop.pkl'}", flush=True)


if __name__ == "__main__":
    main()
