#!/usr/bin/env python
"""Prepare BioLiP2 dev evaluation inputs for BURROW and baselines.

Writes (git-ignored inputs, small tables tracked):
  results/dev/biolip/inputs/<entry_id>.cif      polymer-only representative chain (all methods)
  results/dev/biolip/inputs.list                 list for baseline wrappers
  results/dev/biolip/dev_sites_all.tsv           site table, all sites (ions incl.)  -> ranker `all`
  results/dev/biolip/dev_sites_small.tsv         site table, non-ion sites           -> ranker `small`
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.biolip import entries, write_chain_input, write_site_table  # noqa: E402
from pocketdetect.datasets.trainset import load_entry  # noqa: E402

OUT = ROOT / "results/dev/biolip"


def _w(args):
    eid, npz = args
    f = OUT / "inputs" / f"{eid}.cif"
    if not f.exists():
        write_chain_input(load_entry(npz), f)
    return str(f)


def main():
    m = entries("dev")
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    (OUT / "inputs" / ".gitignore").write_text("*\n")
    with ProcessPoolExecutor(16) as ex:
        files = list(ex.map(_w, zip(m.entry_id, m.npz), chunksize=8))
    (OUT / "inputs.list").write_text("\n".join(files) + "\n")
    for v in ("all", "small"):
        df = write_site_table(m, OUT / f"dev_sites_{v}.tsv", v, OUT / "inputs")
        print(v, "structures", df.structure_id.nunique(), "sites", len(df), "ion sites", int(df.is_ion.sum()))


if __name__ == "__main__":
    main()
