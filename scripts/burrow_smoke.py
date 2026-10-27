#!/usr/bin/env python
"""2-minute end-to-end smoke test (called by `scripts/reproduce_all.sh smoke`).

1. verify the FREEZE v2 model files against results/dev/FREEZE_v2.md (sha256);
2. build 5 single-chain inputs of Track A DEV entries from the local PDB mirror (polymer-only);
3. `burrow predict` with the reference implementation and with the fast backend (CPU) — outputs
   must be byte-identical;
4. score them against the entries' pooled sites with the LIGYSIS site protocol (pocketdetect.site_eval).
Exit code 0 = all checks passed.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import re
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    t0 = time.time()
    from pocketdetect.burrow.biolip import sites, write_chain_input
    from pocketdetect.burrow.cli import DEFAULT_MODELS, _stem, main as cli
    from pocketdetect.datasets.trainset import load_entry, read_manifest
    from pocketdetect.site_eval import evaluate_sites, summarize_sites
    from pocketdetect.sites import Site

    # 1. model integrity
    md = (ROOT / "results/dev/FREEZE_v2.md").read_text()
    want = dict(re.findall(r"\| (results/dev/freeze_v2/models/\S+) \| ([0-9a-f]{64}) \|", md))
    bad = [p for p, h in want.items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    assert want and not bad, f"model sha256 mismatch: {bad}"
    print(f"[smoke] {len(want)} frozen v2 models verified ({time.time() - t0:.0f}s)")
    assert "freeze_v2" in DEFAULT_MODELS[("all", "default")]

    # 2. inputs
    m = read_manifest("dev").head(5)
    tmp = Path(tempfile.mkdtemp(prefix="burrow_smoke_"))
    files, by = [], {}
    for eid, npz in zip(m.entry_id, m.npz):
        e = load_entry(npz)
        f = write_chain_input(e, tmp / "inputs" / f"{eid}.cif")
        files.append(str(f))
        ss = sites(e, "all")
        by[eid] = [Site(eid, s["site_id"], s["center"], s["residues"], len(ss), True, s["atoms"]) for s in ss]
    print(f"[smoke] {len(files)} inputs written to {tmp / 'inputs'} ({time.time() - t0:.0f}s)")

    # 3. reference vs fast backend
    lst = tmp / "in.list"
    lst.write_text("\n".join(files) + "\n")
    for b in ("reference", "fast"):
        rc = cli(["predict", f"@{lst}", "--out", str(tmp / b), "--backend", b, "--workers", "2", "--no-points"])
        assert rc == 0, f"{b} backend failed"
    for f in files:
        s = _stem(f)
        assert (tmp / "reference" / f"{s}.csv").read_text() == (tmp / "fast" / f"{s}.csv").read_text(), s
    print(f"[smoke] reference and fast backend outputs identical ({time.time() - t0:.0f}s)")

    # 4. evaluation
    r = evaluate_sites(by, str(tmp / "fast"))
    s = summarize_sites(r, n_boot=0).set_index("metric")["value"]
    print(f"[smoke] {int(s['n_scored_sites'])} sites: top-N+2 {100 * s['recall_topn2']:.1f}  "
          f"DCA4 top-N+2 {100 * s['recall_ldca_topn2']:.1f}  max recall {100 * s['recall_max']:.1f}  "
          f"pockets/structure {r.structures.n_pred.mean():.1f}")
    print(f"[smoke] PASSED in {time.time() - t0:.0f}s (outputs in {tmp})")


if __name__ == "__main__":
    main()
