#!/usr/bin/env python
"""v3 dev signal (ii): BURROW predictions on the DEV assembly-1 inputs (Track A stored assemblies)
for several output rules (v1 = no floor, v2 = floor 0.03/min_keep 3, aKn = min_keep K x #chains),
optionally with a different model JSON.  Pockets are computed once per input; the rules only differ
in the output step.  Writes results/dev/biolip/asm/preds/<prefix>_<rule>/<entry>__asm1.csv.

  python scripts/burrow_asm_dev_predict.py --model results/dev/freeze_v2/models/burrow_all_default_s0.json --prefix dev
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_asm_oof import rule_fc  # noqa: E402
from pocketdetect.burrow import pipeline as pl  # noqa: E402
from pocketdetect.burrow.model import BurrowModel, FormationConfig  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402

B = ROOT / "results/dev/biolip/asm"
_G = {}


def _job(f):
    gt = json.loads(Path(f).read_text())
    sid, n = gt["pdb_id"], len(gt["chains"])
    P = load_protein(gt["asm_path"], structure_id=sid)
    m = _G["model"]
    captured = {}
    orig = pl.finalize

    def fin(pk, rs, fc):  # capture candidates + scores once, emit every rule
        captured["pk"], captured["rs"] = pk, rs
        return orig(pk, rs, fc)

    pl.finalize = fin
    try:
        base = pl.predict_protein(P, m)
    finally:
        pl.finalize = orig
    from scipy.spatial import cKDTree
    tree = cKDTree(P.coords)
    for rule in _G["rules"]:
        fc = replace(FormationConfig(**m.formation), **{k: getattr(rule_fc(rule, n), k) for k in ("min_score", "min_keep")})
        keep = orig(captured["pk"], captured["rs"], fc)
        pk = captured["pk"]
        pred = pl.Prediction(sid, np.array([pk[i].center for i in keep]).reshape(-1, 3), np.asarray(captured["rs"])[keep],
                             [pl.output_residues(pk[i], tree, P) for i in keep], [np.zeros((0, 3))] * len(keep), {})
        pl.write_prediction(B / "preds" / f"{_G['prefix']}_{rule}", pred, points=False)
    return sid, base.timings.get("total", 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prefix", default="dev")
    ap.add_argument("--rules", default="v1,v2,a3n")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    _G.update(model=BurrowModel.load(a.model), rules=a.rules.split(","), prefix=a.prefix)
    files = sorted(str(p) for p in (B / "gt" / a.split).glob("*.json"))
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(_job, files, chunksize=1))
    print(len(res), "inputs predicted")


if __name__ == "__main__":
    main()
