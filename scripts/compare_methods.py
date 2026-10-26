#!/usr/bin/env python
"""Paired comparison of two evaluated methods (outputs of scripts/evaluate.py on the same
manifest + ligand rules).

  python scripts/compare_methods.py --a results/eval/ours/coach420 --b results/eval/p2rank/coach420 \
      [--metrics success_DCA4_topn,success_DCA4_topn2] [--n-boot 10000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pocketdetect.evaluate import compare_methods
from pocketdetect.seeds import BOOTSTRAP_SEED


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=Path, help="eval dir of method A")
    ap.add_argument("--b", required=True, type=Path, help="eval dir of method B")
    ap.add_argument("--metrics", default="success_DCA4_topn,success_DCA4_topn2,success_DCC4_topn,success_DCC4_topn2")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--out", type=Path, default=None, help="optional CSV output")
    a = ap.parse_args(argv)
    ia = json.loads((a.a / "run_info.json").read_text())
    ib = json.loads((a.b / "run_info.json").read_text())
    if ia.get("gt_cache_key") != ib.get("gt_cache_key"):
        raise SystemExit("ground truth differs between the two evaluations (manifest/rules/code); refusing to compare")
    la = pd.read_csv(a.a / "ligands.csv")
    lb = pd.read_csv(a.b / "ligands.csv")
    rows = [compare_methods(la, lb, m.strip(), n_boot=a.n_boot, seed=a.seed) for m in a.metrics.split(",") if m.strip()]
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df.to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
