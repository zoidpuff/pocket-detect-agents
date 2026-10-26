#!/usr/bin/env python
"""Build the "naive + drop pockets with mean pLDDT < 70" baseline (PREREG O5 comparator) from any
method's predictions on the full AF models.

  python scripts/af_plddt_filter_baseline.py --pred-dir results/preds/p2rank/af_set \
      --out results/preds/p2rank+plddt70/af_set [--threshold 70] [--sites data/processed/gt/af_set/sites.tsv]

Pocket pLDDT = mean residue pLDDT (CA B-factor) over the pocket residues, or over residues with any
heavy atom ≤ 6 Å of the centre when the method gives no residues (pocketdetect.af_eval). The output
directory is a normal prediction directory (score it with run_test_eval.py --benchmark af_set).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketdetect.af_eval import transform_pred_dir  # noqa: E402
from pocketdetect.sites import read_sites  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=70.0)
    ap.add_argument("--sites", type=Path, default=ROOT / "data/processed/gt/af_set/sites.tsv", help="site table providing structure paths")
    a = ap.parse_args(argv)
    st = read_sites(a.sites).drop_duplicates("structure_id")
    log = transform_pred_dir(a.pred_dir, a.out, dict(zip(st.structure_id, st.structure_path)), a.threshold)
    log.to_csv(a.out / "_plddt_filter_log.csv", index=False)
    print(f"{len(log)} structures, pockets in {int(log.n_in.sum())} -> out {int(log.n_out.sum())}, missing {int(log.missing.sum())}")


if __name__ == "__main__":
    main()
