#!/usr/bin/env python
"""Dev-set evaluation (no git gating, no ledger). Refuses registered *test* ground truths — use
scripts/run_test_eval.py for those.

  python scripts/run_dev_eval.py --sites <dev site table .tsv> --pred-dir <dir> --out results/dev/<tag> \
      [--cluster-sources ligysis] [--n-boot 1000]
  python scripts/run_dev_eval.py --ds <dev P2Rank .ds> --pred-dir <dir> --out results/dev/<tag> [--rules p2rank_strict]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketdetect.benchmarks import REGISTRY, Benchmark, run_benchmark  # noqa: E402

TEST_PATHS = {b.abspath.resolve() for b in REGISTRY.values() if b.split == "test"}
TEST_DIR_MARKERS = ("gt/ligysis_lbs", "gt/cryptobench_test", "gt/pocketwatch_test", "gt/af_set", "gt/dimer", "coach420", "holo4k")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sites", type=Path)
    g.add_argument("--ds", type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rules", default="p2rank_strict")
    ap.add_argument("--cluster-sources", default="", help="comma list of test-union sources for cluster units (default: per-structure units)")
    ap.add_argument("--headline", default="")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args(argv)
    gt = (a.sites or a.ds).resolve()
    if gt in TEST_PATHS or any(m in str(gt) for m in TEST_DIR_MARKERS):
        raise SystemExit(f"refused: {gt} is a test-set ground truth; use scripts/run_test_eval.py")
    kind = "sites" if a.sites else "ligands"
    head = tuple(x for x in a.headline.split(",") if x) or (("recall_topn2", "P1K") if kind == "sites" else ("success_DCA4_topn", "success_DCA4_topn2"))
    b = Benchmark(name=f"dev:{gt.stem}", kind=kind, path=str(gt), cluster_sources=tuple(x for x in a.cluster_sources.split(",") if x),
                  headline=head, split="dev", rules=a.rules)
    info = run_benchmark(b, a.pred_dir, a.out, n_boot=a.n_boot, jobs=a.jobs, gt_path=gt)
    print(json.dumps(info["headline"], indent=2))


if __name__ == "__main__":
    main()
