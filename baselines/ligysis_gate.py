#!/usr/bin/env python
"""Reproduction gate for our own runs of P2Rank / fpocket / fpocket+PRANK on LIGYSIS vs the published rows
(results/validation/ligysis_reproduction/reproduction.csv, notebook-12 numbers). TRANS-frame, centre-based metrics.
Writes results/baselines/ligysis_gate.md.   .venv/bin/python baselines/ligysis_gate.py"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "baselines/outputs/ligysis"
pub = pd.read_csv(ROOT / "results/validation/ligysis_reproduction/reproduction.csv").set_index("method")
CASES = [("p2rank_2.4.2", "P2Rank"), ("fpocket_4.2.3", "fpocket"), ("fpocket_4.2.3+PRANK_2.4.2", "fpocket+PRANK"),
         ("fpocket_4.0.3", "fpocket"), ("fpocket_4.0.3+PRANK_2.4.2", "fpocket+PRANK"),
         ("p2rank_2.5.1", "P2Rank"), ("fpocket_4.2.3+PRANK_2.5.1", "fpocket+PRANK"), ("fpocket_4.0.3+PRANK_2.5.1", "fpocket+PRANK")]
def md_table(df):
    h = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False))


rows = []
for v, m in CASES:
    f = OUT / v / "eval/trans_frame/summary.csv"
    if not f.exists():
        continue
    s = pd.read_csv(f).set_index("metric")["value"]
    r = {"variant": v, "published row": m}
    ok = True
    for k, sk, pk in (("top-N", "recall_topn", "topN_pub"), ("top-N+2", "recall_topn2", "topN2_pub"), ("max recall", "recall_max", "max_recall_pub")):
        ours, p = 100 * s[sk], 100 * pub.loc[m, pk]
        r[k] = f"{ours:.1f} / {p:.1f} ({ours - p:+.1f})"
        ok &= abs(ours - p) <= 2.0
    r["P1K"] = f"{100*s['P1K']:.1f} / {100*pub.loc[m,'P1K_pub']:.1f}"
    r["gate (±2.0 pts)"] = ("PASS" if ok else "FAIL") if v in ("p2rank_2.4.2", "fpocket_4.2.3", "fpocket_4.2.3+PRANK_2.4.2") else "n/a (not the paper version)"
    rows.append(r)
df = pd.DataFrame(rows)
md = ("# Baseline reproduction gate on LIGYSIS (our runs vs published)\n\nOur runs (this repo's wrappers, authors' TRANS-frame ligand-free chains, 2,775 chains) scored with the LBS-comparison protocol "
      "(site_eval, 6,721 scored sites, DCC<=12 closest pocket) vs published numbers (ours / published (diff, points)).\n\n" + md_table(df) + "\n")
(ROOT / "results/baselines").mkdir(parents=True, exist_ok=True)
(ROOT / "results/baselines/ligysis_gate.md").write_text(md)
print(md)
