#!/usr/bin/env python
"""Aggregate final dev evaluations (results/dev/biolip/eval_final_{all,small}.csv) into a markdown table:
baselines, then each BURROW preset as mean over seeds with [min, max] and per-seed paired Δ vs P2Rank."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

B = Path(__file__).resolve().parents[1] / "results/dev/biolip"
M = ["recall_topn", "recall_topn2", "recall_ldca_topn2", "P1K"]
LAB = {"recall_topn": "top-N", "recall_topn2": "top-N+2 (DCC12)", "recall_ldca_topn2": "DCA4 top-N+2", "P1K": "P1K"}


def val(x):
    return float(str(x).split()[0])


def main():
    out = []
    for v in ("all", "small"):
        f = B / f"eval_final_{v}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        out.append(f"\n### ranker `{v}` on dev_sites_{v} ({int(d.n_sites.iloc[0])} sites)\n")
        out.append("| method | pockets | " + " | ".join(LAB[m] for m in M) + " | Δ top-N+2 vs P2Rank | Δ DCA4 | Δ P1K |")
        out.append("|---" * (len(M) + 5) + "|")
        for _, r in d.iterrows():
            if not re.match(r"(default|accurate)_s\d", r.method):
                out.append(f"| {r.method} | {r.pockets_mean} | " + " | ".join(str(r[m]) for m in M) + " | " +
                           " | ".join(str(r.get(f"d_{m}", "")) if pd.notna(r.get(f"d_{m}")) else "–"
                                      for m in ("recall_topn2", "recall_ldca_topn2", "P1K")) + " |")
        for p in ("default", "accurate"):
            g = d[d.method.str.match(rf"{p}_s\d")]
            if not len(g):
                continue
            cells = []
            for m in M:
                xs = g[m].map(val)
                cells.append(f"{xs.mean():.1f} [{xs.min():.1f}–{xs.max():.1f}]")
            deltas = []
            for m in ("recall_topn2", "recall_ldca_topn2", "P1K"):
                deltas.append("; ".join(str(x).split()[0] + " " + " ".join(str(x).split()[1:]) for x in g[f"d_{m}"]))
            out.append(f"| **BURROW {p}** (seeds {', '.join(g.method.str[-1])}; mean [min–max]) | "
                       f"{g.pockets_mean.mean():.1f} | " + " | ".join(cells) + " | " + " | ".join(deltas) + " |")
    txt = "\n".join(out)
    print(txt)
    (B / "final_table.md").write_text(txt + "\n")


if __name__ == "__main__":
    main()
