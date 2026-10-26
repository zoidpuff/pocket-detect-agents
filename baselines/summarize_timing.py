#!/usr/bin/env python
"""Markdown table of per-structure wall times (timing.csv total_s = cleaning + tool) for several output dirs.

  .venv/bin/python baselines/summarize_timing.py baselines/outputs/bench/fpocket_j16 baselines/outputs/bench/p2rank_batch_j1 ...
Columns are the directory basenames; rows are structures sorted by number of atoms (after cleaning).
"""
import csv
import sys
from pathlib import Path


def main(dirs):
    data, meta = {}, {}
    for d in dirs:
        name = Path(d).name
        for r in csv.DictReader(open(Path(d) / "timing.csv")):
            if r["n_atoms"]:
                meta[r["stem"]] = (int(r["n_atoms"]), int(r["n_chains"]))
            meta.setdefault(r["stem"], (0, 0))
            data.setdefault(r["stem"], {})[name] = (r["status"], float(r["total_s"]) if r["total_s"] else float("nan"))
    names = [Path(d).name for d in dirs]
    print("| structure | atoms | chains | " + " | ".join(names) + " |")
    print("|---|---:|---:|" + "---:|" * len(names))
    for s in sorted(meta, key=lambda s: meta[s][0]):
        cells = []
        for n in names:
            st, t = data.get(s, {}).get(n, ("-", float("nan")))
            cells.append(f"{t:.2f}" if st == "ok" else ("FAIL" if st != "-" else "-"))
        print(f"| {s} | {meta[s][0]} | {meta[s][1]} | " + " | ".join(cells) + " |")
    tot = []
    for n in names:
        tot.append(f"{sum(v[n][1] for v in data.values() if n in v and v[n][0]=='ok'):.1f}")
    print("| sum of per-structure s | | | " + " | ".join(tot) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
