#!/usr/bin/env python
"""Before/after table of two ``scripts/bench_final.py`` runs (same session) -> results/perf/bench_t1.md.

    python scripts/bench_t1_compare.py results/perf/bench_t1_before.json results/perf/bench_t1_after.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    B, A = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "results/perf/bench_t1.md"
    cw, gw, lg = A.get("cpu_workers"), A.get("gpu_workers"), A.get("large")
    gpus = A.get("gpus", "")
    L = [f"## Track T1: GPU path and per-structure latency, before / after (same session, {B['date']} -> {A['date']})\n",
         f"Before = commit {B['commit']}; after = {A['commit']}. Same node ({A['host']}), same 200 LIGYSIS speed "
         f"chains, `scripts/bench_final.py` run back to back for both trees (CPU workers <= {cw}; GPUs {gpus}, "
         f"{gw} CPU workers per GPU). Outputs are byte-identical between the two trees (see the equivalence "
         "section). Throughput rows include process start-up, parsing and writing.\n",
         "| preset | measurement | before | after | after / before |", "|---|---|---|---|---|"]

    def drv(R, p, lab):
        for d in R["presets"][p]["driver"]:
            if d["label"] == lab:
                return d
        return None

    for p in A["presets"]:
        if p not in B["presets"]:
            continue
        sb, sa = B["presets"][p]["cpu1"], A["presets"][p]["cpu1"]
        rows = [("1 core warm, excl. parsing, median / p90 ms", f"{sb['total']['median']} / {sb['total']['p90']}",
                 f"{sa['total']['median']} / {sa['total']['p90']}", sb["total"]["median"] / sa["total"]["median"]),
                ("1 core warm, incl. parsing, median / p90 ms",
                 f"{sb['total_incl_parse']['median']} / {sb['total_incl_parse']['p90']}",
                 f"{sa['total_incl_parse']['median']} / {sa['total_incl_parse']['p90']}",
                 sb["total_incl_parse"]["median"] / sa["total_incl_parse"]["median"]),
                ("parse (median ms)", f"{sb['parse']['median']}", f"{sa['parse']['median']}",
                 sb["parse"]["median"] / sa["parse"]["median"])]
        for lab, name in (("cpu1w", "CLI, 1 worker, 200 chains: wall s (ms / structure)"),
                          ("cpuN", f"CLI, {cw} CPU workers, 2,000 structures: structures/s"),
                          ("gpu1", f"CLI, 1 GPU ({gw} workers), 2,000 structures: structures/s"),
                          ("gpu3", f"CLI, 3 GPUs ({gw} workers each), 2,000 structures: structures/s"),
                          ("cpuN_large", f"CLI, {cw} CPU workers, {200 * lg:,} structures: structures/s"),
                          ("gpu3_large", f"CLI, 3 GPUs ({gw} workers each), {200 * lg:,} structures: structures/s")):
            db, da = drv(B, p, lab), drv(A, p, lab)
            if db is None or da is None:
                continue
            ok = "" if (db["ok"] == db["n"] and da["ok"] == da["n"]) else f" (ok {db['ok']}/{da['ok']} of {da['n']})"
            if lab == "cpu1w":
                rows.append((name + ok, f"{db['wall_s']} ({1e3 * db['wall_s'] / db['n']:.0f})",
                             f"{da['wall_s']} ({1e3 * da['wall_s'] / da['n']:.0f})", db["wall_s"] / da["wall_s"]))
            else:
                rows.append((name + ok, f"{db['structures_per_s']}", f"{da['structures_per_s']}",
                             da["structures_per_s"] / db["structures_per_s"]))
        for name, b, a, r in rows:
            L.append(f"| {p} | {name} | {b} | {a} | {r:.2f}x |")
    L.append("")
    L.append("Node load (1/5/15 min) before run: " + f"{B['load_start']} -> {B['load_end']}; after run: "
             f"{A['load_start']} -> {A['load_end']}. *after / before*: speed-up factor (time ratio for latency rows, "
             "throughput ratio for throughput rows).\n")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
