#!/usr/bin/env python
"""Final like-for-like timing table (one session, same node, same 200 LIGYSIS speed chains):
BURROW fast / default / accurate (CLI default models, `all` ranker, fast backend) vs P2Rank 2.5.1 and
fpocket 4.2.3 re-measured in the same session -> results/perf/bench_final.json + results/perf/bench_final.md
(included in docs/PERF.md by ``scripts/bench_burrow.py --md-only``).

    nohup python scripts/bench_final.py > results/perf/bench_final.log 2>&1 &

Measurements (sequential, nothing else of ours running):
  1. 1 core, warm, in-process per-structure timings (median / p90; excl. and incl. parsing);
  2. CLI wall time on the 200 chains with 1 worker (process start-up + model load + writing included;
     comparable to P2Rank batch mode with one JVM and 1 thread);
  3. CLI throughput on speed200 x 10: 32 CPU workers, 1 GPU (10 workers), 3 GPUs (10 workers each);
  4. P2Rank 2.5.1 batch (one JVM, 1 thread) and fpocket 4.2.3 (1 process) on the same 200 chains
     (`baselines/run_p2rank.py` / `run_fpocket.py`, as in results/baselines/SPEED.md).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from bench_burrow import bench_driver, bench_inprocess, load_avg, speed200, summarize  # noqa: E402
from pocketdetect.burrow.cli import DEFAULT_MODELS  # noqa: E402

PY = sys.executable
PRESETS = ("fast", "default", "accurate")
LIST = "baselines/ligysis_speed200.txt"
BOUT = ROOT / "baselines/outputs/speed200_final"


def loadavg3():
    return " ".join(open("/proc/loadavg").read().split()[:3])


def timed(cmd):
    t = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return round(time.perf_counter() - t, 2), r


def timing_rows(d):
    with open(d / "timing.csv") as f:
        return list(csv.DictReader(f))


def q(v):
    v = np.asarray(v, float)
    return dict(median=round(float(np.median(v)), 1), p90=round(float(np.quantile(v, 0.9)), 1), mean=round(float(v.mean()), 1))


def baselines(R):
    BOUT.mkdir(parents=True, exist_ok=True)
    out = {}
    d = BOUT / "p2rank_2.5.1_batch_j1"
    shutil.rmtree(d, ignore_errors=True)
    l0 = loadavg3()
    wall, r = timed([PY, "baselines/run_p2rank.py", f"@{LIST}", "-o", str(d), "--version", "2.5.1", "-j", "1"])
    rows = timing_rows(d)
    ok = [x for x in rows if x["status"] == "ok"]
    out["p2rank"] = dict(wall_s=wall, ok=len(ok), n=len(rows), load=[l0, loadavg3()],
                         clean_ms=q([1e3 * float(x["clean_s"]) for x in ok]),
                         tool_ms_amortised=round(1e3 * float(ok[0]["tool_s"]), 1) if ok else None)
    print("p2rank", out["p2rank"], flush=True)
    d = BOUT / "fpocket_4.2.3_j1"
    shutil.rmtree(d, ignore_errors=True)
    l0 = loadavg3()
    wall, r = timed([PY, "baselines/run_fpocket.py", f"@{LIST}", "-o", str(d), "--version", "4.2.3", "-j", "1"])
    rows = timing_rows(d)
    ok = [x for x in rows if x["status"] == "ok"]
    out["fpocket"] = dict(wall_s=wall, ok=len(ok), n=len(rows), load=[l0, loadavg3()],
                          clean_ms=q([1e3 * float(x["clean_s"]) for x in ok]),
                          tool_ms=q([1e3 * float(x["tool_s"]) for x in ok]),
                          total_ms=q([1e3 * float(x["total_s"]) for x in ok]))
    print("fpocket", out["fpocket"], flush=True)
    R["baselines"] = out


def write_md(R, path):
    B = R["presets"]
    L = [f"## Final like-for-like timing (one session: {R['date']}, host {R['host']}, commit {R['commit']})\n",
         "Same 200 LIGYSIS chains for every tool (`baselines/ligysis_speed200.txt`, median 1,803 atoms), same node "
         "(AMD EPYC 9554, 4x RTX 4090), measured back to back by `scripts/bench_final.py`; nothing else of this "
         "project ran on the node. **Shared node:** other users' jobs ran concurrently. Node load (1/5/15 min) is "
         "given for every measurement. BURROW = the CLI default models for each preset with the `all` ranker "
         "(fast: `results/dev/freeze_fast`, default/accurate: `results/dev/freeze_v2`), fast backend, BLAS/Numba "
         "pinned to 1 thread per process.\n",
         "| method | 1 core warm, excl. parsing: median / p90 ms | 1 core warm, incl. parsing: median / p90 ms | "
         "1 process on 200 chains, wall s (amortised ms; start-up incl.) | 32 CPU workers, structures/s | "
         "1 GPU (10 workers), structures/s | 3 GPUs (10 workers each), structures/s |",
         "|---|---|---|---|---|---|---|"]
    for p in PRESETS:
        s = B[p]["cpu1"]
        dr = {d["label"]: d for d in B[p]["driver"]}
        f = lambda k: f"{dr[k]['structures_per_s']}" + ("" if dr[k]["ok"] == dr[k]["n"] else f" ({dr[k]['ok']}/{dr[k]['n']} ok)") \
            if k in dr else "–"
        one = dr.get("cpu1w")
        L.append(f"| BURROW {p} | {s['total']['median']} / {s['total']['p90']} | {s['total_incl_parse']['median']} / "
                 f"{s['total_incl_parse']['p90']} | {one['wall_s']} ({1e3 * one['wall_s'] / max(one['ok'], 1):.0f}) | "
                 f"{f('cpu32')} | {f('gpu1')} | {f('gpu3')} |")
    b = R.get("baselines", {})
    if "p2rank" in b:
        x = b["p2rank"]
        L.append(f"| P2Rank 2.5.1 (batch, one JVM, 1 thread) | – (batch only) | – (batch only) | {x['wall_s']} "
                 f"({1e3 * x['wall_s'] / max(x['ok'], 1):.0f}) | – | – | – |")
    if "fpocket" in b:
        x = b["fpocket"]
        L.append(f"| fpocket 4.2.3 (1 process) | – | {x['tool_ms']['median']} / {x['tool_ms']['p90']} (fpocket call) | "
                 f"{x['wall_s']} ({1e3 * x['wall_s'] / max(x['ok'], 1):.0f}) | – | – | – |")
    L.append("")
    L.append("Loads (1/5/15 min) per measurement: " + "; ".join(
        f"BURROW {p} 1-core {B[p]['cpu1_load'][0]} -> {B[p]['cpu1_load'][1]}" for p in PRESETS) + "; " + "; ".join(
        f"{d['label']} {p} {d['load'][0]} -> {d['load'][1]}" for p in PRESETS for d in B[p]["driver"]) +
        "".join(f"; {k} {v['load'][0]} -> {v['load'][1]}" for k, v in b.items()) + ".\n")
    L.append("Columns: *1 core warm* = per-structure wall time in one warm process (first structure run twice "
             "before timing), excl. or incl. reading/parsing the PDB file; *1 process on 200 chains* = complete CLI run "
             "with one worker (`python -m pocketdetect.burrow.fast predict --workers 1`; interpreter start, imports, "
             "model load, parsing, writing) and, for the baselines, the project wrappers with the same input list "
             "(P2Rank: gemmi cleaning + one JVM over all 200 files, JVM start included; fpocket: gemmi cleaning + one "
             "fpocket process per structure, sequential); throughput columns = speed200 x 10 (2,000 structures) "
             "through the BURROW CLI incl. parsing and writing, process start-up included. Baseline per-structure "
             "medians: P2Rank batch mode gives only the amortised time (JVM total / 200); fpocket's column is the "
             "fpocket call per structure (it includes fpocket's own parsing), gemmi cleaning median "
             f"{b.get('fpocket', {}).get('clean_ms', {}).get('median', '–')} ms in addition. Multi-thread baseline "
             "numbers (P2Rank 8/32 threads 49/47 ms, fpocket 32 processes 39 ms per structure) are from the earlier "
             "session in results/baselines/SPEED.md (load 20-68).\n")
    Path(path).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,3")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "results/perf/bench_final.json")
    ap.add_argument("--md", type=Path, default=ROOT / "results/perf/bench_final.md")
    ap.add_argument("--md-only", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    a = ap.parse_args()
    if a.md_only:
        write_md(json.loads(a.out.read_text()), a.md)
        return
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src/pocketdetect/burrow"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    R = dict(date=time.strftime("%Y-%m-%d %H:%M"), host=platform.node(), commit=commit + (" + uncommitted fast-backend "
             "changes" if dirty else ""), load_start=loadavg3(), presets={})
    paths = speed200()
    gpus = [g for g in a.gpus.split(",") if g]
    tmp = Path(tempfile.mkdtemp(prefix="burrow_final_"))
    l1 = tmp / "speed200.txt"
    l1.write_text("\n".join(p for _, p in paths) + "\n")
    lN = tmp / f"speed200x{a.repeat}.txt"
    lN.write_text("\n".join([p for _, p in paths] * a.repeat) + "\n")
    # 1) 1 core warm, all presets back to back
    for p in PRESETS:
        mp = ROOT / DEFAULT_MODELS[("all", p)]
        l0 = loadavg3()
        rows = bench_inprocess(paths, mp, "fast")
        R["presets"][p] = dict(model=str(mp.relative_to(ROOT)), cpu1=summarize(rows), cpu1_load=[l0, loadavg3()], driver=[])
        print(p, "cpu1", R["presets"][p]["cpu1"]["total"], R["presets"][p]["cpu1"]["total_incl_parse"], flush=True)
    # 2) baselines, 1 thread, right after
    if not a.skip_baselines:
        baselines(R)
    # 3) CLI runs
    configs = [("cpu1w", l1, "cpu", 1, 1), ("cpu32", lN, "cpu", 32, 1)]
    if gpus:
        configs += [("gpu1", lN, f"cuda:{gpus[0]}", 10, 32), ("gpu3", lN, "cuda:" + ",".join(gpus), 10, 32)]
    for lab, lst, dev, w, b in configs:
        for p in PRESETS:
            l0 = loadavg3()
            d = bench_driver(lst, tmp / f"{p}_{lab}", dev, w, p, b)
            d.update(label=lab, load=[l0, loadavg3()])
            R["presets"][p]["driver"].append(d)
            print(p, d, flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    R["load_end"] = loadavg3()
    a.out.write_text(json.dumps(R, indent=1))
    write_md(R, a.md)
    print("done", flush=True)


if __name__ == "__main__":
    main()
