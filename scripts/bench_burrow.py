#!/usr/bin/env python
"""BURROW performance benchmark: reference vs fast backend (CPU 1/16/32 threads, GPU single/batched,
multi-GPU driver) on dev structures -> results/perf/bench_burrow.json + docs/PERF.md.

    nohup python scripts/bench_burrow.py --n 200 --gpus 0,1,3 > results/perf/bench_burrow.log 2>&1 &

All CPU measurements pin BLAS/OpenMP/Numba to the stated thread count.  Medians over structures.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
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
PY = sys.executable
MODEL = ROOT / "results/dev/models/burrow_chen11_s0.json"


def sample_paths(n, seed_idx=2):
    from pocketdetect.burrow.devdata import read_manifest
    from pocketdetect.seeds import SEEDS
    df = read_manifest(ROOT / "results/dev/manifests/all.csv")
    df = df.sample(min(n, len(df)), random_state=SEEDS[seed_idx])
    return list(zip(df.structure_id, df.structure_path))


def med(rows, key):
    v = [r[key] for r in rows if key in r]
    return float(np.median(v)) if v else float("nan")


def stage_table(rows):
    keys = sorted({k for r in rows for k in r if isinstance(r[k], float)})
    return {k: round(med(rows, k), 2) for k in keys}


def bench_inprocess(paths, threads, which, n_ref=None):
    """Per-structure stage timings in this process (warm: one structure run first)."""
    import numba
    numba.set_num_threads(threads)
    from pocketdetect.burrow.model import BurrowModel
    from pocketdetect.burrow.prep import load_protein
    from pocketdetect.burrow.pipeline import predict_protein as ref_predict
    from pocketdetect.burrow.fast.pipeline import FastModel, predict_protein as fast_predict
    m = BurrowModel.load(MODEL)
    fm = FastModel.from_model(m)
    P0 = load_protein(paths[0][1], structure_id=paths[0][0])
    run = (lambda P: ref_predict(P, m)) if which == "reference" else (lambda P: fast_predict(P, fm))
    run(P0)
    rows = []
    for sid, path in paths[: n_ref or len(paths)]:
        t0 = time.perf_counter()
        P = load_protein(path, structure_id=sid)
        parse = (time.perf_counter() - t0) * 1e3
        pr = run(P)
        r = {k: float(v) for k, v in pr.timings.items()}
        r.update(parse=parse, total_incl_parse=parse + r["total"], n_atoms=float(len(P.coords)))
        rows.append(r)
    return rows


COLD = r"""
import os, sys, time, json
t0 = time.perf_counter()
sys.path.insert(0, %(src)r)
from pocketdetect.burrow.prep import load_protein
from pocketdetect.burrow.fast.pipeline import FastModel, predict_protein
fm = FastModel.load(%(model)r)
t1 = time.perf_counter()
P = load_protein(%(path)r)
t2 = time.perf_counter()
predict_protein(P, fm)
t3 = time.perf_counter()
P = load_protein(%(path)r); t4 = time.perf_counter(); predict_protein(P, fm); t5 = time.perf_counter()
print(json.dumps(dict(import_and_model_ms=(t1-t0)*1e3, first_parse_ms=(t2-t1)*1e3, first_predict_ms=(t3-t2)*1e3,
                      warm_total_ms=(t5-t3)*1e3)))
"""


def bench_cold(path):
    out = {}
    with tempfile.TemporaryDirectory() as nc:
        env = dict(os.environ, NUMBA_CACHE_DIR=nc, NUMBA_NUM_THREADS="1")
        code = COLD % dict(src=str(ROOT / "src"), model=str(MODEL), path=path)
        for label in ("cold_compile", "cached"):
            t = time.perf_counter()
            r = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True, check=True)
            out[label] = json.loads(r.stdout.strip().splitlines()[-1])
            out[label]["process_wall_ms"] = (time.perf_counter() - t) * 1e3
    return out


def bench_gpu_inprocess(paths, device, batch_sizes=(1, 8, 32)):
    import torch
    from pocketdetect.burrow.features import BurrowConfig
    from pocketdetect.burrow.prep import load_protein
    from pocketdetect.burrow.fast import gpu
    from pocketdetect.burrow.fast.pipeline import FastModel, predict_from_points
    fm = FastModel.load(MODEL)
    cfg = BurrowConfig()
    Ps = [load_protein(p, structure_id=s) for s, p in paths]
    t = time.perf_counter()
    items = [gpu.prep(P, cfg) for P in Ps]
    prep_ms = (time.perf_counter() - t) * 1e3 / len(items)
    gpu.featurize_batch(items[:2], cfg, device, fm)
    out = dict(host_prep_ms_per_structure=round(prep_ms, 2))
    res = None
    for bs in batch_sizes:
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        res = []
        for s in range(0, len(items), bs):
            res += gpu.featurize_batch(items[s:s + bs], cfg, device, fm)
        out[f"gpu_features_and_scores_ms_per_structure_batch{bs}"] = round((time.perf_counter() - t) * 1e3 / len(items), 2)
    t = time.perf_counter()
    for it, (ps, sc) in zip(items, res):
        predict_from_points(it.P, ps, fm, score=sc)
    out["host_post_ms_per_structure"] = round((time.perf_counter() - t) * 1e3 / len(items), 2)
    out["peak_gpu_mem_GB"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    return out


def bench_driver(list_file, out_dir, device, workers, batch=32):
    shutil.rmtree(out_dir, ignore_errors=True)
    t = time.perf_counter()
    cmd = [PY, "-m", "pocketdetect.burrow.fast", "predict", f"@{list_file}", "--out", str(out_dir), "--device", device,
           "--workers", str(workers), "--batch", str(batch), "--ranker", "small", "--no-points"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    wall = time.perf_counter() - t
    rows = [l for l in (out_dir / "timing.csv").read_text().splitlines()[1:]] if (out_dir / "timing.csv").exists() else []
    n_ok = sum(",ok," in l for l in rows)
    return dict(device=device, workers=workers, batch=batch, n=len(rows), ok=n_ok, wall_s=round(wall, 1),
                structures_per_s=round(n_ok / wall, 1), structures_per_min=round(60 * n_ok / wall),
                stderr_tail=r.stderr.strip().splitlines()[-1:] if r.stderr else [])


def run_driver(R, a, paths):
    gpus = [g for g in a.gpus.split(",") if g]
    lst = a.out.parent / "bench_list.txt"
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text("\n".join([p for _, p in paths] * a.repeat) + "\n")
    tmp = Path(tempfile.mkdtemp(prefix="burrow_bench_"))
    R["driver"] = []
    configs = [("cpu", 16, 1), ("cpu", 32, 1)]
    if gpus:
        configs += [(f"cuda:{gpus[0]}", 10, 32)]
    if len(gpus) > 1:
        configs += [("cuda:" + ",".join(gpus), 10, 32)]
    for dev, w, b in configs:
        print("driver", dev, w, flush=True)
        R["driver"].append(bench_driver(lst, tmp / f"{dev.replace(':', '_').replace(',', '_')}_{w}", dev, w, b))
        print(R["driver"][-1], flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


def write_md(R, path):
    L = ["# BURROW performance (reference vs fast backend)\n",
         f"Generated by `scripts/bench_burrow.py` on {R['date']} ({R['host']}). Raw: `results/perf/bench_burrow.json`. "
         f"Dev structures: {R['n']} random entries of `results/dev/manifests/all.csv` (median {R['median_atoms']:.0f} "
         "heavy atoms). Model: `results/dev/models/burrow_chen11_s0.json` (250 trees x 63 leaves point model). "
         "**Caveat:** shared node (load 40-100 from other jobs during the run) -> absolute CPU numbers are pessimistic.\n",
         "## Equivalence\n",
         "The fast backend reproduces the reference outputs exactly (scores, centres, residue sets, points; "
         "`scripts/verify_burrow_fast.py`):\n"]
    for k, v in R.get("verify", {}).items():
        L.append(f"- {k}: {v}")
    L.append("\n## 1 core, warm (ms per structure, median)\n")
    L.append(f"| stage | reference ({R['n_ref']} structures) | fast (same {R['n_ref']}) | fast (all {R['n']}) |")
    L.append("|---|---|---|---|")
    ref, fs, fast = R["cpu1_reference"], R.get("cpu1_fast_same_subset", {}), R["cpu1_fast"]
    for k in sorted(set(ref) | set(fast)):
        L.append(f"| {k} | {ref.get(k, float('nan')):.1f} | {fs.get(k, float('nan')):.1f} | {fast.get(k, float('nan')):.1f} |")
    L.append("\n`features` = feat_points (solvent-shell grid) + feat_prefilter (dual grid + PSP) + feat_metal + "
             "feat_rays + feat_neighbors + feat_misc; `total` excludes parsing.\n")
    L.append("## Threads (fast backend, Numba threads for the parallel kernels; latency per structure, median ms)\n")
    for t, rows in R["threads"].items():
        L.append(f"- {t} threads: total {rows.get('total', float('nan')):.1f} (features {rows.get('features', float('nan')):.1f}, "
                 f"point score {rows.get('point_score', float('nan')):.1f}, pockets {rows.get('pockets', float('nan')):.1f}, "
                 f"rank {rows.get('rank', float('nan')):.1f})")
    L.append("\n## Cold start (fresh process, 1 thread)\n")
    for k, v in R["cold"].items():
        L.append(f"- {k}: " + ", ".join(f"{a}={b:.0f} ms" for a, b in v.items()))
    L.append("\n## GPU (in-process, one RTX 4090)\n")
    for k, v in R["gpu_inprocess"].items():
        L.append(f"- {k}: {v}")
    L.append("\n## End-to-end throughput (`python -m pocketdetect.burrow.fast predict`, incl. process start-up, "
             "parsing and writing; the 200-structure list repeated %d times)\n" % R.get("repeat", 1))
    L.append("| device | CPU workers | batch | structures | wall s | structures/s | structures/min |")
    L.append("|---|---|---|---|---|---|---|")
    for d in R["driver"]:
        L.append(f"| {d['device']} | {d['workers']} | {d['batch']} | {d['ok']}/{d['n']} | {d['wall_s']} | "
                 f"{d['structures_per_s']} | {d['structures_per_min']} |")
    L.append("\n## Notes\n")
    L.extend(f"- {n}" for n in R["notes"])
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-ref", type=int, default=30)
    ap.add_argument("--gpus", default="0,1,3")
    ap.add_argument("--out", type=Path, default=ROOT / "results/perf/bench_burrow.json")
    ap.add_argument("--md", type=Path, default=ROOT / "docs/PERF.md")
    ap.add_argument("--repeat", type=int, default=10, help="driver runs process the list this many times")
    ap.add_argument("--only-driver", action="store_true", help="re-run only the driver section (reuse JSON)")
    a = ap.parse_args()
    paths = sample_paths(a.n)
    if a.only_driver:
        R = json.loads(a.out.read_text())
        run_driver(R, a, paths)
        R["repeat"] = a.repeat
        a.out.write_text(json.dumps(R, indent=1))
        write_md(R, a.md)
        print("done", flush=True)
        return
    R = dict(date=time.strftime("%Y-%m-%d %H:%M"), host=platform.node(), n=len(paths), n_ref=a.n_ref, notes=[])
    print("cpu1 fast", flush=True)
    rows_fast = bench_inprocess(paths, 1, "fast")
    R["median_atoms"] = med(rows_fast, "n_atoms")
    R["cpu1_fast"] = stage_table(rows_fast)
    print("cpu1 reference", flush=True)
    R["cpu1_reference"] = stage_table(bench_inprocess(paths, 1, "reference", a.n_ref))
    R["cpu1_fast_same_subset"] = stage_table(rows_fast[:a.n_ref])
    R["threads"] = {}
    for t in (16, 32):
        print("threads", t, flush=True)
        R["threads"][str(t)] = stage_table(bench_inprocess(paths[:40], t, "fast"))
    print("cold", flush=True)
    R["cold"] = bench_cold(paths[0][1])
    gpus = [g for g in a.gpus.split(",") if g]
    if gpus:
        print("gpu in-process", flush=True)
        R["gpu_inprocess"] = bench_gpu_inprocess(paths[:96], f"cuda:{gpus[0]}")
    run_driver(R, a, paths)
    vf = ROOT / "results/perf"
    R["verify"] = {}
    for f in sorted(vf.glob("verify_*.json")):
        R["verify"][f.stem] = json.loads(f.read_text())["summary"]
    R["notes"] = [
        "Fast backend = exact re-implementation: Numba ports of formation / residues / descriptors / NMS / metal "
        "candidates / metal geometry, LightGBM-free forests (bit-identical), fp64-exact Triton kernels on GPU.",
        "The reference ray march and neighbour sums are already Numba; exact re-implementations gave no CPU speed-up, "
        "so the CPU fast path reuses them. Reaching <=70 ms/core needs method-side changes (fewer rays / shorter "
        "max length / fewer scored points / smaller point model), see the stage table.",
        "GPU mode: one process per GPU; each has its own pool of single-threaded CPU workers for parsing, host "
        "prep (canonical frame, metal candidates) and post-processing (formation, ranking, NMS, writing).",
        "Thread budget: <= 32 worker processes in total (shared 4096-thread user limit).",
    ]
    R["repeat"] = a.repeat
    a.out.write_text(json.dumps(R, indent=1))
    write_md(R, a.md)
    print("done", flush=True)


if __name__ == "__main__":
    main()
