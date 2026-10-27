#!/usr/bin/env python
"""BURROW performance benchmark for the frozen presets: reference vs fast backend, CPU 1 core
(warm / cold, incl. / excl. parsing), CPU worker pools, GPU (1 and 3 GPUs) -> results/perf/
bench_burrow.json + docs/PERF.md.

    nohup python scripts/bench_burrow.py > results/perf/bench_burrow.log 2>&1 &

Structure sets:
  * ``speed200``: the 200 LIGYSIS chains of ``baselines/ligysis_speed200.txt`` (same inputs as the
    P2Rank / fpocket timings in results/baselines/SPEED.md);
  * ``biolip_dev``: 300 random chains of the BioLiP dev inputs (results/dev/biolip/inputs_audited.csv).
All CPU measurements pin BLAS/OpenMP/Numba to 1 thread per process.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
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
PRESETS = {"default": ROOT / "results/dev/freeze/models/burrow_all_default_s0.json",
           "accurate": ROOT / "results/dev/freeze/models/burrow_all_accurate_s0.json"}


def speed200():
    lines = (ROOT / "baselines/ligysis_speed200.txt").read_text().splitlines()
    paths = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
    return [(Path(p).name.split(".")[0], p) for p in paths]


def biolip_dev(n=300):
    from pocketdetect.burrow.devdata import read_manifest
    from pocketdetect.seeds import SEEDS
    df = read_manifest(ROOT / "results/dev/biolip/inputs_audited.csv").sample(n, random_state=SEEDS[3])
    return [(s, str(p if Path(p).is_absolute() else ROOT / p)) for s, p in zip(df.structure_id, df.structure_path)]


def stats(rows, key):
    v = np.array([r[key] for r in rows if key in r], float)
    return dict(median=round(float(np.median(v)), 1), p90=round(float(np.quantile(v, 0.9)), 1)) if len(v) else {}


def bench_inprocess(paths, model_path, which):
    """Per-structure timings in this process, 1 thread, warm (first structure run twice)."""
    from pocketdetect.burrow.model import BurrowModel
    from pocketdetect.burrow.pipeline import predict_protein as ref_predict
    from pocketdetect.burrow.prep import load_protein
    from pocketdetect.burrow.fast.pipeline import FastModel, predict_protein as fast_predict
    m = BurrowModel.load(model_path)
    fm = FastModel.from_model(m)
    run = (lambda P: ref_predict(P, m)) if which == "reference" else (lambda P: fast_predict(P, fm))
    try:   # the fast CLI parses with burrow.fast.io (identical Protein); older trees: reference loader
        from pocketdetect.burrow.fast.io import load_protein as fast_load
    except ImportError:
        fast_load = None
    load = (lambda p, s: fast_load(p, s)) if (which == "fast" and fast_load is not None) else \
        (lambda p, s: load_protein(p, structure_id=s))
    run(load(paths[0][1], paths[0][0]))
    rows = []
    for sid, path in paths:
        t0 = time.perf_counter()
        P = load(path, sid)
        parse = (time.perf_counter() - t0) * 1e3
        pr = run(P)
        r = {k: float(v) for k, v in pr.timings.items()}
        r.update(parse=parse, total_incl_parse=parse + r["total"], n_atoms=float(len(P.coords)),
                 n_points=float(sum(len(x) for x in pr.points)))
        rows.append(r)
    return rows


def summarize(rows):
    out = {k: stats(rows, k) for k in ("total", "total_incl_parse", "parse", "features", "point_score", "pockets", "rank")}
    out["stages_median"] = {k: round(float(np.median([r[k] for r in rows if k in r])), 1)
                            for k in sorted({k for r in rows for k in r if k.startswith("feat_")})}
    out["n"] = len(rows)
    out["median_atoms"] = float(np.median([r["n_atoms"] for r in rows]))
    return out


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
print(json.dumps(dict(import_and_model_ms=(t1-t0)*1e3, first_parse_ms=(t2-t1)*1e3, first_predict_ms=(t3-t2)*1e3,
                      first_structure_incl_startup_ms=(t3-t0)*1e3)))
"""


def bench_cold(path, model_path):
    out = {}
    with tempfile.TemporaryDirectory() as nc:
        env = dict(os.environ, NUMBA_CACHE_DIR=nc)
        code = COLD % dict(src=str(ROOT / "src"), model=str(model_path), path=path)
        for label in ("cold_jit_compile", "cached_jit"):
            r = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True, check=True)
            out[label] = json.loads(r.stdout.strip().splitlines()[-1])
    # the project's on-disk numba cache (normal installed use)
    code = COLD % dict(src=str(ROOT / "src"), model=str(model_path), path=path)
    r = subprocess.run([PY, "-c", code], env=dict(os.environ), capture_output=True, text=True, check=True)
    out["installed_cache"] = json.loads(r.stdout.strip().splitlines()[-1])
    return out


def bench_driver(list_file, out_dir, device, workers, preset, batch=32):
    shutil.rmtree(out_dir, ignore_errors=True)
    t = time.perf_counter()
    cmd = [PY, "-m", "pocketdetect.burrow.fast", "predict", f"@{list_file}", "--out", str(out_dir), "--device", device,
           "--workers", str(workers), "--batch", str(batch), "--ranker", "all", "--preset", preset, "--no-points"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    wall = time.perf_counter() - t
    tf = out_dir / "timing.csv"
    rows = tf.read_text().splitlines()[1:] if tf.exists() else []
    n_ok = sum(",ok," in l for l in rows)
    return dict(preset=preset, device=device, workers=workers, batch=batch, n=len(rows), ok=n_ok, wall_s=round(wall, 1),
                structures_per_s=round(n_ok / wall, 1), structures_per_min=round(60 * n_ok / wall),
                stderr_tail=r.stderr.strip().splitlines()[-1:] if r.stderr else [])


T1_CHANGES = [
    "GPU driver: CPU workers start first (forkserver with the pipeline pre-imported); while the GPU process "
    "imports torch, creates the CUDA context and warms up the kernels (~3-5 s), the workers predict whole "
    "structures on the CPU path (identical outputs), so no worker idles; if the input runs out first the GPU is "
    "never touched; if the GPU cannot be initialised the run finishes on the CPU path.",
    "Model load: flattened forests cached on disk (`~/.cache/pocketdetect/fastmodel/<sha256>.npz`, override "
    "`BURROW_FAST_CACHE`), so workers no longer import LightGBM / scikit-learn / pandas (worker start-up "
    "2.5 s -> 0.9 s).",
    "Parsing: `burrow/fast/io.py` builds the Protein from `gemmi.FlatStructure` arrays (vectorised residue "
    "classification, altloc and hydrogen filters; per-(element, atom, residue) chemistry from the reference "
    "functions, cached) - identical to `protein_from_structure(load_structure(...))`; ~3x faster; general "
    "per-atom fallback when a guard fails (8-character names, unexpected residue grouping). AF / `--compact` "
    "runs keep the reference loader (they need the full Structure).",
    "Metal candidates: the clash test uses an any-within kernel instead of building neighbour lists "
    "(5.3 -> 2.9 ms; identical candidates).",
    "GPU neighbour sums: the float64 `wcn` sums are accumulated in the Triton neighbour kernel instead of "
    "torch pair lists (default preset GPU featurisation 4.6 -> 2.5 ms / structure).",
    "Driver bookkeeping: O(1) back-pressure on the post-processing queue (was O(n) per batch).",
]


def write_md(R, path):
    L = ["# BURROW performance — frozen presets (fast backend vs reference)\n",
         f"Generated by `scripts/bench_burrow.py` on {R['date']} (host {R['host']}, AMD EPYC 9554, 4x RTX 4090; "
         f"node load average (1/5/15 min) at start {R['load_start']}, at end {R['load_end']}"
         f"{'; throughput runs ' + R['driver_load_start'] + ' -> ' + R['driver_load_end'] if 'driver_load_start' in R else ''}"
         " — **shared node**: other users' jobs ran "
         "concurrently, so all CPU numbers are pessimistic and noisy). Raw: `results/perf/bench_burrow.json`.\n",
         "Presets (`results/dev/FREEZE.md`): **default** = 30 rays, 6/10/16 Å, 3 Å sub-lattice, score floor 0.7, "
         "250x63 point model (`burrow_all_default_s0`); **accurate** = 60 rays, 6/10/16/24 Å, 2 Å sub-lattice, all "
         "points scored (`burrow_all_accurate_s0`). Structure sets: `speed200` = the 200 LIGYSIS chains timed for "
         "P2Rank/fpocket in `results/baselines/SPEED.md`; `biolip_dev` = 300 random BioLiP dev chains.\n",
         "## Equivalence (fast backend == reference)\n",
         "`scripts/verify_burrow_fast.py` against the frozen v1 reference (worktree at b6a896b): 64 random BioLiP dev "
         "structures per standard-mode run, 40 random AF dev models (`results/dev_af/af_dev_entries.csv`) per AF-mode "
         "run; `exact` = identical scores, centres, residue sets and pocket points for every structure.\n",
         "| model | mode | device | structures | with pLDDT 60/70 atoms (w = 0.5/0.75) | top-10 identical | exact |",
         "|---|---|---|---|---|---|---|"]
    for k, v in sorted(R.get("verify", {}).items()):
        L.append(f"| {v.get('model', k)} | {'AF' if v.get('af_mode') else 'standard'} | {v['device']} | {v['n']} | "
                 f"{v.get('n_with_exact_w075_or_w05_atoms', '–')} | {v['top10_identical']} | {v['exact']} |")
    L.append("\n**Fix 2026-10-26 22:10 (GPU AF mode):** Triton passes Python float kernel arguments as fp32. The "
             "first-hit threshold ln 2 (and the PSP opacity threshold) therefore reached the fp64 ray-march kernel "
             "rounded. Atoms with pLDDT exactly 70 have w = 0.75 and mu = ln 2 (float32), so tau == float32(ln 2) "
             "decided first-hit differently from the CPU reference. depth / hit_* features differed for 6 of 40 human "
             "AF models in AF mode; standard mode was unaffected (w = 1). Both thresholds are now passed as fp64 "
             "tensors, other scalars are checked to be exact in fp32, and a regression test with exact 0.5 / 0.75 "
             "weights was added (`tests/test_burrow_fast_equivalence.py::test_af_soft_weights_identical`). "
             "Re-run of the GPU half on 2026-10-26 23:45 (main 348f26f, physical GPU 1): all 6 GPU equivalence tests "
             "pass (`test_af_soft_weights_identical`, `test_gpu_batch_identical`, `test_gpu_features_identical`; "
             "default and accurate presets). A failure reported earlier came from running on a full GPU 0 (CUDA "
             "out-of-memory), not from a numerical difference.\n")
    L.append("**Fix 2026-10-27 (GPU, `--preset fast`):** the GPU ray march received 1/spacing as a Python float "
             "(fp32 in Triton); 1/1.5 Å is not fp32-exact, so the fast preset (1.5 Å grid) was rejected on GPU and "
             "the CLI fell back to the reference implementation. 1/spacing is now passed as an fp64 tensor (as the "
             "CPU reference computes it). Verified exact vs the reference on 40 random BioLiP dev structures for "
             "`burrow_{all,small}_fast_s0` (GPUs 1/2) and 40 AF dev models in AF mode (`all`, GPU 1); regression "
             "tests: `fast` added to the parametrised equivalence tests (CPU + GPU) and "
             "`test_gpu_march_non_fp32_spacing`.\n")
    t1 = ROOT / "results/perf/bench_t1.md"
    if t1.exists():
        L.append(t1.read_text())
        L.append("**What changed in T1 (all in `burrow/fast/`, outputs byte-identical):**\n")
        L += [f"- {x}" for x in T1_CHANGES]
        L.append("\nRe-verification after T1 (`scripts/verify_burrow_fast.py`, 40 random dev structures / AF dev "
                 "models per run, fast backend vs the reference implementation; `results/perf/verify_t1/`):\n")
        L += ["| model | mode | device | structures | with pLDDT 60/70 atoms | top-10 identical | exact |",
              "|---|---|---|---|---|---|---|"]
        for f in sorted((ROOT / "results/perf/verify_t1").glob("*.json")):
            v = json.loads(f.read_text())["summary"]
            L.append(f"| {v['model']} | {'AF' if v.get('af_mode') else 'standard'} | {v['device']} | {v['n']} | "
                     f"{v.get('n_with_exact_w075_or_w05_atoms', '–')} | {v['top10_identical']} | {v['exact']} |")
        io = ROOT / "results/perf/verify_fast_io.json"
        if io.exists():
            v = json.loads(io.read_text())["summary"]
            L.append(f"\nFast loader: `scripts/verify_burrow_fast_io.py` on {v['n']:,} structures (BioLiP dev, apo-dev, "
                     "AF dev, 1,500 whole PDB entries from the train manifest (mmCIF with ligands, waters, altlocs), "
                     f"1,000 human AFDB models, speed200): every Protein field identical (values and dtypes) for "
                     f"{v['identical']:,} / {v['n']:,}; median {v['ref_ms_median']} -> {v['fast_ms_median']} ms "
                     "(16 parallel processes).")
        e2e = ROOT / "results/perf/e2e_t1.txt"
        if e2e.exists():
            L.append("\nEnd-to-end CLI check (before tree vs after tree, every output file byte-compared, "
                     "timing.csv excluded):\n")
            L.append("```\n" + e2e.read_text().strip() + "\n```\n")
    fin = ROOT / "results/perf/bench_final.md"
    if fin.exists():
        L.append(fin.read_text())
    for preset in ("default", "accurate"):
        P = R["presets"][preset]
        L.append(f"\n## {preset.upper()} preset — 1 core, warm (ms per structure)\n")
        L.append("| set | backend | n | median atoms | total excl. parsing (median / p90) | incl. parsing (median / p90) "
                 "| parse | features | point score | pockets | rank |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for key, s in P["cpu1"].items():
            st, be = key.rsplit("/", 1)
            g = lambda n: s[n]["median"] if s.get(n) else float("nan")
            L.append(f"| {st} | {be} | {s['n']} | {s['median_atoms']:.0f} | {s['total']['median']} / {s['total']['p90']} | "
                     f"{s['total_incl_parse']['median']} / {s['total_incl_parse']['p90']} | {g('parse')} | {g('features')} | "
                     f"{g('point_score')} | {g('pockets')} | {g('rank')} |")
        fs = P["cpu1"].get("speed200/fast", {}).get("stages_median", {})
        if fs:
            L.append("\nFeature sub-stages (fast, speed200, median ms): " + ", ".join(f"{k[5:]} {v}" for k, v in fs.items()))
        L.append("\nCold start (fresh process, first structure): " + "; ".join(
            f"{lab}: startup+model {v['import_and_model_ms']:.0f} ms, first predict {v['first_predict_ms']:.0f} ms"
            for lab, v in P["cold"].items()))
        L.append(f"\n### {preset.upper()} — throughput incl. parsing + writing (`python -m pocketdetect.burrow.fast "
                 f"predict`, speed200 x {R['repeat']}, process start-up included)\n")
        L.append("| device | CPU workers | batch | structures ok | wall s | structures/s | structures/min |")
        L.append("|---|---|---|---|---|---|---|")
        for d in P["driver"]:
            L.append(f"| {d['device']} | {d['workers']} | {d['batch']} | {d['ok']}/{d['n']} | {d['wall_s']} | "
                     f"{d['structures_per_s']} | {d['structures_per_min']} |")
    L.append("\n## Context: baselines on the same 200 chains (results/baselines/SPEED.md, same node, load 20-68)\n")
    L += ["| tool | setting | ms / structure (amortised) |", "|---|---|---|",
          "| P2Rank 2.5.1 | batch, one JVM, 1 thread | 295 |", "| P2Rank 2.5.1 | batch, one JVM, 8 / 32 threads | 49 / 47 |",
          "| fpocket 4.2.3 | 1 process | 620 |", "| fpocket 4.2.3 | 32 processes | 39 |",
          "| fpocket 4.2.3 + PRANK 2.5.1 | sequential | 774 |"]
    L.append("\nFor comparison, BURROW amortised ms/structure = 1000 / (structures/s) of the driver table above.\n")
    L.append("## Notes\n")
    L.extend(f"- {n}" for n in R["notes"])
    Path(path).write_text("\n".join(L) + "\n")


def load_avg():
    return " ".join(f"{x:.0f}" for x in os.getloadavg())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,3")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--n-ref", type=int, default=40)
    ap.add_argument("--out", type=Path, default=ROOT / "results/perf/bench_burrow.json")
    ap.add_argument("--md", type=Path, default=ROOT / "docs/PERF.md")
    ap.add_argument("--md-only", action="store_true")
    ap.add_argument("--only-driver", action="store_true", help="re-run only the throughput section")
    a = ap.parse_args()
    if a.md_only:
        R = json.loads(a.out.read_text())
        R["verify"] = {f.stem: json.loads(f.read_text())["summary"] for f in sorted((ROOT / "results/perf").glob("verify_*.json")) if f.stem != "verify_fast_io"}
        write_md(R, a.md)
        return
    sets = {"speed200": speed200(), "biolip_dev": biolip_dev(300)}
    if a.only_driver:
        R = json.loads(a.out.read_text())
        R["repeat"] = a.repeat
        gpus = [g for g in a.gpus.split(",") if g]
        lst = a.out.parent / "bench_list.txt"
        lst.write_text("\n".join([p for _, p in sets["speed200"]] * a.repeat) + "\n")
        tmp = Path(tempfile.mkdtemp(prefix="burrow_bench_"))
        R["driver_load_start"] = load_avg()
        for preset in PRESETS:
            R["presets"][preset]["driver"] = [
                bench_driver(lst, tmp / f"{preset}_{d.replace(':', '_').replace(',', '_')}", d, w, preset, b)
                for d, w, b in [("cpu", 16, 1), ("cpu", 32, 1)] + ([(f"cuda:{gpus[0]}", 10, 32),
                                                                     ("cuda:" + ",".join(gpus), 10, 32)] if gpus else [])]
            print(preset, R["presets"][preset]["driver"], flush=True)
        R["driver_load_end"] = load_avg()
        shutil.rmtree(tmp, ignore_errors=True)
        lst.unlink()
        R["verify"] = {f.stem: json.loads(f.read_text())["summary"]
                       for f in sorted((ROOT / "results/perf").glob("verify_*.json")) if f.stem != "verify_fast_io"}
        a.out.write_text(json.dumps(R, indent=1))
        write_md(R, a.md)
        print("done", flush=True)
        return
    R = dict(date=time.strftime("%Y-%m-%d %H:%M"), host=platform.node(), load_start=load_avg(), repeat=a.repeat,
             presets={}, notes=[])
    gpus = [g for g in a.gpus.split(",") if g]
    lst = a.out.parent / "bench_list.txt"
    lst.write_text("\n".join([p for _, p in sets["speed200"]] * a.repeat) + "\n")
    tmp = Path(tempfile.mkdtemp(prefix="burrow_bench_"))
    for preset, mp in PRESETS.items():
        P = R["presets"][preset] = dict(cpu1={}, driver=[])
        for sname, paths in sets.items():
            print(preset, sname, "fast", flush=True)
            P["cpu1"][f"{sname}/fast"] = summarize(bench_inprocess(paths, mp, "fast"))
        print(preset, "reference", flush=True)
        P["cpu1"][f"speed200/reference (first {a.n_ref})"] = summarize(bench_inprocess(sets["speed200"][:a.n_ref], mp, "reference"))
        P["cpu1"][f"speed200/fast (first {a.n_ref})"] = summarize(bench_inprocess(sets["speed200"][:a.n_ref], mp, "fast"))
        print(preset, "cold", flush=True)
        P["cold"] = bench_cold(sets["speed200"][0][1], mp)
        configs = [("cpu", 16, 1), ("cpu", 32, 1)]
        if gpus:
            configs += [(f"cuda:{gpus[0]}", 10, 32), ("cuda:" + ",".join(gpus), 10, 32)]
        for dev, w, b in configs:
            print(preset, "driver", dev, w, flush=True)
            P["driver"].append(bench_driver(lst, tmp / f"{preset}_{dev.replace(':', '_').replace(',', '_')}_{w}", dev, w,
                                            preset, b))
            print(P["driver"][-1], flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    lst.unlink()
    R["load_end"] = load_avg()
    R["verify"] = {f.stem: json.loads(f.read_text())["summary"] for f in sorted((ROOT / "results/perf").glob("verify_*.json")) if f.stem != "verify_fast_io"}
    R["notes"] = [
        "Fast backend: exact re-implementation (Numba formation / residues / descriptors / NMS / metal candidates & "
        "geometry, LightGBM-free forests, fp64-exact Triton kernels on GPU). It reuses the reference Numba ray "
        "march and neighbour sums on CPU (exact re-implementations were not faster).",
        "GPU mode: one process per GPU, each with --workers single-threaded CPU processes for parsing, host prep "
        "(canonical frame, metal candidates) and post-processing (formation, ranking, NMS, writing). The GPUs are "
        "not the bottleneck (~6 ms of GPU time per structure); throughput is bound by the CPU workers.",
        "Worker budget: <= 32 single-threaded processes (shared 4096-thread user limit); 3 GPUs x 10 workers = 30.",
        "`reference` rows are the committed `burrow.pipeline.predict_protein` on the same structures (paired).",
        "Cold start: `cold_jit_compile` = empty Numba cache (first use after install / code change); `cached_jit` and "
        "`installed_cache` = normal start of a new process.",
    ]
    a.out.write_text(json.dumps(R, indent=1))
    write_md(R, a.md)
    print("done", flush=True)


if __name__ == "__main__":
    main()
