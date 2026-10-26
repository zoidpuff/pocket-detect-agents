#!/usr/bin/env python
"""Benchmark the geometry primitives (CPU 1/N threads, GPU single + batched, JIT warm-up).

Usage::

    .venv/bin/python scripts/bench_geometry.py [--threads 64] [--out results/geometry_bench.json]
                                               [--md docs/GEOMETRY_BENCH.md]

Structures are read from the local wwPDB mirror with gemmi (first model, protein
polymer heavy atoms, altloc 'first', no waters/ligands).  All timings are wall
clock minimum over repeats, in ms, excluding file parsing (reported separately).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

PDB = "/cluster/work/beltrao/shared/rsyncPDB/data/structures/divided/mmCIF"
IDS = ["1crn", "1ubq", "1stp", "3ptb", "1hsg", "4dfr", "1ake", "1hvr", "1m17", "6lu7", "1a28", "2rh1",
       "3eml", "4hhb", "1fin", "2hyy", "1sqn", "3pbl", "1jff", "1aon"]


def load(pid):
    import gemmi
    t = time.perf_counter()
    st = gemmi.read_structure(f"{PDB}/{pid[1:3]}/{pid}.cif.gz")
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    xyz, el, nm, rn = [], [], [], []
    nres = 0
    for ch in st[0]:
        for res in ch:
            if not gemmi.find_tabulated_residue(res.name) or not gemmi.find_tabulated_residue(res.name).is_amino_acid():
                continue
            nres += 1
            for a in res:
                xyz.append(a.pos.tolist())
                el.append(a.element.name)
                nm.append(a.name)
                rn.append(res.name)
    from pocketdetect.geometry import atom_class_matrix, vdw_radii
    el = np.array(el)
    props = atom_class_matrix(el, np.array(nm), np.array(rn))[0]
    return dict(id=pid, coords=np.array(xyz, np.float32), radii=vdw_radii(el), props=props, n_res=nres,
                load_ms=(time.perf_counter() - t) * 1e3)


def tmin(f, reps):
    best = np.inf
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t)
    return best * 1e3, out


def cpu_primitives(s, reps):
    """Per-primitive and pipeline timings (ms).  Primitives run in the input frame."""
    import pocketdetect.geometry as g
    c, r, p = s["coords"], s["radii"], s["props"]
    cfg = g.FeatureConfig()
    res = {}
    res["sas"], sas = tmin(lambda: g.sas_points(c, r, cfg.probe, cfg.n_sphere), reps)
    res["grid"], grid = tmin(lambda: g.grid_candidates(c, r, cfg.grid_spacing, cfg.probe, cfg.grid_max_dist), reps)
    res["occ"], occ = tmin(lambda: g.occupancy_grid(c, r, None, cfg.occ_spacing), reps)
    res["psp_grid"], _ = tmin(lambda: g.psp_counts(grid.points, occ, cfg.psp_length, cfg.psp_dirs), reps)
    pts = sas.points
    res["rays_sas"], b = tmin(lambda: g.ray_buriedness(pts, grid=occ, n_rays=cfg.n_rays, lengths=cfg.ray_lengths,
                                                       step=cfg.ray_step), reps)
    res["neighbors_sas"], _ = tmin(lambda: g.neighbor_features(pts, c, p, cfg.nb_radii, cfg.sigmas), reps)
    buried = pts[b.blocked[:, 1] > 0.6]
    res["cluster_sl"], _ = tmin(lambda: g.single_linkage(buried, 2.0), reps)
    for mode in ("sas", "cascade"):
        mcfg = g.FeatureConfig(points=mode)
        best = None
        for _ in range(reps):
            f = g.compute_point_features(c, r, p, None, mcfg)
            if best is None or f.timings["total"] < best.timings["total"]:
                best = f
        res[f"pipe_{mode}"] = best.timings["total"]
        res[f"pipe_{mode}_stages"] = {k: round(v, 2) for k, v in best.timings.items() if k != "n_bricks"}
        res[f"n_{mode}"] = int(len(best.points))
    res["n_grid"] = int(len(grid.points))
    return res


def gpu_primitives(s, reps):
    import torch
    import pocketdetect.geometry as g
    dev = torch.device("cuda")
    cfg = g.FeatureConfig()
    c = torch.from_numpy(s["coords"]).to(dev)
    r = torch.from_numpy(s["radii"]).to(dev)
    p = torch.from_numpy(s["props"]).to(dev)

    def tm(f):
        best = np.inf
        out = None
        for _ in range(reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = f()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
        return best * 1e3, out
    res = {}
    res["sas"], sas = tm(lambda: g.sas_points_torch(c, r, None, cfg.probe, cfg.n_sphere))
    res["grid"], grid = tm(lambda: g.grid_candidates_torch(c, r, None, cfg.grid_spacing, cfg.probe, cfg.grid_max_dist))
    res["occ"], occ = tm(lambda: g.TorchOccupancyGrid(c, r, None, None, cfg.occ_spacing))
    res["psp_grid"], _ = tm(lambda: g.psp_counts_torch(grid["points"], None, occ, cfg.psp_length, cfg.psp_dirs))
    pts = sas["points"]
    res["rays_sas"], b = tm(lambda: g.ray_buriedness_torch(pts, None, grid=occ, n_rays=cfg.n_rays,
                                                           lengths=cfg.ray_lengths, step=cfg.ray_step))
    res["neighbors_sas"], _ = tm(lambda: g.neighbor_features_torch(pts, c, p, None, None, cfg.nb_radii, cfg.sigmas))
    buried = pts[b["blocked"][:, 1] > 0.6]
    res["cluster_sl"], _ = tm(lambda: g.single_linkage_torch(buried, 2.0))
    one = [dict(coords=s["coords"], radii=s["radii"], props=s["props"])]
    for mode in ("sas", "cascade"):
        res[f"pipe_{mode}"], _ = tm(lambda: g.compute_point_features_batch_torch(one, g.FeatureConfig(points=mode)))
    return res


# ----------------------------------------------------------------------------- throughput
_WORK = None
_MODE = "sas"


def _worker_init(structs, mode):
    os.environ["NUMBA_NUM_THREADS"] = "1"
    import numba
    numba.set_num_threads(1)
    global _WORK, _MODE
    _WORK, _MODE = structs, mode


def _worker(i):
    import pocketdetect.geometry as g
    s = _WORK[i % len(_WORK)]
    g.compute_point_features(s["coords"], s["radii"], s["props"], None, g.FeatureConfig(points=_MODE))
    return i


def cpu_throughput(pool_structs, n_items, n_proc, mode):
    """Structures/s with ``n_proc`` single-threaded worker processes (spawned: no fork after OpenMP)."""
    ctx = mp.get_context("spawn")
    # single-threaded BLAS/OpenMP/numba in the workers (inherited env; else 64 x 64 threads > RLIMIT_NPROC)
    single = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}
    saved = {k: os.environ.get(k) for k in single}
    os.environ.update(single)
    try:
        with ctx.Pool(n_proc, initializer=_worker_init, initargs=(pool_structs, mode)) as pool:
            pool.map(_worker, range(n_proc), chunksize=1)          # import + JIT cache load in every worker
            t = time.perf_counter()
            pool.map(_worker, range(n_items), chunksize=1)
            dt = time.perf_counter() - t
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return n_items / dt


def gpu_throughput(structs, batch_size, mode):
    import torch
    import pocketdetect.geometry as g
    cfg = g.FeatureConfig(points=mode)
    g.compute_point_features_batch_torch(structs[:batch_size], cfg)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(0, len(structs), batch_size):
        g.compute_point_features_batch_torch(structs[i:i + batch_size], cfg)   # includes H2D + D2H copies
    torch.cuda.synchronize()
    return len(structs) / (time.perf_counter() - t)


def rotation_check(structs, n_rot=3):
    """Mean |Δ buriedness(12 Å)| between rotated copies: lab-frame rays vs canonical-frame pipeline."""
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    import pocketdetect.geometry as g
    rots = Rotation.random(n_rot, random_state=0).as_matrix()
    raw, canon = [], []
    for s in structs:
        c, r = s["coords"], s["radii"]
        pts = g.sas_points(c, r).points
        b = [g.ray_buriedness((pts @ M.T).astype(np.float32), (c @ M.T).astype(np.float32), r).blocked[:, 1]
             for M in rots]
        raw += [np.abs(b[0] - b[i]).mean() for i in range(1, n_rot)]
        f = [g.compute_point_features((c @ M.T).astype(np.float32), r) for M in rots]
        for i in range(1, n_rot):
            d, j = cKDTree(f[0].points @ rots[0] @ rots[i].T).query(f[i].points)
            ok = d < 1e-2
            canon.append(np.abs(f[i].blocked[ok, 1] - f[0].blocked[j[ok], 1]).mean())
    return {"lab-frame rays, same physical points (mean |Δb12|)": round(float(np.mean(raw)), 4),
            "canonical-frame pipeline (mean |Δb12|)": float(f"{np.mean(canon):.2e}")}


def huge_check(ids, threads):
    import numba
    import resource
    import torch
    import pocketdetect.geometry as g
    rows = []
    numba.set_num_threads(threads)
    for pid in ids:
        s = load(pid)
        row = dict(id=pid, n_atoms=int(len(s["coords"])), n_res=s["n_res"],
                   extent=[round(float(x)) for x in np.ptp(s["coords"], 0)], load_s=round(s["load_ms"] / 1e3, 1))
        for mode in ("sas", "cascade"):
            cfg = g.FeatureConfig(points=mode)
            t = time.perf_counter()
            f = g.compute_point_features(s["coords"], s["radii"], s["props"], None, cfg)
            row[f"cpu{threads}_{mode}_s"] = round(time.perf_counter() - t, 2)
            row[f"bricks_{mode}"] = f.timings["n_bricks"]
            row[f"n_{mode}"] = int(len(f.points))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                g.compute_point_features_batch_torch([s], cfg)
                torch.cuda.synchronize()
                row[f"gpu_{mode}_s"] = round(time.perf_counter() - t, 2)
                row[f"gpu_{mode}_peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        row["cpu_maxrss_GB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
        rows.append(row)
        print(row, flush=True)
    return rows


# ----------------------------------------------------------------------------- JIT warm-up
WARM_SNIPPET = r"""
import time, json, numpy as np
t0 = time.perf_counter()
import pocketdetect.geometry as g
t1 = time.perf_counter()
rng = np.random.default_rng(0)
c = rng.uniform(0, 20, (500, 3)).astype(np.float32); r = np.full(500, 1.7, np.float32)
out = {"import_ms": (t1 - t0) * 1e3}
for name, f in [("sas", lambda: g.sas_points(c, r)), ("grid", lambda: g.grid_candidates(c, r)),
                ("buriedness", lambda: g.ray_buriedness(c[:10], c, r)),
                ("psp", lambda: g.psp_counts(c[:10], g.occupancy_grid(c, r))),
                ("neighbors", lambda: g.neighbor_features(c[:10], c, None, sigmas=(2, 4))),
                ("cluster", lambda: (g.single_linkage(c, 2.0), g.mean_shift(c[:50], 3.0))),
                ("pipeline_cascade", lambda: g.compute_point_features(c, r, None, None, g.FeatureConfig(points="cascade")))]:
    t = time.perf_counter(); f(); out[name + "_ms"] = (time.perf_counter() - t) * 1e3
if GPU:
    import torch
    t = time.perf_counter(); torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    out["cuda_init_ms"] = (time.perf_counter() - t) * 1e3
    s = [dict(coords=c, radii=r, props=np.ones((500, 3), np.float32))]
    t = time.perf_counter(); g.compute_point_features_batch_torch(s, g.FeatureConfig(points="cascade")); torch.cuda.synchronize()
    out["gpu_pipeline_first_call_ms"] = (time.perf_counter() - t) * 1e3
print(json.dumps(out))
"""


def jit_warmup(gpu: bool):
    res = {}
    with tempfile.TemporaryDirectory() as nc, tempfile.TemporaryDirectory() as tc:
        env = dict(os.environ, NUMBA_CACHE_DIR=nc, TRITON_CACHE_DIR=tc)
        code = "GPU = %s\n" % gpu + WARM_SNIPPET
        for label in ("cold", "cached"):
            t = time.perf_counter()
            out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
            res[label] = json.loads(out.stdout.strip().splitlines()[-1])
            res[label]["process_wall_ms"] = (time.perf_counter() - t) * 1e3
    return res


# ----------------------------------------------------------------------------- report
PRIMS = ["sas", "grid", "occ", "psp_grid", "rays_sas", "neighbors_sas", "cluster_sl", "pipe_sas", "pipe_cascade"]


def write_md(R, path):
    L = []
    hw = R["hardware"]
    L.append("# Geometry primitives benchmark\n")
    L.append(f"Generated by `scripts/bench_geometry.py` on {R['date']}. CPU: {hw['cpu']} "
             f"({hw['cpus_affinity']} cores in our affinity), GPU: {hw['gpu']}; numba threading layer "
             f"`{hw['threading_layer']}`. Raw numbers: `{R['json_path']}`.\n")
    L.append("## API in one paragraph\n")
    L.append("`pocketdetect.geometry`: `sas_points`, `grid_candidates`, `occupancy_grid` + `ray_buriedness` "
             "(multi-scale, soft occlusion via per-atom weights), `psp_counts` (LIGSITE prefilter), "
             "`cascade_mask`, `neighbor_features` / `protrusion` (radius + Gaussian sums of any (N,F) atom "
             "property matrix, e.g. `atom_class_matrix`), `smooth_point_values`, `single_linkage` / "
             "`mean_shift` / `quick_shift`, `canonical_frame`, `brick_boxes`; each has a `*_torch` twin taking "
             "a `batch` vector (Triton fast path, PyTorch fallback). `compute_point_features` (CPU) and "
             "`compute_point_features_batch_torch` (GPU, many structures per call) run the whole pipeline "
             "configured by `FeatureConfig`. Tests: `tests/test_geometry_*.py`.\n")
    L.append("**Caveat:** the node is shared (other agents/jobs run concurrently on the same cores), so "
             "multi-threaded CPU timings are noisy; all values are the minimum over repeats. Structure parsing "
             "(gemmi) is excluded and listed separately (`load_ms` in the JSON).\n")
    L.append("Settings = `FeatureConfig` defaults: " + ", ".join(f"`{k}={v}`" for k, v in R["config"].items()) + ".\n")
    L.append("Columns: `sas` SAS points (30/atom); `grid` solvent-shell grid candidates (1 Å, <= 5 Å from an "
             "atom centre, probe 1.4 Å); `occ` absorption grid; `psp_grid` LIGSITE 7-direction PSP count on all "
             "grid points; `rays_sas` 60-ray buriedness at L = 8/12/20 Å on the SAS points; `neighbors_sas` "
             "13 atom classes x (5 radii + 2 Gaussians) on the SAS points; `cluster_sl` single linkage (2 Å) of "
             "SAS points with b12 > 0.6. `pipe_sas` / `pipe_cascade` = complete per-point feature pipeline "
             "(`compute_point_features`, canonical PCA frame, points + PSP + rays + neighbours); "
             "cascade = grid shell -> PSP >= 2 or 2 Å sub-lattice -> rays/neighbours on survivors.\n")
    for key, title in R["tables"]:
        L.append(f"\n## {title} — ms per structure\n")
        L.append("| PDB | res | atoms | SAS pts | grid pts | cascade pts | " + " | ".join(PRIMS) + " |")
        L.append("|---|---|---|---|---|---|" + "---|" * len(PRIMS))
        for s in R["structures"]:
            row = s.get(key)
            if row is None:
                continue
            c1 = s["cpu1"]
            L.append(f"| {s['id']} | {s['n_res']} | {s['n_atoms']} | {c1['n_sas']} | {c1['n_grid']} | "
                     f"{c1['n_cascade']} | " + " | ".join(f"{row[p]:.1f}" for p in PRIMS) + " |")
    L.append("\n## Cascade pipeline stages, CPU 1 thread (ms)\n")
    L.append("| PDB | res | points | prefilter (occ+PSP) | buriedness | neighbors | total |")
    L.append("|---|---|---|---|---|---|---|")
    for s in R["structures"]:
        st = s["cpu1"]["pipe_cascade_stages"]
        L.append(f"| {s['id']} | {s['n_res']} | {st['points']:.1f} | {st['prefilter']:.1f} | "
                 f"{st['buriedness']:.1f} | {st['neighbors']:.1f} | {st['total']:.1f} |")
    L.append("\n## Batched GPU and throughput\n")
    for k, v in R["batched"].items():
        L.append(f"- {k}: {v}")
    L.append("\n## Rotation invariance (3ptb, 1a28, 6lu7; 3 random rotations)\n")
    for k, v in R["rotation"].items():
        L.append(f"- {k}: {v}")
    L.append("\n## Huge assemblies (bricking; CPU %d threads)\n" % R["huge_threads"])
    if R["huge"]:
        keys = list(R["huge"][0].keys())
        L.append("| " + " | ".join(keys) + " |")
        L.append("|" + "---|" * len(keys))
        for row in R["huge"]:
            L.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
    L.append("\n## JIT / warm-up (fresh subprocess, ms)\n")
    for k, v in R["jit"].items():
        L.append(f"- {k}: " + ", ".join(f"{a}={b:.0f}" for a, b in v.items()))
    L.append("\n## Notes\n")
    L.extend(f"- {n}" for n in R["notes"])
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results/geometry_bench.json")
    ap.add_argument("--md", default="docs/GEOMETRY_BENCH.md")
    ap.add_argument("--n-throughput", type=int, default=512)
    ap.add_argument("--huge", nargs="*", default=["1aon", "4v6x", "3j3q"])
    ap.add_argument("--md-only", action="store_true", help="re-render the markdown from an existing JSON")
    args = ap.parse_args()
    if args.md_only:
        write_md(json.load(open(args.out)), args.md)
        return

    import numba
    import torch
    import pocketdetect.geometry as g
    gpu = torch.cuda.is_available()

    structs = [load(p) for p in IDS]
    s0 = structs[0]
    for mode in ("sas", "cascade"):
        g.compute_point_features(s0["coords"], s0["radii"], s0["props"], None, g.FeatureConfig(points=mode))
    R = dict(date=time.strftime("%Y-%m-%d %H:%M"), json_path=args.out,
             config={k: v for k, v in g.FeatureConfig().__dict__.items()}, structures=[], batched={}, notes=[])
    R["hardware"] = dict(cpu=platform.processor() or "AMD EPYC 9554", cpus_affinity=len(os.sched_getaffinity(0)),
                         gpu=torch.cuda.get_device_name(0) if gpu else "none",
                         threading_layer=numba.threading_layer())
    R["tables"] = [("cpu1", "CPU, 1 thread")] + [(f"cpu{t}", f"CPU, {t} threads") for t in args.threads]
    if gpu:
        R["tables"].append(("gpu", "GPU, one structure per call (Triton fast path)"))
        cpu_primitives(s0, 1)
        gpu_primitives(s0, 1)
    for s in structs:
        e = dict(id=s["id"], n_res=s["n_res"], n_atoms=int(len(s["coords"])), load_ms=round(s["load_ms"], 1))
        for t in [1] + args.threads:
            numba.set_num_threads(t)
            e[f"cpu{t}"] = cpu_primitives(s, args.reps)
        if gpu:
            e["gpu"] = gpu_primitives(s, 5)
        print(s["id"], e["n_atoms"], {k: round(v, 1) for k, v in e["cpu1"].items() if not isinstance(v, dict)},
              {k: round(v, 1) for k, v in e.get("gpu", {}).items()}, flush=True)
        R["structures"].append(e)
    numba.set_num_threads(1)

    feats = [dict(coords=s["coords"], radii=s["radii"], props=s["props"]) for s in structs]
    if gpu:
        for mode in ("sas", "cascade"):
            cfg = g.FeatureConfig(points=mode)
            g.compute_point_features_batch_torch(feats, cfg)
            torch.cuda.synchronize()
            best = np.inf
            for _ in range(3):
                t = time.perf_counter()
                g.compute_point_features_batch_torch(feats, cfg)
                torch.cuda.synchronize()
                best = min(best, time.perf_counter() - t)
            R["batched"][f"GPU, all 20 structures in one call, {mode} (ms)"] = round(best * 1e3, 1)
    pool = [f for f, s in zip(feats, structs) if s["id"] != "1aon"]
    work = [pool[i % len(pool)] for i in range(args.n_throughput)]
    mean_res = np.mean([s["n_res"] for s in structs if s["id"] != "1aon"])
    R["batched"]["throughput set"] = (f"{len(work)} structures (the 19 entries without 1aon, repeated; mean "
                                      f"{mean_res:.0f} residues); parsing excluded")
    for mode in ("sas", "cascade"):
        if gpu:
            for bs in (16, 64, 256):
                sps = gpu_throughput(work, bs, mode)
                R["batched"][f"GPU throughput {mode}, batch={bs} (structures/s, 1 GPU)"] = round(sps, 1)
                print("gpu", mode, bs, sps, flush=True)
        sps = cpu_throughput(pool, len(work), max(args.threads), mode)
        R["batched"][f"CPU throughput {mode}, {max(args.threads)} processes x 1 thread (structures/s)"] = round(sps, 1)
        print("cpu", mode, sps, flush=True)
    R["rotation"] = rotation_check([s for s in structs if s["id"] in ("3ptb", "1a28", "6lu7")])
    R["huge_threads"] = 16
    R["huge"] = huge_check(args.huge, 16)
    R["jit"] = jit_warmup(gpu)
    R["notes"] = [
        "GPU pipeline rows include host->device copy of the atoms, PCA framing on the host and device->host "
        "copy of all features; per-primitive GPU rows start from device tensors.",
        "GPU fast path = Triton kernels (SAS burial test, voxel stamping, PSP, ray marching, neighbour "
        "accumulation); pure-PyTorch fallbacks (use_triton=False) agree with them (tested) but are 5-150x slower.",
        "Single-structure GPU calls are launch/sync-latency bound; batching many structures per call is how "
        "the GPU reaches its throughput.",
        "Rays use step = voxel = 1 Å by default (review round 1); the occlusion radius is the bare vdW radius "
        "(`radius_pad=0`), to be tuned on dev data.",
        "numba kernels are compiled with cache=True: the 'cold' JIT cost is paid once per installation "
        "(or after source changes); 'cached' is the per-process start-up cost afterwards.",
        "CPU multi-thread scaling is limited by the shared node (other jobs on the same cores) and OpenMP "
        "barrier spinning at 64 threads; for throughput use one single-threaded process per core.",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(R, open(args.out, "w"), indent=1, default=float)
    write_md(R, args.md)


if __name__ == "__main__":
    main()
