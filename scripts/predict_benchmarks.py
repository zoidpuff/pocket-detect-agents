#!/usr/bin/env python
"""Produce BURROW predictions on all pre-registered test benchmarks for a frozen model set.
NEVER run before the feature freeze (PREREG test gating). Scoring is done separately by
scripts/run_prereg_analysis.py (which calls the gated scripts/run_test_eval.py).

  nohup .venv/bin/python scripts/predict_benchmarks.py --version v1 --freeze results/dev/biolip/FREEZE.md \
      [--presets default,accurate] [--seeds 0,1,2] [--benchmarks all] [--device cuda:0,1,3] > logs/predict_v1.log 2>&1 &
  .venv/bin/python scripts/predict_benchmarks.py --version v1 --freeze ... --plan     # print the plan only

PREREG mapping (DESIGN §2/§9): ranker `all` for LIGYSIS full, AF set, dimers; `small` for LIGYSIS_NI,
COACH420(mlig), HOLO4K(mlig), CryptoBench, PocketWatch; `--af-mode` (+ AF ranker) for PocketWatch and
the AF set. Preset `default` is primary, `accurate` secondary. One run per training seed SEEDS[:3]
(model files burrow_<ranker>_<preset>_s<i>.json listed with sha256 in FREEZE.md).

Inputs are identical to the baselines' (baselines/bench_inputs.py → baselines/outputs/inputs/<bench>;
dimers: data/processed/dimer/inputs/assembly). LIGYSIS inputs are the authors' TRANS-frame chains; the
predictions are mapped to the asymmetric-unit frame exactly as for the baselines.

Output: results/test/<version>/preds/<bench>/<variant>/{raw/, preds/[, preds_orig/], run.json}
        results/test/<version>/preds/index.json  (bench -> variant -> scored prediction dir, models, timings)
        results/test/<version>/preds/TIMINGS.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketdetect.seeds import SEEDS  # noqa: E402

PY = str(ROOT / ".venv/bin/python")
INPUTS = ROOT / "baselines/outputs/inputs"
# benchmark -> (input dir, ranker, af_mode, prediction-file suffix)
PLAN = {
    "ligysis": (INPUTS / "ligysis", "all", False, ""),
    "ligysis_ni": (INPUTS / "ligysis", "small", False, ""),
    "coach420_mlig": (INPUTS / "coach420_mlig", "small", False, ".pdb"),
    "holo4k_mlig": (INPUTS / "holo4k_mlig", "small", False, ".pdb"),
    "cryptobench": (INPUTS / "cryptobench", "small", False, ""),
    "pocketwatch": (INPUTS / "pocketwatch", "small", True, ""),
    "af_set": (INPUTS / "af_set", "all", True, ""),
    "dimer": (ROOT / "data/processed/dimer/inputs/assembly", "all", False, ""),
}
MODEL_RE = re.compile(r"burrow_(all|small)_(default|accurate)_s(\d+)\.json$")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_freeze(path: Path) -> dict:
    """Model files listed in FREEZE.md as `<path> ... <sha256>` (any markdown layout); every listed
    file must exist and match its checksum. Returns {"models": {(ranker,preset,seed): path},
    "af_rankers": {(preset,seed)|None: path}, "sha256": {path: sha}}."""
    txt = path.read_text()
    found = {}
    for line in txt.splitlines():
        paths = re.findall(r"[\w./+-]+\.json", line)
        shas = re.findall(r"\b[0-9a-f]{64}\b", line)
        for p in paths:
            found[p] = shas[0] if shas else None
    models, afr, shas = {}, {}, {}
    for p, sh in found.items():
        fp = (ROOT / p) if not Path(p).is_absolute() else Path(p)
        if not fp.exists():
            continue
        actual = sha256(fp)
        if sh is None or sh != actual:
            raise SystemExit(f"FREEZE.md: checksum missing or mismatching for {p} (listed {sh}, actual {actual})")
        shas[str(fp)] = actual
        m = MODEL_RE.search(fp.name)
        if m:
            models[(m.group(1), m.group(2), int(m.group(3)))] = fp
        elif fp.name.startswith("afr_"):
            ms = re.search(r"_s(\d+)", fp.name)
            mp = re.search(r"(default|accurate)", fp.name)
            afr[(mp.group(1) if mp else None, int(ms.group(1)) if ms else None)] = fp
    if not models:
        raise SystemExit(f"no burrow_<ranker>_<preset>_s<i>.json models found in {path}")
    return {"models": models, "af_rankers": afr, "sha256": shas}


def pick_af_ranker(afr: dict, preset: str, seed: int):
    for k in ((preset, seed), (None, seed), (preset, None), (None, None)):
        if k in afr:
            return afr[k]
    return None


CODE = {"root": ROOT, "env": dict(os.environ)}  # code that runs BURROW (frozen worktree via --code-root)


def set_code_root(root: Path) -> None:
    CODE["root"] = root
    CODE["env"] = {**os.environ, "PYTHONPATH": str(root / "src")}


def fast_supports_af_ranker() -> bool:
    r = subprocess.run([PY, "-m", "pocketdetect.burrow.fast", "predict", "-h"], cwd=CODE["root"], env=CODE["env"], capture_output=True, text=True)
    return "--af-ranker" in r.stdout


def run_burrow(inputs: Path, out: Path, model: Path, ranker: str, af: bool, af_ranker, device: str, workers: int,
               backend: str, log) -> tuple[float, int]:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if backend == "fast":
        cmd = [PY, "-m", "pocketdetect.burrow.fast", "predict", str(inputs), "--out", str(out), "--model", str(model),
               "--ranker", ranker, "--device", device, "--workers", str(workers), "--no-points"]
        if af:
            cmd.append("--af-mode")
            if af_ranker is not None:
                cmd += ["--af-ranker", str(af_ranker)]
        rc = subprocess.run(cmd, cwd=CODE["root"], env=CODE["env"], stdout=log, stderr=log).returncode
    else:  # reference CLI, sharded over processes (needed when the fast backend lacks the AF ranker)
        files = sorted(str(p) for p in inputs.iterdir() if not p.name.startswith("."))
        shards = [files[i::workers] for i in range(workers)]
        procs = []
        for i, sh in enumerate(shards):
            if not sh:
                continue
            lst = out / f".shard_{i}.txt"
            lst.write_text("\n".join(sh) + "\n")
            sub = out / f".shard_{i}"
            cmd = [PY, "-m", "pocketdetect.burrow", "predict", f"@{lst}", "--out", str(sub), "--model", str(model), "--ranker", ranker,
                   "--threads", "1", "--no-points"]
            if af:
                cmd.append("--af-mode")
                if af_ranker is not None:
                    cmd += ["--af-ranker", str(af_ranker)]
            procs.append(subprocess.Popen(cmd, cwd=CODE["root"], stdout=log, stderr=log, env={**CODE["env"], "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}))
        rc = max(p.wait() for p in procs) if procs else 0
        timing = ["stem,status,n_atoms,n_pockets,total_ms,parse_ms,note"]
        for i in range(workers):
            sub = out / f".shard_{i}"
            if not sub.exists():
                continue
            for f in sub.glob("*.csv"):
                if f.name == "timing.csv":
                    timing += f.read_text().splitlines()[1:]
                else:
                    shutil.move(str(f), out / f.name)
            shutil.rmtree(sub)
            (out / f".shard_{i}.txt").unlink(missing_ok=True)
        (out / "timing.csv").write_text("\n".join(timing) + "\n")
    return time.time() - t0, rc


def finalize(bench: str, vdir: Path, suffix: str) -> Path:
    """raw/<stem>.csv -> preds/<structure_id>.csv (+ LIGYSIS frame mapping). Returns the scored dir."""
    raw, pdir = vdir / "raw", vdir / "preds"
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir()
    for f in raw.glob("*.csv"):
        if f.name == "timing.csv" or f.name.endswith(".points.csv"):
            continue
        stem = f.stem
        if bench == "af_set" and stem.endswith("-model_v6"):
            stem = stem[: -len("-model_v6")]
        shutil.copy(f, pdir / f"{stem}{suffix}.csv")
    if bench in ("ligysis", "ligysis_ni"):
        from pocketdetect.ligysis import trans_preds_to_original

        odir = vdir / "preds_orig"
        if odir.exists():
            shutil.rmtree(odir)
        trans_preds_to_original(pdir, odir)
        return odir
    return pdir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--freeze", type=Path, default=None, help="FREEZE.md (frozen BURROW models); or use --prefix/--model-all/--model-small")
    ap.add_argument("--prefix", default=None, help="descriptive model set: variant = <prefix>_<ranker>_s0[_af] (preset label 'default')")
    ap.add_argument("--model-all", type=Path, default=None)
    ap.add_argument("--model-small", type=Path, default=None)
    ap.add_argument("--presets", default="default,accurate")
    ap.add_argument("--seeds", default="0,1,2", help="indices into SEEDS (model files _s<i>)")
    ap.add_argument("--benchmarks", default="all")
    ap.add_argument("--device", default="cuda:0,1,3")
    ap.add_argument("--af-workers", type=int, default=32, help="CPU workers for --af-mode runs on cpu")
    ap.add_argument("--af-device", default="cpu", help="device for --af-mode runs (v1: the GPU AF path is not exact -> cpu)")
    ap.add_argument("--redo", default="", help="regex over '<bench>/<variant>': discard and regenerate matching runs")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--backend", choices=["auto", "fast", "reference"], default="auto",
                    help="auto: fast backend unless AF-mode needs an AF ranker the fast CLI cannot take")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--code-root", type=Path, default=ROOT, help="git worktree whose src/ runs BURROW (frozen commit); must be clean")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._+-]+", a.version) or a.version.startswith("_"):
        raise SystemExit("bad version name")
    if a.prefix:
        if not (a.model_all and a.model_small):
            raise SystemExit("--prefix needs --model-all and --model-small")
        fz = {"models": {("all", "default", 0): a.model_all.resolve(), ("small", "default", 0): a.model_small.resolve()}, "af_rankers": {},
              "sha256": {str(a.model_all.resolve()): sha256(a.model_all), str(a.model_small.resolve()): sha256(a.model_small)}}
        a.presets, a.seeds = "default", "0"
    elif a.freeze:
        fz = parse_freeze(a.freeze)
    else:
        raise SystemExit("--freeze or --prefix required")
    set_code_root(a.code_root.resolve())
    code_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE["root"], capture_output=True, text=True).stdout.strip()
    code_dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=CODE["root"], capture_output=True, text=True).stdout.strip()
    if a.code_root.resolve() != ROOT and code_dirty:
        raise SystemExit(f"code root {a.code_root} has modified tracked files:\n{code_dirty}")
    chk = subprocess.run([PY, "-c", "import pocketdetect, sys; print(pocketdetect.__file__)"], cwd=CODE["root"], env=CODE["env"], capture_output=True, text=True).stdout.strip()
    if not chk.startswith(str(CODE["root"])):
        raise SystemExit(f"pocketdetect resolves to {chk}, not inside {CODE['root']}")
    seeds = [int(s) for s in a.seeds.split(",")]
    benches = list(PLAN) if a.benchmarks == "all" else a.benchmarks.split(",")
    fast_af = fast_supports_af_ranker()
    jobs = []
    for preset in a.presets.split(","):
        for seed in seeds:
            for b in benches:
                inp, ranker, af, suffix = PLAN[b]
                model = fz["models"].get((ranker, preset, seed))
                if model is None:
                    raise SystemExit(f"FREEZE.md lacks burrow_{ranker}_{preset}_s{seed}.json")
                afr = pick_af_ranker(fz["af_rankers"], preset, seed) if af else None
                backend = a.backend
                if backend == "auto":
                    backend = "reference" if (af and afr is not None and not fast_af) else "fast"
                variant = (f"{a.prefix}_{ranker}_s{seed}" if a.prefix else f"burrow_{ranker}_{preset}_s{seed}") + ("_af" if af else "")
                jobs.append(dict(bench=b, variant=variant, inputs=str(inp), ranker=ranker, preset=preset, seed=seed, seed_value=SEEDS[seed],
                                 af_mode=af, model=str(model), af_ranker=str(afr) if afr else None, backend=backend, suffix=suffix))
    out_root = ROOT / "results/test" / a.version / "preds"
    if a.plan:
        for j in jobs:
            print(json.dumps(j))
        return
    out_root.mkdir(parents=True, exist_ok=True)
    idx_path = out_root / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {"version": a.version, "freeze": str(a.freeze), "sha256": {}, "runs": {}}
    index["sha256"].update(fz["sha256"])
    if index.get("code_commit") not in (None, code_commit):
        raise SystemExit(f"index.json was produced with code {index['code_commit']}, not {code_commit}")
    index.update(code_commit=code_commit, code_root=str(CODE["root"]))
    commit = code_commit
    for j in jobs:
        vdir = out_root / j["bench"] / j["variant"]
        key = f"{j['bench']}/{j['variant']}"
        if a.redo and re.search(a.redo, key) and key in index["runs"]:
            print(f"redo {key}: discarding previous run", flush=True)
            index["runs"].pop(key)
            shutil.rmtree(vdir, ignore_errors=True)
        done = index["runs"].get(key)
        if done and done.get("rc") == 0 and done.get("n_pred_files") == done.get("n_inputs"):
            print(f"skip {key} (done)", flush=True)
            continue
        if (vdir / "raw").exists():
            shutil.rmtree(vdir / "raw")
        with open(out_root / "predict.log", "a") as log:
            log.write(f"\n==== {key} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.flush()
            dev = a.af_device if j["af_mode"] else a.device
            nw = a.af_workers if (j["af_mode"] and dev == "cpu") else a.workers
            wall, rc = run_burrow(Path(j["inputs"]), vdir / "raw", Path(j["model"]), j["ranker"], j["af_mode"], j["af_ranker"],
                                  dev, nw, j["backend"], log)
        scored = finalize(j["bench"], vdir, j["suffix"])
        n_in = sum(1 for p in Path(j["inputs"]).iterdir() if not p.name.startswith("."))
        n_out = len(list(scored.glob("*.csv")))
        rec = {**j, "wall_s": round(wall, 1), "rc": rc, "n_inputs": n_in, "n_pred_files": n_out, "pred_dir": str(scored.relative_to(ROOT)),
               "commit": commit, "model_sha256": sha256(Path(j["model"])), "af_ranker_sha256": sha256(Path(j["af_ranker"])) if j["af_ranker"] else None,
               "device": (dev if j["backend"] == "fast" else "cpu") + f" (workers {nw})", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (vdir / "run.json").write_text(json.dumps(rec, indent=2))
        index["runs"][key] = rec
        idx_path.write_text(json.dumps(index, indent=2))
        print(f"{key}: rc={rc} {n_out}/{n_in} files, {wall:.0f}s ({j['backend']})", flush=True)
    lines = ["# BURROW test predictions — wall-clock timings", "", "| benchmark | variant | backend/device | n inputs | n pred files | wall s | structures/s | rc |", "|---|---|---|---|---|---|---|---|"]
    for k, r in sorted(index["runs"].items()):
        lines.append(f"| {r['bench']} | {r['variant']} | {r['backend']} {r['device']} | {r['n_inputs']} | {r['n_pred_files']} | {r['wall_s']} | "
                     f"{r['n_inputs'] / max(r['wall_s'], 1e-9):.1f} | {r['rc']} |")
    (out_root / "TIMINGS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
