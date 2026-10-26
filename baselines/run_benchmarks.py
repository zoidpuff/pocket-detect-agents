#!/usr/bin/env python
"""Run baseline predictors on a benchmark and score them with the project harness.

  .venv/bin/python baselines/run_benchmarks.py --bench ligysis --variant p2rank_2.4.2 [--jobs 28] [--no-run] [--no-eval]

Layout: baselines/outputs/<bench>/<variant>/{raw/ (wrapper common-format output + timing.csv), preds/<structure_id>.csv, eval/}
Variants: p2rank_2.4.2 | p2rank_2.5.1 | p2rank_2.5.1_af (-c alphafold) | fpocket_4.0.3 | fpocket_4.2.3 |
          fpocket_prank (run once per bench: writes fpocket_<v>, fpocket_<v>+prank_<p> variants; ligysis: fpocket 4.2.3 (the version the LIGYSIS authors used; verified by pocket counts) with PRANK 2.4.2 and 2.5.1,
          other benches: fpocket 4.2.3 with PRANK 2.5.1)
Benchmarks: ligysis cryptobench pocketwatch af_set coach420_mlig holo4k_mlig dimer_assembly dimer_chains
ligysis predictions are made on the authors' TRANS-frame chains; they are scored (a) in the TRANS frame against the authors' site table
(reproduction gate, centre-based metrics only) and (b) mapped back to the wwPDB frame against data/processed/gt/ligysis_lbs (C1-strict etc.).
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

ROOT = C.PROJECT
OUT = C.BASELINES / "outputs"
PY = str(ROOT / ".venv/bin/python")
DS_BENCH = {"coach420_mlig", "holo4k_mlig"}


def input_dir(bench):
    if bench == "dimer_assembly":
        return ROOT / "data/processed/dimer/inputs/assembly"
    if bench == "dimer_chains":
        return ROOT / "data/processed/dimer/inputs/chains"
    return OUT / "inputs" / bench


def load_avg():
    return os.getloadavg()


def sh(cmd, log):
    t = time.time()
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t


def run_tool(bench, variant, jobs, log):
    ind = input_dir(bench)
    vdir = OUT / bench / variant
    raw = vdir / "raw"
    tmp = OUT / "tmp"
    tmp.mkdir(exist_ok=True)
    W = str(C.BASELINES)
    if variant == "p2rank_ours":  # P2Rank 2.5.1 retrained on our training set (baselines/p2rank_ours; results/baselines/P2RANK_OURS.md)
        cmd = [PY, f"{W}/run_p2rank.py", str(ind), "-o", str(raw), "-j", str(jobs), "--version", "2.5.1", "--tmp", str(tmp),
               "--model", "../../../p2rank_ours/model/FastRandomForest.model"]
        return {variant: sh(cmd, log)}
    if variant.startswith("p2rank"):
        ver = variant.split("_")[1]
        cmd = [PY, f"{W}/run_p2rank.py", str(ind), "-o", str(raw), "-j", str(jobs), "--version", ver, "--tmp", str(tmp)]
        if variant.endswith("_af"):
            cmd += ["--config", "alphafold"]
        return {variant: sh(cmd, log)}
    if variant.startswith("fpocket_") and "prank" not in variant:
        ver = variant.split("_")[1]
        cmd = [PY, f"{W}/run_fpocket.py", str(ind), "-o", str(raw), "-j", str(jobs), "--version", ver, "--tmp", str(tmp)]
        return {variant: sh(cmd, log)}
    if variant == "grasp":  # scPDB-trained (train_full: all of sc-PDB) - leakage caveat
        cmd = [PY, f"{W}/run_grasp.py", str(ind), "-o", str(raw), "-j", str(min(jobs, 24)), "--device", os.environ.get("GPU", "2"), "--tmp", str(tmp)]
        return {variant: sh(cmd, log)}
    if variant == "puresnet":  # PDB-wide training set (61.7k ids) - leakage caveat
        cmd = [f"{W}/tools/puresnet/venv/bin/python", f"{W}/run_puresnet.py", str(ind), "-o", str(raw), "-j", str(min(jobs, 24)),
               "--device", os.environ.get("GPU", "2"), "--tmp", str(tmp)]
        return {variant: sh(cmd, log)}
    if variant == "fpocket_prank":
        fv, pv = ("4.2.3", "2.4.2,2.5.1") if bench == "ligysis" else ("4.2.3", "2.5.1")
        base = OUT / bench / f"fpocket_{fv}_prank"
        fpo = OUT / bench / f"fpocket_{fv}" / "raw"
        cmd = [PY, f"{W}/run_fpocket_prank.py", str(ind), "-o", str(base), "-j", str(jobs), "--fpocket-version", fv,
               "--p2rank-version", pv, "--fpocket-out", str(fpo), "--tmp", str(tmp)]
        res = sh(cmd, log)
        # normalise layout: <bench>/fpocket_<v>+PRANK_<p>/raw
        for p in pv.split(","):
            src = base / f"prank_{p}" if "," in pv else base
            dst = OUT / bench / f"fpocket_{fv}+PRANK_{p}" / "raw"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
        return {"fpocket_prank": res}
    raise SystemExit(f"unknown variant {variant}")


def make_preds(bench, vdir):
    """<structure_id>.csv files expected by the harness; ligysis: also back-mapped frame."""
    raw = vdir / "raw"
    suffix = ".pdb" if bench in DS_BENCH else ""
    pdir = vdir / "preds"
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir()
    n = 0
    for f in raw.glob("*.pockets.csv"):
        stem = f.name[: -len(".pockets.csv")]
        if bench == "af_set" and stem.endswith("-model_v6"):  # wrappers that resolve symlinks name outputs after the AFDB file
            stem = stem[: -len("-model_v6")]
        shutil.copy(f, pdir / f"{stem}{suffix}.csv")
        n += 1
    if bench == "ligysis":
        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np
        import pandas as pd
        from pocketdetect.ligysis import load_transforms, to_original_frame
        import csv
        import gemmi
        csv.field_size_limit(1 << 30)
        T = load_transforms()
        # trans.pdb residues carry label_seq_id numbering; the site table uses author ids -> map per chain
        srows = {}
        for r in csv.DictReader(open(ROOT / "data/processed/gt/ligysis_lbs/sites.tsv"), delimiter="\t"):
            srows.setdefault(r["structure_id"], (r["structure_path"], r["input_chains"]))
        odir = vdir / "preds_orig"
        if odir.exists():
            shutil.rmtree(odir)
        odir.mkdir()
        for f in pdir.glob("*.csv"):
            df = pd.read_csv(f, dtype={"residues": str}, keep_default_na=False)
            R, c0 = T[f.stem]
            path, auth_chain = srows[f.stem]
            lab2auth = {r.label_seq: C.fmt_res(auth_chain, r.seqid.num, r.seqid.icode)
                        for r in gemmi.read_structure(path)[0][auth_chain] if r.label_seq is not None}
            if len(df):
                # residues with an insertion code in the trans file (non-SEQRES numbering) cannot be mapped and are dropped
                df["residues"] = [" ".join(lab2auth[int(t.split(":")[1])] for t in rs.split()
                                           if t.split(":")[1].lstrip("-").isdigit() and int(t.split(":")[1]) in lab2auth)
                                  for rs in df["residues"]]
                xyz = to_original_frame(df[["center_x", "center_y", "center_z"]].to_numpy(float), R, c0)
                df[["center_x", "center_y", "center_z"]] = np.round(xyz, 3)
            df.to_csv(odir / f.name, index=False)
    return n


def evaluate(bench, vdir, jobs, log):
    sys.path.insert(0, str(ROOT / "src"))
    from pocketdetect.benchmarks import REGISTRY, run_benchmark
    ev = vdir / "eval"
    info = {}
    t = time.time()
    if bench == "ligysis":
        info["orig"] = run_benchmark("ligysis", vdir / "preds_orig", ev / "orig_frame", jobs=jobs)
        # gate: TRANS-frame, authors' site table (residues are UniProt numbers -> residue metrics not meaningful)
        from pocketdetect.site_eval import SiteProtocol, evaluate_sites, summarize_sites
        from pocketdetect.sites import read_sites, sites_by_structure
        tab = ROOT / "results/validation/ligysis_reproduction/site_table_trans.tsv"
        by = sites_by_structure(read_sites(tab))
        res = evaluate_sites(by, str(vdir / "preds"), SiteProtocol())
        s = summarize_sites(res, n_boot=0)
        (ev / "trans_frame").mkdir(parents=True, exist_ok=True)
        s.to_csv(ev / "trans_frame" / "summary.csv", index=False)
    elif bench == "dimer_chains":
        out = ev
        out.mkdir(parents=True, exist_ok=True)
        rc, _ = sh([PY, str(ROOT / "scripts/eval_dimer.py"), "--pred-dir", str(vdir / "preds"), "--mode", "chain",
                    "--chain-set", "all", "--out", str(out)], log)
        info["rc"] = rc
    elif bench == "dimer_assembly":
        info = run_benchmark("dimer", vdir / "preds", ev, jobs=jobs)
    else:
        info = run_benchmark(bench, vdir / "preds", ev, jobs=jobs)
    (ev / "wall_s.txt").write_text(f"{time.time() - t:.1f}\n")
    return info


def record_run(bench, variant, ver_desc, n_inputs, wall, load0, load1, rc):
    p = ROOT / "results/baselines/RUNS.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("# Baseline runs (appended by baselines/run_benchmarks.py)\n\n| date (UTC) | benchmark | variant | tool version | inputs | n inputs | wall s | node load before/after | rc | output dir |\n|---|---|---|---|---|---|---|---|---|---|\n")
    with open(p, "a") as f:
        f.write(f"| {dt.datetime.utcnow():%Y-%m-%d %H:%M} | {bench} | {variant} | {ver_desc} | baselines/outputs/inputs/{bench} (see baselines/bench_inputs.py) | {n_inputs} | {wall:.0f} | {load0:.1f}/{load1:.1f} | {rc} | baselines/outputs/{bench}/{variant} |\n")


DESC = {"p2rank_2.4.2": "P2Rank 2.4.2 default", "p2rank_2.5.1": "P2Rank 2.5.1 default", "p2rank_2.5.1_af": "P2Rank 2.5.1 -c alphafold", "p2rank_ours": "P2Rank 2.5.1 retrained on our train set",
        "fpocket_4.0.3": "fpocket 4.0.3 (-O0 build)", "fpocket_4.2.3": "fpocket 4.2.3", "fpocket_prank": "fpocket + P2Rank rescore", "grasp": "GrASP train_full (scPDB)", "puresnet": "PUResNet v2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--jobs", type=int, default=28)
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    (OUT / a.bench).mkdir(parents=True, exist_ok=True)
    log = OUT / a.bench / f"{a.variant}.log"
    variants = [a.variant]
    ninp = len(list(input_dir(a.bench).iterdir()))
    if not a.no_run:
        l0 = load_avg()[0]
        t = time.time()
        res = run_tool(a.bench, a.variant, a.jobs, log)
        wall = time.time() - t
        for k, (rc, _) in res.items():
            record_run(a.bench, k, DESC.get(a.variant, a.variant), ninp, wall, l0, load_avg()[0], rc)
    if a.variant == "fpocket_prank":
        fv, pv = ("4.2.3", ["2.4.2", "2.5.1"]) if a.bench == "ligysis" else ("4.2.3", ["2.5.1"])
        variants = [f"fpocket_{fv}"] + [f"fpocket_{fv}+PRANK_{p}" for p in pv]
    for v in variants:
        vdir = OUT / a.bench / v
        if not (vdir / "raw").exists():
            print("missing", vdir)
            continue
        n = make_preds(a.bench, vdir)
        print(f"{a.bench}/{v}: {n}/{ninp} prediction files")
        if not a.no_eval:
            info = evaluate(a.bench, vdir, a.jobs, log)
            print(json.dumps(info.get("headline", info.get("orig", {}).get("headline", info)), default=str)[:600])


if __name__ == "__main__":
    main()
