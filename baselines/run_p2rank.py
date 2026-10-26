#!/usr/bin/env python
"""P2Rank 2.5.1 wrapper -> common pocket format (see baselines/README.md).

  .venv/bin/python baselines/run_p2rank.py <structs...|dir|@list> -o OUT [-j N] [--config default|alphafold] [--mode batch|per-file]

Modes
  batch     (default) one JVM, dataset file, `-threads N` (fastest for many structures)
  per-file  one JVM per structure, -threads 1, N processes in parallel (gives true per-structure wall time)
Configs: default (uses B-factor feature), alphafold (`-c alphafold`; no B-factor, for AF/NMR/cryo-EM models).
Both are conservation-free.
"""
import gzip
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

P2 = C.TOOLS / "p2rank"   # reassigned in main() by --version
VERSIONS = {"2.5.1": "p2rank", "2.4.2": "p2rank_2.4.2"}
JAVA_HOME = C.TOOLS / "jdk"


def java_cmd(heap_gb):
    cp = f"{P2}/bin/p2rank.jar:{P2}/bin/lib/*"
    return [str(JAVA_HOME / "bin/java"), f"-Xmx{heap_gb}g", "-cp", cp, "cz.siret.prank.program.Main"]


def env():
    e = dict(os.environ)
    e["INSTALL_DIR"] = str(P2)
    e["JAVA_HOME"] = str(JAVA_HOME)
    return e


def parse_points(gz_path):
    pts = []
    with gzip.open(gz_path, "rt") as f:
        for l in f:
            if not l.startswith("HETATM"):
                continue
            # P2Rank's pseudo-PDB is not strictly fixed-width (rank field grows for rank >= 100), so parse
            # the rank from the tokens after 'STP <chain>' and xyz from the fixed-width tail
            # (3x 8.3f followed by occupancy '  0.50' and score ' 0.013').
            r = int(l.split("STP", 1)[1].split()[1])
            if r > 0:  # 0 = non-pocket SAS points
                t = l.rstrip("\n")[-36:-12]
                pts.append((r, float(t[0:8]), float(t[8:16]), float(t[16:24])))
    return pts


def convert(raw_dir: Path, fname: str, out: Path, stem: str, chain_map=None):
    """fname = cleaned file name (<stem>.pdb|cif); raw_dir = P2Rank output dir."""
    pockets = []
    with open(raw_dir / f"{fname}_predictions.csv") as f:
        header = [h.strip() for h in f.readline().split(",")]
        for l in f:
            if not l.strip():
                continue
            v = [x.strip() for x in l.rstrip("\n").split(",", len(header) - 1)]
            d = dict(zip(header, v))
            pockets.append(dict(rank=int(d["rank"]), score=float(d["score"]),
                                center=(float(d["center_x"]), float(d["center_y"]), float(d["center_z"])),
                                residues=[C.parse_p2rank_res(t) for t in d["residue_ids"].split()]))
    pf = raw_dir / "visualizations" / "data" / f"{fname}_points.pdb.gz"
    pts = parse_points(pf) if pf.exists() else None
    C.write_pockets(out, stem, pockets, pts, chain_map)
    return len(pockets)


def p2_args(args, outdir, threads):
    a = ["predict", "-o", str(outdir), "-threads", str(threads),
         "-visualizations", "true", "-vis_copy_proteins", "false", "-vis_generate_proteins", "false",
         "-log_to_file", "false"]
    if P2.name != "p2rank_2.4.2":  # `vis_renderers` only exists since 2.5
        a += ["-vis_renderers", "[]"]
    if args.config != "default":
        a += ["-c", args.config]
    if getattr(args, "model", None):
        a += ["-m", args.model]  # path relative to <install>/models/ (e.g. a retrained model; see baselines/p2rank_ours)
    return a


def run_one(job):
    """per-file mode worker"""
    path, args = job
    stem = C.stem_of(path)
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix=f"p2r_{stem}_", dir=args.tmp) as td:
            td = Path(td)
            f, info, tc = C.prepare(path, td, args)
            t1 = time.time()
            raw = td / "raw"
            r = subprocess.run(java_cmd(args.heap) + p2_args(args, raw, 1) + ["-f", str(f)], env=env(),
                               capture_output=True, text=True)
            tt = time.time() - t1
            if r.returncode != 0 or not (raw / f"{f.name}_predictions.csv").exists():
                raise RuntimeError("p2rank failed: " + (r.stdout + r.stderr)[-300:])
            n = convert(raw, f.name, Path(args.out), stem, info.get('chain_map'))
            if args.keep_raw:
                import shutil
                shutil.copytree(raw, Path(args.out) / "raw" / stem, dirs_exist_ok=True)
        return dict(stem=stem, status="ok", clean_s=round(tc, 3), tool_s=round(tt, 3),
                    total_s=round(time.time() - t0, 3), note=f"{n} pockets", **info)
    except Exception as e:
        return dict(stem=stem, status="error", total_s=round(time.time() - t0, 3), note=str(e)[:300])


def prep_job(job):
    path, args, td = job
    try:
        f, info, tc = C.prepare(path, Path(td), args)
        return (C.stem_of(path), f, info, tc, None)
    except Exception as e:
        return (C.stem_of(path), None, {}, 0, str(e)[:300])


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--config", default="default", choices=["default", "alphafold"])
    ap.add_argument("--mode", default="batch", choices=["batch", "per-file"])
    ap.add_argument("--heap", type=int, default=None, help="JVM heap GB (default 2 + 1.5*threads, max 100)")
    ap.add_argument("--tmp", default=None, help="scratch dir (default system tmp)")
    ap.add_argument("--model", default=None, help="custom model file relative to <p2rank>/models/ (e.g. ../../p2rank_ours/model/FastRandomForest.model)")
    ap.add_argument("--version", default="2.5.1", choices=list(VERSIONS), help="P2Rank release (2.4.2 for LIGYSIS reproduction)")
    args = ap.parse_args()
    global P2
    P2 = C.TOOLS / VERSIONS[args.version]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = C.todo(C.expand_inputs(args.inputs), out, args.overwrite)
    if not inputs:
        print("nothing to do")
        return
    if args.heap is None:
        args.heap = int(min(100, 2 + 1.5 * args.procs)) if args.mode == "batch" else 3
    if args.mode == "per-file":
        rows = C.pmap(run_one, [(p, args) for p in inputs], args.procs)
        C.append_timing(out, rows)
        print(f"done: {sum(r['status']=='ok' for r in rows)}/{len(rows)} ok")
        return
    with tempfile.TemporaryDirectory(prefix="p2r_batch_", dir=args.tmp) as td:
        td = Path(td)
        t0 = time.time()
        (td / "in").mkdir()
        prep = C.pmap(prep_job, [(p, args, str(td / "in")) for p in inputs], args.procs)
        ok = [p for p in prep if p[4] is None]
        t_clean = time.time() - t0
        rows = [dict(stem=p[0], status="error", note=p[4]) for p in prep if p[4]]
        ds = td / "batch.ds"
        ds.write_text("\n".join(str(p[1]) for p in ok) + "\n")
        raw = td / "raw"
        t1 = time.time()
        r = subprocess.run(java_cmd(args.heap) + p2_args(args, raw, args.procs) + [str(ds)], env=env(),
                           capture_output=True, text=True)
        t_tool = time.time() - t1
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-2000:], file=sys.stderr)
        for stem, f, info, tc, _ in ok:
            try:
                n = convert(raw, f.name, out, stem, info.get('chain_map'))
                rows.append(dict(stem=stem, status="ok", clean_s=round(tc, 3), tool_s=round(t_tool / len(ok), 3),
                                 total_s=round(tc + t_tool / len(ok), 3),
                                 note=f"{n} pockets; batch tool_s = total JVM wall / n (threads={args.procs})", **info))
            except Exception as e:
                rows.append(dict(stem=stem, status="error", note=f"no output: {e}"[:300]))
        if args.keep_raw:
            import shutil
            shutil.copytree(raw, out / "raw", dirs_exist_ok=True)
        C.append_timing(out, rows)
        print(f"done: {sum(r['status']=='ok' for r in rows)}/{len(rows)} ok; clean {t_clean:.1f}s, p2rank {t_tool:.1f}s wall")


if __name__ == "__main__":
    main()
