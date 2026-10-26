#!/usr/bin/env python
"""fpocket + PRANK rescoring (P2Rank `rescore`) -> common pocket format.

  .venv/bin/python baselines/run_fpocket_prank.py <structs|dir|@list> -o OUT [-j N] [--fpocket-version 4.0.3|4.2.3]
        [--p2rank-version 2.5.1|2.4.2] [--rescore-config default_rescore|rescore_2024]

Step 1: fpocket (single-threaded, N processes in parallel) on the cleaned structure (default parameters);
step 2: one JVM `prank rescore` (dataset with PARAM.PREDICTION_METHOD=fpocket) re-scores/re-ranks fpocket pockets with the
PRANK model (default config `default_rescore`, trained on chen11-fpocket.ds in P2Rank 2.x).
Output as run_p2rank.py (score = PRANK score, center/residues/SAS-points from P2Rank's rescored predictions).
Note: fpocket 4.0.3 only runs when built at -O0 (the stock -O2 -pg build segfaults on modern gcc) - done in tools/fpocket_4.0.3.
"""
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import run_p2rank as R
import run_fpocket as F


def fp_job(job):
    path, td, args = job
    stem = C.stem_of(path)
    t0 = time.time()
    try:
        f, info, tc = C.prepare(path, Path(td), args)
        t1 = time.time()
        C.run_cmd([str(F.FPOCKET), "-f", str(f)], timeout=args.timeout, cwd=td)
        od = Path(td) / f"{f.stem}_out"
        if not (od / f"{f.stem}_out.pdb").exists():
            raise RuntimeError("fpocket produced no output")
        return (stem, f, info, tc, time.time() - t1, None)
    except Exception as e:
        return (stem, None, {}, 0, 0, str(e)[:300])


def from_rescored_csv(td, f, raw, out, stem, chain_map):
    import csv
    tmp = Path(tempfile.mkdtemp(prefix="fpk_geo_", dir=td))
    F.convert(Path(td) / f"{f.stem}_out", f.stem, tmp, "Score", None)  # ranks == fpocket pocket numbers
    geo = {p["rank"]: p for p in C.read_pockets_csv(tmp / f"{stem}.pockets.csv")}
    pts = {}
    for r in csv.DictReader(open(tmp / f"{stem}.points.csv")):
        pts.setdefault(int(r["rank"]), []).append((float(r["x"]), float(r["y"]), float(r["z"])))
    pockets, points = [], []
    for r in csv.reader(open(raw / f"{f.name}_rescored.csv")):
        if not r or not r[0].startswith("pocket"):
            continue
        orig = int(r[0].strip().split(".")[1])
        new_rank, score = int(r[2]), float(r[1])
        g = geo[orig]
        pockets.append(dict(rank=new_rank, score=score, center=g["center"], residues=g["residues"]))
        points += [(new_rank, *v) for v in pts.get(orig, [])]
    C.write_pockets(out, stem, pockets, points, chain_map)
    return len(pockets)


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--fpocket-version", default="4.0.3", choices=list(F.VERSIONS))
    ap.add_argument("--p2rank-version", default="2.5.1",
                    help="comma list of P2Rank versions (2.5.1,2.4.2). With >1 version the outputs go to OUT/prank_<version>/")
    ap.add_argument("--fpocket-out", default=None, help="also write plain fpocket results (common format) to this dir")
    ap.add_argument("--rescore-config", default="default_rescore", choices=["default_rescore", "rescore_2024"])
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--heap", type=int, default=None)
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    F.FPOCKET = C.TOOLS / F.VERSIONS[args.fpocket_version] / "bin" / "fpocket"
    versions = args.p2rank_version.split(",")
    for v in versions:
        if v not in R.VERSIONS:
            ap.error(f"unknown P2Rank version {v}")
    args.config = "default"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp_out = Path(args.fpocket_out) if args.fpocket_out else None
    outs = {v: (out / f"prank_{v}" if len(versions) > 1 else out) for v in versions}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    inputs = C.todo(C.expand_inputs(args.inputs), outs[versions[0]], args.overwrite)
    inputs.sort(key=lambda p: -p.stat().st_size)
    if not inputs:
        print("nothing to do")
        return
    heap = args.heap or int(min(100, 2 + 1.5 * args.procs))
    with tempfile.TemporaryDirectory(prefix="fpr_", dir=args.tmp) as td:
        td = Path(td)
        res = C.pmap(fp_job, [(p, str(td), args) for p in inputs], args.procs)
        ok = [r for r in res if r[5] is None]
        err_rows = [dict(stem=r[0], status="error", note=r[5]) for r in res if r[5]]
        if fp_out is not None:
            fp_out.mkdir(parents=True, exist_ok=True)
            frows = list(err_rows)
            for stem, f, info, tc, tf, _ in ok:
                try:
                    n = F.convert(td / f"{f.stem}_out", f.stem, fp_out, "Score", info.get("chain_map"))
                    frows.append(dict(stem=stem, status="ok", clean_s=round(tc, 3), tool_s=round(tf, 3),
                                      total_s=round(tc + tf, 3), note=f"{n} pockets", **info))
                except Exception as e:
                    frows.append(dict(stem=stem, status="error", note=f"fpocket convert: {e}"[:300]))
            C.append_timing(fp_out, frows)
        ds = td / "rescore.ds"
        ds.write_text("PARAM.PREDICTION_METHOD=fpocket\n\nHEADER: prediction protein\n\n" + "\n".join(
            f"{td}/{f.stem}_out/{f.stem}_out.pdb {f}" for _, f, *_ in ok) + "\n")
        for v in versions:
            R.P2 = C.TOOLS / R.VERSIONS[v]
            o = outs[v]
            raw = td / f"raw_{v}"
            a = R.p2_args(args, raw, args.procs)
            a[0] = "rescore"
            a += ["-c", args.rescore_config] if args.rescore_config != "default_rescore" else []
            t1 = time.time()
            r = subprocess.run(R.java_cmd(heap) + a + [str(ds)], env=R.env(), capture_output=True, text=True)
            tr = time.time() - t1
            if r.returncode != 0:
                print((r.stdout + r.stderr)[-1500:], file=sys.stderr)
            rows = list(err_rows)
            for stem, f, info, tc, tf, _ in ok:
                try:
                    if (raw / f"{f.name}_predictions.csv").exists():
                        n = R.convert(raw, f.name, o, stem, info.get("chain_map"))
                    else:  # P2Rank 2.4.2 `rescore` only writes <file>_rescored.csv: geometry from the fpocket output
                        n = from_rescored_csv(td, f, raw, o, stem, info.get("chain_map"))
                    rows.append(dict(stem=stem, status="ok", clean_s=round(tc, 3), tool_s=round(tf + tr / len(ok), 3),
                                     total_s=round(tc + tf + tr / len(ok), 3),
                                     note=f"{n} pockets; tool_s = fpocket + rescore JVM wall/n", **info))
                except Exception as e:
                    rows.append(dict(stem=stem, status="error", note=f"no rescore output: {e}"[:300]))
            if args.keep_raw:
                shutil.copytree(raw, o / "raw" / "_rescore", dirs_exist_ok=True)
            C.finish(o, rows, f"(P2Rank {v} rescore JVM {tr:.1f}s)")


if __name__ == "__main__":
    main()
