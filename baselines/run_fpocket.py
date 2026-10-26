#!/usr/bin/env python
"""fpocket 4.2.3 wrapper -> common pocket format (see baselines/README.md).

  .venv/bin/python baselines/run_fpocket.py <structs...|dir|@list> -o OUT [-j N] [--score fpocket|drug] [--fpocket-args "-i 10"]

fpocket is single-threaded; -j N runs N structures in parallel. Score = fpocket 'Score' (info.txt) by default
(ranking = fpocket's own order); --score drug ranks by 'Druggability Score' instead.
Center = mean of the pocket's alpha-sphere centers; residues = residues of pocket atoms (pocketN_atm.pdb);
points = alpha-sphere centers (pocketN_vert.pqr).
"""
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

FPOCKET = C.TOOLS / "fpocket_src" / "bin" / "fpocket"   # reassigned in main() by --version
VERSIONS = {"4.2.3": "fpocket_src", "4.0.3": "fpocket_4.0.3"}


def parse_info(path):
    res = {}
    cur = None
    for l in open(path):
        m = re.match(r"Pocket (\d+) :", l)
        if m:
            cur = int(m[1])
            res[cur] = {}
        elif cur and ":" in l:
            k, v = l.split(":", 1)
            try:
                res[cur][k.strip()] = float(v)
            except ValueError:
                pass
    return res


def convert(outdir: Path, stem: str, out: Path, score_key="Score", chain_map=None):
    info = parse_info(outdir / f"{stem}_info.txt")
    pockets, points = [], []
    for n, d in info.items():
        atm = outdir / "pockets" / f"pocket{n}_atm.pdb"
        vert = outdir / "pockets" / f"pocket{n}_vert.pqr"
        res = {}
        for l in open(atm):
            if l.startswith(("ATOM", "HETATM")):
                res[(l[21], int(l[22:26]), l[26].strip())] = None
        residues = sorted(C.fmt_res(c, r, i) for (c, r, i) in res)
        xyz = []
        for l in open(vert):
            if l.startswith(("ATOM", "HETATM")):
                xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
        if not xyz:
            continue
        cx = tuple(sum(v[k] for v in xyz) / len(xyz) for k in range(3))
        pockets.append(dict(orig=n, score=d[score_key], center=cx, residues=residues, xyz=xyz))
    pockets.sort(key=lambda p: (-p["score"], p["orig"]))
    for rk, p in enumerate(pockets, 1):
        p["rank"] = rk
        points += [(rk, *v) for v in p["xyz"]]
    C.write_pockets(out, stem, pockets, points, chain_map)
    return len(pockets)


def run_one(job):
    path, args = job
    stem = C.stem_of(path)
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix=f"fpk_{stem}_", dir=args.tmp) as td:
            td = Path(td)
            f, info, tc = C.prepare(path, td, args)
            t1 = time.time()
            cmd = [str(FPOCKET), "-f", str(f), "-w", "p"] + args.fpocket_args.split()
            C.run_cmd(cmd, timeout=args.timeout, cwd=td)
            tt = time.time() - t1
            od = td / f"{f.stem}_out"
            if not od.exists():  # fpocket names output <file stem>_out (cif: <stem>_out too in 4.2)
                cands = list(td.glob("*_out"))
                od = cands[0]
            key = "Score" if args.score == "fpocket" else "Druggability Score"
            n = convert(od, od.name[:-4], Path(args.out), key, info.get("chain_map"))
            if args.keep_raw:
                shutil.copytree(od, Path(args.out) / "raw" / stem, dirs_exist_ok=True)
        return dict(stem=stem, status="ok", clean_s=round(tc, 3), tool_s=round(tt, 3),
                    total_s=round(time.time() - t0, 3), note=f"{n} pockets", **info)
    except Exception as e:
        return dict(stem=stem, status="error", total_s=round(time.time() - t0, 3), note=str(e)[:300])


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--score", default="fpocket", choices=["fpocket", "drug"])
    ap.add_argument("--fpocket-args", default="", help="extra fpocket CLI args, e.g. '-i 10 -m 3.0'")
    ap.add_argument("--timeout", type=int, default=3600, help="per-structure timeout in s")
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--version", default="4.2.3", choices=list(VERSIONS))
    args = ap.parse_args()
    global FPOCKET
    FPOCKET = C.TOOLS / VERSIONS[args.version] / "bin" / "fpocket"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = C.todo(C.expand_inputs(args.inputs), out, args.overwrite)
    # largest first for better load balance
    inputs.sort(key=lambda p: -p.stat().st_size)
    rows = C.pmap(run_one, [(p, args) for p in inputs], args.procs)
    C.finish(out, rows)


if __name__ == "__main__":
    main()
