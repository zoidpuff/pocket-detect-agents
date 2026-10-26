#!/usr/bin/env python
"""pyKVFinder 0.9.5 wrapper -> common pocket format (see baselines/README.md).

  python baselines/run_pykvfinder.py <structs|dir|@list> -o OUT [-j N] [--step 0.6 --probe-in 1.4 --probe-out 4.0 ...]

pyKVFinder is a purely geometric cavity detector (two rolling probes, KVFinder algorithm; no learning, no ranking).
Defaults of pyKVFinder are used (step 0.6 A, probe_in 1.4, probe_out 4.0, removal_distance 2.4,
volume_cutoff 5 A^3, SES surface). Cavities are ranked by volume (--score area ranks by area).
center = centroid of cavity voxels, points = cavity voxels (subsampled by stride to <= --max-points per cavity),
residues = pyKVFinder's interface residues (chain+resnum; it does not report insertion codes, so every residue
sharing that chain+number, i.e. all insertion-code variants, is listed).
Parallelism: -j N processes, each running pyKVFinder with --threads (default 1).
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

TOOL = C.TOOLS / "pykvfinder"
C.reexec_in_venv(TOOL / "venv")

import numpy as np


def res_index(pdb):
    import gemmi
    idx = {}
    for ch in gemmi.read_structure(str(pdb))[0]:
        for r in ch:
            idx.setdefault((ch.name, str(r.seqid.num)), []).append(C.fmt_res(ch.name, r.seqid.num, r.seqid.icode))
    return idx


def run_one(job):
    path, args = job
    stem = C.stem_of(path)
    t0 = time.time()
    try:
        import pyKVFinder
        import logging
        logging.disable(logging.WARNING)
        with tempfile.TemporaryDirectory(prefix=f"kvf_{stem}_", dir=args.tmp) as td:
            td = Path(td)
            f, info, tc = C.prepare(path, td, args)
            if f.suffix != ".pdb":
                raise RuntimeError("pyKVFinder wrapper needs PDB input (too many atoms/chains for PDB format)")
            t1 = time.time()
            r = pyKVFinder.run_workflow(str(f), step=args.step, probe_in=args.probe_in, probe_out=args.probe_out,
                                        removal_distance=args.removal_distance, volume_cutoff=args.volume_cutoff,
                                        nthreads=args.threads, surface=args.surface)
            tt = time.time() - t1
            pockets, points = [], []
            if r is not None and r.ncav > 0:
                origin = np.asarray(r._vertices[0], dtype=float)
                idx = res_index(f)
                sc = r.volume if args.score == "volume" else r.area
                order = sorted(r.volume, key=lambda k: (-sc[k], k))
                # one pass over the grid: voxel indices grouped by cavity label (np.argwhere per cavity is O(grid * ncav))
                flat = r.cavities.ravel()
                nz = np.flatnonzero(flat >= 2)
                lab = flat[nz]
                srt = np.argsort(lab, kind="stable")
                nz, lab = nz[srt], lab[srt]
                u, start = np.unique(lab, return_index=True)
                bounds = {int(l): (int(x), int(y)) for l, x, y in zip(u, start, list(start[1:]) + [len(lab)])}
                for rk, key in enumerate(order, 1):
                    label = 2 + (ord(key[2]) - 65) + 26 * (ord(key[1]) - 65)  # KAA=2, KAB=3 ... KBA=28
                    if label not in bounds:
                        continue
                    a, b = bounds[label]
                    vox = np.stack(np.unravel_index(nz[a:b], r.cavities.shape), axis=1)
                    xyz = origin + vox * args.step
                    res = []
                    for num, ch, _ in r.residues.get(key, []):
                        res += idx.get((ch, str(num)), [])
                    pockets.append(dict(rank=rk, score=sc[key], center=xyz.mean(0), residues=sorted(set(res))))
                    stride = max(1, int(np.ceil(len(xyz) / args.max_points)))
                    points += [(rk, *p) for p in xyz[::stride]]
            C.write_pockets(Path(args.out), stem, pockets, points, info.get("chain_map"))
        return dict(stem=stem, status="ok", clean_s=round(tc, 3), tool_s=round(tt, 3),
                    total_s=round(time.time() - t0, 3), note=f"{len(pockets)} pockets", **info)
    except Exception as e:
        return dict(stem=stem, status="error", total_s=round(time.time() - t0, 3), note=f"{type(e).__name__}: {e}"[:300])


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--step", type=float, default=0.6)
    ap.add_argument("--probe-in", type=float, default=1.4)
    ap.add_argument("--probe-out", type=float, default=4.0)
    ap.add_argument("--removal-distance", type=float, default=2.4)
    ap.add_argument("--volume-cutoff", type=float, default=5.0)
    ap.add_argument("--surface", default="SES", choices=["SES", "SAS"])
    ap.add_argument("--score", default="volume", choices=["volume", "area"])
    ap.add_argument("--threads", type=int, default=1, help="pyKVFinder OpenMP threads per structure")
    ap.add_argument("--max-points", type=int, default=3000)
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = C.todo(C.expand_inputs(args.inputs), out, args.overwrite)
    inputs.sort(key=lambda p: -p.stat().st_size)
    rows = C.pmap(run_one, [(p, args) for p in inputs], args.procs)
    C.finish(out, rows)


if __name__ == "__main__":
    main()
