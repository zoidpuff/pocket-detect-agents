#!/usr/bin/env python
"""VN-EGNN (original release, ml-jku/vnegnn @1407eec + HF weights fses91/VNEGNN-MODEL, trained on scPDB subset) wrapper.

  .venv/bin/python baselines/run_vnegnn.py <structs...|dir|@list> -o OUT [--device 2|cuda:2|cpu] [-j N] [--seed 0]

Clean (gemmi) -> vnegnn_worker.py in tools/vnegnn/venv (ESM2-650M embeddings + 8 virtual-node EGNN + MeanShift).
Models are loaded once per worker process; -j N launches N worker processes sharing the device (each holds ESM2 ~3 GB).
Score = mean virtual-node confidence of the cluster (higher = better); center = mean virtual-node position;
residues = residues with any heavy atom within --res-radius (default 6 A) of the center (derived, not native);
points = the raw virtual-node positions of the cluster. Prediction is stochastic through the random rotation of the
initial virtual-node sphere -> fixed with --seed (default 0).
"""
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

ENVPY = C.TOOLS / "vnegnn" / "venv" / "bin" / "python"
WORKER = C.BASELINES / "vnegnn_worker.py"


def prep_job(job):
    path, args, td = job
    stem = C.stem_of(path)
    try:
        f, info, tc = C.prepare(path, Path(td), args)
        if f.suffix != ".pdb":
            raise RuntimeError("worker reads PDB only (structure has >62 chains / >99999 atoms / resnum>9999)")
        return stem, f, info, tc, None
    except Exception as e:  # noqa
        return stem, None, {}, 0.0, f"{type(e).__name__}: {e}"[:300]


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res-radius", type=float, default=6.0)
    ap.add_argument("--max-esm-len", type=int, default=3000, help="above this many residues ESM runs per chain")
    ap.add_argument("--split-chains", action="store_true", help="predict each chain separately, pool virtual nodes, MeanShift (authors' HOLO4K protocol; default: whole structure as in predict.py)")
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = C.todo(C.expand_inputs(args.inputs), out, args.overwrite)
    if not inputs:
        print("nothing to do")
        return
    dev = args.device.replace("cuda:", "")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TORCH_HOME=str(C.TOOLS / "vnegnn" / "torch_cache"))
    env["CUDA_VISIBLE_DEVICES"] = "" if dev == "cpu" else dev
    rows = []
    with tempfile.TemporaryDirectory(prefix="vnegnn_", dir=args.tmp) as td:
        td = Path(td)
        (td / "in").mkdir()
        prep = C.pmap(prep_job, [(p, args, str(td / "in")) for p in inputs], args.procs)
        ok = [p for p in prep if p[4] is None]
        rows += [dict(stem=p[0], status="error", note=p[4]) for p in prep if p[4]]
        n = max(1, min(args.procs, len(ok)))
        groups = [ok[i::n] for i in range(n)]

        def run(i):
            g = groups[i]
            lst, res = td / f"files{i}.txt", td / f"res{i}.json"
            lst.write_text("\n".join(str(p[1]) for p in g) + "\n")
            cmd = [str(ENVPY), str(WORKER), "--list", str(lst), "--out", str(res), "--device",
                   "cpu" if dev == "cpu" else "cuda:0", "--seed", str(args.seed),
                   "--res-radius", str(args.res_radius), "--max-esm-len", str(args.max_esm_len)]
            if args.split_chains:
                cmd.append("--split-chains")
            C.run_cmd(cmd, env=env)
            return json.load(open(res))

        results = {}
        with ThreadPoolExecutor(n) as ex:
            futs = [ex.submit(run, i) for i in range(n)]
            for i, fu in enumerate(futs):
                try:
                    results.update(fu.result())
                except Exception as e:  # noqa
                    rows += [dict(stem=p[0], status="error", note=f"worker failed: {e}"[:300]) for p in groups[i]]
        for stem, f, info, tc, _ in ok:
            r = results.get(stem)
            if r is None:
                continue
            if r["status"] != "ok":
                rows.append(dict(stem=stem, status="error", note=r.get("note", "")))
                continue
            tm = r["timing"]
            tool = tm["esm_s"] + tm["model_s"] + tm["cluster_s"]
            pockets = [dict(rank=p["rank"], score=p["score"], center=p["center"], residues=p["residues"]) for p in r["pockets"]]
            pts = [(p["rank"], *q) for p in r["pockets"] for q in p["points"]]
            C.write_pockets(out, stem, pockets, pts, info.get("chain_map"))
            rows.append(dict(stem=stem, status="ok", clean_s=round(tc, 3), tool_s=round(tool, 3),
                             total_s=round(tc + tool, 3),
                             note=f"{len(pockets)} pockets; esm={tm['esm_s']:.2f}s model={tm['model_s']:.2f}s "
                                  f"cluster={tm['cluster_s']:.2f}s (model load {tm['load_s']:.1f}s per worker excluded)",
                             **info))
    C.finish(out, rows)


if __name__ == "__main__":
    main()
