#!/usr/bin/env python
"""GrASP (tiwarylab/GrASP @0a01251, GATv2 'train_full' model, trained on scPDB) wrapper -> common pocket format.

  .venv/bin/python baselines/run_grasp.py <structs...|dir|@list> -o OUT [-j N] [--device 2|cuda:2|cpu] [--chunk 64]

Stages: clean structure (gemmi; ligands/waters/H/altlocs/nucleic stripped) -> grasp_worker.py in the py3.7 GrASP
env (tools/grasp/env): OpenBabel-2.4.1 protonation, rdkit/mdtraj featurisation, GATv2 atom scoring on GPU, then site
clustering with GrASP's paper defaults (average linkage 15 A, p>0.3, score=sum p^2, hull centre).
-j N = parallel featurisation processes (the CPU-bound stage; GrASP featurisation is ~O(n^2), minutes for >20k atoms).
The GPU model is loaded once per --chunk of structures. Score = sum of squared atom probabilities of the site.
points file = surface atoms of each predicted site.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

ENVPY = C.TOOLS / "grasp" / "env" / "bin" / "python"
WORKER = C.BASELINES / "grasp_worker.py"


MODELS = {
    "train_full": "train_full/trained_model_s_train_full_ag_multi_1680643832.8660116/cv_0/epoch_49",
    "coach420_mlig": "coach420_mlig/trained_model_s_coach420_mlig_ag_multi_1672951806.257937/cv_0/epoch_49",
    "holo4k_mlig": "holo4k_mlig/trained_model_s_holo4k_mlig_ag_multi_1672951807.304454/cv_0/epoch_49",
    "cv": "cv/trained_model_ag_multi_1672951703.4392676/cv_0/epoch_49",
}


def prep_job(job):
    path, args, td = job
    stem = C.stem_of(path)
    try:
        f, info, tc = C.prepare(path, Path(td), args)
        if f.suffix != ".pdb":
            raise RuntimeError("GrASP needs PDB input (structure has >62 chains / >99999 atoms / resnum>9999); not supported")
        return stem, f, info, tc, None
    except Exception as e:  # noqa
        return stem, None, {}, 0.0, f"{type(e).__name__}: {e}"[:300]


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--device", default="0", help="GPU index / cuda:N / cpu (sets CUDA_VISIBLE_DEVICES for the worker)")
    ap.add_argument("--chunk", type=int, default=64, help="structures per worker invocation")
    ap.add_argument("--model", default="train_full", choices=list(MODELS),
                    help="train_full: all of scPDB (default; NOT similarity-filtered vs COACH420/HOLO4K); coach420_mlig / holo4k_mlig: "
                         "scPDB filtered by UniProt vs that test set (leakage-controlled for it); cv: scPDB CV fold 0")
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--threshold", type=float, default=0.3, help="atom probability threshold (GrASP default 0.3)")
    ap.add_argument("--eps", type=float, default=15.0, help="average-linkage distance threshold in A (GrASP default 15)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = C.todo(C.expand_inputs(args.inputs), out, args.overwrite)
    if not inputs:
        print("nothing to do")
        return
    dev = args.device.replace("cuda:", "")
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if dev == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = dev
    rows = []
    for i0 in range(0, len(inputs), args.chunk):
        chunk = inputs[i0:i0 + args.chunk]
        with tempfile.TemporaryDirectory(prefix="grasp_", dir=args.tmp) as td:
            td = Path(td)
            (td / "in").mkdir()
            env["HOME"] = str(td)
            prep = C.pmap(prep_job, [(p, args, str(td / "in")) for p in chunk], args.procs)
            ok = [p for p in prep if p[4] is None]
            rows += [dict(stem=p[0], status="error", note=p[4]) for p in prep if p[4]]
            if not ok:
                continue
            (td / "files.txt").write_text("\n".join(str(p[1]) for p in ok) + "\n")
            t0 = time.time()
            cmd = [str(ENVPY), str(WORKER), "--list", str(td / "files.txt"), "--work", str(td / "work"),
                   "--out", str(td / "res.json"), "--procs", str(args.procs),
                   "--device", "cpu" if dev == "cpu" else "cuda:0",
                   "--threshold", str(args.threshold), "--eps", str(args.eps), "--model", MODELS[args.model]]
            try:
                C.run_cmd(cmd, env=env)
            except Exception as e:  # noqa
                rows += [dict(stem=p[0], status="error", note=f"worker failed: {e}"[:300]) for p in ok]
                continue
            wall = time.time() - t0
            res = json.load(open(td / "res.json"))
            for stem, f, info, tc, _ in ok:
                r = res.get(stem)
                if r is None or r["status"] != "ok":
                    rows.append(dict(stem=stem, status="error", note=(r or {}).get("note", "missing result")))
                    continue
                tm = r["timing"]
                tool = tm["parse_s"] + tm["infer_s"] + tm["cluster_s"] + tm["batch_graph_s"] / tm["batch_n"]
                pockets = [dict(rank=p["rank"], score=p["score"], center=p["center"], residues=p["residues"]) for p in r["pockets"]]
                pts = [(p["rank"], *q) for p in r["pockets"] for q in p["points"]]
                C.write_pockets(out, stem, pockets, pts, info.get("chain_map"))
                rows.append(dict(stem=stem, status="ok", clean_s=round(tc, 3), tool_s=round(tool, 3),
                                 total_s=round(tc + tool, 3),
                                 note=f"{len(pockets)} sites; parse={tm['parse_s']:.1f}s infer={tm['infer_s']:.2f}s "
                                      f"cluster={tm['cluster_s']:.2f}s graph={tm['batch_graph_s'] / tm['batch_n']:.2f}s; "
                                      f"chunk wall {wall:.0f}s for {len(ok)} structs (procs={args.procs})",
                                 **info))
    C.finish(out, rows)


if __name__ == "__main__":
    main()
