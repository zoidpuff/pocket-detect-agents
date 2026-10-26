"""Fast BURROW batch prediction driver (CPU process pool or multi-GPU).

    python -m pocketdetect.burrow.fast predict <files|dirs|@list> --out OUT
           [--device cpu | cuda:0 | cuda:0,1,3] [--workers N] [--batch 32]
           [--af-mode] [--ranker all|small|geometry] [--preset default|accurate] [--model M.json]
           [--no-points]

Outputs are identical to ``python -m pocketdetect.burrow predict`` (same writer, same files:
``OUT/<stem>.csv``, ``OUT/<stem>.points.csv``, ``OUT/timing.csv``), computed by the fast backend.

* CPU: ``--workers`` single-threaded processes, one structure each.
* GPU: one process per listed GPU; each owns a pool of ``--workers`` CPU processes that parse and
  prepare structures (and run pocket formation / ranking / writing) while the GPU featurizes and
  scores batches of ``--batch`` structures.  Input files are distributed round-robin over GPUs.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

_SINGLE = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}
ROOT = Path(__file__).resolve().parents[4]
HEADER = "stem,status,n_atoms,n_pockets,total_ms,parse_ms,features_ms,point_score_ms,pockets_ms,rank_ms,device,note"

# ---------------------------------------------------------------- worker state (per process)
_W = {}


def _init_worker(model_path, ranker, af_mode, out, points):
    os.environ.update(_SINGLE)
    import numba
    numba.set_num_threads(1)
    from .pipeline import FastModel
    _W.update(fm=FastModel.load(model_path) if model_path else FastModel.from_model(None), ranker=ranker,
              af=af_mode, out=Path(out), points=points)


def _load(path, stem, af):
    from ...structure import load_structure
    from ..af import af_protein
    from ..prep import protein_from_structure
    s = load_structure(path, structure_id=stem)
    if af:
        P, w, st = af_protein(s)
        return P, w, f"pruned_res={st['n_res_pruned']} soft_atoms={st['n_soft_atoms']}"
    return protein_from_structure(s), None, ""


def _row(stem, P, pred, total, parse, note, device):
    t = pred.timings
    g = lambda k: f"{t.get(k, 0.0):.1f}"
    return (f"{stem},ok,{len(P.coords)},{len(pred.scores)},{total:.1f},{parse:.1f},{g('features')},"
            f"{g('point_score')},{g('pockets')},{g('rank')},{device},{note}")


def _cpu_task(args):
    path, stem = args
    from ..pipeline import write_prediction
    from .pipeline import predict_protein
    try:
        t0 = time.perf_counter()
        P, w, note = _load(path, stem, _W["af"])
        t1 = time.perf_counter()
        pred = predict_protein(P, _W["fm"], weights=w, ranker=_W["ranker"])
        write_prediction(_W["out"], pred, points=_W["points"])
        return _row(stem, P, pred, (time.perf_counter() - t0) * 1e3, (t1 - t0) * 1e3, note, "cpu")
    except Exception as e:  # never abort a batch
        return f"{stem},error,,,,,,,,,cpu,{type(e).__name__}: {str(e)[:200]}"


def _prep_task(args):
    path, stem = args
    from .gpu import prep
    try:
        t0 = time.perf_counter()
        P, w, note = _load(path, stem, _W["af"])
        t1 = time.perf_counter()
        it = prep(P, _W["fm"].cfg, w)
        return stem, it, note, (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3, None
    except Exception as e:
        return stem, None, "", 0.0, 0.0, f"{type(e).__name__}: {str(e)[:200]}"


def _post_task(args):
    stem, P, ps, score, note, parse_ms, prep_ms, gpu_ms, device = args
    from ..pipeline import write_prediction
    from .pipeline import predict_from_points
    try:
        t0 = time.perf_counter()
        tm = {"features": prep_ms + gpu_ms}
        pred = predict_from_points(P, ps, _W["fm"], _W["ranker"], score=score, timings=tm)
        write_prediction(_W["out"], pred, points=_W["points"])
        total = parse_ms + prep_ms + gpu_ms + (time.perf_counter() - t0) * 1e3
        return _row(stem, P, pred, total, parse_ms, note, device)
    except Exception as e:
        return f"{stem},error,,,,,,,,,{device},{type(e).__name__}: {str(e)[:200]}"


# ---------------------------------------------------------------- GPU runner (one process per GPU)
def gpu_runner(device, tasks, out, model_path, ranker, af_mode, points, workers, batch, rows_file):
    os.environ.update(_SINGLE)
    import torch
    from .gpu import featurize_batch
    from .pipeline import FastModel
    fm = FastModel.load(model_path) if model_path else FastModel.from_model(None)
    cfg = fm.cfg
    dev = torch.device(device)
    ctx = mp.get_context("spawn")
    rows, pend = [], []
    with ctx.Pool(workers, initializer=_init_worker, initargs=(model_path, ranker, af_mode, str(out), points)) as pool:
        buf = []

        def flush():
            t0 = time.perf_counter()
            res = featurize_batch([b[1] for b in buf], cfg, dev, None if ranker == "geometry" else fm)
            per = (time.perf_counter() - t0) * 1e3 / len(buf)
            for (stem, it, note, parse_ms, prep_ms), (ps, sc) in zip(buf, res):
                pend.append(pool.apply_async(_post_task, ((stem, it.P, ps, sc, note, parse_ms, prep_ms, per,
                                                           device),)))
            buf.clear()

        for stem, it, note, parse_ms, prep_ms, err in pool.imap(_prep_task, tasks, chunksize=1):
            if err:
                rows.append(f"{stem},error,,,,,,,,,{device},{err}")
                continue
            buf.append((stem, it, note, parse_ms, prep_ms))
            if len(buf) >= batch:
                flush()
        if buf:
            flush()
        rows += [p.get() for p in pend]
    Path(rows_file).write_text("\n".join(rows) + ("\n" if rows else ""))


# ---------------------------------------------------------------- CLI
def _model_path(a):
    if a.ranker == "geometry":
        return None
    from ..cli import DEFAULT_MODELS
    return str(a.model or ROOT / DEFAULT_MODELS[(a.ranker, a.preset)])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="burrow-fast", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cpu", help="cpu | cuda:N | cuda:0,1,3")
    p.add_argument("--workers", type=int, default=8, help="CPU worker processes (per GPU in GPU mode)")
    p.add_argument("--batch", type=int, default=32, help="structures per GPU batch")
    p.add_argument("--af-mode", action="store_true")
    p.add_argument("--ranker", choices=["all", "small", "geometry"], default="all")
    p.add_argument("--preset", choices=["default", "accurate"], default="default")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--no-points", action="store_true")
    a = ap.parse_args(argv)
    from ..cli import _expand, _stem
    os.environ.update(_SINGLE)
    paths = _expand(a.inputs)
    tasks = [(x, _stem(x)) for x in paths]
    a.out.mkdir(parents=True, exist_ok=True)
    model_path = _model_path(a)
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    if a.device == "cpu":
        with ctx.Pool(a.workers, initializer=_init_worker,
                      initargs=(model_path, a.ranker, a.af_mode, str(a.out), not a.no_points)) as pool:
            rows = list(pool.imap(_cpu_task, tasks, chunksize=1))
    else:
        ids = a.device.replace("cuda:", "").split(",")
        devs = [f"cuda:{i}" for i in ids]
        procs, files = [], []
        for gi, d in enumerate(devs):
            f = a.out / f".rows_{gi}.csv"
            files.append(f)
            pr = ctx.Process(target=gpu_runner, args=(d, tasks[gi::len(devs)], a.out, model_path, a.ranker, a.af_mode,
                                                       not a.no_points, a.workers, a.batch, str(f)))
            pr.start()
            procs.append(pr)
        for pr in procs:
            pr.join()
        rows = []
        for f in files:
            if f.exists():
                rows += [r for r in f.read_text().splitlines() if r]
                f.unlink()
    wall = time.perf_counter() - t0
    (a.out / "timing.csv").write_text("\n".join([HEADER] + rows) + "\n")
    n_ok = sum(1 for r in rows if ",ok," in r)
    print(f"[burrow-fast] {n_ok}/{len(tasks)} ok in {wall:.1f} s ({len(tasks) / max(wall, 1e-9):.1f} structures/s) "
          f"-> {a.out}", file=sys.stderr)
    return 0 if n_ok == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
