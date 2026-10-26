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

import numpy as np
import os
import sys
import time
from pathlib import Path

_SINGLE = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}
ROOT = Path(__file__).resolve().parents[4]
HEADER = "stem,status,n_atoms,n_pockets,total_ms,parse_ms,features_ms,point_score_ms,pockets_ms,rank_ms,device,note"

# ---------------------------------------------------------------- worker state (per process)
_W = {}


def _init_worker(model_path, ranker, af_mode, out, points, compact=0):
    os.environ.update(_SINGLE)
    import numba
    numba.set_num_threads(1)
    from .pipeline import FastModel
    _W.update(fm=FastModel.load(model_path) if model_path else FastModel.from_model(None), ranker=ranker,
              af=af_mode, out=Path(out), points=points, compact=compact)


def _load(path, stem, af):
    """Protein (+ AF weights, note); with ``--compact`` also the per-residue pLDDT of the *full*
    (unpruned) protein, used for the pocket-pLDDT summary columns."""
    from ...structure import load_structure
    from ..af import af_protein, residue_plddt
    from ..prep import protein_from_structure
    s = load_structure(path, structure_id=stem)
    info = None
    if _W.get("compact"):
        sp = s.select(s.protein_heavy_mask())
        pl = residue_plddt(sp).astype(np.float32)
        info = dict(rid={r: i for i, r in enumerate(sp.residue_ids.tolist())}, pl=pl, n_res=int(len(pl)),
                    f50=float((pl < 50).mean()) if len(pl) else 0.0, f70=float((pl < 70).mean()) if len(pl) else 0.0,
                    mean=float(pl.mean()) if len(pl) else 0.0)
    if af:
        P, w, st = af_protein(s)
        return P, w, f"pruned_res={st['n_res_pruned']} soft_atoms={st['n_soft_atoms']}", info
    return protein_from_structure(s), None, "", info


def _compact(stem, P, pred, info, row):
    """Top-k pockets as TSV lines (+ one structure line)."""
    k = _W["compact"]
    lines = []
    for r, (c, sc, res) in enumerate(zip(pred.centers[:k], pred.scores[:k], pred.residues[:k]), 1):
        idx = sorted(info["rid"][x] for x in res if x in info["rid"])
        pv = info["pl"][idx] if idx else np.zeros(0, np.float32)
        rs = " ".join(sorted(res, key=_res_key))
        lines.append(f"{stem}\t{r}\t{sc:.5g}\t{c[0]:.3f}\t{c[1]:.3f}\t{c[2]:.3f}\t{len(res)}\t"
                     f"{pv.mean() if len(pv) else float('nan'):.2f}\t{pv.min() if len(pv) else float('nan'):.2f}\t{rs}")
    struct = (f"{stem}\t{info['n_res']}\t{len(set(P.res_index.tolist()))}\t{len(P.coords)}\t{info['mean']:.2f}\t"
              f"{info['f50']:.4f}\t{info['f70']:.4f}\t{len(pred.scores)}")
    return dict(row=row, pockets=lines, struct=struct)


def _res_key(r):
    from ..pipeline import _res_key as rk
    return rk(r)


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
        P, w, note, info = _load(path, stem, _W["af"])
        t1 = time.perf_counter()
        pred = predict_protein(P, _W["fm"], weights=w, ranker=_W["ranker"])
        if not _W.get("compact"):
            write_prediction(_W["out"], pred, points=_W["points"])
        row = _row(stem, P, pred, (time.perf_counter() - t0) * 1e3, (t1 - t0) * 1e3, note, "cpu")
        return _compact(stem, P, pred, info, row) if _W.get("compact") else row
    except Exception as e:  # never abort a batch
        return f"{stem},error,,,,,,,,,cpu,{type(e).__name__}: {str(e)[:200]}"


def _prep_task(args):
    path, stem = args
    from .gpu import prep
    try:
        t0 = time.perf_counter()
        P, w, note, info = _load(path, stem, _W["af"])
        t1 = time.perf_counter()
        it = prep(P, _W["fm"].cfg, w)
        return stem, (it, info), note, (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3, None
    except Exception as e:
        return stem, None, "", 0.0, 0.0, f"{type(e).__name__}: {str(e)[:200]}"


def _post_task(args):
    stem, P, ps, score, note, parse_ms, prep_ms, gpu_ms, device, info = args
    from ..pipeline import write_prediction
    from .pipeline import predict_from_points
    try:
        t0 = time.perf_counter()
        tm = {"features": prep_ms + gpu_ms}
        pred = predict_from_points(P, ps, _W["fm"], _W["ranker"], score=score, timings=tm)
        if not _W.get("compact"):
            write_prediction(_W["out"], pred, points=_W["points"])
        total = parse_ms + prep_ms + gpu_ms + (time.perf_counter() - t0) * 1e3
        row = _row(stem, P, pred, total, parse_ms, note, device)
        return _compact(stem, P, pred, info, row) if _W.get("compact") else row
    except Exception as e:
        return f"{stem},error,,,,,,,,,{device},{type(e).__name__}: {str(e)[:200]}"


# ---------------------------------------------------------------- GPU runner (one process per GPU)
def gpu_runner(device, tasks, out, model_path, ranker, af_mode, points, workers, batch, rows_file, compact=0):
    os.environ.update(_SINGLE)
    import torch
    from .gpu import featurize_batch
    from .pipeline import FastModel
    fm = FastModel.load(model_path) if model_path else FastModel.from_model(None)
    cfg = fm.cfg
    dev = torch.device(device)
    ctx = mp.get_context("spawn")
    rows, pend = [], []
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(model_path, ranker, af_mode, str(out), points, compact)) as pool:
        buf = []

        def flush():
            t0 = time.perf_counter()
            res = featurize_batch([b[1][0] for b in buf], cfg, dev, None if ranker == "geometry" else fm)
            per = (time.perf_counter() - t0) * 1e3 / len(buf)
            for (stem, (it, info), note, parse_ms, prep_ms), (ps, sc) in zip(buf, res):
                pend.append(pool.apply_async(_post_task, ((stem, it.P, ps, sc, note, parse_ms, prep_ms, per,
                                                           device, info),)))
            buf.clear()

        # bounded pipeline: at most `window` prepared structures / pending post-tasks in memory
        from collections import deque
        window = max(2 * batch, 2 * workers)
        todo = iter(tasks)
        q = deque()

        def fill():
            while len(q) < window:
                t = next(todo, None)
                if t is None:
                    return
                q.append(pool.apply_async(_prep_task, (t,)))

        fill()
        while q:
            stem, it, note, parse_ms, prep_ms, err = q.popleft().get()
            fill()
            if err:
                rows.append(f"{stem},error,,,,,,,,,{device},{err}")
                continue
            buf.append((stem, it, note, parse_ms, prep_ms))
            if len(buf) >= batch:
                flush()
                busy = [p for p in pend if not p.ready()]
                while len(busy) > window:          # back-pressure on the post-processing queue
                    busy[0].wait()
                    busy = [p for p in busy if not p.ready()]
        if buf:
            flush()
        rows += [p.get() for p in pend]
    _dump(rows, rows_file)


def _dump(results, rows_file):
    """Write timing rows (and compact pocket / structure lines) of one runner."""
    rows = [r["row"] if isinstance(r, dict) else r for r in results]
    Path(rows_file).write_text("\n".join(rows) + ("\n" if rows else ""))
    comp = [r for r in results if isinstance(r, dict)]
    if comp:
        Path(str(rows_file) + ".pockets").write_text("".join(l + "\n" for r in comp for l in r["pockets"]))
        Path(str(rows_file) + ".structs").write_text("".join(r["struct"] + "\n" for r in comp))


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
    p.add_argument("--compact", type=int, default=0, metavar="K",
                   help="write only OUT/pockets.tsv (top-K pockets: centre, score, residues, mean/min pocket "
                        "pLDDT) and OUT/structures.tsv instead of per-structure files")
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
                      initargs=(model_path, a.ranker, a.af_mode, str(a.out), not a.no_points, a.compact)) as pool:
            res = list(pool.imap(_cpu_task, tasks, chunksize=1))
        f = a.out / ".rows_cpu.csv"
        _dump(res, f)
        files = [f]
    else:
        ids = a.device.replace("cuda:", "").split(",")
        devs = [f"cuda:{i}" for i in ids]
        procs, files = [], []
        for gi, d in enumerate(devs):
            f = a.out / f".rows_{gi}.csv"
            files.append(f)
            pr = ctx.Process(target=gpu_runner, args=(d, tasks[gi::len(devs)], a.out, model_path, a.ranker, a.af_mode,
                                                       not a.no_points, a.workers, a.batch, str(f), a.compact))
            pr.start()
            procs.append(pr)
        for pr in procs:
            pr.join()
    rows = []
    PH = "stem\trank\tscore\tcenter_x\tcenter_y\tcenter_z\tn_res\tmean_plddt\tmin_plddt\tresidues"
    SH = "stem\tn_res\tn_res_modelled\tn_atoms\tmean_plddt\tfrac_plddt_lt50\tfrac_plddt_lt70\tn_pockets"
    pk, st = [], []
    for f in files:
        if f.exists():
            rows += [r for r in f.read_text().splitlines() if r]
            f.unlink()
        for suf, acc in ((".pockets", pk), (".structs", st)):
            g = Path(str(f) + suf)
            if g.exists():
                acc.append(g.read_text())
                g.unlink()
    if a.compact:
        (a.out / "pockets.tsv").write_text(PH + "\n" + "".join(pk))
        (a.out / "structures.tsv").write_text(SH + "\n" + "".join(st))
    wall = time.perf_counter() - t0
    (a.out / "timing.csv").write_text("\n".join([HEADER] + rows) + "\n")
    n_ok = sum(1 for r in rows if ",ok," in r)
    print(f"[burrow-fast] {n_ok}/{len(tasks)} ok in {wall:.1f} s ({len(tasks) / max(wall, 1e-9):.1f} structures/s) "
          f"-> {a.out}", file=sys.stderr)
    return 0 if n_ok == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
