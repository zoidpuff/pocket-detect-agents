"""BURROW command line (unified): ``burrow predict`` routes to the fast backend (CPU process pool or
multi-GPU, ``src/pocketdetect/burrow/fast``) and falls back automatically to the reference
implementation (``pipeline.predict_protein``) if the fast backend is unavailable or fails on an input.

Inputs: structure files (PDB/mmCIF, .gz ok), directories (all such files) or ``@list`` files.
Output per structure: ``<out>/<stem>.csv`` (rank,score,center_x,center_y,center_z,residues) and
``<out>/<stem>.points.csv`` (rank,x,y,z), plus ``<out>/timing.csv``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# (ranker, preset) -> model JSON (the JSON also carries its feature design: rays, lengths, sub-lattice,
# score floor).  Phase 2: BioLiP2-trained dev models; frozen models will move to a release dir.
DEFAULT_MODELS = {  # FREEZE v2 (results/dev/FREEZE_v2.md; D13): v1 models + output score floor 0.03, seed SEEDS[0]
    ("all", "default"): "results/dev/freeze_v2/models/burrow_all_default_s0.json",
    ("small", "default"): "results/dev/freeze_v2/models/burrow_small_default_s0.json",
    ("all", "accurate"): "results/dev/freeze_v2/models/burrow_all_accurate_s0.json",
    ("small", "accurate"): "results/dev/freeze_v2/models/burrow_small_accurate_s0.json",
    # additional speed preset (results/dev/FREEZE_fast.md): coarser grid, fewer rays, smaller point model
    ("all", "fast"): "results/dev/freeze_fast/models/burrow_all_fast_s0.json",
    ("small", "fast"): "results/dev/freeze_fast/models/burrow_small_fast_s0.json",
}
# AF-mode ranker (owned by the AF-mode developer, burrow/af_rank.py); retrained after the feature freeze.
DEFAULT_AF_RANKER = None  # decision D8: --af-mode default = standard ranker (pruning + soft occlusion only)
REGION_RULE = "c16_0.005"   # burrow.output_rules; selected on train-CV (results/dev/biolip/V3b_CYCLE.md)
_EXT = (".pdb", ".ent", ".cif", ".mmcif", ".pdb.gz", ".ent.gz", ".cif.gz", ".mmcif.gz")


def _expand(inputs):
    out = []
    for x in inputs:
        if x.startswith("@"):
            out += [l.strip() for l in Path(x[1:]).read_text().splitlines() if l.strip()]
        elif Path(x).is_dir():
            out += sorted(str(p) for p in Path(x).iterdir() if p.name.lower().endswith(_EXT))
        else:
            out.append(x)
    return out


def _stem(path):
    n = Path(path).name
    for e in sorted(_EXT, key=len, reverse=True):
        if n.lower().endswith(e):
            return n[: -len(e)]
    return Path(n).stem


USAGE = """burrow predict <files|dirs|@list> --out OUT [--ranker all|small|geometry] [--preset default|accurate|fast]
               [--min-score none|<float>] [--af-mode [--af-ranker PATH]] [--model M.json]
               [--device cpu|cuda:N[,M,...]] [--workers N] [--backend auto|fast|reference] [--no-points]"""


def _model_file(a):
    root = Path(__file__).resolve().parents[3]
    return None if a.ranker == "geometry" else Path(a.model or root / DEFAULT_MODELS[(a.ranker, a.preset)])


def _min_score_override(model_path: Path, min_score, tmpdir: Path) -> Path:
    """Copy of the model JSON with the output floor overridden (the fast backend reads the floor
    from the model's ``formation``)."""
    import json
    d = json.loads(Path(model_path).read_text())
    d["formation"] = dict(d["formation"], min_score=None if str(min_score).lower() == "none" else float(min_score))
    out = tmpdir / ("model_minscore_" + Path(model_path).name)
    out.write_text(json.dumps(d))
    return out


def _run_reference(a, paths, out: Path):
    """Reference implementation (pure CPU, one structure at a time)."""
    from ..structure import load_structure
    from .af import af_protein
    from .experimental import is_experimental, predict_protein_experimental
    from .model import BurrowModel
    from .pipeline import predict_protein, write_prediction
    from .prep import protein_from_structure

    model = None
    mp_ = _model_file(a)
    if mp_ is not None:
        model = BurrowModel.load(mp_)
        if a.min_score is not None:
            model.formation = dict(model.formation,
                                   min_score=None if str(a.min_score).lower() == "none" else float(a.min_score))
    af_ranker = None
    afp = a.af_ranker or DEFAULT_AF_RANKER
    if a.af_mode and model is not None and afp is not None:
        if Path(afp).exists():
            from .af_rank import AFRanker
            af_ranker = AFRanker.load(afp)
        else:
            print(f"[burrow] AF ranker {afp} not found: --af-mode uses pruning + soft occlusion with the "
                  "standard ranker", file=sys.stderr)
    rows = []
    for path in paths:
        stem = _stem(path)
        try:
            t0 = time.perf_counter()
            s = load_structure(path, structure_id=stem)
            if a.af_mode:
                P, w, st = af_protein(s)
                note = f"pruned_res={st['n_res_pruned']} soft_atoms={st['n_soft_atoms']}"
            else:
                P, w, note = protein_from_structure(s), None, ""
            t1 = time.perf_counter()
            if af_ranker is not None:
                from .af_rank import predict_protein_af
                pred = predict_protein_af(P, model, af_ranker, weights=w)
            elif is_experimental(model):
                pred = predict_protein_experimental(P, model, weights=w)
            else:
                pred = predict_protein(P, model, weights=w, ranker="geometry" if model is None else "learned")
            write_prediction(out, pred, points=not a.no_points)
            t2 = time.perf_counter()
            rows.append(f"{stem},ok,{len(P.coords)},{len(pred.scores)},{(t2 - t0) * 1e3:.1f},{(t1 - t0) * 1e3:.1f},"
                        f"reference,{note}")
        except Exception as e:  # never abort a batch; record the failure
            rows.append(f"{stem},error,,,,,reference,{type(e).__name__}: {str(e)[:200]}")
            print(f"[burrow] {stem}: {type(e).__name__}: {e}", file=sys.stderr)
    return rows


def _run_fast(a, paths, out: Path):
    """Fast backend (src/pocketdetect/burrow/fast; CPU process pool or multi-GPU). Returns the set of
    stems it produced successfully."""
    import tempfile
    from .fast import run as fast_run
    with tempfile.TemporaryDirectory() as td:
        lst = Path(td) / "inputs.list"
        lst.write_text("\n".join(paths) + "\n")
        argv = ["predict", f"@{lst}", "--out", str(out), "--device", a.device, "--workers", str(a.workers),
                "--ranker", a.ranker]                      # the model path (below) selects the preset
        mp_ = _model_file(a)
        if mp_ is not None and a.min_score is not None:
            mp_ = _min_score_override(mp_, a.min_score, Path(td))
        if mp_ is not None:
            argv += ["--model", str(mp_)]
        if a.af_mode:
            argv.append("--af-mode")
        if a.no_points:
            argv.append("--no-points")
        fast_run.main(argv)
    ok = set()
    tf = out / "timing.csv"
    if tf.exists():
        for line in tf.read_text().splitlines()[1:]:
            f = line.split(",")
            if len(f) > 1 and f[1] == "ok":
                ok.add(f[0])
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="burrow", description=__doc__ + "\n" + USAGE,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--af-mode", action="store_true", help="pLDDT pruning + soft occlusion (AF/predicted models)")
    p.add_argument("--af-ranker", type=Path, default=None,
                   help="optional AF-mode ranker JSON (default: none -> standard ranker; D8); reference backend only")
    p.add_argument("--ranker", choices=["all", "small", "geometry"], default="all")
    p.add_argument("--preset", choices=["default", "accurate", "fast"], default="default",
                   help="default | accurate (slower, +1-2 pt on non-ion sites) | fast (speed preset, FREEZE_fast.md)")
    p.add_argument("--model", type=Path, default=None, help="override model JSON")
    p.add_argument("--min-score", default=None,
                   help="override the model's output score floor (v2 models: 0.03); 'none' = output the full "
                        "NMS'd pocket list (up to 50; recommended for exhaustive discovery)")
    p.add_argument("--output-rule", choices=["v2", "region"], default="v2",
                   help="v2 (default): score floor 0.03 after 3 pockets; region: v2 plus the best pocket (score >= "
                        "0.005) of every surface region > 16 Å from kept pockets — more coverage for exhaustive "
                        "discovery (dev evidence only, results/dev/biolip/V3b_CYCLE.md)")
    p.add_argument("--device", default="cpu", help="cpu | cuda:N | cuda:0,1 (GPU = fast backend)")
    p.add_argument("--workers", type=int, default=1, help="CPU worker processes (per GPU in GPU mode)")
    p.add_argument("--backend", choices=["auto", "fast", "reference"], default="auto",
                   help="auto (default): fast backend with automatic fallback to the reference implementation")
    p.add_argument("--threads", type=int, default=1, help="Numba threads (reference backend)")
    p.add_argument("--no-points", action="store_true")
    a = ap.parse_args(argv)
    os.environ.setdefault("NUMBA_NUM_THREADS", str(a.threads))
    if a.output_rule != "v2":
        if a.min_score is not None:
            ap.error("--output-rule region cannot be combined with --min-score")
        a.min_score = "none"          # full list from either backend, then the post-filter below
    paths = _expand(a.inputs)
    a.out.mkdir(parents=True, exist_ok=True)
    backend = a.backend
    if backend != "reference" and _model_file(a) is not None:
        import json
        meta = json.loads(_model_file(a).read_text()).get("meta", {})
        if any(meta.get(f) for f in ("apo_feats", "if_feats")):
            print("[burrow] experimental model features: using the reference backend", file=sys.stderr)
            backend = "reference"
    if backend != "reference" and a.af_ranker is not None:
        print("[burrow] --af-ranker is supported by the reference backend only: using it", file=sys.stderr)
        backend = "reference"
    done, rows = set(), []
    if backend in ("auto", "fast"):
        try:
            done = _run_fast(a, paths, a.out)
            rows = [l for l in (a.out / "timing.csv").read_text().splitlines()[1:] if l.split(",")[0] in done]
        except Exception as e:
            if backend == "fast":
                raise
            print(f"[burrow] fast backend unavailable ({type(e).__name__}: {e}); using the reference "
                  "implementation", file=sys.stderr)
    todo = [x for x in paths if _stem(x) not in done]
    if todo and backend == "fast":
        print(f"[burrow] fast backend failed on {len(todo)} input(s)", file=sys.stderr)
    elif todo:
        if done:
            print(f"[burrow] {len(todo)} input(s) re-run with the reference implementation", file=sys.stderr)
        rows += _run_reference(a, todo, a.out)
    if a.output_rule == "region":
        from .output_rules import filter_prediction_files
        for x in paths:
            if (a.out / f"{_stem(x)}.csv").exists():
                filter_prediction_files(a.out, _stem(x), REGION_RULE, 300.0)
    header = "stem,status,n_atoms,n_pockets,total_ms,parse_ms,backend_or_timings..."
    (a.out / "timing.csv").write_text("\n".join([header] + rows) + "\n")
    n_ok = sum(1 for r in rows if ",ok," in r)
    print(f"[burrow] {n_ok}/{len(paths)} ok -> {a.out}")
    return 0 if n_ok == len(paths) else 1


if __name__ == "__main__":
    sys.exit(main())
