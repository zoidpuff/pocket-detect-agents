"""``burrow predict <inputs...> --out <dir> [--af-mode] [--ranker all|small|geometry] [--device cpu|cuda:0]``

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
DEFAULT_MODELS = {  # trained on the D6-audited BioLiP2 train set, seed SEEDS[0]
    ("all", "default"): "results/dev/biolip/models/burrow_all_default_s0.json",
    ("small", "default"): "results/dev/biolip/models/burrow_small_default_s0.json",
    ("all", "accurate"): "results/dev/biolip/models/burrow_all_accurate_s0.json",
    ("small", "accurate"): "results/dev/biolip/models/burrow_small_accurate_s0.json",
}
# AF-mode ranker (owned by the AF-mode developer, burrow/af_rank.py); retrained after the feature freeze.
DEFAULT_AF_RANKER = "results/dev_af/models/afr_final_afpdb_pl.json"  # one-line update after the freeze
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="burrow", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--af-mode", action="store_true", help="pLDDT pruning + soft occlusion + AF ranker")
    p.add_argument("--af-ranker", type=Path, default=None,
                   help=f"AF-mode ranker JSON (default {DEFAULT_AF_RANKER}; missing -> standard ranker + warning)")
    p.add_argument("--ranker", choices=["all", "small", "geometry"], default="all")
    p.add_argument("--preset", choices=["default", "accurate"], default="default")
    p.add_argument("--model", type=Path, default=None, help="override model JSON")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no-points", action="store_true")
    a = ap.parse_args(argv)
    os.environ.setdefault("NUMBA_NUM_THREADS", str(a.threads))
    from ..structure import load_structure
    from .af import af_protein
    from .model import BurrowModel
    from .pipeline import predict_protein, write_prediction
    from .prep import protein_from_structure

    if a.device != "cpu":
        print(f"[burrow] device {a.device}: GPU backend not implemented in phase 1, running on CPU", file=sys.stderr)
    model = None
    if a.ranker != "geometry":
        root = Path(__file__).resolve().parents[3]
        mp = a.model or root / DEFAULT_MODELS[(a.ranker, a.preset)]
        model = BurrowModel.load(mp)
    af_ranker = None
    if a.af_mode and model is not None:
        root = Path(__file__).resolve().parents[3]
        afp = a.af_ranker or root / DEFAULT_AF_RANKER
        if Path(afp).exists():
            from .af_rank import AFRanker
            af_ranker = AFRanker.load(afp)
        else:
            print(f"[burrow] AF ranker {afp} not found: --af-mode uses pruning + soft occlusion with the "
                  "standard ranker", file=sys.stderr)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = ["stem,status,n_atoms,n_pockets,total_ms,parse_ms,note"]
    for path in _expand(a.inputs):
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
            else:
                pred = predict_protein(P, model, weights=w, ranker="geometry" if model is None else "learned")
            write_prediction(a.out, pred, points=not a.no_points)
            t2 = time.perf_counter()
            rows.append(f"{stem},ok,{len(P.coords)},{len(pred.scores)},{(t2 - t0) * 1e3:.1f},{(t1 - t0) * 1e3:.1f},{note}")
        except Exception as e:  # never abort a batch; record the failure
            rows.append(f"{stem},error,,,,,{type(e).__name__}: {str(e)[:200]}")
            print(f"[burrow] {stem}: {type(e).__name__}: {e}", file=sys.stderr)
    (a.out / "timing.csv").write_text("\n".join(rows) + "\n")
    n_ok = sum(1 for r in rows[1:] if ",ok," in r)
    print(f"[burrow] {n_ok}/{len(rows) - 1} ok -> {a.out}")
    return 0 if n_ok == len(rows) - 1 else 1


if __name__ == "__main__":
    sys.exit(main())
