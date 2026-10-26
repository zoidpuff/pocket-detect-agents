#!/usr/bin/env python
"""Evaluate pocket predictions against ligand ground truth.

Usage:
  python scripts/evaluate.py --dataset-manifest data/manifests/coach420.csv \
      --pred-dir results/preds/p2rank/coach420 --out results/eval/p2rank/coach420 \
      --ligand-rules p2rank [--criteria DCA4,DCC4] [--n-boot 1000] [--jobs 16]

Manifest CSV columns: structure_id, structure_path [, ligand_spec] [, chains]
  * structure_path: absolute or relative to the manifest file's directory
  * ligand_spec: optional P2Rank-style explicit ligand list (e.g. "ATP,MG[A_501]"); empty = rules
  * chains: optional comma-separated author chain ids to keep ("" or "*" = all)

Outputs in --out: ligands.csv, proteins.csv, pockets.csv, summary.csv, ground_truth.csv,
run_info.json, and a ground-truth cache (gt_cache.pkl) reused when manifest+rules+code match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

import pocketdetect.ligands as ligmod
from pocketdetect.evaluate import DEFAULT_CRITERIA, evaluate
from pocketdetect.ligands import build_ground_truth, get_rules
from pocketdetect.seeds import BOOTSTRAP_SEED


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, comment=None)
    for c in ("structure_id", "structure_path"):
        if c not in df.columns:
            raise SystemExit(f"manifest {path} lacks column {c!r}")
    if df["structure_id"].duplicated().any():
        raise SystemExit(f"duplicate structure_id in {path}: {df.loc[df.structure_id.duplicated(), 'structure_id'].tolist()[:5]}")
    base = path.resolve().parent
    df["structure_path"] = [str(p if Path(p).is_absolute() else (base / p)) for p in df["structure_path"]]
    if "ligand_spec" not in df.columns:
        df["ligand_spec"] = ""
    if "chains" not in df.columns:
        df["chains"] = ""
    return df


def _gt_job(args):
    sid, path, rules, spec, chains = args
    try:
        ch = [c for c in chains.split(",") if c] if chains and chains != "*" else None
        return build_ground_truth(path, structure_id=sid, rules=rules, spec=spec or None, chains=ch), None
    except Exception as e:  # report, do not silently drop
        return None, f"{sid}: {type(e).__name__}: {e}"


def build_all(df: pd.DataFrame, rules: str, jobs: int):
    tasks = list(zip(df.structure_id, df.structure_path, [rules] * len(df), df.ligand_spec, df.chains))
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            res = list(ex.map(_gt_job, tasks, chunksize=8))
    else:
        res = [_gt_job(t) for t in tasks]
    gts = [g for g, e in res if g is not None]
    errors = [e for g, e in res if e is not None]
    return gts, errors


def cache_key(manifest: Path, rules: str) -> str:
    h = hashlib.sha256()
    h.update(manifest.read_bytes())
    h.update(json.dumps(get_rules(rules).as_dict(), sort_keys=True, default=str).encode())
    for mod in ("ligands.py", "structure.py"):
        h.update((Path(ligmod.__file__).parent / mod).read_bytes())
    return h.hexdigest()


def gt_table(gts) -> pd.DataFrame:
    rows = []
    for g in gts:
        for l in g.ligands + g.other_ligands:
            rows.append(
                {
                    "structure_id": g.structure_id,
                    "ligand_id": l.ligand_id,
                    "name": l.name,
                    "category": l.category,
                    "n_atoms": l.n_atoms,
                    "center_x": l.center[0],
                    "center_y": l.center[1],
                    "center_z": l.center[2],
                    "contact_dist": l.contact_dist,
                    "center_dist": l.center_dist,
                    "binding_residues": " ".join(l.binding_residues),
                }
            )
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-manifest", required=True, type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ligand-rules", default="p2rank")
    ap.add_argument("--criteria", default=",".join(DEFAULT_CRITERIA), help="comma list, first = primary (e.g. DCA4,DCC4,LDCC12)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gt-cache", type=Path, default=None, help="ground-truth cache file (default <out>/gt_cache.pkl)")
    ap.add_argument("--allow-gt-errors", action="store_true", help="continue if some structures fail to load (reported)")
    a = ap.parse_args(argv)

    t0 = time.time()
    get_rules(a.ligand_rules)
    df = read_manifest(a.dataset_manifest)
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.gt_cache or (a.out / "gt_cache.pkl")
    key = cache_key(a.dataset_manifest, a.ligand_rules)
    gts, errors = None, []
    if cache.exists():
        with open(cache, "rb") as f:
            obj = pickle.load(f)
        if obj.get("key") == key:
            gts, errors = obj["gts"], obj["errors"]
    if gts is None:
        gts, errors = build_all(df, a.ligand_rules, a.jobs)
        with open(cache, "wb") as f:
            pickle.dump({"key": key, "gts": gts, "errors": errors}, f)
    if errors:
        print(f"[evaluate] {len(errors)} structures failed ground-truth extraction:", file=sys.stderr)
        for e in errors[:20]:
            print("   ", e, file=sys.stderr)
        if not a.allow_gt_errors:
            raise SystemExit("aborting (use --allow-gt-errors to continue)")
    t_gt = time.time() - t0

    criteria = [c.strip() for c in a.criteria.split(",") if c.strip()]
    res = evaluate(gts, a.pred_dir, criteria=criteria, n_boot=a.n_boot, seed=a.seed)
    res.save(a.out)
    gt_table(gts).to_csv(a.out / "ground_truth.csv", index=False)
    info = dict(res.info)
    info.update(
        {
            "manifest": str(a.dataset_manifest),
            "pred_dir": str(a.pred_dir),
            "ligand_rules": a.ligand_rules,
            "rules_def": get_rules(a.ligand_rules).as_dict(),
            "gt_errors": errors,
            "gt_cache_key": key,
            "seconds_ground_truth": round(t_gt, 2),
            "seconds_total": round(time.time() - t0, 2),
        }
    )
    with open(a.out / "run_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    s = res.summary
    show = s[s.metric.str.match(r"success_.*_(top1|topn|topn2)$") | s.metric.isin(["ap", "ap_pooled", "n_proteins", "n_ligands"])]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"[evaluate] missing prediction files: {info['n_pred_missing']} / {info['n_structures']}; "
          f"structures without relevant ligands (excluded): {info['n_structures_no_ligands']}")


if __name__ == "__main__":
    main()
