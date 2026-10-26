#!/usr/bin/env python
"""Feature freeze: write the frozen BURROW model JSONs (SEEDS[:3] x {all, small} x {default, accurate})
with full provenance, and results/dev/FREEZE.md (paths + sha256).

Point models / feature designs come from the p3_* runs (audited D6 train set).  If a phase-3 ranker
variant is selected (--ranker-variant), the ranker is retrained on the run's cached train candidates
(out-of-fold point scores) with that variant's extra features; V0 keeps the run's own ranker.

  python scripts/burrow_freeze.py --ranker-variant V0_base
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_ranker_cv import VARIANTS, build_X, train_ranker  # noqa: E402
from pocketdetect.burrow.model import BurrowModel  # noqa: E402
from pocketdetect.burrow.propensity import PROP_NAMES  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402

OUT = ROOT / "results/dev/freeze"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(seed_idx):
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    return dict(
        train_manifest_sha256=sha(ROOT / "data/processed/train/manifest.csv"),
        dev_manifest_sha256=sha(ROOT / "data/processed/dev/manifest.csv"),
        test_union_sha256=sha(ROOT / "data/manifests/test_union.tsv"),
        train_dev_audit_exclusions_sha256=sha(ROOT / "data/manifests/train_dev_audit_exclusions.tsv"),
        git_commit=git, git_dirty_tracked=dirty, seed=SEEDS[seed_idx], seed_idx=seed_idx,
        trainset_snapshot="results/dev/cache/trainset_snapshot/audited_d6 (manifests == data/processed after D6)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranker-variant", default="V0_base", choices=list(VARIANTS))
    a = ap.parse_args(argv)
    v = VARIANTS[a.ranker_variant]
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    lines = ["# FREEZE — BURROW models at feature freeze (2026-10-26 23:00)", "",
             f"Ranker variant: {a.ranker_variant}. Default = 30 rays, 6/10/16 Å, 3 Å sub-lattice, score floor 0.7, "
             "250x63 point model; accurate = 60 rays, 6/10/16/24 Å, 2 Å sub-lattice, all points scored. "
             "Details: docs/METHOD.md, results/dev/biolip/PHASE2.md, PHASE3.md.", "",
             "| model | sha256 |", "|---|---|"]
    for preset in ("default", "accurate"):
        for var in ("all", "small"):
            for si in range(3):
                run = f"p3_{var}_{preset}_s{si}"
                # V0: the phase-2 models (the p3 retrains of 5/12 configs were verified bit-identical:
                # same point/ranker model strings and 579/579 identical dev prediction files)
                src = ROOT / f"results/dev/biolip/models/burrow_{var}_{preset}_s{si}.json"
                m = BurrowModel.load(src if a.ranker_variant == "V0_base" else ROOT / f"results/dev/biolip/models/{run}.json")
                meta = dict(m.meta)
                if a.ranker_variant != "V0_base":
                    pdir = ROOT / "results/dev/cache/pockets" / run
                    data = pickle.loads((pdir / "train.pkl").read_bytes())
                    pm = pickle.loads((pdir / "meta.pkl").read_bytes())
                    sids = [s for s in data if len(data[s][0])]
                    prop = None
                    if v.get("prop"):
                        prop = pickle.loads((pdir / "prop.pkl").read_bytes())["train"]
                        meta["propensity"] = json.loads(
                            (ROOT / f"results/dev/biolip/propensity_{run}.json").read_text())["full"]
                    Xs, extra = build_X(data, sids, pm["fn"], v["ctx"], prop)
                    X = np.concatenate(Xs)
                    q = np.concatenate([data[s][1] for s in sids])
                    rankers = [train_ranker(X, q, [len(x) for x in Xs], v, si if e == 0 else (si + 3 * e) % 10)
                               for e in range(v["ens"])]
                    m.rank_model = rankers[0].model_to_string()
                    m.rank_features = list(pm["fn"]) + list(extra) + (PROP_NAMES if v.get("prop") else [])
                    meta["context"] = v["ctx"]
                    meta["rank_ensemble"] = [r.model_to_string() for r in rankers[1:]]
                meta.update(preset=preset, ranker_variant=a.ranker_variant, provenance=provenance(si),
                            source_model=str(src.relative_to(ROOT)) if a.ranker_variant == "V0_base" else run,
                            output_residue_rule="points<=4A; fallback atoms<=6A of centre; else nearest atom")
                m.meta = meta
                name = f"burrow_{var}_{preset}_s{si}.json"
                m.save(OUT / "models" / name)
                lines.append(f"| results/dev/freeze/models/{name} | {sha(OUT / 'models' / name)} |")
                print(name, flush=True)
    (ROOT / "results/dev/FREEZE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
