#!/usr/bin/env python
"""FREEZE v2 (decisions D12/D13) = FREEZE v1 models + output score floor tau = 0.03 (min_keep 3).

tau was selected on TRAIN cluster-CV by scripts/burrow_tau_cv.py (results/dev/biolip/tau_cv_p3_*.csv).
Writes results/dev/freeze_v2/models/*.json and results/dev/FREEZE_v2.md (sha256 of every model and of
its v1 source).  ``--out-dir`` writes elsewhere (used to verify byte-identical regeneration).

  python scripts/burrow_freeze_v2.py [--out-dir DIR] [--no-md]
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.model import BurrowModel  # noqa: E402

TAU, MIN_KEEP = 0.03, 3


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/dev/freeze_v2/models")
    ap.add_argument("--no-md", action="store_true")
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    lines = ["# FREEZE v2 — BURROW = FREEZE v1 models + output score floor (H5, decision D12)", "",
             "Each model = the v1 model of the same name (results/dev/freeze/models/, unchanged point models and "
             f"rankers) with `formation.min_score = {TAU}`, `formation.min_keep = {MIN_KEEP}` (pockets with ranker "
             f"score < {TAU} are dropped once {MIN_KEEP} pockets are kept). tau selected on TRAIN cluster-CV only "
             "(results/dev/biolip/tau_cv_p3_{small,all}_default_s0.csv; joint criterion over both rankers), one dev "
             "confirmation (results/dev/biolip/V2_CYCLE.md). Same tau for all/small and default/accurate. "
             f"Regenerate with scripts/burrow_freeze_v2.py (git HEAD at writing: {git}).", "",
             "| model | sha256 | v1 source sha256 |", "|---|---|---|"]
    for preset in ("default", "accurate"):
        for var in ("all", "small"):
            for si in range(3):
                n = f"burrow_{var}_{preset}_s{si}.json"
                src = Path("results/dev/freeze/models") / n          # relative path recorded in meta
                m = BurrowModel.load(ROOT / src)
                m.formation = dict(m.formation, min_score=TAU, min_keep=MIN_KEEP)
                m.meta = dict(m.meta, freeze="v2", v2_variant="H5_output_score_floor", v2_tau=TAU, v2_min_keep=MIN_KEEP,
                              v2_source_model=str(src), v2_source_sha256=sha(ROOT / src),
                              v2_selection="train cluster-CV (scripts/burrow_tau_cv.py), D12; dev confirmation V2_CYCLE.md")
                out = a.out_dir / n
                m.save(out)
                lines.append(f"| results/dev/freeze_v2/models/{n} | {sha(out)} | {sha(ROOT / src)} |")
    if not a.no_md:
        (ROOT / "results/dev/FREEZE_v2.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
