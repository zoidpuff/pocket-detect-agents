#!/usr/bin/env python
"""Additional `fast` preset freeze: models trained with the fast feature design (scripts/burrow_train_biolip.py
runs named fast<CFG>_<ranker>_s<i>), with the v2 output rule (score floor 0.03, min_keep 3) and provenance.

  python scripts/burrow_freeze_fast.py --cfg C6 --seeds 0[,1,2]
Writes results/dev/freeze_fast/models/burrow_{all,small}_fast_s<i>.json and results/dev/FREEZE_fast.md.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_freeze import provenance  # noqa: E402
from pocketdetect.burrow.model import BurrowModel  # noqa: E402

OUT = ROOT / "results/dev/freeze_fast/models"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for si in [int(x) for x in a.seeds.split(",")]:
        for var in ("all", "small"):
            src = ROOT / f"results/dev/biolip/models/fast{a.cfg}_{var}_s{si}.json"
            m = BurrowModel.load(src)
            m.formation = dict(m.formation, min_score=0.03, min_keep=3)
            m.meta = dict(m.meta, preset="fast", fast_cfg=a.cfg, source_model=str(src.relative_to(ROOT)),
                          output_rule="v2 floor 0.03 / min_keep 3", provenance=provenance(si))
            out = OUT / f"burrow_{var}_fast_s{si}.json"
            m.save(out)
            rows.append(f"| results/dev/freeze_fast/models/{out.name} | {sha(out)} |")
    fc = m.meta["feature_cfg"]
    txt = ["# FREEZE fast — additional speed preset (`burrow predict --preset fast`)", "",
           f"Feature design `{a.cfg}`: grid spacing {fc['spacing']} Å, {fc['n_rays']} rays, lengths {fc['lengths']} Å, "
           f"sub-lattice {fc['subsample']}, shell ≤ {fc['max_dist']} Å, PSP length {fc['psp_length']} Å, metal candidates "
           f"{'on' if fc['metal'] else 'off'}; point model {m.meta.get('point_rounds')} trees × {m.meta.get('point_leaves')} leaves, "
           f"points scored if b ≥ {m.meta.get('score_floor')}; pocket link {m.formation['link']} Å, min volume "
           f"{m.formation['min_vol']} points; v2 output rule (floor 0.03, min_keep 3). Trained on the D6-audited BioLiP2 "
           "train set (same as v2). Not a pre-registered model. Selection and dev numbers: results/dev/biolip/FAST_PRESET.md.",
           "", "| model | sha256 |", "|---|---|"] + rows
    (ROOT / "results/dev/FREEZE_fast.md").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
