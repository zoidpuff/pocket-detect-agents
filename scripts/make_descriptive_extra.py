#!/usr/bin/env python
"""Build the --extra JSON for scripts/run_prereg_analysis.py: descriptive rows (no claims) for the
ablation models (A1 fold-disjoint, A2 CHEN11-only, A3 shuffled labels; paired vs v1 seed 0), A1 with
the v2 output floor (paired vs v2 seed 0) and P2Rank retrained on our data (paired vs P2Rank 2.5.1
stock and vs v2 seed 0).

  .venv/bin/python scripts/make_descriptive_extra.py > results/test/descriptive/extra.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P2RANK_OURS_MODEL = ROOT / "baselines/p2rank_ours/model/FastRandomForest.model"
BASE_DIR = {"ligysis": "ligysis", "ligysis_ni": "ligysis", "coach420_mlig": "coach420_mlig", "holo4k_mlig": "holo4k_mlig",
            "cryptobench": "cryptobench", "pocketwatch": "pocketwatch", "af_set": "af_set", "dimer": "dimer_assembly"}


def counterpart(variant: str) -> str:
    """desc.abl_<x>_<ranker>_s0[_af] -> burrow_<ranker>_default_s0[_af]"""
    m = re.search(r"_(all|small)_s0(_af)?$", variant)
    return f"burrow_{m.group(1)}_default_s0{m.group(2) or ''}"


def main():
    out = []
    for vdir, ref_version in (("descriptive_v1code", "v1"), ("descriptive_v2code", "v2")):
        idx_path = ROOT / "results/test" / vdir / "preds/index.json"
        if not idx_path.exists():
            continue
        idx = json.loads(idx_path.read_text())
        for key, r in sorted(idx["runs"].items()):
            if r["rc"] != 0 or r["n_pred_files"] != r["n_inputs"]:
                sys.exit(f"incomplete run {vdir}/{key}")
            refs = [{"version": ref_version, "variant": counterpart(r["variant"])}]
            models = [r["model"]] + ([r["af_ranker"]] if r.get("af_ranker") else [])
            if "afranker" in r["variant"]:  # AF ranker row: vs the PREREG v2 variant of that benchmark and vs P2Rank -c alphafold
                refs = [{"version": "v2", "variant": "burrow_small_default_s0_af" if r["bench"] == "pocketwatch" else "burrow_all_default_s0_af"},
                        {"baseline": "p2rank_2.5.1_af"}]
            out.append({"name": r["variant"], "bench": r["bench"], "pred_dir": r["pred_dir"], "models": models,
                        "code_commit": idx["code_commit"], "compare_to": refs})
    PLAN = {"ligysis": ("all", False), "ligysis_ni": ("small", False), "coach420_mlig": ("small", False), "holo4k_mlig": ("small", False),
            "cryptobench": ("small", False), "pocketwatch": ("small", True), "af_set": ("all", True), "dimer": ("all", False)}
    for b, (ranker, af) in PLAN.items():
        d = ROOT / "baselines/outputs" / BASE_DIR[b] / "p2rank_ours" / ("preds_orig" if b in ("ligysis", "ligysis_ni") else "preds")
        if not d.exists():
            continue
        out.append({"name": "desc.p2rank_ours", "bench": b, "pred_dir": str(d.relative_to(ROOT)), "models": [str(P2RANK_OURS_MODEL)],
                    "code_commit": None,
                    "compare_to": [{"baseline": "p2rank_2.5.1"}, {"version": "v2", "variant": f"burrow_{ranker}_default_s0" + ("_af" if af else "")}]})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
