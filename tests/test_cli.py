"""End-to-end test of scripts/evaluate.py and scripts/compare_methods.py on synthetic files."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from conftest import PLUS, backbone, build, ligand_res, write

from pocketdetect.evaluate import write_predictions

ROOT = Path(__file__).resolve().parents[1]
C0 = np.array([10.0, 0.0, 3.5])


def run(script, *args):
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr + r.stdout
    return r


def test_cli_end_to_end(tmp_path):
    sdir = tmp_path / "structs"
    sdir.mkdir()
    rows = []
    for i in range(4):
        extra = [ligand_res("A", "LIG", 900, C0 + PLUS)] if i < 3 else []  # s3: no ligand
        write(build(backbone("A", 1, 10) + extra), sdir / f"s{i}.pdb")
        rows.append({"structure_id": f"s{i}", "structure_path": f"structs/s{i}.pdb", "ligand_spec": "", "chains": ""})
    man = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(man, index=False)

    good, bad = tmp_path / "pred_good", tmp_path / "pred_bad"
    good.mkdir(), bad.mkdir()
    for i in range(2):  # s2 has no prediction file -> counted as zero predictions
        write_predictions(good / f"s{i}.csv", [C0 + [0, 0, 1.0], [50, 50, 50]], residues=["A:3 A:4", ""])
        write_predictions(bad / f"s{i}.csv", [[50, 50, 50], C0 + [0, 0, 1.0]])

    out_g, out_b = tmp_path / "eval_good", tmp_path / "eval_bad"
    run("evaluate.py", "--dataset-manifest", man, "--pred-dir", good, "--out", out_g, "--ligand-rules", "p2rank", "--n-boot", 200, "--jobs", 2)
    run("evaluate.py", "--dataset-manifest", man, "--pred-dir", bad, "--out", out_b, "--ligand-rules", "p2rank", "--n-boot", 200, "--jobs", 1)
    for f in ("ligands.csv", "proteins.csv", "pockets.csv", "summary.csv", "ground_truth.csv", "run_info.json", "gt_cache.pkl"):
        assert (out_g / f).exists(), f
    info = json.loads((out_g / "run_info.json").read_text())
    assert info["n_pred_missing"] == 2 and info["n_structures_no_ligands"] == 1 and info["n_ligands"] == 3
    s = pd.read_csv(out_g / "summary.csv").set_index(["metric", "average"])["value"]
    assert s[("success_DCA4_top1", "ligand")] == 2 / 3
    sb = pd.read_csv(out_b / "summary.csv").set_index(["metric", "average"])["value"]
    assert sb[("success_DCA4_top1", "ligand")] == 0 and sb[("success_DCA4_topn2", "ligand")] == 2 / 3
    gt = pd.read_csv(out_g / "ground_truth.csv")
    assert set(gt.ligand_id) == {"LIG|A:900"} and (gt.category == "relevant").all()

    # second run reuses the ground-truth cache
    run("evaluate.py", "--dataset-manifest", man, "--pred-dir", good, "--out", out_g, "--n-boot", 0)
    r = run("compare_methods.py", "--a", out_g, "--b", out_b, "--metrics", "success_DCA4_top1,success_DCA4_topn2", "--n-boot", 500, "--out", tmp_path / "cmp.csv")
    cmp = pd.read_csv(tmp_path / "cmp.csv").set_index("metric")
    assert cmp.loc["success_DCA4_top1", "diff_ligand"] == 2 / 3
    assert cmp.loc["success_DCA4_topn2", "diff_ligand"] == 0
