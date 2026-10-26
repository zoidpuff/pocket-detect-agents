"""Test-gating scripts: version cap, dirty-tree refusal, dev script refusing test ground truths."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_test_eval as rte  # noqa: E402

ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}


def test_version_cap(tmp_path, monkeypatch):
    led = tmp_path / "L.md"
    rows = rte.LEDGER_HEADER + "".join(
        f"| 2026-10-27T0{i}:00:00Z | v{i} | no | ligysis | M | abc | x | y | z | g | p | 0 | h | d |\n" for i in range(1, 4))
    led.write_text(rows)
    monkeypatch.setattr(rte, "LEDGER", led)
    rte.check_versions("v2", False, "ligysis", "M")  # re-running an existing version is allowed
    rte.check_versions("v1", False, "ligysis", "OTHER")  # other method unaffected
    with pytest.raises(SystemExit, match="3 non-final"):
        rte.check_versions("v4", False, "ligysis", "M")
    rte.check_versions("final", True, "ligysis", "M")
    led.write_text(rows + "| 2026-10-27T05:00:00Z | final | yes | ligysis | M | abc | x | y | z | g | p | 0 | h | d |\n")
    with pytest.raises(SystemExit, match="already"):
        rte.check_versions("final2", True, "ligysis", "M")


def test_dirty_tree_refused(tmp_path):
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not dirty:
        pytest.skip("tree is clean; refusal path not exercised")
    (tmp_path / "p").mkdir()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_test_eval.py"), "--benchmark", "cryptobench", "--pred-dir", str(tmp_path / "p"),
                        "--method", "x", "--version", "t", "--model", "none"], capture_output=True, text=True, env=ENV)
    assert r.returncode != 0 and "refused" in (r.stderr + r.stdout)


def test_dev_script_refuses_test_gt(tmp_path):
    (tmp_path / "p").mkdir()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_dev_eval.py"), "--sites", str(ROOT / "data/processed/gt/cryptobench_test/sites.tsv"),
                        "--pred-dir", str(tmp_path / "p"), "--out", str(tmp_path / "o")], capture_output=True, text=True, env=ENV)
    assert r.returncode != 0 and "refused" in (r.stderr + r.stdout)


def test_dev_eval_runs(tmp_path):
    import pandas as pd

    from pocketdetect.evaluate import write_predictions
    from pocketdetect.sites import write_sites

    df = pd.DataFrame({"structure_id": ["a", "a", "b"], "site_id": ["0", "1", "0"], "center_x": [0.0, 30.0, 0.0], "center_y": [0.0] * 3,
                       "center_z": [0.0] * 3, "residues": ["", "", ""], "n_sites": [2, 2, 1], "scored": [1, 1, 1]})
    write_sites(df, tmp_path / "dev_sites.tsv")
    (tmp_path / "p").mkdir()
    write_predictions(tmp_path / "p/a.csv", [[1, 0, 0], [50, 0, 0]])
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_dev_eval.py"), "--sites", str(tmp_path / "dev_sites.tsv"), "--pred-dir", str(tmp_path / "p"),
                        "--out", str(tmp_path / "o"), "--n-boot", "50"], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr
    s = pd.read_csv(tmp_path / "o/summary.csv").set_index("metric")["value"]
    assert s["recall_topn2"] == pytest.approx(1 / 3) and s["n_pred_missing"] == 1
