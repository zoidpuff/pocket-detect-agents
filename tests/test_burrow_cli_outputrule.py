"""`burrow predict --output-rule region` (optional, non-default; results/dev/biolip/V3b_CYCLE.md).

* default output (`--output-rule v2`) is byte-identical to the frozen v2 dev predictions;
* `--output-rule region` reproduces the v3b dev files (c16_0.005 applied to the full list),
  with the reference implementation and with the fast backend (post-filter after either backend);
* `--output-rule region` cannot be combined with `--min-score`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pocketdetect.burrow.cli import main

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "results/dev/biolip"
V2 = B / "v2_preds/H5_small_default_s0"
REGION = B / "v3b/preds/default_small_biolip_c16_0.005"
LIST = B / "inputs_audited.list"

pytestmark = pytest.mark.skipif(not (LIST.exists() and V2.exists() and REGION.exists()),
                                reason="dev inputs / reference prediction files not available")


def _inputs(n=5):
    return [ROOT / l for l in LIST.read_text().split()[:n]]


def _run(tmp_path, name, *extra):
    files = _inputs()
    lst = tmp_path / f"{name}.list"
    lst.write_text("\n".join(map(str, files)) + "\n")
    out = tmp_path / name
    assert main(["predict", f"@{lst}", "--out", str(out), "--ranker", "small", "--workers", "2", "--no-points",
                 *extra]) == 0
    return out, [f.stem for f in files]


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_default_is_v2(tmp_path, backend):
    out, stems = _run(tmp_path, f"v2_{backend}", "--backend", backend)
    for s in stems:
        assert (out / f"{s}.csv").read_text() == (V2 / f"{s}.csv").read_text(), s


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_region_rule_reproduces_dev_files(tmp_path, backend):
    out, stems = _run(tmp_path, f"region_{backend}", "--backend", backend, "--output-rule", "region")
    for s in stems:
        assert (out / f"{s}.csv").read_text() == (REGION / f"{s}.csv").read_text(), s


def test_region_rule_rejects_min_score(tmp_path):
    with pytest.raises(SystemExit):
        main(["predict", str(_inputs(1)[0]), "--out", str(tmp_path / "x"), "--output-rule", "region",
              "--min-score", "0.1"])
