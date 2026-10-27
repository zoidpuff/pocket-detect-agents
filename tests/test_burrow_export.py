"""pocketdetect.burrow.export.pocket_representations: pockets identical to the frozen pipeline, consistent
point / descriptor tables, and unchanged default CLI predictions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pocketdetect.burrow.cli import DEFAULT_MODELS
from pocketdetect.burrow.export import pocket_representations
from pocketdetect.burrow.model import BurrowModel
from pocketdetect.burrow.pipeline import predict_protein
from pocketdetect.burrow.prep import load_protein

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/raw/p2rank_datasets/current/chen11/a.001.001.001_1s69a.pdb"
V2 = ROOT / "results/dev/biolip/v2_preds/H5_small_default_s0"
LIST = ROOT / "results/dev/biolip/inputs_audited.list"

pytestmark = pytest.mark.skipif(not (REAL.exists() and (ROOT / DEFAULT_MODELS[("all", "default")]).exists()),
                                reason="test structure or frozen models not available")


@pytest.mark.parametrize("ranker,preset", [("all", "default"), ("small", "default"), ("all", "fast")])
def test_pockets_identical_to_predict_protein(ranker, preset):
    rep = pocket_representations(REAL, ranker=ranker, preset=preset)
    m = BurrowModel.load(ROOT / DEFAULT_MODELS[(ranker, preset)])
    ref = predict_protein(load_protein(REAL), m)
    pk = rep["pockets"]
    assert len(pk) == len(ref.scores) > 0
    assert np.allclose(pk[["center_x", "center_y", "center_z"]].to_numpy(), ref.centers)
    assert np.allclose(pk.score.to_numpy(), ref.scores)
    assert list(pk.residues) == [" ".join(sorted(r)) for r in ref.residues]
    for r, pts in zip(pk["rank"], ref.points):        # point sets of every pocket
        assert np.allclose(rep["points"].loc[rep["pocket_points"][r], ["x", "y", "z"]].to_numpy(), pts)


def test_tables_are_consistent():
    rep = pocket_representations(REAL, ranker="all", min_score=None)
    pts, pk = rep["points"], rep["pockets"]
    assert len(rep["point_feature_names"]) == 84 and len(rep["descriptor_names"]) == 76
    assert all(n in pts.columns for n in rep["point_feature_names"])
    assert all(n in pk.columns for n in rep["descriptor_names"])
    assert len(pk) >= len(pocket_representations(REAL, ranker="all")["pockets"])   # floor only removes pockets
    for r, n in zip(pk["rank"], pk.n_points):
        assert len(rep["pocket_points"][r]) == n == pk.loc[pk["rank"] == r, "n_pts"].iloc[0]


@pytest.mark.skipif(not (V2.exists() and LIST.exists()), reason="dev predictions not available")
def test_default_cli_predictions_unchanged(tmp_path):
    """Adding the export module does not change the frozen default output (v2 dev files)."""
    from pocketdetect.burrow.cli import main
    f = ROOT / LIST.read_text().split()[0]
    assert main(["predict", str(f), "--out", str(tmp_path), "--ranker", "small", "--no-points"]) == 0
    assert (tmp_path / f"{f.stem}.csv").read_text() == (V2 / f"{f.stem}.csv").read_text()
