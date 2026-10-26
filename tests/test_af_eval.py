import numpy as np
import pytest
from conftest import backbone, build, write

from pocketdetect.af_eval import drop_low_plddt, hallucination_counts, pocket_plddt, residue_plddt, transform_pred_dir
from pocketdetect.evaluate import Predictions, read_predictions, write_predictions
from pocketdetect.sites import Site
from pocketdetect.structure import load_structure


def af_like(tmp_path):
    res = backbone("A", 1, 20)
    for i, r in enumerate(res):  # residues 1-10 confident (90), 11-20 disordered (30)
        for a in r.atoms:
            a.b = 90.0 if i < 10 else 30.0
    return load_structure(write(build(res), tmp_path / "af.cif"))


def P(centers, residues):
    c = np.asarray(centers, float)
    return Predictions("af", c, np.arange(len(c), 0, -1.0), np.arange(1, len(c) + 1), [frozenset(r.split()) for r in residues])


def test_pocket_plddt_residues_and_centre_fallback(tmp_path):
    s = af_like(tmp_path)
    pl = residue_plddt(s)
    assert pl["A:1"] == 90.0 and pl["A:15"] == 30.0
    pred = P([[0, 0, 0], [0, 0, 0], [3.8 * 14, 0, 3], [500, 0, 0]], ["A:1 A:2", "A:9 A:12", "", ""])
    v = pocket_plddt(pred, s)
    assert v[0] == 90.0 and v[1] == pytest.approx(60.0)
    assert v[2] == 30.0  # centre-only pocket over residues 14-15 (6 A rule)
    assert np.isnan(v[3])  # nothing within 6 A
    f = drop_low_plddt(pred, v, 70)
    assert f.n == 2 and list(f.ranks) == [1, 2] and np.allclose(f.centers[1], [500, 0, 0])  # NaN kept


def test_hallucination_counts(tmp_path):
    s = af_like(tmp_path)
    site = Site("af", "0", np.array([0.0, 0, 0]), frozenset(), 1, True)
    pred = P([[3.8 * 15, 0, 3], [0, 0, 3], [3.8 * 19, 0, 3]], ["", "", ""])  # low, high(site), low (15 A apart)
    v = pocket_plddt(pred, s)
    h = hallucination_counts(pred, v, [site])
    assert h["n_low"] == 2 and h["n_hallucinated"] == 2 and h["n_hallucinated_topn2"] == 2 and h["n_pockets_topn2"] == 3
    # a low-pLDDT pocket that matches a site is not a hallucination
    site2 = Site("af", "1", np.array([3.8 * 15, 0, 3]), frozenset(), 2, True)
    assert hallucination_counts(pred, v, [site, site2])["n_hallucinated"] == 1


def test_transform_pred_dir(tmp_path):
    s_path = tmp_path / "af.cif"
    af_like(tmp_path)
    (tmp_path / "in").mkdir()
    write_predictions(tmp_path / "in/af.csv", [[3.8 * 15, 0, 3], [0, 0, 3]], [0.9, 0.5])
    log = transform_pred_dir(tmp_path / "in", tmp_path / "out", {"af": str(s_path), "missing": str(s_path)}, 70)
    out = read_predictions(tmp_path / "out/af.csv")
    assert out.n == 1 and np.allclose(out.centers[0], [0, 0, 3]) and out.scores[0] == 0.5
    assert log.set_index("structure_id").loc["missing", "missing"]
