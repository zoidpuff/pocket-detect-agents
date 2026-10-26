import math

import numpy as np
import pandas as pd
import pytest
from conftest import PLUS, backbone, build, ligand_res, make_gt, make_ligand, write

from pocketdetect.evaluate import (
    Predictions,
    compare_methods,
    empty_predictions,
    evaluate,
    evaluate_structure,
    parse_criterion,
    pooled_average_precision,
    read_predictions,
    write_predictions,
)
from pocketdetect.ligands import build_ground_truth


def preds(sid, centers, scores=None, residues=None):
    centers = np.asarray(centers, float).reshape(-1, 3)
    m = len(centers)
    scores = np.arange(m, 0, -1.0) if scores is None else np.asarray(scores, float)
    res = [frozenset(r.split()) if isinstance(r, str) else frozenset(r) for r in (residues or [""] * m)]
    return Predictions(sid, centers, scores, np.arange(1, m + 1), res)


def one(gt, pr, **kw):
    lig, prot, pk = evaluate_structure(gt, pr, **kw)
    return pd.DataFrame(lig), prot, pk


L1 = make_ligand(PLUS, "LIG|A:1")


@pytest.mark.parametrize("d,ok", [(3.9, True), (4.0, True), (4.1, False)])
def test_distance_thresholds_inclusive(d, ok):
    lig, prot, _ = one(make_gt("s", [L1]), preds("s", [[0, 0, d]]))
    assert lig.loc[0, "dca_top1"] == pytest.approx(d)
    assert lig.loc[0, "dcc_top1"] == pytest.approx(d)
    assert lig.loc[0, "success_DCA4_top1"] == ok
    assert lig.loc[0, "success_DCC4_top1"] == ok
    assert lig.loc[0, "rank_DCA4"] == (1 if ok else -1)


def test_dca_vs_dcc():
    lig, _, _ = one(make_gt("s", [L1]), preds("s", [[4.5, 0, 0]]))
    assert lig.loc[0, "dca_top1"] == pytest.approx(3.5)
    assert lig.loc[0, "dcc_top1"] == pytest.approx(4.5)
    assert lig.loc[0, "success_DCA4_top1"] and not lig.loc[0, "success_DCC4_top1"]


def test_topn_and_topn2_and_pocket_precision_and_ap():
    L2 = make_ligand(PLUS + [30, 0, 0], "LIG|A:2")
    gt = make_gt("s", [L1, L2])
    pr = preds("s", [[15, 0, 0], [0, 0, 1], [30, 0, 1]])
    lig, prot, pk = one(gt, pr)
    assert list(lig["rank_DCA4"]) == [2, 3]
    assert list(lig["success_DCA4_top1"]) == [False, False]
    assert list(lig["success_DCA4_topn"]) == [True, False]  # n = 2
    assert list(lig["success_DCA4_top3"]) == [True, True]
    assert list(lig["success_DCA4_topn2"]) == [True, True]
    assert prot["pocket_precision_top1"] == 0.0
    assert prot["pocket_precision_topn"] == pytest.approx(0.5)
    assert prot["pocket_precision_top3"] == pytest.approx(2 / 3)
    assert prot["pocket_precision_topn2"] == pytest.approx(2 / 3)  # only 3 pockets exist
    assert prot["ap"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert list(pk["tp_greedy"]) == [False, True, True]


def test_no_predictions():
    lig, prot, pk = one(make_gt("s", [L1]), empty_predictions("s"))
    assert not lig.filter(like="success_").to_numpy().any()
    assert lig.loc[0, "rank_DCA4"] == -1 and math.isnan(lig.loc[0, "dca_top1"])
    assert math.isnan(prot["pocket_precision_top1"]) and prot["ap"] == 0.0
    assert len(pk) == 0


def test_more_ligands_than_predictions():
    ligs = [make_ligand(PLUS + [30 * i, 0, 0], f"LIG|A:{i+1}") for i in range(3)]
    lig, prot, _ = one(make_gt("s", ligs), preds("s", [[0, 0, 1]]))
    assert lig["success_DCA4_topn"].mean() == pytest.approx(1 / 3)
    assert lig["success_DCA4_topn2"].mean() == pytest.approx(1 / 3)
    assert prot["pocket_precision_topn"] == 1.0
    assert prot["ap"] == pytest.approx(1 / 3)


def test_rank_is_authoritative_and_ties(tmp_path):
    p = tmp_path / "s.csv"
    # file order differs from rank order; equal scores
    pd.DataFrame(
        {"rank": [2, 1], "score": [0.5, 0.5], "center_x": [0, 50], "center_y": [0, 0], "center_z": [1, 0], "residues": ["", ""]}
    ).to_csv(p, index=False)
    pr = read_predictions(p)
    assert list(pr.ranks) == [1, 2] and pr.centers[0, 0] == 50
    lig, _, _ = one(make_gt("s", [L1]), pr)
    assert lig.loc[0, "rank_DCA4"] == 2 and not lig.loc[0, "success_DCA4_top1"]
    pd.DataFrame({"rank": [1, 1], "score": [1, 1], "center_x": [0, 0], "center_y": [0, 0], "center_z": [0, 0], "residues": ["", ""]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="duplicate ranks"):
        read_predictions(p)


def test_read_predictions_missing_column(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame({"rank": [1], "score": [1], "center_x": [0], "center_y": [0], "center_z": [0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="lacks columns"):
        read_predictions(p)
    p.write_text("rank,score,center_x,center_y,center_z,residues\n")
    assert read_predictions(p).n == 0


def test_duplicate_centers():
    lig, prot, pk = one(make_gt("s", [L1]), preds("s", [[0, 0, 1], [0, 0, 1]]))
    assert lig.loc[0, "success_DCA4_topn"]
    assert prot["pocket_precision_topn2"] == 1.0  # both pockets lie at a ligand
    assert list(pk["tp_greedy"]) == [True, False]  # the duplicate is a FP for AP
    assert prot["ap"] == 1.0


def test_residue_overlap_multichain_insertion_codes():
    T = ("A:52A", "B:12", "A:53")
    lig = make_ligand(PLUS, "LIG|A:1", residues=T)
    universe = ["A:52A", "A:53", "A:99", "B:12"] + [f"A:{i}" for i in range(1, 7)]
    gt = make_gt("s", [lig], universe)
    pr = preds("s", [[0, 0, 1]], residues=["A:52A B:12 A:99 C:1"])
    ldf, prot, _ = one(gt, pr)
    # C:1 is not a protein residue of this structure -> reported, counted as FP, never TN
    assert prot["n_pred_res_unknown"] == 1
    assert ldf.loc[0, "res_jaccard_matched"] == pytest.approx(2 / 5)  # {52A,B12} / {52A,B12,53,99,C:1}
    assert ldf.loc[0, "res_irel_matched"] == pytest.approx(2 / 3)
    assert ldf.loc[0, "res_precision_matched"] == pytest.approx(2 / 4)
    tp, fp, fn, tn = 2, 2, 1, 10 - 4
    assert (prot["res_topn2_tp"], prot["res_topn2_fp"], prot["res_topn2_fn"], prot["res_topn2_tn"]) == (tp, fp, fn, tn)
    assert prot["res_topn2_precision"] == pytest.approx(2 / 4)
    assert prot["res_topn2_recall"] == pytest.approx(2 / 3)
    assert prot["res_topn2_mcc"] == pytest.approx((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    # unmatched pocket -> overlap 0
    ldf, _, _ = one(gt, preds("s", [[0, 0, 20]], residues=["A:52A B:12"]))
    assert ldf.loc[0, "res_jaccard_matched"] == 0.0 and ldf.loc[0, "res_jaccard_best_topn2"] == pytest.approx(2 / 3)
    # no residue lists -> NaN, not 0
    ldf, prot, _ = one(gt, preds("s", [[0, 0, 1]]))
    assert math.isnan(ldf.loc[0, "res_jaccard_matched"]) and math.isnan(prot["res_topn2_mcc"])


def test_ldcc_closest_matching():
    pr = preds("s", [[0, 0, 10], [0, 0, 2]])
    ldf, _, _ = one(make_gt("s", [L1]), pr, criteria=["DCC12", "LDCC12"])
    assert ldf.loc[0, "rank_DCC12"] == 1 and ldf.loc[0, "rank_LDCC12"] == 2
    assert ldf.loc[0, "success_DCC12_top1"] and not ldf.loc[0, "success_LDCC12_top1"]
    assert ldf.loc[0, "success_LDCC12_topn2"]
    with pytest.raises(ValueError):
        parse_criterion("XYZ4")


def test_hydrogen_excluded_from_dca(tmp_path):
    c0 = np.array([10.0, 0.0, 3.5])
    lig = ligand_res("A", "LIG", 900, np.vstack([c0 + PLUS, c0 + [0, 0, 1.0]]), ["C"] * 5 + ["H"])
    path = write(build(backbone("A", 1, 10) + [lig]), tmp_path / "h.cif")
    pr = preds("h", [c0 + [0, 0, 4.8]])
    gt = build_ground_truth(path, structure_id="h", rules="p2rank")
    ldf, _, _ = one(gt, pr)
    assert ldf.loc[0, "ligand_n_atoms"] == 5 and not ldf.loc[0, "success_DCA4_top1"]
    gt = build_ground_truth(path, structure_id="h", rules="p2rank_strict")
    ldf, _, _ = one(gt, pr)
    assert ldf.loc[0, "ligand_n_atoms"] == 6 and ldf.loc[0, "success_DCA4_top1"]


def summary_value(res, metric, average):
    s = res.summary
    return s[(s.metric == metric) & (s.average == average)].iloc[0]


def test_ligand_vs_protein_average_and_bootstrap():
    L2 = make_ligand(PLUS + [30, 0, 0], "LIG|A:2")
    gts = [make_gt("p1", [L1, L2]), make_gt("p2", [L1]), make_gt("p3", [])]
    pr = {"p1": preds("p1", [[0, 0, 1], [30, 0, 1]]), "p2": preds("p2", [[0, 0, 20]])}
    res = evaluate(gts, pr, n_boot=500)
    assert res.info["n_structures_no_ligands"] == 1
    assert res.info["n_pred_missing"] == 1  # p3 has no prediction entry
    r = summary_value(res, "success_DCA4_topn", "ligand")
    assert r.value == pytest.approx(2 / 3) and r.n == 3
    assert r.ci_low <= r.value <= r.ci_high
    r = summary_value(res, "success_DCA4_topn", "protein")
    assert r.value == pytest.approx(0.5) and r.n == 2
    # determinism
    res2 = evaluate(gts, pr, n_boot=500)
    pd.testing.assert_frame_equal(res.summary, res2.summary)
    # all-success -> degenerate CI
    res3 = evaluate(gts[:1], pr, n_boot=200)
    r = summary_value(res3, "success_DCA4_topn", "ligand")
    assert (r.value, r.ci_low, r.ci_high) == (1.0, 1.0, 1.0)


def test_pooled_ap():
    sc = np.array([0.9, 0.5, 0.8, 0.7])
    tp = np.array([True, False, False, True])
    assert pooled_average_precision(sc, tp, 2) == pytest.approx((1 + 2 / 3) / 2)
    # tie at the top: one threshold -> precision 1/2 at recall 1
    assert pooled_average_precision(np.array([0.9, 0.9]), np.array([True, False]), 1) == pytest.approx(0.5)
    assert pooled_average_precision(np.array([0.9]), np.array([True]), 2) == pytest.approx(0.5)  # missed ligand
    gts = [make_gt("p1", [L1]), make_gt("p2", [L1])]
    pr = {"p1": preds("p1", [[0, 0, 1], [9, 9, 9]], [0.9, 0.5]), "p2": preds("p2", [[9, 9, 9], [0, 0, 1]], [0.8, 0.7])}
    res = evaluate(gts, pr, n_boot=100)
    assert summary_value(res, "ap_pooled", "pooled").value == pytest.approx((1 + 2 / 3) / 2)
    assert summary_value(res, "ap", "protein").value == pytest.approx((1 + 0.5) / 2)


def test_compare_methods():
    rows_a, rows_b = [], []
    for i in range(20):
        rows_a.append({"structure_id": f"p{i}", "ligand_id": "L", "success_DCA4_topn": 1.0})
        rows_b.append({"structure_id": f"p{i}", "ligand_id": "L", "success_DCA4_topn": float(i % 2)})
    a, b = pd.DataFrame(rows_a), pd.DataFrame(rows_b)
    out = compare_methods(a, b, "success_DCA4_topn", n_boot=2000)
    assert out["diff_ligand"] == pytest.approx(0.5)
    assert out["mcnemar_a_only"] == 10 and out["mcnemar_b_only"] == 0
    assert out["p_mcnemar_exact"] == pytest.approx(2 * 0.5**10)
    assert out["diff_ligand_ci"][0] > 0 and out["p_bootstrap_ligand"] < 0.01
    same = compare_methods(a, a, "success_DCA4_topn", n_boot=200)
    assert same["diff_ligand"] == 0 and same["p_mcnemar_exact"] == 1.0 and same["p_bootstrap_ligand"] == 1.0
    # a ligand missing from B's table counts as a miss for B (failures = misses), not dropped
    out2 = compare_methods(a, b.iloc[:-1], "success_DCA4_topn", n_boot=200)
    assert out2["n_ligands"] == 20 and out2["n_filled_as_miss"] == 1
    assert out2["b_ligand"] == pytest.approx(9 / 20)  # p19 (a hit for B) became a miss
    # grouped resampling: all structures in one unit -> degenerate CI
    g = {f"p{i}": "u0" for i in range(20)}
    out3 = compare_methods(a, b, "success_DCA4_topn", n_boot=200, groups=g)
    assert out3["n_units"] == 1 and out3["diff_ligand_ci"][0] == out3["diff_ligand_ci"][1] == pytest.approx(0.5)


def test_write_read_roundtrip(tmp_path):
    p = tmp_path / "x.csv"
    write_predictions(p, [[1, 2, 3], [4, 5, 6]], [0.7, 0.2], ["A:1 A:2A", ""])
    pr = read_predictions(p)
    assert pr.structure_id == "x" and pr.n == 2
    assert pr.residues[0] == frozenset({"A:1", "A:2A"}) and pr.residues[1] == frozenset()
    assert np.allclose(pr.centers, [[1, 2, 3], [4, 5, 6]])
