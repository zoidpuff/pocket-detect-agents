"""Known-answer tests for the site-table protocol (LIGYSIS LBS-comparison metrics + strict variants)."""

import numpy as np
import pandas as pd
import pytest

from pocketdetect.evaluate import Predictions, empty_predictions
from pocketdetect.ligysis import LIGYSIS_RESULTS, filter_redundant
from pocketdetect.site_eval import SiteProtocol, compare_site_methods, evaluate_sites, pooled_precision, summarize_sites
from pocketdetect.sites import Site, read_sites, sites_by_structure, write_sites


def site(sid, i, c, res=(), n=3, scored=True, atoms=None, nres=None, group=None):
    meta = {}
    if nres is not None:
        meta["n_residues_structure"] = nres
    if group is not None:
        meta["group"] = group
    return Site(sid, str(i), np.asarray(c, float), frozenset(res), n, scored, None if atoms is None else np.asarray(atoms, float), meta)


def preds(sid, centers, scores=None, residues=None):
    centers = np.asarray(centers, float).reshape(-1, 3)
    m = len(centers)
    scores = np.arange(m, 0, -1.0) if scores is None else np.asarray(scores, float)
    res = [frozenset(r.split()) for r in (residues or [""] * m)]
    return Predictions(sid, centers, scores, np.arange(1, m + 1), res)


def S(res, **kw):
    return summarize_sites(res, n_boot=0, **kw).set_index("metric")["value"]


def test_rounding_and_threshold():
    st = {"a": [site("a", 0, [0, 0, 0], n=1)]}
    for x, ok in ((12.004, True), (12.006, False)):
        r = evaluate_sites(st, {"a": preds("a", [[x, 0, 0]])})
        assert r.sites.loc[0, "recall_topn"] == ok
    r = evaluate_sites(st, {"a": preds("a", [[12.006, 0, 0]])}, SiteProtocol(round_decimals=None))
    assert not r.sites.loc[0, "recall_topn"]


def test_closest_vs_anypocket_vs_1to1_and_N():
    # N = 3 (incl. one unscored site); scored sites at x=0 and x=30
    st = {"a": [site("a", 0, [0, 0, 0]), site("a", 1, [30, 0, 0]), site("a", 2, [60, 0, 0], scored=False)]}
    pr = preds("a", [[10, 0, 0], [50, 50, 50], [51, 50, 50], [52, 50, 50], [1, 0, 0], [31, 0, 0]])
    r = evaluate_sites(st, {"a": pr})
    s0, s1 = r.sites.iloc[0], r.sites.iloc[1]
    assert s0.rank_closest == 5 and s0.dcc_closest == 1.0
    assert not s0.recall_topn and s0.recall_topn2  # closest pocket has rank 5 = N+2
    assert s0.recall_anyp_topn  # rank-1 pocket is within 12 A
    assert s1.rank_closest == 6 and not s1.recall_topn2 and s1.recall_max
    v = S(r)
    assert v["recall_topn2"] == 0.5 and v["recall_max"] == 1.0 and v["n_scored_sites"] == 2


def test_one_to_one_and_ties():
    # one pocket midway between two sites, both within 12 A: any-pocket recalls both, 1-to-1 only one
    st = {"a": [site("a", 0, [0, 0, 0], n=2), site("a", 1, [10, 0, 0], n=2)]}
    r = evaluate_sites(st, {"a": preds("a", [[5, 0, 0]])})
    assert r.sites.recall_anyp_topn.sum() == 2 and r.sites.recall_1to1_topn.sum() == 1
    # two pockets at the same distance: larger IRel wins, then better rank
    st = {"b": [site("b", 0, [0, 0, 0], res=["A:1", "A:2"], n=1)]}
    r = evaluate_sites(st, {"b": preds("b", [[3, 0, 0], [0, 3, 0]], residues=["A:9", "A:1 A:2"])})
    assert r.sites.loc[0, "rank_closest"] == 2 and r.sites.loc[0, "irel_closest"] == 1.0
    r = evaluate_sites(st, {"b": preds("b", [[3, 0, 0], [0, 3, 0]])})
    assert r.sites.loc[0, "rank_closest"] == 1


def test_missing_predictions_are_misses():
    st = {"a": [site("a", 0, [0, 0, 0], n=1)], "b": [site("b", 0, [0, 0, 0], n=1)]}
    r = evaluate_sites(st, {"a": preds("a", [[1, 0, 0]])})
    v = S(r)
    assert v["recall_topn"] == 0.5 and v["n_pred_missing"] == 1
    assert bool(r.structures.set_index("structure_id").loc["b", "pred_missing"])


def test_dca_variants():
    atoms = [[0, 0, 0], [6, 0, 0]]
    st = {"a": [site("a", 0, [3, 0, 0], n=1, atoms=atoms)]}
    r = evaluate_sites(st, {"a": preds("a", [[20, 0, 0], [9.9, 0, 0]])})
    row = r.sites.iloc[0]
    assert row.dca_closest == pytest.approx(3.9) and row.rank_closest_dca == 2
    assert row.recall_ldca_topn2 and not row.recall_ldca_topn  # N=1: rank 2 only within N+2


def test_pooled_precision_and_one_to_one():
    t = np.array([1, 1, 0, 1, 0, 0], bool)
    assert pooled_precision(t, None, 4, 1) == (0.75, 3.0)
    p, n = pooled_precision(t, np.array([2, 0, 1, 1, 1, 1]), 4, 1)  # weighted: 1,1,0,1 at 4
    assert p == 0.75 and n == 3.0
    st = {"a": [site("a", 0, [0, 0, 0], n=1)], "b": [site("b", 0, [0, 0, 0], n=1)]}
    pr = {"a": preds("a", [[1, 0, 0], [2, 0, 0], [40, 0, 0]], [0.9, 0.8, 0.1]), "b": preds("b", [[30, 0, 0], [1, 0, 0]], [0.85, 0.7])}
    r = evaluate_sites(st, pr, SiteProtocol(p_at=4, fp_at=1))
    v = S(r)
    # pooled order: a1(T) b1(F) a2(T) b2(T) a3(F): P@4 = 3/4; 1-to-1: a2 duplicates a's site -> FP
    assert v["P1K"] == 0.75 and v["P1K_1to1"] == 0.5
    assert v["TP@1FP"] == 3 and v["TP@1FP_1to1"] == 1


def test_rro_and_residue_metrics():
    st = {"a": [site("a", 0, [0, 0, 0], res=["A:1", "A:2", "A:3", "A:4"], n=2, nres=10),
                site("a", 1, [50, 0, 0], res=["A:9"], n=2, scored=False, nres=10)]}
    r = evaluate_sites(st, {"a": preds("a", [[1, 0, 0]], residues=["A:1 A:2 A:7"])})
    v = S(r)
    assert v["RRO"] == pytest.approx(0.5)
    tp, fp, fn, tn = 2, 1, 3, 10 - 6
    t = r.structures.iloc[0]
    assert (t.res_tp, t.res_fp, t.res_fn, t.res_tn) == (tp, fp, fn, tn)
    assert t.res_f1 == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert t.res_mcc == pytest.approx((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))


def test_filter_redundant_port():
    df = pd.DataFrame({
        "origin": ["X"] * 4, "rep_chain": ["c"] * 4, "ID": [1, 2, 3, 4],
        "up_aas": [[1, 2, 3, 4], [1, 2, 3, 4], [10, 11], [20]],
        "centre": [(0, 0, 0), (20, 0, 0), (4, 0, 0), (40, 0, 0)],
        "SCORE": [0.5, 0.9, 0.1, 0.3],
    })
    out = filter_redundant(df, "SCORE")
    # 1 vs 2: Jaccard 1 and score_1 < score_2 -> 1 dropped (scan of 1 stops, so 1 vs 3 never compared);
    # 2 vs 3: no overlap, 20 A apart -> both kept; 4 isolated
    assert out.ID.tolist() == [2, 3, 4]


def test_cluster_bootstrap_and_compare():
    st = {f"s{i}": [site(f"s{i}", 0, [0, 0, 0], n=1, group=f"g{i // 2}")] for i in range(10)}
    good = {k: preds(k, [[1, 0, 0]]) for k in st}
    half = {k: preds(k, [[1, 0, 0]] if int(k[1:]) % 2 == 0 else [[30, 0, 0]]) for k in st}
    ra, rb = evaluate_sites(st, good), evaluate_sites(st, half)
    sm = summarize_sites(rb, n_boot=300).set_index("metric")
    assert sm.loc["n_groups", "value"] == 5
    # every group holds one hit and one miss -> resampled recall is always exactly 0.5
    assert sm.loc["recall_topn", "ci_low"] == sm.loc["recall_topn", "ci_high"] == 0.5
    c = compare_site_methods(ra, rb, ["recall_topn"], n_boot=300, include_p1k=False).iloc[0]
    assert c["diff"] == 0.5 and c.ci_low == c.ci_high == 0.5 and c.n_groups == 5
    # B missing one structure entirely: its site counts as a miss
    rb2 = evaluate_sites(st, {k: v for k, v in good.items() if k != "s0"})
    c = compare_site_methods(ra, rb2, ["recall_topn"], n_boot=100, include_p1k=False).iloc[0]
    assert c["diff"] == pytest.approx(0.1)


def test_site_table_roundtrip(tmp_path):
    df = pd.DataFrame({"structure_id": ["a", "a"], "site_id": ["0", "1"], "center_x": [0.0, 1.0], "center_y": [0.0, 1.0],
                       "center_z": [0.0, 1.0], "residues": ["A:1 A:2", ""], "n_sites": [2, 2], "scored": [True, False],
                       "ligand_atoms": ["0.000,0.000,0.000 1.000,2.000,3.000", ""]})
    write_sites(df, tmp_path / "s.tsv")
    by = sites_by_structure(read_sites(tmp_path / "s.tsv"))
    assert by["a"][0].residues == {"A:1", "A:2"} and by["a"][1].residues == frozenset()
    assert by["a"][0].ligand_atoms.shape == (2, 3) and by["a"][1].ligand_atoms is None and not by["a"][1].scored


@pytest.mark.skipif(not (LIGYSIS_RESULTS / "ALL_COMPARISONS_EXTENDED6.pkl").exists(), reason="LIGYSIS raw data absent")
def test_ligysis_published_fpocket_prank_exact():
    from pocketdetect.ligysis import load_reference_sites, predictions_from_comparisons, safe_unpickle

    sites = load_reference_sites()
    assert len(sites) == 6882 and sites.scored.sum() == 6721 and sites.structure_id.nunique() == 2775
    comp = safe_unpickle(LIGYSIS_RESULTS / "ALL_COMPARISONS_EXTENDED6.pkl")
    r = evaluate_sites(sites_by_structure(sites), predictions_from_comparisons(comp, "fpocket+PRANK"))
    v = r.sites
    # published (their notebook 12, cells 61/62): top-N 3283, top-N+2 4061 of 6721; P1K 81.7 %
    assert int(v.recall_topn.sum()) == 3283 and int(v.recall_topn2.sum()) == 4061
    assert S(r)["P1K"] == pytest.approx(0.817)
