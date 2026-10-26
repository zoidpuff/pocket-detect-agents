"""Toy tests with known answers for scripts/eval_dimer.py (symmetric-copy dimer evaluation)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_dimer as ed  # noqa: E402
from pocketdetect.evaluate import Predictions, empty_predictions  # noqa: E402


def copy(lid, center, residues, chain_entity, cls="single_chain", coords=None, ion=False):
    counts = {}
    for r in residues:
        counts[r.split(":")[0]] = counts.get(r.split(":")[0], 0) + 1
    ents = sorted({f"E{chain_entity[r.split(':')[0]]}:{r.split(':')[1]}" for r in residues})
    c = np.asarray(center, float)
    return {
        "ligand_id": lid,
        "ccd": lid.split("|")[0],
        "is_ion": ion,
        "center": c.tolist(),
        "coords": (np.asarray(coords, float) if coords is not None else c[None, :]).tolist(),
        "binding_residues": list(residues),
        "chain_counts": counts,
        "entity_residues": ents,
        "site_class": cls,
    }


HOMO = {"A": "1", "B": "1"}


def homodimer_gt(copies, chain_entity=HOMO, primary="A"):
    return {
        "pdb_id": "toy",
        "chains": [{"chain": c, "entity_id": e} for c, e in chain_entity.items()],
        "copies": copies,
        "primary_chain": primary,
    }


def preds(centers, residues=None):
    centers = np.asarray(centers, float).reshape(-1, 3)
    m = len(centers)
    res = [frozenset(r.split()) for r in (residues or [""] * m)]
    return Predictions("toy", centers, np.arange(m, 0, -1.0), np.arange(1, m + 1), res)


# Symmetric single-chain site: copy on A at x=0, copy on B at x=40 (same entity residues 10,11,12)
SYM = [
    copy("LIG|A:900", (0, 0, 0), ["A:10", "A:11", "A:12"], HOMO),
    copy("LIG|B:900", (40, 0, 0), ["B:10", "B:11", "B:12"], HOMO),
]


def run_one(gt, pr, grouping="j50", chain=None, collapse=True):
    g = ed.site_gt(gt, grouping, chain)
    rows, unit = ed.evaluate_unit(g, pr, collapse=collapse)
    return g, rows, unit


def test_symmetric_copies_collapse_to_one_site_and_any_copy_counts():
    g, rows, unit = run_one(homodimer_gt(SYM), preds([(40, 1, 0)], ["B:10 B:11"]))
    assert g.n_sites == 1 and len(g.copy_ids) == 2
    assert rows[0]["success_DCC12_topn2"] and rows[0]["rank_DCC12"] == 1
    assert unit["tp_topn2"] == 1 and unit["fp_topn2"] == 0


def test_prediction_collapse_merges_symmetric_predictions():
    # predictions on both copies: entity sets identical -> second dropped by the collapse
    pr = preds([(0, 1, 0), (40, 1, 0)], ["A:10 A:11 A:12", "B:10 B:11 B:12"])
    _, rows, unit = run_one(homodimer_gt(SYM), pr)
    assert unit["n_pred_kept"] == 1 and unit["n_pred_collapsed"] == 1
    assert unit["tp_topn2"] == 1 and unit["fp_topn2"] == 0 and unit["neutral_topn2"] == 0


def test_duplicate_hit_of_matched_site_is_neutral_not_fp():
    pr = preds([(0, 1, 0), (40, 1, 0)], ["A:10 A:11 A:12", "B:10 B:11 B:12"])
    _, rows, unit = run_one(homodimer_gt(SYM), pr, collapse=False)
    assert unit["n_pred_kept"] == 2
    assert unit["tp_topn2"] == 1 and unit["neutral_topn2"] == 1 and unit["fp_topn2"] == 0


def test_non_hitting_prediction_is_fp_and_topk_cutoff():
    # N = 1 -> top-N+2 = 3. Hit only at rank 4.
    far = [(100, 0, 0), (0, 100, 0), (0, 0, 100)]
    res = ["A:50", "A:60", "A:70", "A:10 A:11"]
    _, rows, unit = run_one(homodimer_gt(SYM), preds(far + [(0, 2, 0)], res))
    r = rows[0]
    assert r["rank_DCC12"] == 4
    assert not r["success_DCC12_topn"] and not r["success_DCC12_topn2"] and r["success_DCC12_all"]
    assert unit["fp_topn2"] == 3 and unit["tp_topn2"] == 0


def test_dca_vs_dcc_on_elongated_ligand():
    atoms = [(x, 0, 0) for x in range(0, 31, 2)]  # 30 A long ligand, centroid (15,0,0)
    gt = homodimer_gt([copy("BIG|A:900", (15, 0, 0), ["A:10", "A:11", "A:12"], HOMO, coords=atoms)])
    _, rows, _ = run_one(gt, preds([(31, 2, 0)], ["A:99"]))  # 16 A from centroid, ~2.2 A from end atom
    assert rows[0]["success_DCA4_topn2"] and not rows[0]["success_DCC12_topn2"]
    _, rows, _ = run_one(gt, preds([(15, 11, 0)], ["A:99"]))  # 11 A from centroid, >4 A from all atoms
    assert rows[0]["success_DCC12_topn2"] and not rows[0]["success_DCA4_topn2"]


def test_strict_vs_j50_grouping():
    # copies differ by one residue (Jaccard 3/4): strict -> 2 sites, j50 -> 1 site
    cs = [
        copy("LIG|A:900", (0, 0, 0), ["A:10", "A:11", "A:12"], HOMO),
        copy("LIG|B:900", (40, 0, 0), ["B:10", "B:11", "B:12", "B:13"], HOMO),
    ]
    gt = homodimer_gt(cs)
    assert ed.site_gt(gt, "strict").n_sites == 2
    assert ed.site_gt(gt, "j50").n_sites == 1
    # with strict GT + collapse, the second copy cannot be hit (ceiling < 1); with j50 it is 1/1
    pr = preds([(0, 1, 0), (40, 1, 0)], ["A:10 A:11 A:12", "B:10 B:11 B:12 B:13"])
    _, rows, _ = run_one(gt, pr, grouping="strict")
    assert sum(r["success_DCC12_topn2"] for r in rows) == 1
    _, rows, _ = run_one(gt, pr, grouping="j50")
    assert sum(r["success_DCC12_topn2"] for r in rows) == 1 and len(rows) == 1


def test_interface_site_copies_and_class():
    # 2-fold interface site, two copies: A:10-12 + B:50-52 and B:10-12 + A:50-52
    c1 = copy("LIG|A:900", (0, 0, 0), ["A:10", "A:11", "A:12", "B:50", "B:51", "B:52"], HOMO, cls="homo_interface")
    c2 = copy("LIG|B:900", (20, 0, 0), ["B:10", "B:11", "B:12", "A:50", "A:51", "A:52"], HOMO, cls="homo_interface")
    g, rows, unit = run_one(homodimer_gt([c1, c2]), preds([(20, 3, 0)], ["A:51 B:11"]))
    assert g.n_sites == 1 and g.sites.iloc[0]["site_class"] == "homo_interface"
    assert rows[0]["success_DCC12_topn2"]


def test_hetero_entities_do_not_collapse():
    ce = {"A": "1", "B": "2"}
    cs = [
        copy("LIG|A:900", (0, 0, 0), ["A:10", "A:11", "A:12"], ce),
        copy("LIG|B:900", (40, 0, 0), ["B:10", "B:11", "B:12"], ce),
    ]
    gt = homodimer_gt(cs, chain_entity=ce)
    assert ed.site_gt(gt, "j50").n_sites == 2
    pr = preds([(0, 1, 0), (40, 1, 0)], ["A:10 A:11 A:12", "B:10 B:11 B:12"])
    _, rows, unit = run_one(gt, pr)
    assert unit["n_pred_kept"] == 2 and all(r["success_DCC12_topn2"] for r in rows)


def test_chain_mode_keeps_only_relevant_copies():
    cs = SYM + [copy("ION|B:901", (60, 0, 0), ["B:80", "B:81"], HOMO, ion=True)]
    g = ed.site_gt(homodimer_gt(cs), "j50", chain="A")
    assert g.copy_ids == ["LIG|A:900"] and g.n_sites == 1
    g = ed.site_gt(homodimer_gt(cs), "j50", chain="B")
    assert set(g.copy_ids) == {"LIG|B:900", "ION|B:901"} and g.n_sites == 2
    # minor contact (1 residue of 6) does not make a copy chain-relevant
    assert not ed.chain_relevant({"A": 5, "B": 1}, "B") and ed.chain_relevant({"A": 2, "B": 2}, "B")


def test_empty_and_missing_predictions_are_misses():
    g = ed.site_gt(homodimer_gt(SYM), "j50")
    rows, unit = ed.evaluate_unit(g, empty_predictions("toy", missing=True))
    assert not rows[0]["success_DCC12_all"] and unit["pred_missing"]


def test_empty_residue_sets_never_merge():
    pr = preds([(0, 1, 0), (40, 1, 0)])  # no residues
    kept, _ = ed.collapse_predictions(pr, HOMO)
    assert kept.tolist() == [0, 1]


def test_residues_near_centre_fallback():
    coords = np.array([[0, 0, 0], [5.9, 0, 0], [6.1, 0, 0]], float)
    ids = np.array(["A:1", "A:2", "A:3"], dtype=object)
    (r,) = ed.residues_near(np.zeros((1, 3)), coords, ids)
    assert r == frozenset({"A:1", "A:2"})


def test_oracle_and_summary_ceiling():
    cs = [
        copy("LIG|A:900", (0, 0, 0), ["A:10", "A:11", "A:12"], HOMO),
        copy("LIG|B:900", (40, 0, 0), ["B:10", "B:11", "B:12", "B:13"], HOMO),
    ]
    gts = {"toy": homodimer_gt(cs)}
    out = {}
    for grouping in ("j50", "strict"):
        g = ed.site_gt(gts["toy"], grouping)
        rows, unit = ed.evaluate_unit(g, ed.oracle_predictions(g, gts["toy"]))
        import pandas as pd

        s = ed.summarize(pd.DataFrame(rows), pd.DataFrame([unit]), n_boot=0)
        out[grouping] = float(s[(s.metric == "recall_DCC12_topn2") & (s.stratum == "all")].value.iloc[0])
    assert out["j50"] == pytest.approx(1.0) and out["strict"] == pytest.approx(0.5)
