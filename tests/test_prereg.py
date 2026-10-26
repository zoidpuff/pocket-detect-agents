"""Decision logic of the pre-registered analysis (pocketdetect.prereg)."""

import numpy as np
import pandas as pd
import pytest

from pocketdetect.prereg import Rows, decide_noninferiority, holm, paired


def site_rows(vals, groups, metric="recall_topn2"):
    n = len(vals)
    return Rows("sites", pd.DataFrame({"structure_id": [f"s{i}" for i in range(n)], "site_id": ["0"] * n, "group": groups,
                                       metric: vals, "pred_missing": [False] * n}), ["structure_id", "site_id"])


def test_holm_step_down():
    h = holm({"a": 0.001, "b": 0.02, "c": 0.008}, alpha=0.025)
    assert h["a"]["rejected"] and h["c"]["rejected"] and h["b"]["rejected"]  # 0.001<=0.025/3, 0.008<=0.0125, 0.02<=0.025
    h = holm({"a": 0.001, "b": 0.03, "c": 0.02}, alpha=0.025)
    assert h["a"]["rejected"] and not h["c"]["rejected"] and not h["b"]["rejected"]  # 0.02 > 0.0125 stops the sequence


def test_noninferiority_decisions():
    assert decide_noninferiority({"ci_low": -1.5, "ci_high": 3}, -2) == "pass"
    assert decide_noninferiority({"ci_low": -5, "ci_high": -2.5}, -2) == "fail"
    assert decide_noninferiority({"ci_low": -3, "ci_high": 1}, -2) == "inconclusive"


def test_paired_self_and_known_difference():
    g = [f"u{i // 2}" for i in range(40)]
    A = site_rows([1.0] * 40, g)
    c = paired(A, A, "recall_topn2", n_boot=200)
    assert c["diff"] == 0 and c["ci_low"] == 0 and c["ci_high"] == 0
    B = site_rows([1.0, 0.0] * 20, g)  # every unit: one hit, one miss
    c = paired(A, B, "recall_topn2", n_boot=200)
    assert c["diff"] == pytest.approx(50) and c["ci_low"] == pytest.approx(50) and c["n_units"] == 20


def test_paired_missing_rows_are_misses_and_undefined_dropped():
    g = ["u0", "u1", "u2", "u3"]
    A = site_rows([1.0, 1.0, np.nan, 1.0], g)
    B = site_rows([1.0, 1.0, np.nan, 1.0], g)
    B.rows = B.rows.iloc[:3]  # s3 missing for B -> miss
    c = paired(A, B, "recall_topn2", n_boot=100)
    assert c["n_undefined_dropped"] == 1 and c["n_rows"] == 3 and c["diff"] == pytest.approx(100 / 3)
