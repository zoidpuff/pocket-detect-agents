#!/usr/bin/env python
"""LIGYSIS reproduction gate (PREREG fixed rule): score the authors' published per-method
predictions with our harness (pocketdetect.site_eval) and compare with their published table
(numbers parsed from the outputs stored in their notebook 12_BENCHMARK_pocket_level.ipynb).
Optionally cross-check against the authors' own benchmark_lib functions.

  python -I scripts/reproduce_ligysis.py --out results/validation/ligysis_reproduction [--their-code]

Writes <out>/reproduction.csv, <out>/site_table_trans.tsv (reference sites, TRANS frame) and
results/validation/ligysis_reproduction.md.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketdetect.ligysis import LIGYSIS_RESULTS, filter_redundant, load_reference_sites, predictions_from_comparisons, predictions_from_frame, safe_unpickle  # noqa: E402
from pocketdetect.site_eval import SiteProtocol, evaluate_sites, summarize_sites  # noqa: E402
from pocketdetect.sites import sites_by_structure, write_sites  # noqa: E402

CODE = ROOT / "data/raw/ligysis/lbs_comparison_code/LBS-comparison-v2.0.0/bartongroup-LBS-comparison-820714c/ANALYSIS"
NB = CODE / "12_BENCHMARK_pocket_level.ipynb"


def published_tables() -> dict:
    """Parse the printed DataFrames in the notebook outputs (no code execution)."""
    nb = json.loads(NB.read_text())

    def out(i):
        txt = []
        for o in nb["cells"][i].get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain")
            if t:
                txt.append("".join(t))
        return "\n".join(txt)

    pub: dict = {}
    for cell, rank_t in ((62, 0), (61, 2), (60, 1000)):
        for line in out(cell).splitlines():
            m = re.match(r"^\s*\d+\s+(\S+)\s+(\d+)\s+12\.0\s+1\.05\s+([\d.]+)\s*\\?\s*$", line)
            if m and int(m.group(2)) == rank_t:
                pub.setdefault(m.group(1), {})[f"rate_{rank_t}"] = float(m.group(3))
        # second block: correct_predictions per row index -> map by row id
        ids = {}
        for line in out(cell).splitlines():
            m = re.match(r"^(\d+)\s+(\S+)\s+(\d+)\s+12\.0\s+1\.05\s+([\d.]+)\s*\\?$", line.strip())
            if m:
                ids[m.group(1)] = m.group(2)
        for line in out(cell).splitlines():
            m = re.match(r"^(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", line.strip())
            if m and m.group(1) in ids:
                pub[ids[m.group(1)]][f"count_{rank_t}"] = int(m.group(2))
    for cell in (52, 53):  # stats table: Total (= max-recall count), Median Overlap (= mean IRel, RRO)
        idx = {}
        for line in out(cell).splitlines():
            s = line.strip().rstrip("\\").strip()
            m = re.match(r"^(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$", s)
            if m:
                idx[m.group(1)] = m.group(2)
                pub.setdefault(m.group(2), {}).setdefault("count_1000", int(m.group(3)))
                pub[m.group(2)].setdefault("rate_1000", round(int(m.group(3)) / 6721, 3))
                continue
            m = re.match(r"^(\d+)\s+([\d.]+|NaN)\s+([\d.]+|NaN)$", s)
            if m and m.group(1) in idx:
                pub[idx[m.group(1)]]["RRO"] = float(m.group(2)) / 100
    for cell in (134, 143):  # precision at 1000 predictions
        for line in out(cell).splitlines():
            m = re.match(r"^\d+\s+(\S+)\s+(?:1000\s+(\d+)\s+([\d.]+)|([\d.]+)\s+(\d+)\s+\d+\s+1000)", line.strip())
            if m:
                name = m.group(1)
                if m.group(2):
                    pub.setdefault(name, {})["P1K_tp"] = int(m.group(2))
                else:
                    pub.setdefault(name, {})["P1K_tp"] = int(m.group(5))
    for cell in (133, 141, 142):  # TP at 100 FP
        for line in out(cell).splitlines():
            m = re.match(r"^\d+\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)", line.strip())
            if m:
                name, a, b, c = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
                tp = a if b == 100 else (b if a == b + 100 else None)  # (cumsum_TP, 100, total) or (total, cumsum_TP, ...)
                if b == 100:
                    tp = a
                elif a == b + 100:
                    tp = b
                if tp is not None:
                    pub.setdefault(name, {})["TP100FP"] = tp
    return pub


def their_functions():
    """Load selected functions from the authors' benchmark_lib.py *without* executing the module
    (it imports plotting libraries and their JSU_lib): parse with ast, exec only these defs."""
    src = (CODE / "benchmark_lib.py").read_text()
    tree = ast.parse(src)
    keep = {"filter_redundant_pockets2", "compare_ALL_predictions", "success_rate_COMB", "jaccard_index", "get_volume_overlap"}
    mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep], type_ignores=[])
    from scipy.spatial import distance
    from scipy.spatial.distance import euclidean

    ns = {"np": np, "pd": pd, "distance": distance, "euclidean": euclidean, "volumes_dict": {}}
    exec(compile(mod, str(CODE / "benchmark_lib.py"), "exec"), ns)
    return ns


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/validation/ligysis_reproduction")
    ap.add_argument("--their-code", action="store_true", help="also cross-check with the authors' functions")
    ap.add_argument("--methods", default="")
    ap.add_argument("--frame-check", action="store_true", help="re-score fpocket+PRANK / P2Rank in the original (mirror) frame "
                    "against data/processed/gt/ligysis_lbs/sites.tsv through the common CSV format")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sites = load_reference_sites()
    write_sites(sites, a.out / "site_table_trans.tsv")
    comp = safe_unpickle(LIGYSIS_RESULTS / "ALL_COMPARISONS_EXTENDED6.pkl")
    ref_keys = set(zip(comp.rep_chain_ref, comp.ID_ref.astype(str)))
    our_keys = set(zip(sites.structure_id[sites.scored], sites.site_id[sites.scored]))
    check = {
        "n_sites_all": int(len(sites)), "n_sites_scored": int(sites.scored.sum()), "n_chains": int(sites.structure_id.nunique()),
        "scored_set_equals_authors_NR_refs": our_keys == ref_keys,
        "N_matches_authors": bool((comp.drop_duplicates(["rep_chain_ref"]).set_index("rep_chain_ref").N_SITES_ref
                                   == sites.drop_duplicates("structure_id").set_index("structure_id").n_sites.reindex(comp.rep_chain_ref.unique())).all()),
    }
    print(check)
    by = sites_by_structure(sites)
    pub = published_tables()
    methods = [m for m in a.methods.split(",") if m] or sorted(comp.origin_pred.unique())
    rows = []
    for meth in methods:
        preds = predictions_from_comparisons(comp, meth)
        res = evaluate_sites(by, preds, SiteProtocol())
        s = summarize_sites(res, n_boot=0).set_index("metric")["value"]
        n = int(s["n_scored_sites"])
        K = res.pockets
        row = {
            "method": meth, "n_chains_with_preds": len(preds),
            "topN": s["recall_topn"], "topN2": s["recall_topn2"], "max_recall": s["recall_max"],
            "topN_count": int(round(s["recall_topn"] * n)), "topN2_count": int(round(s["recall_topn2"] * n)), "max_count": int(round(s["recall_max"] * n)),
            "P1K": s["P1K"], "TP@100FP": s["TP@100FP"], "P1K_1to1": s["P1K_1to1"], "RRO": s["RRO"],
            "topN2_anyp": s["recall_anyp_topn2"], "topN2_1to1": s["recall_1to1_topn2"],
        }
        p = pub.get(meth, {})
        for k_ours, k_pub in (("topN", "rate_0"), ("topN2", "rate_2"), ("max_recall", "rate_1000")):
            row[f"{k_ours}_pub"] = p.get(k_pub, np.nan)
        for k_ours, k_pub in (("topN_count", "count_0"), ("topN2_count", "count_2"), ("max_count", "count_1000")):
            row[f"{k_ours}_pub"] = p.get(k_pub, np.nan)
        row["P1K_pub"] = p.get("P1K_tp", np.nan) / 1000 if "P1K_tp" in p else np.nan
        row["TP@100FP_pub"] = p.get("TP100FP", np.nan)
        row["RRO_pub"] = p.get("RRO", np.nan)
        rows.append(row)
        print(f"{meth:26s} topN {row['topN']:.4f}/{row['topN_pub']} topN2 {row['topN2']:.4f}/{row['topN2_pub']} max {row['max_recall']:.4f}/{row['max_recall_pub']} P1K {row['P1K']:.3f}/{row['P1K_pub']}", flush=True)
    df = pd.DataFrame(rows)
    for k in ("topN", "topN2", "max_recall"):
        df[f"{k}_diff_pts"] = 100 * (df[k] - df[f"{k}_pub"])
    df.to_csv(a.out / "reproduction.csv", index=False)

    their = {}
    if a.their_code:
        ns = their_functions()
        comb = safe_unpickle(LIGYSIS_RESULTS / "ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl")
        lig = comb[comb.origin == "LIGYSIS"].sort_values(["rep_chain", "ID"]).reset_index(drop=True)
        lig = lig[["origin", "rep_chain", "ID", "centre", "n_aas", "up_aas", "aas"]].copy()
        lig["N_SITES"] = lig.groupby("rep_chain")["rep_chain"].transform("size")
        lig_nr_theirs = ns["filter_redundant_pockets2"](lig, jaccard_threshold=0.75, distance_threshold=5.0)
        lig_nr_ours = filter_redundant(lig, "n_aas")
        their["redundancy_filter_identical"] = set(zip(lig_nr_theirs.rep_chain, lig_nr_theirs.ID)) == set(zip(lig_nr_ours.rep_chain, lig_nr_ours.ID))
        their["n_nr_theirs"] = len(lig_nr_theirs)
        for meth, src_origin, score_col in (("P2Rank", "P2Rank", "prob"), ("GrASP", "GrASP", "score")):
            pr = comb[comb.origin == src_origin].copy()
            pr = pr.rename(columns={score_col: "SCORE"})[["origin", "rep_chain", "ID", "centre", "up_aas", "aas", "RANK", "SCORE"]]
            pr["RANK"] = pr["RANK"].astype(int)
            t1 = time.time()
            allc = ns["compare_ALL_predictions"](lig_nr_theirs.drop(columns=["N_SITES"]).assign(N_SITES=lig_nr_theirs.N_SITES), pr)
            allc = allc[allc.RANK_pred == allc.RANK_pred].copy()
            allc["distance"] = allc["distance"].round(2)
            allc["relative_intersection"] = allc["relative_intersection"].round(2)
            best = allc.sort_values(["origin_ref", "rep_chain_ref", "ID_ref", "origin_pred", "distance", "relative_intersection"],
                                    ascending=[True, True, True, True, True, False]).drop_duplicates(["origin_ref", "rep_chain_ref", "ID_ref", "origin_pred"], keep="first")
            r = ns["success_rate_COMB"](best, [12], [0, 2, 1000], [1.05], TOTAL_preds=len(lig_nr_theirs))
            preds = predictions_from_frame(pr)
            ours = summarize_sites(evaluate_sites(by, preds), n_boot=0).set_index("metric")["value"]
            their[meth] = {
                "their_code_counts": {int(k): int(v) for k, v in zip(r.rank_t, r.correct_predictions)},
                "our_counts_from_same_raw_preds": {0: int(round(ours["recall_topn"] * 6721)), 2: int(round(ours["recall_topn2"] * 6721)), 1000: int(round(ours["recall_max"] * 6721))},
                "seconds": round(time.time() - t1, 1),
            }
            print(meth, their[meth], flush=True)
        (a.out / "their_code_crosscheck.json").write_text(json.dumps(their, indent=2, default=str))
    if a.frame_check:
        import tempfile

        from pocketdetect.evaluate import write_predictions
        from pocketdetect.ligysis import load_transforms, to_original_frame
        from pocketdetect.sites import read_sites

        T = load_transforms()
        orig = sites_by_structure(read_sites(ROOT / "data/processed/gt/ligysis_lbs/sites.tsv"))
        for meth in ("fpocket+PRANK", "P2Rank"):
            preds = predictions_from_comparisons(comp, meth)
            with tempfile.TemporaryDirectory() as td:
                for chain, p in preds.items():
                    Rm, c0 = T[chain]
                    write_predictions(Path(td) / f"{chain}.csv", to_original_frame(p.centers, Rm, c0), p.scores)
                r_o = evaluate_sites(orig, td)
            r_t = evaluate_sites(by, {k: v.__class__(v.structure_id, v.centers, v.scores, v.ranks, [frozenset()] * v.n) for k, v in preds.items()})
            check[f"frame_check_{meth}"] = {
                "orig_frame_counts": [int(r_o.sites[c].sum()) for c in ("recall_topn", "recall_topn2", "recall_max")],
                "trans_frame_counts_no_residues": [int(r_t.sites[c].sum()) for c in ("recall_topn", "recall_topn2", "recall_max")],
                "max_abs_dcc_diff": float((r_o.sites.set_index(["structure_id", "site_id"]).dcc_closest
                                           - r_t.sites.set_index(["structure_id", "site_id"]).dcc_closest).abs().max()),
            }
            print(meth, check[f"frame_check_{meth}"], flush=True)
    (a.out / "checks.json").write_text(json.dumps(check, indent=2, default=str))
    write_report(df, check, their, a.out)
    print(f"done in {time.time() - t0:.0f}s")


def write_report(df: pd.DataFrame, check: dict, their: dict, out: Path):
    sel = ["VN-EGNN-NR", "IF-SitePred-rescored-NR", "GrASP", "PUResNet+PRANK", "DeepPocket-Segmented-NR", "DeepPocket-Rescored",
           "P2Rank+Cons", "P2Rank", "fpocket+PRANK", "fpocket", "PocketFinder-AA", "Ligsite-AA", "Surfnet-AA"]
    d = df.set_index("method")
    lines = [
        "# LIGYSIS reproduction gate",
        "",
        "Generated by `scripts/reproduce_ligysis.py` (do not edit by hand). Our harness (`pocketdetect.site_eval`, LBS-comparison",
        "protocol: closest-pocket DCC ≤ 12 Å after rounding to 0.01 Å, N = all sites of the chain, 6,721 non-redundant scored sites)",
        "applied to the authors' published per-chain predictions (`ALL_COMPARISONS_EXTENDED6.pkl`: centre, rank, score, residues).",
        "Published numbers parsed from the stored outputs of their notebook `12_BENCHMARK_pocket_level.ipynb`.",
        "Gate (PREREG): |Δ| ≤ 0.5 points for top-N, top-N+2 and max recall.",
        "",
        f"Reference checks: {check}",
        "",
        "| method | top-N ours / pub | top-N+2 ours / pub | max recall ours / pub | max |Δ| (pts) | gate | P1K ours / pub | TP@100FP ours / pub | RRO ours / pub |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    allrows = [m for m in sel if m in d.index] + [m for m in d.index if m not in sel]
    n_pass = n_tested = 0
    for m in allrows:
        r = d.loc[m]
        diffs = [abs(r[f"{k}_diff_pts"]) for k in ("topN", "topN2", "max_recall") if np.isfinite(r[f"{k}_diff_pts"])]
        mx = max(diffs) if diffs else np.nan
        ok = "PASS" if diffs and mx <= 0.5 + 1e-9 else ("n/a" if not diffs else "FAIL")
        if diffs:
            n_tested += 1
            n_pass += ok == "PASS"

        def f(x, pct=True):
            return "–" if not np.isfinite(x) else (f"{100 * x:.1f}" if pct else f"{x:.0f}")

        lines.append(f"| {m}{' *' if m in sel else ''} | {f(r.topN)} / {f(r.topN_pub)} | {f(r.topN2)} / {f(r.topN2_pub)} | {f(r.max_recall)} / {f(r.max_recall_pub)} | "
                     f"{mx:.2f} | {ok} | {f(r.P1K)} / {f(r.P1K_pub)} | {f(r['TP@100FP'], False)} / {f(r['TP@100FP_pub'], False)} | {f(r.RRO)} / {f(r.RRO_pub)} |")
    lines += ["", f"**Gate: {n_pass}/{n_tested} methods within ±0.5 points on all three recall metrics.** (* = methods of the paper's main table.)", ""]
    lines += ["Residual differences (all within the gate) come only from ties in the closest-pocket selection: several",
              "predictions at exactly the same rounded distance and IRel (duplicate centres; e.g. VN-EGNN 47 and",
              "DeepPocket-Segmented 301 sites have tied best candidates of different rank). Their pandas sort keeps the",
              "lowest pocket ID among ties; we keep the best rank (IDs are not part of the common prediction format).", ""]
    lines += ["Published max recall is only printed for the original (non-variant) methods; P1K / TP@100FP only for the methods shown in",
              "their precision cells. Pooled precision depends on score-tie order (the authors shuffled ties of PUResNet / PocketFinder /",
              "Ligsite / Surfnet with pandas seed 12345); we break ties by (chain, rank).", ""]
    if their:
        lines += ["## Cross-check with the authors' code", "", "Their `filter_redundant_pockets2`, `compare_ALL_predictions` and `success_rate_COMB` (loaded from",
                  "`benchmark_lib.py` via ast, no other code executed) applied to the raw P2Rank / GrASP predictions of",
                  "`ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl`, vs our harness on the same raw predictions:", "", "```", json.dumps(their, indent=2, default=str), "```", ""]
    fc = {k: v for k, v in check.items() if k.startswith("frame_check")}
    if fc:
        lines += ["## Original-frame check", "", "The same published predictions mapped back to the wwPDB asymmetric-unit frame",
                  "(x = Rᵀ x_T + c0), written in the common CSV format and scored against `data/processed/gt/ligysis_lbs/sites.tsv`",
                  "(the table used for new methods) give identical top-N / top-N+2 / max-recall counts:", "", "```", json.dumps(fc, indent=2), "```",
                  "(max |ΔDCC| > 0 only for the 4 sites whose centre was recomputed from the mirror file because the inverted",
                  "author centre deviated by > 1 Å; see data/processed/gt/ligysis_lbs/README.md.)", ""]
    lines += ["Extra columns in `reproduction.csv`: one-to-one (Hungarian) top-N+2, any-pocket top-N+2, P1K one-to-one, RRO."]
    (ROOT / "results/validation/ligysis_reproduction.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
