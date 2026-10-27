"""Independent LIGYSIS scorer (verifier). Uses the LBS-comparison authors' own functions
(filter_redundant_pockets2, compare_ALL_predictions, extracted from benchmark_lib.py via ast) and
their notebook-12 selection logic. No pocketdetect imports.

Everything is in the authors' TRANS frame: sites/centres from ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl,
BURROW / re-run predictions from the *TRANS-frame* CSVs (input = authors' trans_rep_chains PDBs).

usage: python -I ligysis_authors_eval.py <method_name> <published|csv_dir> <out_prefix>
"""
import ast
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.spatial.distance import euclidean

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vt_safe_pickle import safe_load  # noqa: E402

ROOT = "/cluster/project/beltrao/gmagnus/pocket_detect"
LIB = f"{ROOT}/data/raw/ligysis/lbs_comparison_code/LBS-comparison-v2.0.0/bartongroup-LBS-comparison-820714c/ANALYSIS/benchmark_lib.py"
COMBINED = f"{ROOT}/data/raw/ligysis/lbs_comparison_results/ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl"

# --- load the authors' functions (only the ones we read and need) ---
_src = open(LIB).read()
_tree = ast.parse(_src)
_wanted = {"filter_redundant_pockets2", "compare_ALL_predictions", "get_volume_overlap", "jaccard_index"}
_ns = {"np": np, "pd": pd, "distance": distance, "euclidean": euclidean, "volumes_dict": {}}
for node in _tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in _wanted:
        exec(compile(ast.Module(body=[node], type_ignores=[]), LIB, "exec"), _ns)
filter_redundant_pockets2 = _ns["filter_redundant_pockets2"]
compare_ALL_predictions = _ns["compare_ALL_predictions"]


def load_ligysis():
    comb = safe_load(COMBINED).sort_values(["origin", "rep_chain", "ID"], ascending=[False, True, True]).reset_index(drop=True)
    i_cols0 = ["origin", "rep_chain", "ID", "centre", "n_aas", "RoG", "SASA", "VOL", "up_aas", "aas"]
    L = comb.query('origin == "LIGYSIS"')[i_cols0 + ["ion_prop", "cluster_dist", "monomer_prop"]].copy().reset_index(drop=True)
    L["N_SITES"] = L.groupby("rep_chain")["rep_chain"].transform("size")
    return comb, L


def load_csv_preds(d, name, chains):
    rows = []
    for ch in chains:
        f = os.path.join(d, f"{ch}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, keep_default_na=False)
        for r in df.itertuples():
            res = str(r.residues).split() if r.residues else []
            aas = []
            for x in res:
                num = x.split(":")[1]
                num = "".join(c for c in num if c.isdigit() or c == "-")
                aas.append(int(num))
            rows.append({"origin": name, "rep_chain": ch, "ID": int(r.rank), "RANK": int(r.rank), "SCORE": float(r.score),
                         "centre": (float(r.center_x), float(r.center_y), float(r.center_z)), "aas": aas, "up_aas": aas})
    return pd.DataFrame(rows)


def main():
    name, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    comb, L = load_ligysis()
    L_NR = filter_redundant_pockets2(L, jaccard_threshold=0.75, distance_threshold=5.0)
    assert len(L) == 6882, len(L)
    chains = sorted(L.rep_chain.unique())
    if src == "published":
        P = comb.query('origin == "fpocket+PRANK"')[["origin", "rep_chain", "ID", "centre", "n_aas", "up_aas", "aas", "RANK", "score"]].copy().reset_index(drop=True)
        P = P.rename(columns={"score": "SCORE"})
        P["origin"] = name
        ref = L_NR.copy()  # authors' up_aas on both sides
    else:
        P = load_csv_preds(src, name, chains)
        ref = L_NR.copy()
        ref["up_aas"] = ref["aas"]  # BURROW residues are trans-PDB numbering (= aas); IRel only breaks ties
    n_missing_chains = len(set(chains) - set(P.rep_chain))
    # per-chain call of the authors' function (identical result, much faster than one global call)
    res = []
    Pg = dict(tuple(P.groupby("rep_chain")))
    for ch, refc in ref.groupby("rep_chain"):
        predc = Pg.get(ch, P.iloc[0:0])
        res.append(compare_ALL_predictions(refc, predc))
    A = pd.concat(res)
    A = A.query("RANK_pred == RANK_pred").reset_index(drop=True)
    A.ID_pred = A.ID_pred.astype(int)
    A.RANK_pred = A.RANK_pred.astype(int)
    A.distance = A.distance.round(2)
    A.relative_intersection = A.relative_intersection.round(2)
    B = A.sort_values(["origin_ref", "rep_chain_ref", "ID_ref", "origin_pred", "distance", "relative_intersection"],
                      ascending=[True, True, True, True, True, False], kind="mergesort").drop_duplicates(
        ["origin_ref", "rep_chain_ref", "ID_ref", "origin_pred"], keep="first")
    TOT = len(L_NR)
    def rec(k):
        return int(((B.distance <= 12) & (B.RANK_pred <= B.N_SITES_ref + k)).sum())
    out_d = {"method": name, "src": src, "n_sites_NR": TOT, "n_chains": len(chains), "n_missing_chains": n_missing_chains,
             "n_pockets": len(P), "topN": rec(0), "topN2": rec(2), "max": int((B.distance <= 12).sum())}
    for k in ["topN", "topN2", "max"]:
        out_d[k + "_pct"] = round(100 * out_d[k] / TOT, 2)
    # P1K, authors' definition: pocket TRUE if DCC<=12 to any NR site of its chain; pooled, sorted by SCORE desc
    tp = set(map(tuple, A.query("distance <= 12")[["rep_chain_ref", "ID_pred"]].values.tolist()))
    Ps = P.sort_values(["rep_chain", "RANK"], kind="mergesort").sort_values("SCORE", ascending=False, kind="mergesort")
    true = np.array([(c, i) in tp for c, i in zip(Ps.rep_chain, Ps.ID)])
    out_d["P1K_pct"] = round(100 * true[:1000].mean(), 2)
    out_d["n_score_nan"] = int(P.SCORE.isna().sum())
    # per-site table for paired analyses
    per = L_NR[["rep_chain", "ID", "N_SITES"]].copy()
    hit = B[(B.distance <= 12) & (B.RANK_pred <= B.N_SITES_ref + 2)][["rep_chain_ref", "ID_ref"]]
    hs = set(map(tuple, hit.values.tolist()))
    per["hit_topN2"] = [int((c, i) in hs) for c, i in zip(per.rep_chain, per.ID)]
    per.to_csv(out + "_sites.csv", index=False)
    json.dump(out_d, open(out + ".json", "w"), indent=1)
    print(json.dumps(out_d))


if __name__ == "__main__":
    main()
