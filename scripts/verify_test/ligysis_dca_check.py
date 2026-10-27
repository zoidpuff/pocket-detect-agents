"""Verifier: own minimal C1-strict (DCA<=4, closest-by-DCA pocket, rank<=N+2), C1 (orig frame, LDCC12),
P1K, and TRANS->orig frame consistency of prediction files. No pocketdetect imports.

usage: python -I ligysis_dca_check.py <name> <preds_orig_dir> <raw_trans_dir|-> <out_prefix>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vt_safe_pickle import safe_load  # noqa: E402

ROOT = "/cluster/project/beltrao/gmagnus/pocket_detect"
SITES = os.environ.get("VT_SITES", f"{ROOT}/data/processed/gt/ligysis_lbs/sites.tsv")
RES = f"{ROOT}/data/raw/ligysis/lbs_comparison_results"


def parse_atoms(s):
    if not isinstance(s, str) or not s.strip():
        return None
    return np.array([[float(v) for v in t.split(",")] for t in s.split()])


def read_preds(d, sid):
    f = os.path.join(d, f"{sid}.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f, keep_default_na=False)
    return df.sort_values("rank", kind="mergesort").reset_index(drop=True)


def main():
    name, dorig, draw, out = sys.argv[1:5]
    s = pd.read_csv(SITES, sep="\t", keep_default_na=False, na_values=[""])
    R = safe_load(f"{RES}/PDB_rot_matrices_ALL_CLUST.pkl")
    C0 = safe_load(f"{RES}/PDB_orig_centroids_ALL_CLUST.pkl")
    nmiss = 0
    frame_dev = []
    rows = []
    pooled = []
    npk = []
    nan_scores = 0
    for sid, g in s.groupby("structure_id"):
        p = read_preds(dorig, sid)
        if p is None:
            nmiss += 1
            p = pd.DataFrame(columns=["rank", "score", "center_x", "center_y", "center_z", "residues"])
        npk.append(len(p))
        nan_scores += int(pd.to_numeric(p["score"], errors="coerce").isna().sum())
        P = p[["center_x", "center_y", "center_z"]].to_numpy(float)
        ranks = p["rank"].to_numpy(int)
        if draw != "-" and len(p):
            q = read_preds(draw, sid)
            # authors' convention x_T = R (x - c0)  =>  x = R^T x_T + c0
            Q = q[["center_x", "center_y", "center_z"]].to_numpy(float)
            back = Q @ np.asarray(R[sid]) + np.asarray(C0[sid])
            assert (q["rank"].to_numpy() == ranks).all()
            frame_dev.append(np.abs(back - P).max())
        N = int(g["n_sites"].iloc[0])
        sc = g[g.scored == 1]
        cents = sc[["center_x", "center_y", "center_z"]].to_numpy(float)
        for (_, site), cen in zip(sc.iterrows(), cents):
            row = {"structure_id": sid, "site_id": site.site_id, "N": N}
            if len(P):
                dcc = np.round(np.linalg.norm(P - cen, axis=1), 2)
                i = int(np.lexsort((ranks, dcc))[0])  # closest by DCC, ties -> better rank
                row["hit_ldcc12_topn2"] = int(dcc[i] <= 12 and ranks[i] <= N + 2)
                row["hit_ldcc12_topn"] = int(dcc[i] <= 12 and ranks[i] <= N)
            else:
                row["hit_ldcc12_topn2"] = row["hit_ldcc12_topn"] = 0
            A = parse_atoms(site.ligand_atoms)
            if A is None:
                row["has_lig"] = 0
                row["hit_ldca4_topn2"] = np.nan
            else:
                row["has_lig"] = 1
                if len(P):
                    dca = np.sqrt(((P[:, None, :] - A[None, :, :]) ** 2).sum(-1)).min(1)
                    dca = np.round(dca, 2)
                    i = int(np.lexsort((ranks, dca))[0])
                    row["hit_ldca4_topn2"] = int(dca[i] <= 4 and ranks[i] <= N + 2)
                    row["hit_anydca4_topn2"] = int(((dca <= 4) & (ranks <= N + 2)).any())
                else:
                    row["hit_ldca4_topn2"] = row["hit_anydca4_topn2"] = 0
            rows.append(row)
        # pooled precision: TP if DCC<=12 to any scored site of the structure
        if len(P) and len(cents):
            d = np.round(np.linalg.norm(P[:, None, :] - cents[None], axis=2), 2).min(1)
            tp = d <= 12
        else:
            tp = np.zeros(len(P), bool)
        for sc_, r_, t_ in zip(pd.to_numeric(p["score"]), ranks, tp):
            pooled.append((-sc_, sid, r_, bool(t_)))
    df = pd.DataFrame(rows)
    pooled.sort(key=lambda x: (x[0], x[1], x[2]))
    p1k = np.mean([t for *_, t in pooled[:1000]])
    lig = df[df.has_lig == 1]
    npk = np.array(npk)
    res = {"method": name, "n_scored": len(df), "n_missing_files": nmiss,
           "C1_topn2_pct": round(100 * df.hit_ldcc12_topn2.mean(), 2), "topn_pct": round(100 * df.hit_ldcc12_topn.mean(), 2),
           "n_lig_sites": len(lig), "C1strict_ldca4_topn2_pct": round(100 * lig.hit_ldca4_topn2.mean(), 2),
           "anydca4_topn2_pct": round(100 * lig.hit_anydca4_topn2.mean(), 2), "P1K_pct": round(100 * p1k, 2),
           "pockets_per_chain_mean": round(float(npk.mean()), 2), "pockets_per_chain_median": float(np.median(npk)),
           "pockets_per_chain_min": int(npk.min()), "pockets_per_chain_max": int(npk.max()),
           "n_chains_zero_pockets": int((npk == 0).sum()), "n_nan_scores": nan_scores,
           "frame_maxdev_A": (round(float(np.max(frame_dev)), 4) if frame_dev else None)}
    df.to_csv(out + "_sites.csv", index=False)
    json.dump(res, open(out + ".json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
