"""Verifier: minimal HOLO4K(mlig) / COACH420(mlig) DCA4 top-n (P2Rank explicit-ligand-code rules:
matching HETATM groups, clustered single-linkage 1.7 A, all relevant; atoms incl. H as P2Rank does,
first record per atom name). No pocketdetect imports.

usage: python -I holo4k_dca4.py <ds_file> <pred_dir> <name> [--heavy]
"""
import os
import sys

import gemmi
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

STD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL CSE DAR MSE SEC TYS "
          "A C G I U DA DC DG DI DT".split())
WATER = {"HOH", "DOD", "WAT"}


def ligands(path, codes, heavy):
    st = gemmi.read_structure(path)
    groups = []
    for ch in st[0]:
        for res in ch:
            if res.het_flag != "H" or res.name in STD or res.name in WATER or res.name not in codes:
                continue
            seen, xyz = set(), []
            for a in res:
                if heavy and a.element.name in ("H", "D"):
                    continue
                if a.name in seen:
                    continue
                seen.add(a.name)
                xyz.append(a.pos.tolist())
            if xyz:
                groups.append(np.array(xyz))
    if not groups:
        return []
    allx = np.vstack(groups)
    gid = np.concatenate([[i] * len(g) for i, g in enumerate(groups)])
    pairs = cKDTree(allx).query_pairs(1.7, output_type="ndarray")
    import scipy.sparse as sp
    m = sp.coo_matrix((np.ones(len(pairs)), (gid[pairs[:, 0]], gid[pairs[:, 1]])), shape=(len(groups),) * 2)
    _, lab = connected_components(m, directed=False)
    return [np.vstack([groups[i] for i in range(len(groups)) if lab[i] == k]) for k in np.unique(lab)]


def main():
    ds, pred_dir, name = sys.argv[1:4]
    heavy = "--heavy" in sys.argv
    base = os.path.dirname(ds)
    rows = []
    for line in open(ds):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("HEADER"):
            continue
        prot, codes = line.split()[0], line.split()[1].split(",")
        ligs = ligands(os.path.join(base, prot), set(codes), heavy)
        sid = os.path.basename(prot)
        f = os.path.join(pred_dir, f"{sid}.csv")
        P = pd.read_csv(f).sort_values("rank") if os.path.exists(f) else pd.DataFrame(columns=["rank", "center_x", "center_y", "center_z"])
        C = P[["center_x", "center_y", "center_z"]].to_numpy(float)
        n = len(ligs)
        for L in ligs:
            if len(C):
                dca = np.sqrt(((C[:, None] - L[None]) ** 2).sum(-1)).min(1)
                ok = np.nonzero(dca <= 4.0)[0]
                rk = int(ok[0]) + 1 if len(ok) else -1  # position in rank order
            else:
                rk = -1
            rows.append({"sid": sid, "n": n, "rank": rk})
    df = pd.DataFrame(rows)
    topn = ((df["rank"] > 0) & (df["rank"] <= df["n"])).mean()
    topn2 = ((df["rank"] > 0) & (df["rank"] <= df["n"] + 2)).mean()
    print(f"{name}\tligands={len(df)}\tstructures={df.sid.nunique()}\tDCA4_topn={100*topn:.2f}\tDCA4_topn2={100*topn2:.2f}\theavy={heavy}")


if __name__ == "__main__":
    main()
