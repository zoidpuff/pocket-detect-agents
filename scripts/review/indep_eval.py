#!/usr/bin/env python
"""Independent re-implementation of the LIGYSIS-style site metrics (reviewer; does NOT import pocketdetect).

For each method dir: per-site closest-pocket DCC (rounded 0.01, ties -> larger IRel -> better rank) top-N / top-N+2
recall at 12 A, ldca top-N+2 (pocket with min DCA, DCA<=4, rank<=N+2), anyp top-N+2, Hungarian 1:1 top-N+2,
max recall, pooled P1K and P1K 1:1. Missing file = zero pockets. Paired cluster bootstrap vs a reference
(cluster = site-table `group`), P1K recomputed per resample, own RNG seed (12345).

  python scripts/review/indep_eval.py --sites results/dev/biolip/dev_sites_all.tsv --ref p2rank=DIR name=DIR ... --out X.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def parse_atoms(s):
    if not isinstance(s, str) or not s.strip():
        return None
    return np.array([[float(v) for v in t.split(",")] for t in s.split()])


def read_preds(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    if len(d) == 0:
        return d
    assert d["rank"].is_unique, path
    return d.sort_values("rank").reset_index(drop=True)


def eval_method(sites: pd.DataFrame, pdir: str):
    rows, pocket_rows = [], []
    for sid, ss in sites.groupby("structure_id", sort=True):
        N = int(ss.n_sites.iloc[0])
        p = read_preds(os.path.join(pdir, f"{sid}.csv"))
        if p is None:
            missing = True
            p = pd.DataFrame(columns=["rank", "score", "center_x", "center_y", "center_z", "residues"])
        else:
            missing = False
        P = p[["center_x", "center_y", "center_z"]].to_numpy(float) if len(p) else np.zeros((0, 3))
        ranks = p["rank"].to_numpy(int) if len(p) else np.zeros(0, int)
        # position in rank order (1-based) -- rank column authoritative, gaps allowed: use ordinal position
        pos = np.argsort(np.argsort(ranks, kind="stable"), kind="stable") + 1 if len(p) else ranks
        pres = [set(str(r).split()) if isinstance(r, str) else set() for r in (p["residues"] if len(p) else [])]
        S = ss[["center_x", "center_y", "center_z"]].to_numpy(float)
        dcc = np.round(np.linalg.norm(P[None, :, :] - S[:, None, :], axis=2), 2) if len(p) else np.zeros((len(ss), 0))
        hit_any = np.zeros(len(p), bool)
        for k, (_, s) in enumerate(ss.iterrows()):
            r = dict(structure_id=sid, site_id=s.site_id, group=s.group, missing=missing, N=N)
            sres = set(str(s.residues).split())
            if len(p) == 0:
                r.update(topn=0, topn2=0, maxr=0, anyp2=0, ldca2=0, anyp_dca2=0, dcc_min=np.nan)
                rows.append(r)
                continue
            d = dcc[k]
            irel = np.array([len(pr & sres) / max(1, len(sres)) for pr in pres])
            order = np.lexsort((pos, -irel, d))  # min d, then max irel, then best rank
            b = order[0]
            r["dcc_min"] = d[b]
            ok = d[b] <= 12.0
            r["topn"] = int(ok and pos[b] <= N)
            r["topn2"] = int(ok and pos[b] <= N + 2)
            r["maxr"] = int(ok)
            r["anyp2"] = int(np.any((d <= 12.0) & (pos <= N + 2)))
            la = parse_atoms(s.ligand_atoms)
            dca = np.round(np.linalg.norm(P[:, None, :] - la[None, :, :], axis=2).min(1), 2)
            o2 = np.lexsort((pos, dca))
            r["ldca2"] = int(dca[o2[0]] <= 4.0 and pos[o2[0]] <= N + 2)
            r["anyp_dca2"] = int(np.any((dca <= 4.0) & (pos <= N + 2)))
            rows.append(r)
        # Hungarian 1:1 over top-N+2
        if len(p):
            top = pos <= N + 2
            M = (dcc[:, top] <= 12.0).astype(float)
            if M.size:
                ri, ci = linear_sum_assignment(-M)
                matched = set(ri[M[ri, ci] > 0])
            else:
                matched = set()
            for k in range(len(ss)):
                rows[-len(ss) + k]["one2one2"] = int(k in matched)
            # pockets for pooled precision
            for j in range(len(p)):
                pocket_rows.append(dict(structure_id=sid, group=ss.group.iloc[0], rank=int(pos[j]), score=float(p["score"].iloc[j]),
                                        dmin=float(dcc[:, j].min()), site_closest=int(np.argmin(dcc[:, j])),
                                        hits=tuple(np.where(dcc[:, j] <= 12.0)[0]), nres=len(pres[j])))
        else:
            for k in range(len(ss)):
                rows[-len(ss) + k]["one2one2"] = 0
    return pd.DataFrame(rows), pd.DataFrame(pocket_rows)


def pooled_p1k(pk: pd.DataFrame, weights: dict | None = None, n=1000):
    """pk sorted by score desc, ties structure_id, rank. weights: group -> multiplicity (bootstrap)."""
    if weights is not None:
        w = pk.group.map(weights).fillna(0).to_numpy()
    else:
        w = np.ones(len(pk))
    tp = (pk.dmin.to_numpy() <= 12.0).astype(float)
    cw = np.cumsum(w)
    k = np.searchsorted(cw, n)  # index where cumulative weight reaches n
    k = min(k, len(pk) - 1)
    # take whole items up to n (fractional last item as in multiplicity expansion)
    ww = w[: k + 1].copy()
    excess = cw[k] - n if cw[k] > n else 0
    ww[-1] -= excess
    return float((tp[: k + 1] * ww).sum() / ww.sum())


def pooled_p1k_1to1(pk: pd.DataFrame, n=1000):
    claimed = set()
    tps = []
    for row in pk.itertuples():
        t = 0
        for h in row.hits:
            key = (row.structure_id, h)
            if key not in claimed:
                claimed.add(key)
                t = 1
                break
        tps.append(t)
        if len(tps) >= n:
            break
    return float(np.mean(tps))


def sort_pk(pk):
    return pk.sort_values(["score", "structure_id", "rank"], ascending=[False, True, True]).reset_index(drop=True)


METRICS = ["topn", "topn2", "ldca2", "anyp2", "anyp_dca2", "one2one2", "maxr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("methods", nargs="*")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-dir", default=None)
    a = ap.parse_args()
    sites = pd.read_csv(a.sites, sep="\t")
    sites = sites[sites.scored == 1]
    ms = [a.ref] + a.methods
    res = {}
    for m in ms:
        name, d = m.split("=", 1)
        s, pk = eval_method(sites, d)
        pk = sort_pk(pk)
        res[name] = (s, pk)
        if a.dump_dir:
            os.makedirs(a.dump_dir, exist_ok=True)
            s.to_csv(os.path.join(a.dump_dir, f"{name}_sites.csv"), index=False)
            pk.drop(columns=["hits"]).to_csv(os.path.join(a.dump_dir, f"{name}_pockets.csv"), index=False)
        print(name, "done", file=sys.stderr, flush=True)
    refname = a.ref.split("=", 1)[0]
    groups = sorted(sites.group.unique())
    rng = np.random.default_rng(12345)
    boots = [dict(zip(groups, np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups)))) for _ in range(a.n_boot)]
    out = []

    def stat(s, pk, w):
        if w is None:
            v = {m: s[m].mean() for m in METRICS}
        else:
            ww = s.group.map(w).to_numpy()
            v = {m: float((s[m].to_numpy() * ww).sum() / ww.sum()) for m in METRICS}
        v["P1K"] = pooled_p1k(pk, w)
        return v

    ref_s, ref_pk = res[refname]
    ref_b = [stat(ref_s, ref_pk, w) for w in boots]
    for name, (s, pk) in res.items():
        assert list(s.site_id) == list(ref_s.site_id) and list(s.structure_id) == list(ref_s.structure_id)
        pt = stat(s, pk, None)
        row = dict(method=name, n_sites=len(s), n_structs=s.structure_id.nunique(), n_missing_structs=int(s.groupby("structure_id").missing.first().sum()),
                   pockets_mean=round(pk.groupby("structure_id").size().reindex(s.structure_id.unique(), fill_value=0).mean(), 2),
                   P1K_1to1=round(100 * pooled_p1k_1to1(pk), 1))
        bs = [stat(s, pk, w) for w in boots]
        for m in METRICS + ["P1K"]:
            arr = np.array([b[m] for b in bs])
            row[m] = round(100 * pt[m], 1)
            row[m + "_ci"] = f"[{100*np.percentile(arr,2.5):.1f},{100*np.percentile(arr,97.5):.1f}]"
            if name != refname:
                dd = arr - np.array([b[m] for b in ref_b])
                row["d_" + m] = f"{100*(pt[m]-stat(ref_s, ref_pk, None)[m]):+.1f} [{100*np.percentile(dd,2.5):+.1f},{100*np.percentile(dd,97.5):+.1f}]"
        out.append(row)
        print(row, flush=True)
    pd.DataFrame(out).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
