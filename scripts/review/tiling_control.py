#!/usr/bin/env python
"""Reviewer control: label-free surface tiling pockets on the dev inputs (no learning, no ligand info).

SAS points (probe 1.4 A, 40 Fibonacci points/atom, element vdW radii) -> variants, each capped at 50 pockets:
  tile12_rand_s{0,1,2}  Poisson-disk tiling (random visiting order, min spacing 12 A), ranked randomly
  tile12_bur           same tiles (seed 0), ranked by buriedness = #heavy atoms within 10 A of the centre
  tile10_bur           Poisson-disk tiling at 10 A (seed 0), ranked by buriedness
  burnms10 / burnms12  greedy NMS over ALL SAS points in buriedness order (radius 10 / 12 A)
  burrow_shuffled_s0   BURROW default s0 pockets (saved CLI predictions), ranks randomly permuted
Residues = residues with any heavy atom <= 6 A of the centre (LIGYSIS convention for point-only methods).
Writes results/review_dev/controls/<variant>/<id>.csv
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/review_dev/controls"
VDW = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "SE": 1.9}
CAP = 50


def fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def load(path):
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    xyz, rad, rid = [], [], []
    for ch in st[0]:
        for r in ch:
            for a in r:
                xyz.append(a.pos.tolist())
                rad.append(VDW.get(a.element.name.upper(), 1.8))
                rid.append(f"{ch.name}:{r.seqid.num}{r.seqid.icode.strip()}")
    return np.array(xyz), np.array(rad), np.array(rid)


def sas_points(xyz, rad, probe=1.4, n=40):
    u = fib(n)
    pts = (xyz[:, None, :] + (rad[:, None, None] + probe) * u[None]).reshape(-1, 3)
    owner = np.repeat(np.arange(len(xyz)), n)
    tp = cKDTree(pts)
    keep = np.ones(len(pts), bool)
    for j in range(len(xyz)):
        idx = np.asarray(tp.query_ball_point(xyz[j], rad[j] + probe - 1e-3), dtype=int)
        if idx.size:
            keep[idx[owner[idx] != j]] = False
    return pts[keep]


def poisson(pts, spacing, order):
    acc = np.zeros((0, 3))
    s2 = spacing ** 2
    for i in order:
        p = pts[i]
        if acc.shape[0] == 0 or np.min(np.sum((acc - p) ** 2, axis=1)) >= s2:
            acc = np.vstack([acc, p])
    return acc


def write(d, sid, centres, scores, xyz_tree, rid):
    os.makedirs(d, exist_ok=True)
    o = np.argsort(-scores, kind="stable")[:CAP]
    rows = []
    for rnk, j in enumerate(o, 1):
        res = sorted(set(rid[xyz_tree.query_ball_point(centres[j], 6.0)]))
        rows.append(dict(rank=rnk, score=float(scores[j]), center_x=round(centres[j][0], 3), center_y=round(centres[j][1], 3),
                         center_z=round(centres[j][2], 3), residues=" ".join(res)))
    pd.DataFrame(rows, columns=["rank", "score", "center_x", "center_y", "center_z", "residues"]).to_csv(Path(d) / f"{sid}.csv", index=False)
    return len(rows)


def job(path):
    sid = Path(path).stem
    xyz, rad, rid = load(path)
    t = cKDTree(xyz)
    pts = sas_points(xyz, rad)
    bur = np.array([len(x) for x in t.query_ball_point(pts, 10.0)], float)
    stats = {"id": sid, "n_sas": len(pts)}
    h = int.from_bytes(sid.encode()[:8].ljust(8, b"0"), "little") % (2 ** 32)
    for s in range(3):
        rng = np.random.default_rng([s, h])
        tiles = poisson(pts, 12.0, rng.permutation(len(pts)))
        sc = rng.random(len(tiles))
        stats[f"tile12_rand_s{s}"] = write(OUT / f"tile12_rand_s{s}", sid, tiles, sc, t, rid)
        stats[f"tile12_n_uncapped_s{s}"] = len(tiles)
        if s == 0:
            tb = np.array([len(x) for x in t.query_ball_point(tiles, 10.0)], float)
            stats["tile12_bur"] = write(OUT / "tile12_bur", sid, tiles, tb + 1e-6 * sc, t, rid)
            tiles10 = poisson(pts, 10.0, rng.permutation(len(pts)))
            tb10 = np.array([len(x) for x in t.query_ball_point(tiles10, 10.0)], float)
            stats["tile10_bur"] = write(OUT / "tile10_bur", sid, tiles10, tb10 + 1e-6 * rng.random(len(tiles10)), t, rid)
    jit = np.random.default_rng([99, h]).random(len(pts)) * 1e-3
    order = np.argsort(-(bur + jit), kind="stable")
    for R in (10.0, 12.0):
        c = poisson(pts, R, order)
        cb = np.array([len(x) for x in t.query_ball_point(c, 10.0)], float)
        stats[f"burnms{int(R)}"] = write(OUT / f"burnms{int(R)}", sid, c, cb + 1e-6 * np.arange(len(c))[::-1], t, rid)
    # BURROW shuffled ranks
    b = ROOT / "results/dev/biolip/preds_cli/burrow_all_default_s0" / f"{sid}.csv"
    if b.exists():
        p = pd.read_csv(b)
        rng = np.random.default_rng([7, h])
        perm = rng.permutation(len(p))
        p = p.iloc[perm].reset_index(drop=True)
        p["rank"] = np.arange(1, len(p) + 1)
        p["score"] = np.linspace(1, 0, len(p)) if len(p) else []
        d = OUT / "burrow_shuffled_s0"
        os.makedirs(d, exist_ok=True)
        p.to_csv(d / f"{sid}.csv", index=False)
    return stats


def main():
    files = [l.strip() for l in open(ROOT / "results/dev/biolip/inputs_audited.list") if l.strip()]
    files = [str(ROOT / f) for f in files]
    if len(sys.argv) > 1:
        files = files[: int(sys.argv[1])]
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(job, files, chunksize=4))
    pd.DataFrame(rows).to_csv(OUT / "control_stats.csv", index=False)
    print(pd.DataFrame(rows).describe().T.to_string())


if __name__ == "__main__":
    main()
