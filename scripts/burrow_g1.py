#!/usr/bin/env python
"""Gate G1 (DESIGN §2.2): candidate recall before ranking.

A ligand is recalled if at least one candidate pocket centre has DCC <= 12 Å AND DCA <= 4 Å
(same pocket).  Reports recall (ligand-averaged), #pockets/protein (mean, median, p90), for
BURROW geometry-only candidate generator variants, fpocket 4.2.3 and P2Rank 2.5.1 (all pockets),
plus the point-level ceiling (fraction of ligands with >= 1 candidate point within 4 Å).

  python scripts/burrow_g1.py --feats results/dev/cache/feats/<hash> --out results/dev/G1
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # shared per-user thread limit; LightGBM gets num_threads explicitly

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.devdata import load_gts, read_manifest  # noqa: E402
from pocketdetect.burrow.pockets import assign_residues, generate, nms  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402
from pocketdetect.evaluate import read_predictions  # noqa: E402

VARIANTS = {
    # name: (value expression, thresholds, link, max_pockets, sub_size)
    "bur10_t.5/.6/.7": ("bur10", (0.5, 0.6, 0.7), 2.05, 40, None),
    "mix_t.4/.5/.6/.7": ("mix", (0.4, 0.5, 0.6, 0.7), 2.05, 40, None),
    "mix_t.4/.5/.6/.7_sub8": ("mix", (0.4, 0.5, 0.6, 0.7), 2.05, 40, 8.0),
    "mix_t.4/.5/.6/.7_sub6": ("mix", (0.4, 0.5, 0.6, 0.7), 2.05, 40, 6.0),
    "mix_t.4/.55/.7/.85_sub8": ("mix", (0.4, 0.55, 0.7, 0.85), 2.05, 40, 8.0),
    "mix_t.4/.55/.7/.85_sub6": ("mix", (0.4, 0.55, 0.7, 0.85), 2.05, 40, 6.0),
    "mix_t.4/.55/.7/.85_sub6_n60": ("mix", (0.4, 0.55, 0.7, 0.85), 2.05, 60, 6.0),
}


def value_of(name, F, names):
    col = lambda n: F[:, names.index(n)]
    if name == "mix":
        return (col("bur10") + col("bur16") + col("bur24")) / 3.0
    return col(name)


def hits(centers, gt):
    """Per relevant ligand: recalled (DCC<=12 & DCA<=4 by one pocket), plus rank of first such."""
    out = []
    C = np.asarray(centers, np.float64).reshape(-1, 3)
    for lig in gt.ligands:
        if len(C) == 0:
            out.append((lig.ligand_id, False, -1))
            continue
        dcc = np.linalg.norm(C - lig.center, axis=1)
        dca = cKDTree(lig.coords).query(C, k=1)[0]
        ok = (dcc <= 12) & (dca <= 4)
        out.append((lig.ligand_id, bool(ok.any()), int(np.argmax(ok)) + 1 if ok.any() else -1))
    return out


_G = {}


def _job(sid):
    gt = _G["gts"][sid]
    if not gt.ligands:
        return []
    z = np.load(Path(_G["feats"]) / f"{sid}.npz")
    F, names, pts, X, vol = z["feats"], list(z["names"]), z["points"], z["X"], z["vol"]
    P = load_protein(_G["paths"][sid], structure_id=sid)
    tree = cKDTree(P.coords)
    rows = []
    pt_ceiling = float((z["d_all"] <= 4.0).any())
    for lig in gt.ligands:
        d = cKDTree(lig.coords).query(pts, k=1)[0] if len(pts) else np.array([np.inf])
        rows.append(dict(structure_id=sid, ligand_id=lig.ligand_id, method="point_ceiling", hit=bool((d <= 4).any()),
                         n_pockets=np.nan))
    del pt_ceiling
    for vname, (val, ths, link, nmax, sub) in VARIANTS.items():
        v = value_of(val, F, names)
        pk = generate(X, pts, vol, v, ths, link=link, sub_size=sub)
        assign_residues(pk, pts, tree, P.res_index, P.residue_ids)
        keep = nms(pk, [p.strength for p in pk], max_n=nmax)
        cents = [pk[i].center for i in keep]
        for lid, h, r in hits(cents, gt):
            rows.append(dict(structure_id=sid, ligand_id=lid, method=f"burrow_geo:{vname}", hit=h, n_pockets=len(keep)))
    for m, d in _G["baselines"].items():
        f = Path(d) / f"{sid}.pockets.csv"
        pr = read_predictions(f, sid) if f.exists() else None
        cents = pr.centers if pr is not None else np.zeros((0, 3))
        for lid, h, r in hits(cents, gt):
            rows.append(dict(structure_id=sid, ligand_id=lid, method=m, hit=h, n_pockets=len(cents)))
            rows.append(dict(structure_id=sid, ligand_id=lid, method=m + "@40", hit=h and 0 < r <= 40,
                             n_pockets=min(len(cents), 40)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "results/dev/manifests/all.csv")
    ap.add_argument("--feats", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results/dev/G1")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args(argv)
    df = read_manifest(a.manifest)
    _G.update(gts=load_gts(a.manifest), feats=str(a.feats), paths=dict(zip(df.structure_id, df.structure_path)),
              baselines={"fpocket": str(ROOT / "results/dev/baselines/fpocket"),
                         "p2rank": str(ROOT / "results/dev/baselines/p2rank")})
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = [r for rr in ex.map(_job, list(df.structure_id), chunksize=4) for r in rr]
    res = pd.DataFrame(rows)
    sub = dict(zip(df.structure_id, df.subset))
    res["set"] = res.structure_id.map(lambda s: "chen11" if sub[s] == "chen11" else "joined")
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / "g1_ligands.csv", index=False)
    summ = []
    for (m, st), g in list(res.groupby(["method", "set"])) + [((m, "all"), g) for m, g in res.groupby("method")]:
        pp = g.drop_duplicates("structure_id").n_pockets
        summ.append(dict(method=m, set=st, n_ligands=len(g), recall=round(100 * g.hit.mean(), 2),
                         pockets_mean=round(pp.mean(), 1), pockets_median=pp.median(), pockets_p90=pp.quantile(0.9)))
    summ = pd.DataFrame(summ).sort_values(["set", "method"])
    summ.to_csv(a.out / "g1_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
