#!/usr/bin/env python
"""Gate G1 on the audited BioLiP2 dev set: geometry-only candidates (before any learning/ranking),
<= 40 per chain after LIGYSIS-redundancy NMS (5 Å / Jaccard 0.75), ordered by Σ vol·b; a site is
recalled if one candidate centre has DCC <= 12 Å to the site centre AND DCA <= 4 Å to its ligand
atoms.  Also fpocket 4.2.3 and P2Rank 2.5.1 (all pockets) on the same inputs.

  python scripts/burrow_g1_biolip.py --feats-dir results/dev/cache/feats_biolip/<hash> --tag default
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.model import FormationConfig, geo_value  # noqa: E402
from pocketdetect.burrow.pockets import PointGraph, assign_residues, generate, nms  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402
from pocketdetect.evaluate import read_predictions  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402

B = ROOT / "results/dev/biolip"
_G = {}


def hits(C, ss):
    C = np.asarray(C, float).reshape(-1, 3)
    out = []
    for s in ss:
        if len(C) == 0:
            out.append(False)
            continue
        dcc = np.linalg.norm(C - s.center, axis=1)
        dca = cKDTree(s.ligand_atoms).query(C, k=1)[0]
        out.append(bool(((dcc <= 12) & (dca <= 4)).any()))
    return out


def job(sid):
    ss = _G["by"][sid]
    z = np.load(Path(_G["fd"]) / f"{sid}.npz")
    names = list(z["names"])
    fc = FormationConfig()
    v = geo_value(z["feats"], names)
    pk = generate(z["X"], z["points"], z["vol"], v, fc.geo_thresholds, fc.link, fc.max_extent, fc.split_size,
                  fc.min_vol, sub_size=fc.sub_size, graph=PointGraph(z["X"], fc.link))
    P = load_protein(B / "inputs" / f"{sid}.cif")
    assign_residues(pk, z["points"], cKDTree(P.coords), P.res_index, P.residue_ids)
    from pocketdetect.burrow.model import geo_score
    rows = []
    allc = [p.center for p in pk]
    rows += [dict(method="burrow_geo_pool_uncapped", hit=h, n=len(pk)) for h in hits(allc, ss)]
    for lab, sc in (("strength", [p.strength for p in pk]), ("buriedvol", geo_score(pk, z["vol"], z["feats"], names, fc))):
        keep = nms(pk, sc, 5.0, 0.75, 40)
        rows += [dict(method=f"burrow_geo_{lab}_nms5_cap40", hit=h, n=len(keep)) for h in hits([pk[i].center for i in keep], ss)]
        keep = nms(pk, sc, 5.0, 0.75, None)
        rows += [dict(method=f"burrow_geo_{lab}_nms5_nocap", hit=h, n=len(keep)) for h in hits([pk[i].center for i in keep], ss)]
    for m in ("p2rank", "fpocket"):
        pr = read_predictions(B / "preds" / m / f"{sid}.csv", sid)
        rows += [dict(method=m, hit=h, n=pr.n) for h in hits(pr.centers, ss)]
    return [dict(structure_id=sid, **r) for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats-dir", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    by = sites_by_structure(read_sites(B / "dev_sites_all.tsv"))
    _G.update(by=by, fd=a.feats_dir)
    with ProcessPoolExecutor(6, mp_context=mp.get_context("fork")) as ex:
        rows = [r for rr in ex.map(job, list(by), chunksize=4) for r in rr]
    df = pd.DataFrame(rows)
    ion = {(r.structure_id, r.site_id): r.is_ion for r in pd.read_csv(B / "dev_sites_all.tsv", sep="\t").itertuples()}
    s = df.groupby("method").agg(recall=("hit", "mean"), n_sites=("hit", "size"))
    s["pockets_mean"] = df.drop_duplicates(["structure_id", "method"]).groupby("method").n.mean()
    s["recall"] *= 100
    print(s.round(1).to_string())
    s.round(2).to_csv(B / f"G1_biolip_{a.tag}.csv")


if __name__ == "__main__":
    main()
