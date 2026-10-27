#!/usr/bin/env python
"""Error analysis (DEV only): v2 `small` default s0 on the non-ion BioLiP2-dev sites (dev_sites_small.tsv).

Per site: v2 / P2Rank 2.5.1 / fpocket+PRANK outcome (LIGYSIS closest-pocket top-N+2, DCC 12 Å) and the
v2 miss type:
  closest_rule   some v2 pocket of rank <= N+2 is within 12 Å, but the closest pocket is ranked > N+2
  ranking        a v2 pocket within 12 Å exists, all ranked > N+2
  floor_removed  no v2 pocket within 12 Å, but the full (v1, no floor) list has one
  coverage       no pocket within 12 Å even in the full list
Site descriptors: #residues, ligand heavy atoms, BURROW buriedness of the grid points <= 4 Å of the
ligand (mean blocked@16, depth), ligand class (CCD-based), peptide-adjacent (site residues in Track A's
peptide mask), site WCN relative to the chain median, distance to a chain terminus (residues), #sites/chain.

  python scripts/burrow_nonion_errors.py
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("BURROW_TRAINSET", str(ROOT / "results/dev/cache/trainset_snapshot/audited_d6"))
from pocketdetect.burrow.biolip import entries  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402
from pocketdetect.datasets.trainset import load_entry  # noqa: E402
from pocketdetect.site_eval import evaluate_sites  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402

B = ROOT / "results/dev/biolip"
FEATS = ROOT / "results/dev/cache/feats_biolip/69ddda668a"   # default-preset feature cache (dev inputs)
METHODS = {"v2": B / "v2_preds/H5_small_default_s0", "v1_full": B / "preds_freeze/burrow_small_default_s0",
           "p2rank": B / "preds/p2rank", "fpocket_prank": B / "preds/fpocket_prank"}

NUC_COF = set("ATP ADP AMP ANP ACP AGS APC GTP GDP GMP GNP GSP GCP CTP CDP CMP UTP UDP UMP TTP TMP DTP DCP NAD NAI NAP "
              "NDP FAD FDA FMN COA ACO SCA CAA MLC SAM SAH SFG HEM HEC HEA HEB CLA BCL PLP PMP TPP TDP B12 COB "
              "F42 H4B BH4 MTE THF THG FOL MTX 5GP 3PS U5P C5P A3P PAP PPS 2GP 3GP IMP XMP ADN GUN".split())
SUGAR = set("GLC BGC MAN BMA GAL GLA FUC FUL XYS XYP NAG NDG NGA A2G SIA SLB RAM RHA ARA AHR FRU SUC MAL LAT TRE "
            "GCS GCU BDP IDS SGN GLO MGL KDO NGZ".split())
LIPID = set("PLM MYR OLA OLC STE PAM PEE PEF PGV PLC LHG LMT LDA C8E HTG DMU UND SPH CHL CLR CHD CPS PCF POP 1PE "
            "PG4 P6G PEG PGE 2PE 12P 15P 7PE BNG BOG DDQ LPP MPG 3PE PEU OCT DAO DKA LAU".split())
ADDITIVE = set("GOL EDO DMS ACT FMT PG0 MPD IPA EOH MOH BME DTT TRS EPE MES CIT TLA SO4 PO4 NO3 SCN AZI "
               "ACE NH4 IMD BU3 PGO 1PG FLC MLI".split())


def lig_class(ccds, n_atoms, peptide_adj):
    s = set(ccds)
    if s & NUC_COF:
        return "nucleotide/cofactor"
    if s & SUGAR:
        return "sugar"
    if s & LIPID:
        return "lipid/detergent"
    if (s and s <= ADDITIVE) or n_atoms < 8:
        return "additive/fragment"
    return "drug-like, peptide-adjacent" if peptide_adj else "drug-like/other"


def _site_props(args):
    sid, npz, rows = args
    e = load_entry(npz)
    a = e.arrays
    P = load_protein(B / "inputs" / f"{sid}.cif", structure_id=sid)
    z = np.load(FEATS / f"{sid}.npz")
    names = list(z["names"])
    pts = z["points"]
    ptree = cKDTree(pts) if len(pts) else None
    wcn_res = pd.Series(P.props[:, P.prop_names.index("wcn")]).groupby(P.res_index).mean()
    rid_index = {r: i for i, r in enumerate(P.residue_ids)}
    med = float(np.median(wcn_res.values))
    n_res_tot = len(P.residue_ids)
    mask = set()
    for kind, res in zip(a["mask_kinds"], a["mask_residues"]):
        if kind == "peptide" and res:
            mask |= {f"{e.chain}:{r}" for r in res.split(";")}
    lsite = a["lig_site"]
    out = []
    for r in rows:
        k = int(str(r["site_id"])[1:])
        ccds = [str(c) for c, s, ion in zip(a["lig_ccd"], lsite, a["lig_is_ion"]) if s == k and not ion]
        atoms = np.array([[float(v) for v in t.split(",")] for t in r["ligand_atoms"].split()])
        res = str(r["residues"]).split()
        ri = [rid_index[x] for x in res if x in rid_index]
        near = sorted({j for l in ptree.query_ball_point(atoms, 4.0) for j in l}) if ptree is not None else []
        pep = bool(set(res) & mask)
        out.append(dict(structure_id=sid, site_id=str(r["site_id"]), n_res=len(res), n_lig_atoms=len(atoms),
                        n_sites_chain=int(r["n_sites"]), ligand_class=lig_class(ccds, len(atoms), pep),
                        ccds="&".join(sorted(set(ccds))), peptide_adjacent=pep,
                        bur16=float(np.mean(z["feats"][near, names.index("bur16")])) if near else np.nan,
                        depth=float(np.mean(z["feats"][near, names.index("depth")])) if near else np.nan,
                        rel_wcn=float(np.mean([wcn_res[i] for i in ri]) / med) if ri else np.nan,
                        term_dist=int(min(min(i, n_res_tot - 1 - i) for i in ri)) if ri else -1))
    return out


def main():
    st = read_sites(B / "dev_sites_small.tsv")
    by = sites_by_structure(st)
    groups = {s: ss[0].meta.get("group") for s, ss in by.items()}
    res = {}
    for m, d in METHODS.items():
        r = evaluate_sites(by, str(d), groups=groups).sites
        r["site_id"] = r["site_id"].astype(str)
        res[m] = r.set_index(["structure_id", "site_id"])
    raw = pd.read_csv(B / "dev_sites_small.tsv", sep="\t", dtype={"site_id": str})
    md = entries("dev")
    npz = dict(zip(md.entry_id, md.npz))
    tasks = [(sid, npz[sid], g.to_dict("records")) for sid, g in raw.groupby("structure_id")]
    with ProcessPoolExecutor(12) as ex:
        props = pd.DataFrame([r for rr in ex.map(_site_props, tasks, chunksize=4) for r in rr])
    df = props.set_index(["structure_id", "site_id"])
    for m in METHODS:
        df[f"{m}_hit"] = res[m]["recall_topn2"].astype(bool).reindex(df.index)
    v2, v1 = res["v2"].reindex(df.index), res["v1_full"].reindex(df.index)
    N2 = v2["n_sites"] + 2
    anyp = (v2["rank_first_hit"] > 0) & (v2["rank_first_hit"] <= N2)
    has12 = v2["rank_first_hit"] > 0
    v1has = v1["rank_first_hit"] > 0
    df["v2_outcome"] = np.select([df["v2_hit"], anyp, has12, v1has],
                                 ["hit", "miss: closest-pocket rule", "miss: ranking", "miss: removed by floor"],
                                 "miss: coverage")
    df = df.reset_index()
    df.to_csv(B / "nonion_errors_sites.csv", index=False)
    print(df.v2_outcome.value_counts())
    print(df[["v2_hit", "v1_full_hit", "p2rank_hit", "fpocket_prank_hit"]].mean().round(3))


if __name__ == "__main__":
    main()
