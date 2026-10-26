#!/usr/bin/env python
"""Sanity checks for the AF-mode test set (Track C). Exit code 1 if any hard check fails.

Checks
  1. counts / bookkeeping: every LIGYSIS UniProt and every benchmark site accounted for
     (included, flagged, or on an excluded protein); fragmented proteins listed
  2. model files: present, sha256 == extract index, single chain, numbering 1..L,
     pLDDT (B-factor) in [0, 100] and equal to the GT JSON
  3. residue-type match rate (site level, by reference) and AF-vs-UniProt sequence identity
  4. reproduction of LIGYSIS's own site centres on the representative PDB chains (validates the
     residue mapping; LIGYSIS centre = mean CA of site residues)
  5. pLDDT distributions (protein and site level)
  6. independent recomputation for a fixed random sample of sites: CA centre from the model
     file, residue types vs UniProt, transferred ligand contacts the site
  7. known-answer spot checks: CA2 (P00918) catalytic zinc on His94/His96/His119;
     ESR1 (P03372) estradiol pocket with Glu353/Arg394/His524
  8. harness compatibility + controls via pocketdetect.evaluate: oracle centres (LDCC12
     top-n recall must be 1), predictions 1000 A away (0), oracle shifted by 30 A (< 0.1; may hit
     a neighbouring site), protein centroid (low)

Usage: .venv/bin/python scripts/data/verify_af_set.py [--n-spot 12]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
from build_af_io import af_ground_truth, load_af_dataset, load_af_manifest, read_gt_json  # noqa: E402
from safe_pickle import safe_load  # noqa: E402

from pocketdetect.evaluate import Predictions, evaluate  # noqa: E402

AF = ROOT / "data/processed/af"
LIG_RES = ROOT / "data/raw/ligysis/lbs_comparison_results"
FAILS: list[str] = []
REPORT: dict = {}


def check(name: str, ok: bool, msg: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + msg if msg else ''}")
    REPORT.setdefault("checks", {})[name] = dict(ok=bool(ok), msg=msg)
    if not ok:
        FAILS.append(name)


def model_arrays(path: Path):
    st = gemmi.read_structure(str(path))
    ch = st[0][0]
    out = dict(chains=len(st[0]), nums=[], names=[], ca=[], b=[], atoms=[])
    for r in ch:
        out["nums"].append(r.seqid.num)
        out["names"].append(r.name)
        ca = r.find_atom("CA", "*")
        out["ca"].append([ca.pos.x, ca.pos.y, ca.pos.z])
        out["b"].append(ca.b_iso)
        out["atoms"].append(np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r]))
    out["ca"] = np.array(out["ca"])
    out["b"] = np.array(out["b"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spot", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args()

    ex = pd.read_csv(AF / "extract_index.csv", keep_default_na=False)
    P = pd.read_csv(AF / "af_manifest.csv")
    S = pd.read_csv(AF / "af_sites.csv")
    Lg = pd.read_csv(AF / "af_ligands.csv")
    hs = safe_load(LIG_RES / "LIGYSIS_human_sites_APRIL_2024.pkl")
    preds = safe_load(LIG_RES / "ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl")
    defs = safe_load(LIG_RES / "LIGYSIS_sites_DEF_TRANS.pkl")
    bench = preds[preds.origin == "LIGYSIS"].merge(defs[["rep_chain", "ID", "SITE_NAME", "ACC"]], on=["rep_chain", "ID"])

    # ------------------------------------------------------------------ 1. counts
    print("== 1. counts")
    check("manifest covers all LIGYSIS UniProts", set(P.acc) == set(hs.ACC) and len(P) == hs.ACC.nunique(), f"{len(P)} rows")
    check("extract index covers all LIGYSIS UniProts", set(ex.acc) == set(hs.ACC))
    frag = ex[ex.status == "fragmented"]
    REPORT["extract_status"] = ex.status.value_counts().to_dict()
    REPORT["fragmented_accs"] = frag.acc.tolist()
    REPORT["absent_accs"] = ex[ex.status == "absent"].acc.tolist()
    print("  extract:", REPORT["extract_status"])
    excl = set(P.loc[P.status == "excluded", "acc"])
    s_all = set(S.site_name)
    missing = set(hs.SITE_NAME) - s_all
    check("every LIGYSIS site is in af_sites.csv or on an excluded protein",
          all(n.rsplit("_", 2)[0] in excl for n in missing), f"{len(missing)} sites on excluded proteins")
    bn = set(bench.SITE_NAME)
    b_inc = set(S.loc[S.include & S.in_lbs_benchmark, "site_name"])
    b_flag = set(S.loc[~S.include & S.in_lbs_benchmark, "site_name"])
    b_excl = {n for n in bn if n not in s_all}
    check("benchmark sites partition (6,882)", len(bn) == 6882 and len(b_inc) + len(b_flag) + len(b_excl) == 6882
          and set(S.loc[S.in_lbs_benchmark, "site_name"]) <= bn,
          f"included {len(b_inc)}, flagged {len(b_flag)}, on excluded proteins {len(b_excl)}")
    bench_accs = set(bench.ACC)
    pb = P[P.acc.isin(bench_accs)]
    REPORT["benchmark_proteins"] = dict(total=len(bench_accs), included=int((pb.status == "included").sum()),
                                        excluded=pb[pb.status == "excluded"].reason.value_counts().to_dict())
    REPORT["counts"] = dict(proteins_total=len(P), proteins_included=int((P.status == "included").sum()),
                            proteins_excluded=P[P.status == "excluded"].reason.value_counts().to_dict(),
                            sites_rows=len(S), sites_included=int(S.include.sum()),
                            benchmark_sites_included=len(b_inc), benchmark_sites_flagged=len(b_flag),
                            benchmark_sites_on_excluded_proteins=len(b_excl),
                            site_flags=S["flags"].fillna("").str.replace(r"duplicate_of:\S*", "duplicate", regex=True).value_counts().to_dict())
    print("  counts:", json.dumps(REPORT["counts"]))
    print("  benchmark proteins:", REPORT["benchmark_proteins"])
    check("structure ids unique", P.structure_id.is_unique)

    # ------------------------------------------------------------------ 2. model files
    print("== 2. model files")
    inc = P[P.status == "included"]
    exi = ex.set_index("acc")
    bad_sha, bad_num, bad_b, bad_chain, n_checked = [], [], [], [], 0
    for r in inc.itertuples():
        mp = AF / r.model_path
        if not mp.exists():
            bad_sha.append(r.acc)
            continue
        h = hashlib.sha256(mp.read_bytes()).hexdigest()
        if h != exi.loc[r.acc, "sha256"]:
            bad_sha.append(r.acc)
    check("all included models present with matching sha256", not bad_sha, f"{len(inc)} models; bad {bad_sha[:5]}")
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(inc.acc.to_numpy(), size=min(150, len(inc)), replace=False)
    for acc in sample:
        r = inc[inc.acc == acc].iloc[0]
        m = model_arrays(AF / r.model_path)
        gt = read_gt_json(AF / r.gt_path)
        n_checked += 1
        if m["chains"] != 1:
            bad_chain.append(acc)
        if m["nums"] != list(range(1, len(m["nums"]) + 1)) or gt["residue_numbers"] != m["nums"]:
            bad_num.append(acc)
        if not (np.all((m["b"] >= 0) & (m["b"] <= 100)) and np.allclose(m["b"], gt["plddt"], atol=0.01)):
            bad_b.append(acc)
    check("single chain (150 sampled models)", not bad_chain, str(bad_chain[:5]))
    check("residue numbering 1..L == GT JSON (sample)", not bad_num, str(bad_num[:5]))
    check("pLDDT in [0,100] and == GT JSON (sample)", not bad_b, str(bad_b[:5]))
    check("manifest: numbering_1_to_L for all included", bool(inc.numbering_1_to_L.astype(str).eq("True").all()))

    # ------------------------------------------------------------------ 3. residue types
    print("== 3. residue-type match")
    seq_eq = inc.af_seq_eq_uniprot.astype(str).eq("True")
    REPORT["af_seq_eq_uniprot"] = dict(n=int(seq_eq.sum()), of=len(inc), mismatching=inc.loc[~seq_eq, "acc"].tolist())
    print(f"  AF sequence == UniProt canonical (2026_03): {seq_eq.sum()}/{len(inc)}; differ: {REPORT['af_seq_eq_uniprot']['mismatching']}")
    rt = S.groupby("restype_ref").agg(n=("site_name", "size"), exact=("restype_id", lambda x: float((x == 1).mean())),
                                      ge075=("restype_id", lambda x: float((x >= 0.75).mean())))
    print(rt.to_string())
    REPORT["restype_by_ref"] = rt.reset_index().to_dict("records")
    ok_rate = float((S.restype_id >= 0.75).mean())
    exact_rate = float((S.restype_id == 1).mean())
    REPORT["restype_match_rate"] = dict(exact=exact_rate, ge_075=ok_rate)
    check("site residue-type match rate (>= 0.75 identity) >= 0.97", ok_rate >= 0.97, f"exact {exact_rate:.4f}, >=0.75 {ok_rate:.4f}")
    check("no included site below the residue-type threshold", bool((S.loc[S.include, "restype_id"] >= 0.75).all()))
    okl = Lg[Lg.status == "ok"]
    REPORT["ligand_transfer"] = dict(status=Lg.status.value_counts().to_dict(), match=okl["match"].value_counts().to_dict(),
                                     rmsd_quantiles={str(q): float(okl.rmsd.quantile(q)) for q in (0.5, 0.75, 0.9, 0.95)},
                                     frac_rmsd_le2=float((okl.rmsd <= 2).mean()))
    print("  ligand transfer:", json.dumps(REPORT["ligand_transfer"]))

    # ------------------------------------------------------------------ 4. LIGYSIS centre reproduction
    print("== 4. LIGYSIS centre reproduction on representative chains")
    cc = S.ligysis_centre_check.dropna()
    check("LIGYSIS CA-centroid reproduced (max dev < 0.002 A; stored rounded to 1e-3)", len(cc) > 7000 and cc.max() < 0.002,
          f"n={len(cc)}, max={cc.max():.5f}")

    # ------------------------------------------------------------------ 5. pLDDT distribution
    print("== 5. pLDDT distribution")
    si = S[S.include]
    REPORT["plddt"] = dict(
        site_band_all=si.plddt_band.value_counts().to_dict(),
        site_band_benchmark=si[si.in_lbs_benchmark].plddt_band.value_counts().to_dict(),
        site_mean_quantiles={str(q): float(si.plddt_mean.quantile(q)) for q in (0.05, 0.25, 0.5)},
        protein_frac_lt50_quantiles={str(q): float(inc.frac_plddt_lt50.quantile(q)) for q in (0.5, 0.75, 0.9)},
        frac_sites_near_low=float(si.near_low_plddt.mean()),
        frac_sites_with_prunable_residue=float((si.frac_res_prunable > 0).mean()),
    )
    print("  ", json.dumps(REPORT["plddt"]))
    check("site pLDDT within [0,100]", bool(si.plddt_min.between(0, 100).all() and si.plddt_mean.between(0, 100).all()))
    check("band consistent with mean", bool(((si.plddt_mean >= 90) == (si.plddt_band == ">=90")).all()))

    # ------------------------------------------------------------------ 6. independent recomputation
    print("== 6. spot checks (independent recomputation)")
    up = pd.read_csv(AF / "external/uniprot_human_reviewed.tsv.gz", sep="\t")
    useq = dict(zip(up.Entry, up.Sequence))
    sp = si.sample(n=args.n_spot, random_state=args.seed)
    rows = []
    for r in sp.itertuples():
        pr = P[P.acc == r.acc].iloc[0]
        m = model_arrays(AF / pr.model_path)
        gt = read_gt_json(AF / pr.gt_path)
        site = next(s for s in gt["sites"] if s["site_name"] == r.site_name)
        res = site["residues"]
        ca = m["ca"][[u - 1 for u in res]].mean(0)
        d_ca = float(np.linalg.norm(ca - np.array(site["center_ca"])))
        types = "".join(gemmi.find_tabulated_residue(m["names"][u - 1]).one_letter_code.upper() for u in res)
        uni = "".join(useq[r.acc][u - 1] for u in res) if r.acc in useq and max(res) <= len(useq[r.acc]) else ""
        lig_d = np.nan
        if site["ligands"]:
            sx = np.vstack([m["atoms"][u - 1] for u in res])
            lx = np.vstack([np.asarray(l["coords"]).reshape(-1, 3) for l in site["ligands"] if l["rmsd"] <= 2.0] or [np.zeros((0, 3))])
            if len(lx):
                lig_d = float(cKDTree(sx).query(lx)[0].min())
        rows.append(dict(site=r.site_name, n_res=len(res), center_ca_dev=round(d_ca, 4), af_types=types[:20],
                         uniprot_types=uni[:20], types_eq=types == uni, plddt=round(r.plddt_mean, 1),
                         lig=r.lig_resnames if isinstance(r.lig_resnames, str) else "", lig_min_dist=round(lig_d, 2),
                         rmsd_rep=round(r.rmsd_ca_af_vs_rep, 2) if np.isfinite(r.rmsd_ca_af_vs_rep) else np.nan))
    spot = pd.DataFrame(rows)
    print(spot.to_string(index=False))
    REPORT["spot_checks"] = spot.to_dict("records")
    check("spot: CA centre recomputed from model file", bool((spot.center_ca_dev < 0.002).all()))
    check("spot: transferred ligands contact their site (min dist <= 4.5 A)", bool(spot.lig_min_dist.dropna().le(4.5).all()))

    # ------------------------------------------------------------------ 7. known answers
    print("== 7. known-answer checks")
    g = read_gt_json(AF / P.loc[P.acc == "P00918", "gt_path"].iloc[0])
    m = model_arrays(AF / P.loc[P.acc == "P00918", "model_path"].iloc[0])
    zn_sites = [s for s in g["sites"] if {94, 96, 119} <= set(s["residues"]) and any(l["resname"] == "ZN" for l in s["ligands"])]
    ok7a = False
    if zn_sites:
        s = zn_sites[0]
        zn = np.vstack([np.asarray(l["coords"]).reshape(-1, 3) for l in s["ligands"] if l["resname"] == "ZN"])
        st = gemmi.read_structure(str(AF / g["model_path"]))
        ne = []
        for rr in st[0][0]:
            if rr.seqid.num in (94, 96, 119):
                for a in rr:
                    if a.name in ("NE2", "ND1"):
                        ne.append([a.pos.x, a.pos.y, a.pos.z])
        dmin = cKDTree(np.array(ne)).query(zn)[0]
        ok7a = bool(np.median(dmin) < 3.0)
        print(f"  CA2 zinc site {s['site_name']}: Zn->His N min dist per Zn copy {np.round(dmin, 2).tolist()}")
    check("CA2 Zn site on His94/96/119 with transferred Zn at coordination distance", ok7a)
    g = read_gt_json(AF / P.loc[P.acc == "P03372", "gt_path"].iloc[0])
    e_sites = [s for s in g["sites"] if {353, 394, 524} <= set(s["residues"])]
    ok7b = bool(e_sites) and any(l["rmsd"] <= 2 for l in e_sites[0]["ligands"])
    if e_sites:
        print(f"  ESR1 ligand pocket {e_sites[0]['site_name']}: {len(e_sites[0]['residues'])} residues, ligands "
              f"{[(l['resname'], l['rmsd']) for l in e_sites[0]['ligands']]}")
    check("ESR1 estradiol pocket (E353/R394/H524) present with accepted ligand", ok7b)

    # ------------------------------------------------------------------ 8. harness + controls
    print("== 8. evaluation-harness compatibility and controls")
    gts = load_af_dataset(site_set="benchmark")
    n_sites = sum(len(x.ligands) for x in gts)
    check("loader: benchmark proteins/sites", n_sites == len(b_inc), f"{len(gts)} proteins, {n_sites} sites")

    def preds_from(fn):
        out = {}
        for x in gts:
            c = fn(x)
            out[x.structure_id] = Predictions(x.structure_id, c, -np.arange(len(c), dtype=float), np.arange(1, len(c) + 1),
                                              [frozenset() for _ in range(len(c))])
        return out

    oracle = preds_from(lambda x: np.array([l.center for l in x.ligands]))
    shifted = preds_from(lambda x: np.array([l.center for l in x.ligands]) + np.array([30.0, 0, 0]))
    far = preds_from(lambda x: np.array([l.center for l in x.ligands]) + np.array([1000.0, 0, 0]))
    centroid_cache = {}

    def centroid(x):
        if x.structure_id not in centroid_cache:
            centroid_cache[x.structure_id] = model_arrays(Path(x.path))["ca"].mean(0)[None, :]
        return centroid_cache[x.structure_id]

    cen = preds_from(centroid)
    res = {}
    for nm, pr in (("oracle", oracle), ("oracle_shift30", shifted), ("far_away", far), ("protein_centroid", cen)):
        e = evaluate(gts, pr, criteria=["LDCC12", "DCA4"], n_boot=0)
        sm = e.summary.set_index(["metric", "average"])["value"]
        res[nm] = dict(ldcc12_topn=float(sm[("success_LDCC12_topn", "ligand")]),
                       ldcc12_topn2=float(sm[("success_LDCC12_topn2", "ligand")]),
                       dca4_topn2=float(sm[("success_DCA4_topn2", "ligand")]))
        print(f"  {nm}: {res[nm]}")
    REPORT["controls"] = res
    check("oracle centres: LDCC12 top-n recall == 1", abs(res["oracle"]["ldcc12_topn"] - 1) < 1e-12)
    check("far-away predictions: LDCC12 and DCA4 recall == 0", res["far_away"]["ldcc12_topn2"] == 0 and res["far_away"]["dca4_topn2"] == 0)
    # a centre shifted by 30 A can land on another site of the same protein -> low, not zero
    check("oracle shifted 30 A: LDCC12 top-n+2 recall < 0.1", res["oracle_shift30"]["ldcc12_topn2"] < 0.1)
    check("protein centroid: LDCC12 top-n+2 recall < 0.5", res["protein_centroid"]["ldcc12_topn2"] < 0.5)
    g_ni = load_af_dataset(site_set="benchmark", ions="exclude")
    REPORT["benchmark_NI_sites"] = sum(len(x.ligands) for x in g_ni)
    g_lig = load_af_dataset(site_set="all", center="ligand")
    REPORT["all_sites_with_ligand_centre"] = sum(len(x.ligands) for x in g_lig)
    print(f"  benchmark no-ion sites: {REPORT['benchmark_NI_sites']}; all sites with ligand centre: {REPORT['all_sites_with_ligand_centre']}")

    REPORT["n_fail"] = len(FAILS)
    (AF / "verify_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    print(f"\n{len(FAILS)} failed checks: {FAILS}" if FAILS else "\nALL CHECKS PASSED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
