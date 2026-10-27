#!/usr/bin/env python
"""POST-HOC, DESCRIPTIVE analyses requested by the external report review (not pre-registered;
no claims). Uses the evaluation tables of results/test/{v1,v2}/ and the project bootstrap
(pocketdetect.prereg.paired: paired cluster bootstrap, 30 % sequence clusters, 1,000 resamples,
BOOTSTRAP_SEED).

  .venv/bin/python scripts/posthoc_v2.py   ->  results/test/v2/POSTHOC.md + posthoc.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pocketdetect.prereg import INTERFACE, Rows, load_rows, paired  # noqa: E402
from pocketdetect.seeds import BOOTSTRAP_SEED  # noqa: E402
from run_prereg_analysis import NUC_COF, fold_strata  # noqa: E402

T = ROOT / "results/test"
NB = 1000


def sub(R: Rows, mask) -> Rows:
    return Rows(R.kind, R.rows[mask.to_numpy() if hasattr(mask, "to_numpy") else mask], R.keys, R.pockets, R.eval_dir)


def pair_sub(A, B, metric, labels_a: pd.Series, lab, stratum=None):
    ka = sub(A, labels_a.eq(lab))
    keys = set(map(tuple, ka.rows[A.keys].to_numpy()))
    kb = sub(B, pd.Series([tuple(x) in keys for x in B.rows[B.keys].to_numpy()], index=B.rows.index))
    c = paired(ka, kb, metric, stratum=stratum, n_boot=NB)
    return {k: v for k, v in c.items() if not k.startswith("_")}


def af_ligand_classes(sites: pd.DataFrame) -> pd.Series:
    from pocketdetect.ligysis import LIGYSIS_RESULTS, safe_unpickle

    cpl = safe_unpickle(LIGYSIS_RESULTS / "LIGYSIS_human_chains_per_lig_MAY_2024.pkl")
    st = pd.read_csv(ROOT / "data/processed/gt/af_set/sites.tsv", sep="\t", usecols=["structure_id", "site_id", "ion_prop"], dtype={"site_id": str})
    ion = dict(zip(zip(st.structure_id, st.site_id), st.ion_prop))
    out = []
    for sid, site in zip(sites.structure_id, sites.site_id.astype(str)):
        acc, seg, i = site.rsplit("_", 2)
        names = {k.split("_")[1] for k in cpl.get(f"{acc}_{seg}", {}).get(int(i), {})}
        ip = ion.get((sid, site))
        out.append("ion" if ip == 1 else ("nucleotide-cofactor" if names & NUC_COF else "other"))
    return pd.Series(out, index=sites.index)


def headline(js: dict) -> dict:
    f1 = {f"{r['endpoint']}|{r['seed']}|{r['comparator']}": (round(r["a"], 4), round(r["diff"], 4), round(r["ci_low"], 4), round(r["ci_high"], 4), r["pass"])
          for r in js["family1"]}
    f2 = {f"{r['endpoint']}|{r['seed']}": (round(r["a"], 4), round(r["diff"], 4), round(r["ci_low"], 4), round(r["ci_high"], 4), r["decision"]) for r in js["family2"]}
    return {"family1": f1, "family2": f2, "sequence": js["family1_sequence"], "verdict": js["family2_verdict"]}


def git_json(commit: str, path: str):
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def main():
    res = {"label": "POST-HOC, DESCRIPTIVE (not pre-registered)", "bootstrap": {"n_boot": NB, "seed": BOOTSTRAP_SEED, "units": "30% sequence clusters"}}
    # (1) O5 by ligand class
    A = load_rows("af_set", T / "v2/af_set/burrow_all_default_s0_af")
    B = load_rows("af_set", T / "v2/comparators/af_set/p2rank_2.5.1_af")
    cls3 = af_ligand_classes(A.rows)
    cls2 = cls3.map(lambda c: "ion" if c == "ion" else "non-ion")
    o5 = []
    for name, lab in (("2-class", cls2), ("3-class", cls3)):
        for c in sorted(lab.unique()):
            for m in ("recall_topn2", "recall_ldca_topn2"):
                o5.append({"split": name, "class": c, "metric": m, **pair_sub(A, B, m, lab, c)})
    res["o5_ligand_class"] = o5
    # (2) LIGYSIS_NI
    ni = []
    comps = {"fpocket_4.2.3+PRANK_2.4.2": T / "v2/comparators/ligysis_ni/fpocket_4.2.3+PRANK_2.4.2", "p2rank_2.5.1": T / "v2/comparators/ligysis_ni/p2rank_2.5.1"}
    for ver, var in (("v1", "burrow_small_default_s0"), ("v2", "burrow_small_default_s0"), ("v2", "burrow_small_default_s1"), ("v2", "burrow_small_default_s2")):
        Am = load_rows("ligysis_ni", T / ver / "ligysis_ni" / var)
        for cn, cd in comps.items():
            Bm = load_rows("ligysis_ni", cd)
            for m in ("recall_topn2", "recall_ldca_topn2"):
                c = paired(Am, Bm, m, n_boot=NB)
                ni.append({"method": f"{ver}:{var}", "comparator": cn, "metric": m, **{k: v for k, v in c.items() if not k.startswith("_")}})
    res["ligysis_ni"] = ni
    # (3) O6 fold novelty on interface sites only
    Ad = load_rows("dimer", T / "v2/dimer/burrow_all_default_s0")
    Bd = load_rows("dimer", T / "v2/comparators/dimer/p2rank_2.5.1")
    fs = fold_strata("dimer", Ad.rows.complex_id.unique())
    lab = Ad.rows.complex_id.map(fs).where(Ad.rows.site_class.isin(INTERFACE), "not_interface")
    o6 = []
    for c in sorted(set(lab) - {"not_interface"}):
        o6.append({**pair_sub(Ad, Bd, "success_DCC12_topn2", lab, c), "stratum": c})
    res["o6_interface_fold_novelty"] = o6
    res["o6_note"] = "interface = homo_interface + hetero_interface sites (243 of 807); fold stratum = max Foldseek TM band over the complex's chains"
    # (4) provenance of headline numbers across commits
    prov = {}
    for ver, commits in (("v1", ["8b59de7", "20421b6", "2b367a9", "9521756", "HEAD"]), ("v2", ["2b367a9", "9521756", "HEAD"])):
        hs = {c: git_json(c, f"results/test/{ver}/summary.json") for c in commits}
        ref = headline(hs[commits[0]])
        prov[ver] = {c: ("identical" if headline(js) == ref else "DIFFERENT") if js else "absent" for c, js in hs.items()}
        prov[ver]["reference_commit"] = commits[0]
    res["provenance"] = prov
    # (5) dimer counts
    u = pd.read_csv(T / "v2/dimer/burrow_all_default_s0/units.csv")
    res["dimer_counts"] = {"assemblies_scored": int(len(u)), "sites": int(len(Ad.rows)), "interface_sites": int(Ad.rows.site_class.isin(INTERFACE).sum()),
                           "assemblies_with_interface_site": int(Ad.rows[Ad.rows.site_class.isin(INTERFACE)].complex_id.nunique()),
                           "cluster30_units": int(Ad.rows.group.nunique())}
    (T / "v2/posthoc.json").write_text(json.dumps(res, indent=2, default=str))
    f = lambda x: f"{x:.2f}"  # noqa: E731
    L = ["# POST-HOC, DESCRIPTIVE analyses — v2 (NOT pre-registered; no claims)", "",
         "Requested by the external report review (reviews/report/REPORT_REVIEW.md). Generated by `scripts/posthoc_v2.py` from the scored evaluation tables",
         f"of results/test/{{v1,v2}}; paired cluster bootstrap (30 % sequence clusters, {NB} resamples, seed {BOOTSTRAP_SEED}). Δ in points.", "",
         "## 1. O5 (AF set) by ligand class — v2 seed 0 vs P2Rank 2.5.1 -c alphafold", "",
         "| split | class | metric | n sites | v2 | P2Rank-AF | Δ [95% CI] |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['split']} | {r['class']} | {r['metric']} | {r['n_rows']} | {f(r['a'])} | {f(r['b'])} | {f(r['diff'])} [{f(r['ci_low'])}, {f(r['ci_high'])}] |" for r in o5]
    L += ["", "P1K per ligand class is not reported: P1K pools all predicted pockets, and false-positive pockets have no ligand class, so a per-class "
          "precision is not defined without an arbitrary assignment.", "",
          "## 2. LIGYSIS_NI (4,141 scored non-ion sites) — paired Δ", "", "| method | comparator | metric | method | comparator | Δ [95% CI] |", "|---|---|---|---|---|---|"]
    L += [f"| {r['method']} | {r['comparator']} | {r['metric']} | {f(r['a'])} | {f(r['b'])} | {f(r['diff'])} [{f(r['ci_low'])}, {f(r['ci_high'])}] |" for r in ni]
    L += ["", "## 3. O6 — fold novelty on the 243 interface sites (v2 seed 0 vs P2Rank 2.5.1)", "",
          "| fold stratum (max TM) | n interface sites | v2 | P2Rank | Δ [95% CI] |", "|---|---|---|---|---|"]
    L += [f"| {r['stratum']}{' *' if r['n_rows'] < 100 else ''} | {r['n_rows']} | {f(r['a'])} | {f(r['b'])} | {f(r['diff'])} [{f(r['ci_low'])}, {f(r['ci_high'])}] |" for r in o6]
    L += ["", "(* < 100 sites.) The O6 fold strata in SUMMARY.md were computed over all 807 dimer sites; this table restricts them to the endpoint's interface sites.", "",
          "## 4. Provenance of headline numbers across commits", "",
          "Family-1 (a, Δ, CI, pass for every endpoint × seed × comparator) and family-2 (a, Δ, CI, decision) values compared with the first scoring commit:", ""]
    for ver, d in prov.items():
        L.append(f"- {ver} (reference {d['reference_commit']}): " + ", ".join(f"{c}: {v}" for c, v in d.items() if c != "reference_commit"))
    dc = res["dimer_counts"]
    L += ["", "## 5. Dimer counts", "",
          f"All {dc['assemblies_scored']} assemblies are scored ({dc['sites']} unique sites; {dc['interface_sites']} interface sites on "
          f"{dc['assemblies_with_interface_site']} assemblies). The 283 in SUMMARY's denominator table is the number of **30 % sequence-cluster resampling units** "
          f"({dc['cluster30_units']}) for the dimer benchmark (the 'n structures/units' column reports units for the dimer evaluator), not a count of scored assemblies.", ""]
    (T / "v2/POSTHOC.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
