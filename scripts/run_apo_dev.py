#!/usr/bin/env python
"""CryptoBench train-fold apo-dev diagnostic (DEV ONLY): BURROW (small/all) vs P2Rank 2.5.1 vs fpocket(4.2.3)+PRANK on apo structures,
BURROW-small and P2Rank on the holo counterparts. Writes baselines/outputs/apo_dev/** and results/dev_apo/APO_DEV.md.

  .venv/bin/python scripts/run_apo_dev.py inputs|predict|eval [--workers 24]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baselines"))
GT = ROOT / "data/processed/gt/cryptobench_trainfolds"
OUT = ROOT / "baselines/outputs/apo_dev"
PY = str(ROOT / ".venv/bin/python")
csv.field_size_limit(1 << 30)


def rows(name):
    return list(csv.DictReader(open(GT / name), delimiter="\t"))


def make_inputs():
    import gemmi
    for tab, d in (("sites.tsv", "apo"), ("sites_holo.tsv", "holo")):
        (OUT / "inputs" / d).mkdir(parents=True, exist_ok=True)
        seen = set()
        for r in rows(tab):
            sid = r["structure_id"]
            if sid in seen:
                continue
            seen.add(sid)
            st = gemmi.read_structure(r["structure_path"])
            keep = set(r["input_chains"].split("-"))
            m = st[0]
            for i in range(len(m) - 1, -1, -1):
                if m[i].name not in keep:
                    del m[i]
            if all(len(c.name) == 1 for c in m):
                st.write_pdb(str(OUT / "inputs" / d / f"{sid}.pdb"))
            else:
                st.make_mmcif_document().write_file(str(OUT / "inputs" / d / f"{sid}.cif"))
        print(d, len(seen))


def sh(cmd, log):
    with open(log, "a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)


def predict(workers):
    W = ROOT / "baselines"
    env = dict(os.environ, NUMBA_NUM_THREADS="1")
    jobs = []
    # BURROW: shard inputs over processes (CLI is sequential)
    for var, ranker, sets in (("burrow_small", "small", ("apo", "holo")), ("burrow_all", "all", ("apo",))):
        for s in sets:
            files = sorted((OUT / "inputs" / s).iterdir())
            n = max(1, workers // 4)
            for k in range(n):
                shard = files[k::n]
                lst = OUT / "preds" / f"{var}_{s}_shard{k}.list"
                lst.parent.mkdir(parents=True, exist_ok=True)
                lst.write_text("\n".join(map(str, shard)) + "\n")
                cmd = [PY, "-m", "pocketdetect.burrow", "predict", f"@{lst}", "--out", str(OUT / "preds" / f"{var}_{s}"), "--ranker", ranker, "--threads", "1"]
                jobs.append(cmd)
    procs = []
    for cmd in jobs:
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(OUT / "burrow.log", "a"), stderr=subprocess.STDOUT))
    for s in ("apo", "holo"):
        subprocess.run([PY, str(W / "run_p2rank.py"), str(OUT / "inputs" / s), "-o", str(OUT / "raw" / f"p2rank_2.5.1_{s}"), "-j", str(max(4, workers // 2)),
                        "--version", "2.5.1"], stdout=open(OUT / "p2rank.log", "a"), stderr=subprocess.STDOUT)
    subprocess.run([PY, str(W / "run_fpocket_prank.py"), str(OUT / "inputs" / "apo"), "-o", str(OUT / "raw" / "fpocket_4.2.3_prank_apo"), "-j", str(max(4, workers // 2)),
                    "--fpocket-version", "4.2.3", "--p2rank-version", "2.5.1", "--fpocket-out", str(OUT / "raw" / "fpocket_4.2.3_apo")],
                   stdout=open(OUT / "fpocket.log", "a"), stderr=subprocess.STDOUT)
    for p in procs:
        p.wait()


def eval_all():
    import numpy as np
    import pandas as pd
    from pocketdetect.site_eval import SiteProtocol, compare_site_methods, evaluate_sites, summarize_sites
    from pocketdetect.sites import read_sites, sites_by_structure
    import shutil

    def preds_dir(name):
        """normalised <structure_id>.csv dir"""
        d = OUT / "pred_norm" / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        if name.startswith("burrow"):
            for f in (OUT / "preds" / name).glob("*.csv"):
                if not f.name.endswith((".points.csv", "timing.csv")):
                    shutil.copy(f, d / f.name)
        else:
            raw = OUT / "raw" / name
            for f in raw.glob("*.pockets.csv"):
                shutil.copy(f, d / (f.name[: -len(".pockets.csv")] + ".csv"))
        return d

    methods_apo = {"BURROW-small": "burrow_small_apo", "BURROW-all": "burrow_all_apo", "P2Rank 2.5.1": "p2rank_2.5.1_apo", "fpocket+PRANK": "fpocket_4.2.3_prank_apo", "fpocket": "fpocket_4.2.3_apo"}
    methods_holo = {"BURROW-small": "burrow_small_holo", "P2Rank 2.5.1": "p2rank_2.5.1_holo"}
    tabs = {"apo_cryptic": ("sites.tsv", methods_apo), "apo_noncryptic": ("sites_noncryptic.tsv", methods_apo), "holo": ("sites_holo.tsv", methods_holo)}
    results, summ = {}, []
    for tname, (tf, meths) in tabs.items():
        st = read_sites(GT / tf)
        by = sites_by_structure(st)
        uni = {r["structure_id"]: r["uniprot"].split()[0] for r in rows(tf)}
        for mname, d in meths.items():
            if not (OUT / "preds" / d).exists() and not (OUT / "raw" / d).exists():
                continue
            res = evaluate_sites(by, str(preds_dir(d)), SiteProtocol(), groups=uni)
            results[(tname, mname)] = res
            s = summarize_sites(res, n_boot=1000).set_index("metric")
            for m in ("recall_topn", "recall_topn2", "recall_ldca_topn2", "recall_max", "P1K", "n_scored_sites", "n_structures"):
                if m in s.index:
                    summ.append(dict(table=tname, method=mname, metric=m, value=s.loc[m, "value"], lo=s.loc[m, "ci_low"], hi=s.loc[m, "ci_high"]))
    S = pd.DataFrame(summ)
    S.to_csv(ROOT / "results/dev_apo/summary_long.csv", index=False)
    comp = []
    for tname in tabs:
        ms = [m for (t, m) in results if t == tname]
        for a in ms:
            for b in ms:
                if a < b and ("BURROW-small" in (a, b) or "BURROW-all" in (a, b)) and not (a.startswith("BURROW") and b.startswith("BURROW") and False):
                    c = compare_site_methods(results[(tname, a)], results[(tname, b)], metrics=("recall_topn2",) if tname == "apo_noncryptic" else ("recall_topn2", "recall_ldca_topn2"), n_boot=1000)
                    c.insert(0, "table", tname); c.insert(1, "A", a); c.insert(2, "B", b)
                    comp.append(c)
    # apo -> holo drop per method (unpaired: different structures sets; same apo-holo pairs)
    pd.concat(comp).to_csv(ROOT / "results/dev_apo/paired_comparisons.csv", index=False)
    write_md(S, pd.concat(comp))


def write_md(S, C):
    L = ["# CryptoBench train-fold apo-dev diagnostic (DEV ONLY)\n",
         "Generated by `scripts/run_apo_dev.py`. GT: `data/processed/gt/cryptobench_trainfolds/` (README there). Folds 0-3 of the CryptoBench train split (885 apo structures;",
         "never the test split). Recall = site hit by a pocket of rank <= N+2 (N = #sites of the structure), closest-pocket protocol (site_eval); DCC12 = centre within 12 A of the",
         "site-residue CA centroid; DCA4 = pocket with minimum distance-to-ligand-atom <= 4 A, ligand = holo ligand transferred into the apo frame (CA Kabsch on pocket residues).",
         "CIs: bootstrap over UniProt units (no sequence-cluster map exists for train folds); paired differences use the same resamples. Models: BURROW default presets, seed-0 models",
         "from results/dev/biolip/models (trained on BioLiP2 with the CryptoBench-excluded train set), P2Rank 2.5.1 default, fpocket 4.2.3 (+PRANK 2.5.1).\n"]
    for t, title in (("apo_cryptic", "Apo, cryptic sites"), ("apo_noncryptic", "Apo, non-cryptic (auxiliary) pockets"), ("holo", "Holo counterparts")):
        sub = S[S.table == t]
        if sub.empty:
            continue
        L.append(f"\n## {title}\n")
        ns = sub[sub.metric == "n_scored_sites"].value.max()
        L.append(f"{int(ns)} scored sites.\n")
        L.append("| method | DCC12 top-N+2 | DCC12 top-N | DCA4 top-N+2 | max recall | P1K |\n|---|---|---|---|---|---|")
        for m in sub.method.unique():
            def g(metric):
                r = sub[(sub.method == m) & (sub.metric == metric)]
                if r.empty:
                    return "-"
                r = r.iloc[0]
                return f"{100*r.value:.1f} [{100*r.lo:.1f}, {100*r.hi:.1f}]"
            L.append(f"| {m} | {g('recall_topn2')} | {g('recall_topn')} | {g('recall_ldca_topn2')} | {g('recall_max')} | {g('P1K')} |")
    L.append("\n## Apo -> holo drop (DCC12 top-N+2, same apo-holo pairs; difference of the two tables)\n")
    L.append("| method | holo | apo | drop (holo - apo) |\n|---|---|---|---|")
    for m in ("BURROW-small", "P2Rank 2.5.1"):
        h = S[(S.table == "holo") & (S.method == m) & (S.metric == "recall_topn2")]
        a = S[(S.table == "apo_cryptic") & (S.method == m) & (S.metric == "recall_topn2")]
        if len(h) and len(a):
            L.append(f"| {m} | {100*h.value.iloc[0]:.1f} | {100*a.value.iloc[0]:.1f} | {100*(h.value.iloc[0]-a.value.iloc[0]):+.1f} |")
    L.append("\n## Paired comparisons (A - B, percentage points, paired UniProt-unit bootstrap)\n")
    L.append("| table | A | B | metric | diff | 95% CI |\n|---|---|---|---|---|---|")
    for r in C.itertuples():
        if r.metric in ("recall_topn2", "recall_ldca_topn2"):
            L.append(f"| {r.table} | {r.A} | {r.B} | {r.metric} | {100*r.diff:+.1f} | [{100*r.ci_low:+.1f}, {100*r.ci_high:+.1f}] |")
    (ROOT / "results/dev_apo/APO_DEV.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inputs", "predict", "eval"])
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"inputs": make_inputs, "predict": lambda: predict(a.workers), "eval": eval_all}[a.stage]()
