#!/usr/bin/env python
"""DESCRIPTIVE analyses (test data; no tuning): (1) PocketMiner-test residue-level ROC-AUC/AUPRC; (2) CryptoBench TEST holo counterparts + apo->holo drop.
Writes results/descriptive/POCKETMINER_CRYPTOBENCH_HOLO.md.   .venv/bin/python scripts/descr_pocketminer_cryptobench_holo.py
Inputs: baselines/outputs/descr/{pm,holo} (see the script header of the runs in the md)."""
from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import gemmi
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.clusters import groups_for_structures  # noqa: E402
from pocketdetect.seeds import BOOTSTRAP_SEED  # noqa: E402
from pocketdetect.site_eval import SiteProtocol, compare_site_methods, evaluate_sites, summarize_sites  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402

PM = ROOT / "baselines/outputs/descr/pm"
HO = ROOT / "baselines/outputs/descr/holo"
D = ROOT / "data/raw/pocketminer/data/pm-dataset"
GT = ROOT / "data/processed/gt/cryptobench_test_holo"
csv.field_size_limit(1 << 30)


# ------------------------------------------------------------------ (1) PocketMiner test
def residue_table():
    ids = np.load(D / "test_apo_ids_with_chainids.npy", allow_pickle=True)
    lab = np.load(D / "test_label_dictionary.npy", allow_pickle=True).item()
    rows = []
    for i in ids:
        i = str(i)
        key = next(k for k in lab if str(k).lower() in (i.lower(), i[:4].lower()))
        st = gemmi.read_structure(str(PM / "inputs" / f"{i}.pdb"))
        res = [(c.name, r.seqid.num, r.seqid.icode.strip()) for c in st[0] for r in c
               if (gemmi.find_tabulated_residue(r.name) is not None and gemmi.find_tabulated_residue(r.name).is_amino_acid())]
        L = np.asarray(lab[key])
        assert len(L) == len(res), (i, len(L), len(res))
        for (ch, n, ic), y in zip(res, L):
            rows.append(dict(protein=i, rid=f"{ch}:{n}{ic}", y=int(y)))
    return pd.DataFrame(rows)


def per_residue_scores(T):
    out = {}
    # PocketMiner raw per-residue probability
    pm = {}
    for i in T.protein.unique():
        for r in csv.DictReader(open(PM / "pocketminer" / f"{i}.residues.csv")):
            pm[(i, f"{r['chain']}:{r['resnum']}")] = float(r["prob"])
    out["PocketMiner"] = np.array([pm.get((p, r), np.nan) for p, r in zip(T.protein, T.rid)])
    # P2Rank 2.5.1 residue score / calibrated probability
    p2s, p2p = {}, {}
    for i in T.protein.unique():
        for r in csv.DictReader(open(PM / "p2rank/raw" / f"{i}.pdb_residues.csv"), skipinitialspace=True):
            k = (i, f"{r['chain']}:{r['residue_label']}")
            p2s[k], p2p[k] = float(r["score"]), float(r["probability"])
    out["P2Rank 2.5.1 (residue score)"] = np.array([p2s.get((p, r), np.nan) for p, r in zip(T.protein, T.rid)])
    out["P2Rank 2.5.1 (residue probability)"] = np.array([p2p.get((p, r), np.nan) for p, r in zip(T.protein, T.rid)])
    # BURROW v2: per-residue = max calibrated pocket score over pockets containing the residue (0 if none; PREREG residue rule)
    for rk in ("small", "all"):
        for fl, tag in (("default", "v2 floor 0.03"), ("nofloor", "--min-score none")):
            sc = {}
            for i in T.protein.unique():
                for r in csv.DictReader(open(PM / f"burrow_{rk}_{fl}" / f"{i}.csv")):
                    for t in r["residues"].split():
                        sc[(i, t)] = max(sc.get((i, t), 0.0), float(r["score"]))
            out[f"BURROW v2 {rk} ({tag})"] = np.array([sc.get((p, r), 0.0) for p, r in zip(T.protein, T.rid)])
    return out


def metrics(T, s, n_boot=1000):
    m = (T.y.values != 2) & ~np.isnan(s)
    y, x, g = T.y.values[m], s[m], T.protein.values[m]
    auc, ap = roc_auc_score(y, x), average_precision_score(y, x)
    prots = np.unique(g)
    idx = {p: np.flatnonzero(g == p) for p in prots}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    A, P = [], []
    for _ in range(n_boot):
        pick = rng.choice(prots, len(prots))
        ii = np.concatenate([idx[p] for p in pick])
        if len(set(y[ii])) < 2:
            continue
        A.append(roc_auc_score(y[ii], x[ii]))
        P.append(average_precision_score(y[ii], x[ii]))
    mac = [roc_auc_score(y[idx[p]], x[idx[p]]) for p in prots if len(set(y[idx[p]])) == 2]
    return dict(n_res=int(m.sum()), n_pos=int(y.sum()), auc=auc, auc_ci=(np.percentile(A, 2.5), np.percentile(A, 97.5)),
                auprc=ap, ap_ci=(np.percentile(P, 2.5), np.percentile(P, 97.5)), macro_auc=float(np.mean(mac)), n_prot_macro=len(mac), cov=float((x != 0).mean()))


def part1():
    T = residue_table()
    S = per_residue_scores(T)
    rows = []
    for name, s in S.items():
        r = metrics(T, s)
        rows.append(f"| {name} | {r['n_res']} ({r['n_pos']} pos) | {r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}] | {r['auprc']:.3f} [{r['ap_ci'][0]:.3f}, {r['ap_ci'][1]:.3f}] | {100*r['cov']:.0f}% |")
    base = metrics(T, np.zeros(len(T)) + np.random.default_rng(0).random(len(T)))
    return ("| method | labelled residues | pooled ROC-AUC [95% CI, protein bootstrap] | AUPRC [95% CI] | residues with score > 0 |\n|---|---|---|---|---|\n" + "\n".join(rows)
            + f"\n\nRandom-score reference: AUC {base['auc']:.3f}, AUPRC {base['auprc']:.3f} (= positive rate {base['n_pos']/base['n_res']:.3f}).")


# ------------------------------------------------------------------ (2) holo counterparts
def norm(src, dst, raw):
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in Path(src).glob("*.pockets.csv" if raw else "*.csv"):
        if f.name.endswith((".points.csv", "timing.csv")):
            continue
        shutil.copy(f, dst / (f.name[: -len(".pockets.csv")] + ".csv" if raw else f.name))
    return dst


def part2():
    ha = read_sites(GT / "sites_holo.tsv")
    aa = read_sites(GT / "sites_apo.tsv")
    by_h, by_a = sites_by_structure(ha), sites_by_structure(aa)
    groups = groups_for_structures(list(by_a), ("cryptobench",), "cluster30")
    T = ROOT / "results/test/v2/preds/cryptobench"
    meth = {
        "BURROW v2 small": (ROOT / "results/test/v2/preds/cryptobench/burrow_small_default_s0/preds", norm(HO / "burrow_small", HO / "norm_burrow", raw=False)),
        "P2Rank 2.5.1": (ROOT / "baselines/outputs/cryptobench/p2rank_2.5.1/preds", norm(HO / "p2rank", HO / "norm_p2rank", raw=True)),
    }
    res, rows, comps = {}, [], []
    for m, (apo_d, holo_d) in meth.items():
        res[m] = (evaluate_sites(by_a, str(apo_d), SiteProtocol(), groups=groups), evaluate_sites(by_h, str(holo_d), SiteProtocol(), groups=groups))
        for state, r in zip(("apo", "holo"), res[m]):
            s = summarize_sites(r, n_boot=1000).set_index("metric")
            cell = lambda k: f"{100*s.loc[k,'value']:.1f} [{100*s.loc[k,'ci_low']:.1f}, {100*s.loc[k,'ci_high']:.1f}]"
            rows.append(f"| {m} | {state} | {int(s.loc['n_scored_sites','value'])} | {cell('recall_topn2')} | {cell('recall_topn')} | {cell('recall_ldca_topn2')} | {cell('recall_max')} | {cell('P1K')} |")
        # paired apo -> holo on identical sites (same structure_id/site_id keys)
        c = compare_site_methods(res[m][1], res[m][0], metrics=("recall_topn2", "recall_ldca_topn2"), n_boot=1000)
        for r in c.itertuples():
            if r.metric in ("recall_topn2", "recall_ldca_topn2"):
                comps.append(f"| {m} | {r.metric} | {100*r.b:.1f} | {100*r.a:.1f} | {100*r.diff:+.1f} [{100*r.ci_low:+.1f}, {100*r.ci_high:+.1f}] |")
    # method comparison on holo (paired)
    cm = compare_site_methods(res["BURROW v2 small"][1], res["P2Rank 2.5.1"][1], metrics=("recall_topn2", "recall_ldca_topn2"), n_boot=1000)
    ca = compare_site_methods(res["BURROW v2 small"][0], res["P2Rank 2.5.1"][0], metrics=("recall_topn2", "recall_ldca_topn2"), n_boot=1000)
    mm = [f"| holo | {r.metric} | {100*r.diff:+.1f} [{100*r.ci_low:+.1f}, {100*r.ci_high:+.1f}] |" for r in cm.itertuples() if r.metric != "P1K"]
    mm += [f"| apo (paired subset) | {r.metric} | {100*r.diff:+.1f} [{100*r.ci_low:+.1f}, {100*r.ci_high:+.1f}] |" for r in ca.itertuples() if r.metric != "P1K"]
    t1 = "| method | structure state | sites | DCC12 top-N+2 | DCC12 top-N | DCA4 top-N+2 | max recall | P1K |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(rows)
    t2 = "| method | metric | apo | holo | drop (holo - apo) [paired 95% CI] |\n|---|---|---|---|---|\n" + "\n".join(comps)
    t3 = "| structures | metric | BURROW v2 small - P2Rank [paired 95% CI] |\n|---|---|---|\n" + "\n".join(mm)
    return t1, t2, t3


if __name__ == "__main__":
    p1 = part1()
    t1, t2, t3 = part2()
    off = pd.read_csv(ROOT / "results/test/v2/cryptobench/burrow_small_default_s0/summary.csv").set_index("metric")
    off_s = f"{100*off.loc['recall_topn2','value']:.1f} [{100*off.loc['recall_topn2','ci_low']:.1f}, {100*off.loc['recall_topn2','ci_high']:.1f}]"
    md = f"""# PocketMiner test residues and CryptoBench TEST holo counterparts (DESCRIPTIVE; no tuning; test data)

Generated by `scripts/descr_pocketminer_cryptobench_holo.py` (builder for holo GT: `scripts/build_gt_cryptobench_test_holo.py`). Model: BURROW FREEZE v2 (default s0 point/rank models, `--ranker small|all`, output floor 0.03 unless noted).

## 1. PocketMiner dataset, test split (35 apo chains; residues labelled 0/1; label 2 excluded)

Per-residue score: PocketMiner = raw per-residue probability (our wrapper, TF2.15 port, as validated earlier: pooled AUC 0.79); P2Rank 2.5.1 = its `_residues.csv` score / calibrated probability;
BURROW v2 = max calibrated pocket score over output pockets containing the residue, 0 if in no pocket (PREREG residue rule; "--min-score none" = full NMS'd list, up to 50 pockets, so more residues are covered).
Inputs = the dataset's cleaned apo structures (hydrogens/ions stripped by the wrappers). CIs: bootstrap over the 35 proteins (1,000 resamples). Ties (many zeros for BURROW) are handled by sklearn (average rank).

{p1}

PocketMiner is a *cryptic*-pocket predictor trained on MD simulations; BURROW and P2Rank are crystal-structure ligand-site predictors, so the comparison is descriptive.

## 2. CryptoBench TEST: holo counterparts

Pairing: for each of the 222 test apo structures the main holo entry (chain-restricted, ligands stripped by the wrappers) is predicted with BURROW v2 small and P2Rank 2.5.1; an apo cryptic site is paired when a member entry uses that holo structure
(233 of 270 official sites, all 222 structures). Site centre on holo = CA centroid of the holo pocket residues; N (top-N) = number of paired sites in both states, so apo/holo tables use identical sites. DCA4: holo = real ligand atoms; apo =
holo ligand transferred into the apo frame by CA Kabsch on the pocket residues (RMSD <= 4 A; 212/233 sites have ligand atoms; sites without ligand atoms count as DCA misses in both states). Apo predictions are the official ones used for O4
(`results/test/v2`, BURROW v2 small default s0; P2Rank 2.5.1 from `baselines/outputs/cryptobench`). For reference the official O4 apo number for BURROW v2 small (all 270 sites): DCC12 top-N+2 = {off_s}.
CIs: 30% sequence-cluster bootstrap (as the registry's CryptoBench units), paired where stated.

{t1}

### Apo -> holo drop (same sites; paired bootstrap)

(DCA4 here counts the 21 sites without ligand atoms as misses in both states, hence lower than the DCA4 column above, whose denominator is the sites with ligand atoms.)

{t2}

### BURROW v2 small minus P2Rank, paired, by structure state

{t3}
"""
    out = ROOT / "results/descriptive/POCKETMINER_CRYPTOBENCH_HOLO.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md)
    print(md)
