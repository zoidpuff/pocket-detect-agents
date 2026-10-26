#!/usr/bin/env python
"""Markdown tables for results/dev_af/AF_DEV_RESULTS.md from the eval CSVs of one run tag.

  python scripts/af_dev_report.py --tag def0 > /tmp/tables.md      (tag '' = the v1 exploration run)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/dev_af"
COLS = [("method", "config"), ("pockets_mean", "pockets"), ("recall_topn2", "DCC12 top-N+2"),
        ("recall_ldca_topn2", "DCA4 top-N+2"), ("P1K", "P1K"), ("prec_top1", "top-1 prec"),
        ("halluc_rate", "halluc % (all)"), ("halluc_rate_topn2", "halluc % (top-N+2)"),
        ("topn2_disorder_rich", "DCC12, disorder-rich prot."), ("d_recall_topn2", "Δ DCC12"),
        ("d_recall_ldca_topn2", "Δ DCA4"), ("d_P1K", "Δ P1K")]


def table(f, title, ref):
    p = D / f
    if not p.exists():
        return f"_{f} missing_\n"
    d = pd.read_csv(p)
    cols = [(c, h) for c, h in COLS if c in d.columns]
    out = [f"**{title}** (`{f}`; Δ = paired cluster-bootstrap difference vs `{ref}`, 95 % CI)\n",
           "| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c, _ in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    t = ap.parse_args().tag
    s = f"_{t}" if t else ""
    print(table(f"eval_ablation{s}.csv", "AF dev — geometric mechanisms, controls, baselines (standard ranker)", "naive"))
    if not t:
        print(table("eval_vs_plddt70.csv", "AF dev — AF mode vs the pLDDT<70 filter baseline", "naive_plddt70"))
        print(table("eval_ranker_v1.csv", "AF dev — AF-aware rankers (prune+soft geometry)", "prune_soft"))
        print(table("eval_combos_dev.csv", "AF dev — combinations", "naive_plddt70"))
        print(table("eval_pdbdev_noop_v1.csv", "PDB dev (579 chains, uniform pLDDT 95) — AF rankers vs standard (no-op / non-inferiority)", "standard"))
        print(table("eval_trainoof.csv", "AF train, out-of-fold point scores (2,844 models) — geometric mechanisms", "naive"))
        print(table("eval_trainoof_vs_plddt70.csv", "AF train OOF — vs the pLDDT<70 filter", "naive_plddt70"))
        print(table("eval_trainoof_rankers.csv", "AF train OOF — cluster-CV rankers", "cv_pdb_std"))
        print(table("eval_trainoof_lam.csv", "AF train OOF — λ geometry + CV AF ranker", "cv_pdb_std"))
        print(table("eval_lam_dev.csv", "AF dev — λ geometry + AF ranker", "naive_plddt70"))
        print(table("eval_pdbchain_same_entries.csv", "PDB chains of the same 278 dev proteins (context)", "-"))
    else:
        print(table(f"eval_ranker{s}.csv", "AF dev — AF-aware rankers (prune+soft geometry)", "naive_plddt70"))
        print(table(f"eval_pdbdev_noop{s}.csv", "PDB dev (uniform pLDDT 95) — AF rankers vs standard", "standard"))
        print(table(f"eval_trainoof{s}.csv", "AF train OOF — geometric mechanisms", "naive"))
        print(table(f"eval_trainoof_vs_plddt70{s}.csv", "AF train OOF — vs pLDDT<70 filter", "naive_plddt70"))
        print(table(f"eval_trainoof_rankers{s}.csv", "AF train OOF — cluster-CV rankers", "cv_pdb_std"))
        nf = D / "logs" / f"noop{s}.log"
        if nf.exists():
            print("No-op check: " + "; ".join(nf.read_text().split("\n")).strip("; ") + "\n")


if __name__ == "__main__":
    main()
