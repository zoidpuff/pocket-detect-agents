#!/usr/bin/env python
"""Fair-data control: stock P2Rank 2.5.1 vs P2Rank retrained on OUR train set (P2Rank-ours) vs BURROW v1 (frozen default s0)
on BioLiP dev (small/all site tables) and CryptoBench train-fold apo-dev. DEV ONLY. Writes results/baselines/P2RANK_OURS.md (+ csv).
    .venv/bin/python scripts/eval_p2rank_ours.py
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.site_eval import SiteProtocol, compare_site_methods, evaluate_sites, summarize_sites  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402

PO = ROOT / "baselines/p2rank_ours"
APO = ROOT / "baselines/outputs/apo_dev"
csv.field_size_limit(1 << 30)


def norm(src, dst, raw=True):
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in Path(src).glob("*.pockets.csv" if raw else "*.csv"):
        if f.name.endswith((".points.csv", "timing.csv")):
            continue
        shutil.copy(f, dst / (f.name[: -len(".pockets.csv")] + ".csv" if raw else f.name))
    return dst


def run_table(name, tab, groups, methods, ref="P2Rank stock"):
    by = sites_by_structure(read_sites(tab))
    res = {m: evaluate_sites(by, str(d), SiteProtocol(), groups=groups(by)) for m, d in methods.items()}
    rows = []
    for m, r in res.items():
        s = summarize_sites(r, n_boot=1000).set_index("metric")
        row = dict(table=name, method=m, n_sites=int(s.loc["n_scored_sites", "value"]),
                   pockets_mean=round(r.pockets.groupby("structure_id").size().reindex(list(by), fill_value=0).mean(), 1))
        for k in ("recall_topn", "recall_topn2", "recall_ldca_topn2", "recall_max", "P1K"):
            if k in s.index:
                row[k] = f"{100*s.loc[k,'value']:.1f} [{100*s.loc[k,'ci_low']:.1f}, {100*s.loc[k,'ci_high']:.1f}]"
        rows.append(row)
    comps = []
    for a, b in (("P2Rank-ours", "P2Rank stock"), ("BURROW-v1", "P2Rank-ours"), ("BURROW-v1", "P2Rank stock")):
        a_ = [m for m in res if m.startswith(a)]
        b_ = [m for m in res if m == b]
        for x in a_:
            for y in b_:
                c = compare_site_methods(res[x], res[y], metrics=("recall_topn2", "recall_ldca_topn2"), n_boot=1000)
                for r in c.itertuples():
                    if r.metric in ("recall_topn2", "recall_ldca_topn2", "P1K"):
                        comps.append(dict(table=name, A=x, B=y, metric=r.metric, diff=f"{100*r.diff:+.1f} [{100*r.ci_low:+.1f}, {100*r.ci_high:+.1f}]"))
    return rows, comps


def main():
    out_rows, out_comps = [], []
    # BioLiP dev: groups = cluster30 stored in the site table
    for v in ("small", "all"):
        meth = {"P2Rank stock": ROOT / "results/dev/biolip/preds/p2rank",
                "P2Rank-ours": norm(PO / "dev_raw", PO / "dev_norm", raw=True) if (PO / "dev_raw").exists() else None,
                f"BURROW-v1 {v} (default s0)": PO / f"preds/burrow_v1_{v}_dev"}
        meth = {k: d for k, d in meth.items() if d is not None}
        r, c = run_table(f"dev_{v}", ROOT / f"results/dev/biolip/dev_sites_{v}.tsv", lambda by: {s: ss[0].meta.get("group") for s, ss in by.items()}, meth)
        out_rows += r; out_comps += c
    # CryptoBench train-fold apo-dev (cryptic sites; UniProt units)
    rows = list(csv.DictReader(open(ROOT / "data/processed/gt/cryptobench_trainfolds/sites.tsv"), delimiter="\t"))
    uni = {r["structure_id"]: r["uniprot"].split()[0] for r in rows}
    for v in ("small", "all"):
        meth = {"P2Rank stock": norm(APO / "raw/p2rank_2.5.1_apo", PO / "apo_stock_norm"),
                "P2Rank-ours": norm(PO / "apo_raw", PO / "apo_ours_norm") if (PO / "apo_raw").exists() else None,
                f"BURROW-v1 {v} (default s0)": PO / f"preds/burrow_v1_{v}_apo"}
        meth = {k: d for k, d in meth.items() if d is not None}
        r, c = run_table(f"apo_cryptic_{v}", ROOT / "data/processed/gt/cryptobench_trainfolds/sites.tsv", lambda by: uni, meth)
        out_rows += r; out_comps += c
    R, C = pd.DataFrame(out_rows), pd.DataFrame(out_comps)
    R.to_csv(ROOT / "results/baselines/p2rank_ours_summary.csv", index=False)
    C.to_csv(ROOT / "results/baselines/p2rank_ours_paired.csv", index=False)
    def md(df):
        return "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False))
    text = (ROOT / "results/baselines/P2RANK_OURS.head.md").read_text() if (ROOT / "results/baselines/P2RANK_OURS.head.md").exists() else "# P2Rank retrained on our training set (fair-data control)\n"
    text += "\n## Results (DEV ONLY; recall % [95% CI], cluster bootstrap on dev (30% clusters) / UniProt units on apo-dev)\n\n" + md(R) + "\n\n## Paired differences (A - B, percentage points)\n\n" + md(C) + "\n"
    (ROOT / "results/baselines/P2RANK_OURS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
