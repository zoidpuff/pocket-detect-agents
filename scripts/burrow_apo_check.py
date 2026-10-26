#!/usr/bin/env python
"""Secondary (DEV-only) check of BURROW model JSONs on the CryptoBench TRAIN-fold apo set
(results/dev_apo/APO_DEV.md; GT data/processed/gt/cryptobench_trainfolds; never the test split).

Predicts the 885 apo inputs (baselines/outputs/apo_dev/inputs/apo) with each model via the CLI path
(predict_protein) into results/dev/biolip/apo_preds/<name>/ and reports top-N+2 (DCC12, closest
pocket) and DCA4 top-N+2 on cryptic and non-cryptic site tables, paired vs P2Rank 2.5.1
(UniProt-cluster bootstrap, as in scripts/run_apo_dev.py).

  python scripts/burrow_apo_check.py name1=model1.json name2=model2.json [--jobs 8]
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.model import BurrowModel  # noqa: E402
from pocketdetect.burrow.pipeline import predict_protein, write_prediction  # noqa: E402
from pocketdetect.burrow.prep import load_protein  # noqa: E402
from pocketdetect.site_eval import compare_site_methods, evaluate_sites, summarize_sites  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402

GT = ROOT / "data/processed/gt/cryptobench_trainfolds"
INP = ROOT / "baselines/outputs/apo_dev/inputs/apo"
P2R = ROOT / "baselines/outputs/apo_dev/pred_norm/p2rank_2.5.1_apo"
OUT = ROOT / "results/dev/biolip/apo_preds"
csv.field_size_limit(1 << 30)
_G = {}


def _job(f):
    m = _G["model"]
    P = load_protein(f, structure_id=Path(f).stem)
    write_prediction(_G["out"], predict_protein(P, m), points=False)
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    files = sorted(str(p) for p in INP.glob("*.pdb"))
    rows = []
    tabs = {t: sites_by_structure(read_sites(GT / f)) for t, f in (("apo_cryptic", "sites.tsv"),
                                                                   ("apo_noncryptic", "sites_noncryptic.tsv"))}
    uni = {}
    for f in ("sites.tsv", "sites_noncryptic.tsv"):
        for r in csv.DictReader(open(GT / f), delimiter="\t"):
            uni[r["structure_id"]] = r["uniprot"].split()[0]
    ref = {t: evaluate_sites(by, str(P2R), groups=uni) for t, by in tabs.items()}
    for spec in a.models:
        name, path = spec.split("=", 1)
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        _G.update(model=BurrowModel.load(path), out=out)
        todo = [f for f in files if not (out / f"{Path(f).stem}.csv").exists()]
        with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
            list(ex.map(_job, todo, chunksize=4))
        for t, by in tabs.items():
            r = evaluate_sites(by, str(out), groups=uni)
            s = summarize_sites(r, n_boot=0).set_index("metric")["value"]
            has_dca = "recall_ldca_topn2" in s.index
            mets = ("recall_topn2", "recall_ldca_topn2") if has_dca else ("recall_topn2",)
            c = compare_site_methods(r, ref[t], metrics=mets, n_boot=1000).set_index("metric")
            sp = summarize_sites(ref[t], n_boot=0).set_index("metric")["value"]
            rows.append(dict(model=name, table=t, n_sites=int(s["n_scored_sites"]), topn2=100 * s["recall_topn2"],
                             p2rank_topn2=100 * sp["recall_topn2"],
                             d_topn2_vs_p2rank=f"{100 * c.loc['recall_topn2', 'diff']:+.1f} [{100 * c.loc['recall_topn2', 'ci_low']:+.1f},{100 * c.loc['recall_topn2', 'ci_high']:+.1f}]",
                             dca4=100 * s["recall_ldca_topn2"] if has_dca else float("nan"),
                             p2rank_dca4=100 * sp["recall_ldca_topn2"] if has_dca else float("nan"),
                             d_dca4_vs_p2rank=f"{100 * c.loc['recall_ldca_topn2', 'diff']:+.1f}" if has_dca else ""))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows).round(1)
    f = ROOT / "results/dev/biolip/apo_check.csv"
    if f.exists():
        df = pd.concat([pd.read_csv(f), df], ignore_index=True)
    df.to_csv(f, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
