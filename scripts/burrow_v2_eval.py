#!/usr/bin/env python
"""v2 research cycle: evaluate one BURROW model JSON against frozen v1 on the three dev sets, paired.

* BioLiP2 dev (primary guard): CLI predictions on results/dev/biolip/inputs, site table
  dev_sites_<variant>.tsv; top-N, top-N+2 (DCC12), DCA4 top-N+2, P1K; paired vs v1 (cluster30).
* apo-dev (target; CryptoBench TRAIN folds, never trained on): apo cryptic / non-cryptic tables;
  top-N+2 (+ DCA4); paired vs v1 (UniProt clusters, as scripts/run_apo_dev.py).
* AF dev (secondary guard): AF-dev models, input variant prune_soft (scripts/af_dev_predict.py,
  predictions in results/dev/biolip/af_preds/), results/dev_af/af_dev_sites_all.tsv; paired vs v1.
Every call appends to results/dev/biolip/v2_eval.csv (the selection log counts these rows).

  python scripts/burrow_v2_eval.py --name H2_scd --model M.json --variant small [--v1 FREEZE_MODEL]
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import multiprocessing as mp
import subprocess
import sys
import time
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

B = ROOT / "results/dev/biolip"
GT_APO = ROOT / "data/processed/gt/cryptobench_trainfolds"
APO_IN = ROOT / "baselines/outputs/apo_dev/inputs/apo"
AF_SITES = ROOT / "results/dev_af/af_dev_sites_all.tsv"
csv.field_size_limit(1 << 30)
_G = {}


def _pred_job(f):
    P = load_protein(f, structure_id=Path(f).stem)
    from pocketdetect.burrow.experimental import is_experimental, predict_protein_experimental
    fn = predict_protein_experimental if is_experimental(_G["model"]) else predict_protein
    write_prediction(_G["out"], fn(P, _G["model"]), points=False)
    return f


def predict_dir(model_path, files, out, jobs):
    out.mkdir(parents=True, exist_ok=True)
    _G.update(model=BurrowModel.load(model_path), out=out)
    todo = [f for f in files if not (out / f"{Path(f).stem}.csv").exists()]
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        list(ex.map(_pred_job, todo, chunksize=4))


def compare(by, groups, d_new, d_ref, metrics):
    rn = evaluate_sites(by, str(d_new), groups=groups)
    rr = evaluate_sites(by, str(d_ref), groups=groups)
    sn = summarize_sites(rn, n_boot=0).set_index("metric")["value"]
    sr = summarize_sites(rr, n_boot=0).set_index("metric")["value"]
    mets = [m for m in metrics if m in sn.index]
    c = compare_site_methods(rn, rr, metrics=mets, n_boot=1000).set_index("metric")
    row = {}
    for m in mets + ["P1K"]:
        if m not in sn.index:
            continue
        row[m] = round(100 * sn[m], 1)
        row[f"v1_{m}"] = round(100 * sr[m], 1)
        if m in c.index:
            row[f"d_{m}"] = f"{100 * c.loc[m, 'diff']:+.1f} [{100 * c.loc[m, 'ci_low']:+.1f},{100 * c.loc[m, 'ci_high']:+.1f}]"
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--variant", choices=["all", "small"], required=True)
    ap.add_argument("--v1", type=Path, default=None)
    ap.add_argument("--sets", default="biolip,apo,apotd,af")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    v1 = a.v1 or ROOT / f"results/dev/freeze/models/burrow_{a.variant}_default_s0.json"
    sets = a.sets.split(",")
    rows = []
    t0 = time.time()
    if "biolip" in sets:
        files = (B / "inputs_audited.list").read_text().split()
        for nm, mp_ in ((a.name, a.model), (f"v1_{a.variant}_default_s0", v1)):
            predict_dir(mp_, [str(ROOT / f) for f in files], B / "v2_preds" / nm, a.jobs)
        by = sites_by_structure(read_sites(B / f"dev_sites_{a.variant}.tsv"))
        g = {s: ss[0].meta.get("group") for s, ss in by.items()}
        r = compare(by, g, B / "v2_preds" / a.name, B / "v2_preds" / f"v1_{a.variant}_default_s0",
                    ["recall_topn", "recall_topn2", "recall_ldca_topn2"])
        rows.append(dict(name=a.name, variant=a.variant, set=f"biolip_dev_{a.variant}", **r))
        print(f"[{time.time() - t0:.0f}s]", rows[-1], flush=True)
    if "apo" in sets:
        files = sorted(str(p) for p in APO_IN.glob("*.pdb"))
        for nm, mp_ in ((a.name, a.model), (f"v1_{a.variant}_default_s0", v1)):
            predict_dir(mp_, files, B / "v2_apo_preds" / nm, a.jobs)
        uni = {}
        for f in ("sites.tsv", "sites_noncryptic.tsv"):
            for r in csv.DictReader(open(GT_APO / f), delimiter="\t"):
                uni[r["structure_id"]] = r["uniprot"].split()[0]
        for t, f in (("apo_cryptic", "sites.tsv"), ("apo_noncryptic", "sites_noncryptic.tsv")):
            by = sites_by_structure(read_sites(GT_APO / f))
            r = compare(by, uni, B / "v2_apo_preds" / a.name, B / "v2_apo_preds" / f"v1_{a.variant}_default_s0",
                        ["recall_topn", "recall_topn2", "recall_ldca_topn2"])
            rows.append(dict(name=a.name, variant=a.variant, set=t, **r))
            print(f"[{time.time() - t0:.0f}s]", rows[-1], flush=True)
    if "apotd" in sets:  # Track A2 apo chains of DEV proteins (extra apo guard)
        D = B / "apo_track_dev"
        files = sorted(str(p) for p in (D / "inputs").glob("*.cif"))
        for nm, mp_ in ((a.name, a.model), (f"v1_{a.variant}_default_s0", v1)):
            predict_dir(mp_, files, B / "v2_apotd_preds" / nm, a.jobs)
        by = sites_by_structure(read_sites(D / f"sites_{a.variant}.tsv"))
        g = {s: ss[0].meta.get("group") for s, ss in by.items()}
        r = compare(by, g, B / "v2_apotd_preds" / a.name, B / "v2_apotd_preds" / f"v1_{a.variant}_default_s0",
                    ["recall_topn", "recall_topn2", "recall_ldca_topn2"])
        rows.append(dict(name=a.name, variant=a.variant, set=f"apo_track_dev_{a.variant}", **r))
        print(f"[{time.time() - t0:.0f}s]", rows[-1], flush=True)
    if "af" in sets:
        for nm, mp_ in ((a.name, a.model), (f"v1_{a.variant}_default_s0", v1)):
            if not (B / "af_preds" / nm).exists():
                subprocess.run([sys.executable, str(ROOT / "scripts/af_dev_predict.py"), "--config", "prune_soft",
                                "--model", str(mp_), "--name", nm, "--out-root", str(B / "af_preds"),
                                "--jobs", str(a.jobs)], check=True, capture_output=True)
        st = read_sites(AF_SITES)
        by = sites_by_structure(st)
        g = {s: ss[0].meta.get("group") for s, ss in by.items()}
        r = compare(by, g, B / "af_preds" / a.name, B / "af_preds" / f"v1_{a.variant}_default_s0",
                    ["recall_topn", "recall_topn2", "recall_ldca_topn2"])
        rows.append(dict(name=a.name, variant=a.variant, set="af_dev_all", **r))
        print(f"[{time.time() - t0:.0f}s]", rows[-1], flush=True)
    df = pd.DataFrame(rows)
    f = B / "v2_eval.csv"
    if f.exists():
        df = pd.concat([pd.read_csv(f), df], ignore_index=True)
    df.to_csv(f, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
