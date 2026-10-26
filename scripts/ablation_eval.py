#!/usr/bin/env python
"""Predict + evaluate BURROW ablation models on BioLiP dev and CryptoBench train-fold apo-dev,
paired against the frozen v1 seed-0 models -> results/ablations/ABLATIONS.md (+ CSV).

  python scripts/ablation_eval.py predict --pred-root <worktree with frozen ref + fast backend>
  python scripts/ablation_eval.py eval
  python scripts/ablation_eval.py test-commands      # writes results/ablations/TEST_COMMANDS.md (not run)

Predictions use the deployable fast CLI (`python -m pocketdetect.burrow.fast predict --model M`,
CPU or one GPU; outputs identical to `burrow predict`), run from a worktree whose reference code is
the frozen v1 code.  Predictions -> data/processed/ablations/preds/<model>/<set>/ (git-ignored).
Metrics (pocketdetect.site_eval, closest-pocket protocol): DCC12 top-N+2 (`recall_topn2`), DCC12
top-N, DCA4 top-N+2 (`recall_ldca_topn2`), P1K; paired cluster bootstrap A - v1 (BioLiP: MMseqs2-30 %
clusters; apo-dev: UniProt units), 1,000 resamples.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = str(ROOT / ".venv/bin/python")
MODELS = ROOT / "results/ablations/models"
PREDS = ROOT / "data/processed/ablations/preds"
APO_GT = ROOT / "data/processed/gt/cryptobench_trainfolds"
SETS = {"biolip_dev": ROOT / "data/processed/ablations/biolip_dev_inputs.list",   # absolute-path copy of inputs_audited.list
        "apo_dev": ROOT / "baselines/outputs/apo_dev/inputs/apo"}
V1 = {v: ROOT / f"results/dev/freeze/models/burrow_{v}_default_s0.json" for v in ("small", "all")}
csv.field_size_limit(1 << 30)
TAG = ""
LABEL = {"none": "v1 re-trained (sanity)", "fold_disjoint": "A1 fold-disjoint training", "chen11": "A2 CHEN11-only training",
         "shuffled": "A3 shuffled labels (negative control)", "drop-rays": "A4 − ray buriedness",
         "drop-hitchem": "A4 − first-hit chemistry", "drop-bbbur": "A4 − backbone-only buriedness",
         "drop-metal": "A4 − metal features", "drop-rescomp": "A4 − residue composition",
         "drop-atomclass": "A4 − atom-class densities"}


def all_models():
    out = {f"v1_{v}_s0": (p, v) for v, p in V1.items()}
    for f in sorted(MODELS.glob("abl_*.json")):
        v = json.loads(f.read_text())["meta"]["variant"]
        out[f.stem] = (f, v)
    return out


def predict(pred_root: Path, workers: int, only=None, device="cpu"):
    env = dict(os.environ, PYTHONPATH=str(pred_root / "src"), NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for name, (mp_, v) in all_models().items():
        if only and name not in only:
            continue
        for sname, src in SETS.items():
            od = PREDS / name / sname
            if (od / "timing.csv").exists():
                continue
            arg = f"@{src}" if src.is_file() else str(src)
            t = time.time()
            r = subprocess.run([PY, "-m", "pocketdetect.burrow.fast", "predict", arg, "--out", str(od), "--device", device,
                                "--workers", str(workers), "--batch", "32", "--model", str(mp_), "--ranker", v, "--no-points"],
                               env=env, cwd=pred_root, capture_output=True, text=True)
            print(name, sname, f"{time.time() - t:.0f}s", r.stderr.strip().splitlines()[-1:] if r.stderr else "", flush=True)


def _site_sets():
    from pocketdetect.sites import read_sites, sites_by_structure
    out = {}
    for v in ("small", "all"):
        by = sites_by_structure(read_sites(ROOT / f"results/dev/biolip/dev_sites_{v}.tsv"))
        out[("biolip_dev", v)] = (by, {sid: ss[0].meta.get("group") for sid, ss in by.items()})
    for tab, f in (("apo_cryptic", "sites.tsv"), ("apo_noncryptic", "sites_noncryptic.tsv")):
        by = sites_by_structure(read_sites(APO_GT / f))
        uni = {r["structure_id"]: r["uniprot"].split()[0] for r in csv.DictReader(open(APO_GT / f), delimiter="\t")}
        out[(tab, "any")] = (by, uni)
    return out


def evaluate(n_boot: int):
    from pocketdetect.site_eval import compare_site_methods, evaluate_sites, summarize_sites
    S = _site_sets()
    rows = []
    cache = {}

    def ev(name, sset, key):
        k = (name, key)
        if k not in cache:
            by, groups = S[key]
            d = PREDS / name / sset
            cache[k] = evaluate_sites(by, str(d), groups=groups)
        return cache[k]

    for name, (_, v) in all_models().items():
        if name.startswith("v1_"):
            continue
        ref = f"v1_{v}_s0"
        for table, sset, key in (("BioLiP dev", "biolip_dev", ("biolip_dev", v)), ("apo-dev cryptic", "apo_dev", ("apo_cryptic", "any")),
                                 ("apo-dev non-cryptic", "apo_dev", ("apo_noncryptic", "any"))):
            if not (PREDS / name / sset).exists() or not (PREDS / ref / sset).exists():
                continue
            ra, rb = ev(name, sset, key), ev(ref, sset, key)
            sa = summarize_sites(ra, n_boot=n_boot).set_index("metric")
            sb = summarize_sites(rb, n_boot=n_boot).set_index("metric")
            mets = ("recall_topn2", "recall_topn", "recall_ldca_topn2") if table != "apo-dev non-cryptic" else ("recall_topn2", "recall_topn")
            c = compare_site_methods(ra, rb, metrics=mets, n_boot=n_boot).set_index("metric")
            lo, hi = [k for k in c.columns if "lo" in k][0], [k for k in c.columns if "hi" in k][0]
            for m in list(mets) + ["P1K"]:
                if m not in sa.index:
                    continue
                rows.append(dict(model=name, ablation=json.loads((MODELS / f"{name}.json").read_text())["meta"]["ablation"],
                                 variant=v, table=table, metric=m, value=100 * sa.loc[m, "value"], ref_value=100 * sb.loc[m, "value"],
                                 diff=100 * c.loc[m, "diff"] if m in c.index else float("nan"),
                                 lo=100 * c.loc[m, lo] if m in c.index else float("nan"),
                                 hi=100 * c.loc[m, hi] if m in c.index else float("nan"),
                                 n_sites=int(sa.loc["n_scored_sites", "value"])))
            print(name, table, flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results/ablations"
    df.to_csv(out / f"ablations_long{TAG}.csv", index=False)
    write_md(df)


def write_md(df):
    meta = {f.stem: json.loads(f.read_text())["meta"] for f in MODELS.glob("abl_*.json")}
    L = ["# BURROW ablations (pre-registered, DESIGN §2.5 / §5) — dev only\n",
         "All models: FREEZE v1 default-preset recipe (30 rays, 6/10/16 Å, 3 Å sub-lattice, score floor 0.7, "
         "250x63 point model, graded ranker 300 rounds), seed SEEDS[0], trained by `scripts/ablation_train.py` "
         "with the frozen code (worktree at 5e13267); predictions with the deployable fast CLI (CPU). Each row is "
         "paired against the frozen v1 seed-0 model of the same variant (`burrow_{small,all}_default_s0`) on the "
         "same structures: Δ = ablation − v1, percentage points, 95 % paired cluster-bootstrap CI (BioLiP dev: "
         "MMseqs2-30 % clusters; apo-dev: UniProt units; 1,000 resamples). DCC12 = closest-pocket centre within 12 Å "
         "of the site centre among the top-N(+2); DCA4 = closest-by-DCA pocket within 4 Å of a ligand atom; P1K = "
         "pooled precision of the top-1 pocket. Dev-only numbers: the test-set rows for A1–A3 are prepared in "
         "`results/ablations/TEST_COMMANDS.md` and are run by the eval agent after the final version is fixed.\n",
         "**Reproducibility check:** re-training v1 `small` with this script (ablation `none`, 16 LightGBM threads, "
         "frozen code) gives a point model and ranker that are byte-identical to the frozen "
         "`burrow_small_default_s0.json` (model strings equal), and its predictions on BioLiP dev equal the method "
         "developer's frozen predictions. The ablation models were trained with 12 LightGBM threads (2 lanes within "
         "the 24-worker budget; `deterministic=True`). Thread count does not change the trees: re-training `none` and 5 "
         "ablations (shuffled, fold-disjoint, CHEN11, no-rays, no-hitchem; `small`) with 16 threads gave model strings "
         "that differ only in the `[num_threads: N]` parameter line, and the predictions are byte-identical "
         "(row `v1 re-trained (sanity) (12 LightGBM threads)`).\n",
         "**Notes:** A1 keeps 4,839 of 8,184 training inputs (entries with Foldseek max(qTM, tTM) ≥ 0.5 to any test chain removed, "
         "`data/processed/train/fold_to_test.tsv`). A2 uses the 251 CHEN11 structures with P2Rank-rule ligands (no ions), so the "
         "`small` and `all` CHEN11 models are identical (same trees; only `meta.variant` differs) — their apo-dev rows coincide, and "
         "their BioLiP dev rows differ only because the `all` site table contains ion sites. A3 permutes point labels and pocket "
         "targets within each protein (OOF scoring and candidate formation as in v1).\n",
         "Feature groups dropped in A4 (point model and the matching pocket descriptors of the ranker; candidate "
         "generation is unchanged because it is not learned):\n"]
    from ablation_train import GROUPS
    for g, (rp, rk) in GROUPS.items():
        L.append(f"- `{g}`: point features `{rp}`; ranker descriptors `{rk}`")
    L.append("")
    for table in ("BioLiP dev", "apo-dev cryptic", "apo-dev non-cryptic"):
        sub = df[df.table == table]
        if sub.empty:
            continue
        L.append(f"\n## {table}\n")
        mets = [m for m in ("recall_topn2", "recall_topn", "recall_ldca_topn2", "P1K") if m in set(sub.metric)]
        hdr = {"recall_topn2": "DCC12 top-N+2", "recall_topn": "DCC12 top-N", "recall_ldca_topn2": "DCA4 top-N+2", "P1K": "P1K"}
        L.append("| ablation | variant | train inputs | " + " | ".join(f"{hdr[m]}: value (Δ vs v1 [95% CI])" for m in mets) + " |")
        L.append("|---|---|---|" + "---|" * len(mets))
        for (nm_, var), g in sub.groupby(["model", "variant"], sort=False):
            abl = g.ablation.iloc[0]
            nm = g.model.iloc[0]
            cells = []
            for m in mets:
                r = g[g.metric == m]
                if r.empty:
                    cells.append("–")
                    continue
                r = r.iloc[0]
                cells.append(f"{r.value:.1f} ({r['diff']:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}])")
            lab = LABEL.get(abl.replace(':', '-'), abl) + (" (12 LightGBM threads)" if nm.endswith("_t12") else "")
            L.append(f"| {lab} | {var} | {meta.get(nm, {}).get('n_train_inputs', '–')} | " + " | ".join(cells) + " |")
        ref = sub.drop_duplicates(["variant", "metric"])
        L.append("\nv1 reference values: " + "; ".join(
            f"{v}: " + ", ".join(f"{hdr[m]} {ref[(ref.variant == v) & (ref.metric == m)].ref_value.iloc[0]:.1f}"
                                 for m in mets if len(ref[(ref.variant == v) & (ref.metric == m)]))
            for v in sorted(ref.variant.unique())) + f" ({int(sub.n_sites.iloc[0])} scored sites).")
    (ROOT / f"results/ablations/ABLATIONS{TAG}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def test_commands():
    L = ["# Test-set prediction commands for ablations A1–A3 (prepared, NOT run)\n",
         "To be run by the eval agent after the final version is fixed, as descriptive rows. Same harness as the "
         "v1 test predictions (`scripts/predict_benchmarks.py`), with the ablation model JSON instead of the frozen "
         "model; CPU or GPU fast backend (outputs identical). Example (one model; repeat for each):\n",
         "```bash"]
    for f in sorted(MODELS.glob("abl_*.json")):
        m = json.loads(f.read_text())["meta"]
        if m["ablation"] not in ("fold_disjoint", "chen11", "shuffled"):
            continue
        L.append(f"# {LABEL[m['ablation']]} ({m['variant']})")
        L.append(f"PYTHONPATH=<worktree at 6a00a61 or later>/src .venv/bin/python -m pocketdetect.burrow.fast predict "
                 f"<test inputs: @list|dir> --out results/test/ablations/{f.stem}/<benchmark> --model {f.relative_to(ROOT)} "
                 f"--ranker {m['variant']} --device cpu --workers 24 --no-points")
    L += ["```", "", "Then evaluate with the same test evaluation script / protocol as v1 (paired vs v1 on identical "
          "structures), reporting the rows as descriptive (ablations are dev-selected, not part of the pre-registered "
          "primary comparison)."]
    (ROOT / "results/ablations/TEST_COMMANDS.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["predict", "eval", "md", "test-commands"])
    ap.add_argument("--pred-root", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default="cpu", help="cpu or one GPU (outputs are identical)")
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--preds-dir", type=Path, default=None)
    ap.add_argument("--out-tag", default="", help="suffix for ABLATIONS<tag>.md / ablations_long<tag>.csv")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    global MODELS, PREDS, TAG
    MODELS = a.models_dir or MODELS
    PREDS = a.preds_dir or PREDS
    TAG = a.out_tag
    if a.stage == "predict":
        predict(a.pred_root, a.workers, a.only, a.device)
    elif a.stage == "eval":
        evaluate(a.n_boot)
    elif a.stage == "md":
        write_md(pd.read_csv(ROOT / f"results/ablations/ablations_long{TAG}.csv"))
    else:
        test_commands()


if __name__ == "__main__":
    main()
