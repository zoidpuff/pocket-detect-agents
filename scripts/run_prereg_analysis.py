#!/usr/bin/env python
"""Pre-registered test analysis (docs/PREREG.md incl. amendments A1–A3) for one frozen version.

  # real run (after the freeze; predictions from scripts/predict_benchmarks.py):
  nohup .venv/bin/python scripts/run_prereg_analysis.py --version v1 > logs/prereg_v1.log 2>&1 &
  # dry run (pipeline validation; a baseline stands in for BURROW; no ledger, no git gating):
  .venv/bin/python scripts/run_prereg_analysis.py --dry-run --stand-in p2rank_2.5.1

Steps
 1. Evaluate every BURROW variant (real: through the gated scripts/run_test_eval.py → TEST_LEDGER;
    dry: pocketdetect.benchmarks.run_benchmark) and every comparator from its baseline prediction
    directory with the *same* code and ground truth (results/test/<v>/comparators/...). The
    published fpocket+PRANK per-chain predictions (LBS-comparison) are mapped to the evaluation
    frame (results/test/_common/published_fpocket+PRANK/preds).
 2. Family 1 (fixed sequence, each seed in SEEDS[:3], preset default, vs published AND re-run
    fpocket 4.2.3+PRANK 2.4.2): C1 recall_topn2 Δ ≥ +2.0 and CI>0 → C1-strict recall_ldca_topn2 CI>0
    → C2 P1K lower CI > −3. Sensitivity: published comparator on the common chain subset.
 3. Family 2 (each seed): O2a/O2b/O3/O4/O5/O6 vs the best (highest point estimate) of the listed
    baseline variants; one-sided bootstrap p for H0 Δ ≤ −2; Holm (α = 0.025 one-sided);
    pass (Holm-rejected) / fail (upper CI < −2) / inconclusive; superiority = lower CI > 0.
    Endpoint verdict = all three seeds pass (reported per seed).
 4. Descriptive: secondary metrics (top-N, max recall, one-to-one, P1K, residue F1/MCC, RRO,
    hallucination), LIGYSIS_NI, accurate preset, strata (fold novelty, ligand class, pLDDT band,
    dimer class), controls (oracle, protein centroid, random surface, surface tiling).
 5. results/test/<v>/SUMMARY.md + summary.json with provenance (commit, model sha256s, PREREG sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketdetect.benchmarks import REGISTRY, read_p2rank_ds, run_benchmark  # noqa: E402
from pocketdetect.clusters import load_clusters, structure_units  # noqa: E402
from pocketdetect.prereg import FAMILY1, FAMILY2, Endpoint, decide_noninferiority, holm, load_rows, paired  # noqa: E402
from pocketdetect.seeds import BOOTSTRAP_SEED, SEEDS  # noqa: E402

BASE = ROOT / "baselines/outputs"
COMMON = ROOT / "results/test/_common"
BENCH_DIR = {"ligysis": "ligysis", "ligysis_ni": "ligysis", "coach420_mlig": "coach420_mlig", "holo4k_mlig": "holo4k_mlig",
             "cryptobench": "cryptobench", "pocketwatch": "pocketwatch", "af_set": "af_set", "dimer": "dimer_assembly"}
SECONDARY = {
    "sites": ["recall_topn", "recall_topn2", "recall_max", "recall_anyp_topn2", "recall_1to1_topn2", "recall_ldca_topn2", "P1K", "P1K_1to1",
              "TP@100FP", "RRO", "res_f1", "res_mcc", "hallucination_rate", "hallucination_rate_topn2"],
    "ligands": ["success_DCA4_topn", "success_DCA4_topn2", "success_DCC4_topn", "success_DCA4_top1", "ap"],
    "dimer": ["recall_DCC12_topn2@interface", "recall_DCA4_topn2@interface", "recall_DCC12_topn2@all", "precision_DCC12_topn2@all",
              "recall_DCC12_topn2@homo_interface", "recall_DCC12_topn2@hetero_interface"],
}
NUC_COF = set("ATP ADP AMP ANP ACP AGS GTP GDP GMP GNP GCP GSP CTP CDP UTP UDP TTP NAD NAI NAP NDP NADH FAD FMN SAM SAH COA ACO "
              "HEM HEC HEA PLP TPP THP BTN FES SF4 F3S MGD PQQ CMP 3PO UPG U5P".split())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def kind_of(bench: str) -> str:
    return REGISTRY[bench].kind


def baseline_pred_dir(bench: str, variant: str) -> Path:
    v = BASE / BENCH_DIR[bench] / variant
    return v / ("preds_orig" if bench in ("ligysis", "ligysis_ni") else "preds")


def published_pred_dir() -> Path:
    """Authors' published fpocket+PRANK per-chain predictions mapped to the evaluation frame (cached)."""
    d = COMMON / "published_fpocket+PRANK" / "preds"
    if d.exists() and len(list(d.glob("*.csv"))) > 2000:
        return d
    from pocketdetect.evaluate import write_predictions
    from pocketdetect.ligysis import LIGYSIS_RESULTS, load_transforms, predictions_from_comparisons, safe_unpickle, to_original_frame

    d.mkdir(parents=True, exist_ok=True)
    comp = safe_unpickle(LIGYSIS_RESULTS / "ALL_COMPARISONS_EXTENDED6.pkl")
    T = load_transforms()
    for chain, p in predictions_from_comparisons(comp, "fpocket+PRANK").items():
        R, c0 = T[chain]
        write_predictions(d / f"{chain}.csv", to_original_frame(p.centers, R, c0), p.scores)
    (d.parent / "README.md").write_text("Published fpocket+PRANK predictions of the LBS-comparison (ALL_COMPARISONS_EXTENDED6.pkl), "
                                        "centres mapped to the asymmetric-unit frame; no residues (their UniProt residue sets are not "
                                        "mapped); chains without predictions have no file (= misses).\n")
    return d


def evaluate(bench: str, pred_dir: Path, out: Path, n_boot: int) -> Path:
    if (out / "eval_info.json").exists():
        return out
    t = time.time()
    run_benchmark(bench, pred_dir, out, n_boot=n_boot, jobs=16)
    print(f"  evaluated {bench} {pred_dir} -> {out} ({time.time() - t:.0f}s)", flush=True)
    return out


def gated_evaluate(version: str, bench: str, variant: str, pred_dir: Path, models: list, n_boot: int) -> Path:
    out = ROOT / "results/test" / version / bench / variant
    if (out / "eval_info.json").exists():
        return out
    cmd = [sys.executable, str(ROOT / "scripts/run_test_eval.py"), "--benchmark", bench, "--pred-dir", str(pred_dir), "--method", variant,
           "--version", version, "--n-boot", str(n_boot)]
    for m in models:
        cmd += ["--model", m]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"run_test_eval failed for {bench}/{variant}:\n{r.stderr}{r.stdout}")
    return out


def summary_values(eval_dir: Path) -> pd.Series:
    s = pd.read_csv(eval_dir / "summary.csv")
    if "average" in s.columns:
        s = s[s.average.isin(["ligand", "pooled", "count"])].drop_duplicates("metric")
    return s.set_index("metric")


# --------------------------------------------------------------------------- controls
def control_structures(bench: str):
    b = REGISTRY[bench]
    if b.kind == "sites":
        from pocketdetect.sites import read_sites, sites_by_structure

        st = read_sites(b.abspath)
        out, centers = [], {}
        for sid, g in st.groupby("structure_id", sort=True):
            r = g.iloc[0]
            chains = str(r["input_chains"]).replace("-", ",").split(",") if "input_chains" in g.columns and str(r["input_chains"]) not in ("", "nan") else None
            out.append((sid, r["structure_path"], chains))
            sc = g[g.scored]
            centers[sid] = sc[["center_x", "center_y", "center_z"]].to_numpy(float)
        return out, centers
    if b.kind == "ligands":
        man = read_p2rank_ds(b.abspath)
        return [(r.structure_id, r.structure_path, None) for r in man.itertuples()], None
    ids = pd.read_csv(ROOT / "data/processed/dimer/inputs_assembly.csv")
    return [(r.structure_id, str(ROOT / "data/processed/dimer" / r.input_path), None) for r in ids.itertuples()], None


def _ctrl_job(a):
    from pocketdetect.controls import write_control

    kind, chunk, out, centers = a
    return write_control(kind, chunk, out, centers=centers)


def make_control(bench: str, kind: str, out: Path, jobs: int = 32) -> Path:
    if out.exists() and any(out.glob("*.csv")):
        return out
    structs, centers = control_structures(bench)
    if kind == "oracle" and centers is None:
        return None
    chunks = [structs[i::jobs] for i in range(jobs)]
    with ProcessPoolExecutor(jobs) as ex:
        list(ex.map(_ctrl_job, [(kind, c, str(out), centers) for c in chunks if c]))
    return out


# --------------------------------------------------------------------------- strata
def fold_strata(bench: str, sids) -> dict:
    t = pd.read_csv(ROOT / "data/manifests/test_fold_novelty.tsv", sep="\t", dtype=str)
    src = {"ligysis": ["ligysis", "ligysis_extra"], "ligysis_ni": ["ligysis", "ligysis_extra"], "coach420_mlig": ["coach420"], "holo4k_mlig": ["holo4k"],
           "cryptobench": ["cryptobench"], "pocketwatch": ["pocketwatch_test"], "af_set": ["af_set"], "dimer": ["dimer"]}[bench]
    t = t[t.source_set.isin(src)]
    lab = dict(zip(t.id, t.stratum))
    ids = pd.Index(t.id)
    out = {}
    for s in sids:
        units = [u for so in src for u in structure_units(so, s, ids)]
        vals = [lab[u] for u in units if u in lab]
        order = {"<0.5 (no_hit)": 0, "<0.5": 0, "0.5-0.7": 1, ">0.7": 2}
        out[s] = {0: "<0.5", 1: "0.5-0.7", 2: ">0.7"}[max(order.get(v, 0) for v in vals)] if vals else "unknown"  # most similar chain decides
    return out


def ligand_class_map() -> dict:
    from pocketdetect.ligysis import LIGYSIS_RESULTS, load_reference_sites, safe_unpickle

    S = load_reference_sites()
    cpl = safe_unpickle(LIGYSIS_RESULTS / "LIGYSIS_human_chains_per_lig_MAY_2024.pkl")
    out = {}
    for r in S.itertuples(index=False):
        names = {k.split("_")[1] for k in cpl.get(r.seg_id, {}).get(int(r.site_id), {})}
        if r.ion_prop == 1:
            c = "ion"
        elif names & NUC_COF:
            c = "nucleotide-cofactor"
        else:
            c = "other"
        out[(r.structure_id, str(r.site_id))] = c
    return out


def stratified(A, B, metric: str, labels: pd.Series, n_boot: int) -> list:
    """Per-stratum paired estimates (rows of A/B filtered by label; strata < 100 rows flagged)."""
    from pocketdetect.prereg import Rows

    rows = []
    for lab in sorted(labels.dropna().unique()):
        ka = A.rows[labels.reindex(A.rows.index).eq(lab).to_numpy()]
        keys = set(map(tuple, ka[A.keys].to_numpy()))
        kb = B.rows[[tuple(x) in keys for x in B.rows[B.keys].to_numpy()]]
        c = paired(Rows(A.kind, ka, A.keys), Rows(B.kind, kb, B.keys), metric, n_boot=n_boot)
        rows.append({**{k: v for k, v in c.items() if not k.startswith("_")}, "stratum": lab, "small_stratum": c["n_rows"] < 100})
    return rows


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stand-in", default="p2rank_2.5.1", help="dry run: baseline variant used in place of BURROW")
    ap.add_argument("--presets", default="default,accurate")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--skip-controls", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.dry_run:
        version = "_dryrun"
    else:
        if not a.version:
            raise SystemExit("--version required (or --dry-run)")
        version = a.version
    OUT = ROOT / "results/test" / version
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    presets = a.presets.split(",")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip())

    # ---- method under test: bench -> variant -> (pred_dir, models)
    from predict_benchmarks import PLAN  # noqa: E402

    method: dict = {}
    if a.dry_run:
        for b, (inp, ranker, af, _sfx) in PLAN.items():
            stand = a.stand_in
            if af and (BASE / BENCH_DIR[b] / f"{a.stand_in}_af").exists():
                stand = f"{a.stand_in}_af"  # AF-mode analogue of the stand-in where it exists
            for p in presets:
                for s in seeds:
                    v = f"burrow_{ranker}_{p}_s{s}" + ("_af" if af else "")
                    method.setdefault(b, {})[v] = (baseline_pred_dir(b, stand), ["none"])
        provenance_models = {"stand_in": a.stand_in}
    else:
        idx = json.loads((OUT / "preds/index.json").read_text())
        for key, r in idx["runs"].items():
            if r.get("rc") != 0:
                print(f"WARNING {key}: rc={r.get('rc')} (failures count as misses)", flush=True)
            models = [r["model"]] + ([r["af_ranker"]] if r.get("af_ranker") else [])
            method.setdefault(r["bench"], {})[r["variant"]] = (ROOT / r["pred_dir"], models)
        provenance_models = idx["sha256"]

    def method_eval(bench, variant):
        pdir, models = method[bench][variant]
        if a.dry_run:
            # all seeds/presets share one stand-in directory: evaluate once
            return evaluate(bench, pdir, OUT / "eval" / bench / f"standin_{pdir.parent.name}", a.n_boot)
        return gated_evaluate(version, bench, variant, pdir, models, a.n_boot)

    comp_cache: dict = {}

    def comparator_eval(bench, variant):
        key = (bench, variant)
        if key not in comp_cache:
            pdir = published_pred_dir() if variant.startswith("published:") else baseline_pred_dir(bench, variant)
            name = variant.replace(":", "_")
            comp_cache[key] = evaluate(bench, pdir, OUT / "comparators" / bench / name, a.n_boot)
        return comp_cache[key]

    def var_name(ep: Endpoint, preset: str, seed: int) -> str:
        return f"burrow_{ep.ranker}_{preset}_s{seed}" + ("_af" if ep.af_mode else "")

    rows_cache: dict = {}

    def rows(bench, d):
        k = (bench, str(d))
        if k not in rows_cache:
            rows_cache[k] = load_rows(bench, d)
        return rows_cache[k]

    result = {"version": version, "dry_run": a.dry_run, "family1": [], "family2": [], "descriptive": {}, "controls": [], "sanity": []}

    # ---- family 1
    print("family 1", flush=True)
    f1_pass = {}
    for ep in FAMILY1:
        recs = []
        for s in seeds:
            A = rows(ep.benchmark, method_eval(ep.benchmark, var_name(ep, "default", s)))
            for cv in ep.comparators:
                B = rows(ep.benchmark, comparator_eval(ep.benchmark, cv))
                c = paired(A, B, ep.metric, n_boot=a.n_boot)
                if ep.name == "C1":
                    ok = c["diff"] >= 2.0 and c["ci_low"] > 0
                elif ep.name == "C1-strict":
                    ok = c["ci_low"] > 0
                else:
                    ok = c["ci_low"] > ep.margin
                rec = {"endpoint": ep.name, "seed": SEEDS[s], "seed_index": s, "comparator": cv, **{k: v for k, v in c.items() if not k.startswith("_")}, "pass": bool(ok)}
                if cv.startswith("published:") and ep.metric != "P1K":
                    cs = paired(A, B, ep.metric, n_boot=a.n_boot, common_subset=True)
                    rec["common_subset"] = {k: v for k, v in cs.items() if k in ("a", "b", "diff", "ci_low", "ci_high", "n_rows")}
                recs.append(rec)
        f1_pass[ep.name] = all(r["pass"] for r in recs)
        result["family1"] += recs
    seq, alive = {}, True
    for ep in FAMILY1:
        seq[ep.name] = ("PASS" if f1_pass[ep.name] else "FAIL") if alive else ("not tested (sequence stopped; " + ("pass" if f1_pass[ep.name] else "fail") + " descriptive)")
        alive = alive and f1_pass[ep.name]
    result["family1_sequence"] = seq

    # ---- family 2
    print("family 2", flush=True)
    f2 = []
    for s in seeds:
        pv, recs = {}, {}
        for ep in FAMILY2:
            A = rows(ep.benchmark, method_eval(ep.benchmark, var_name(ep, "default", s)))
            comps = []
            for cv in ep.comparators:
                if not baseline_pred_dir(ep.benchmark, cv).exists():
                    continue
                B = rows(ep.benchmark, comparator_eval(ep.benchmark, cv))
                comps.append((paired(A, B, ep.metric, stratum=ep.stratum, n_boot=a.n_boot, keep_draws=True), cv))
            best = max(comps, key=lambda x: x[0]["b"])
            c, cv = best
            d = c.pop("_draws")
            pv[ep.name] = float(np.mean(d <= ep.margin))
            recs[ep.name] = {"endpoint": ep.name, "benchmark": ep.benchmark, "metric": ep.metric + (f"@{ep.stratum}" if ep.stratum else ""), "seed": SEEDS[s],
                             "seed_index": s, "best_comparator": cv, "comparator_values": {x[1]: round(x[0]["b"], 3) for x in comps}, **c,
                             "superior": c["ci_low"] > 0}
        H = holm(pv)
        for k, r in recs.items():
            r["p_noninferiority"] = pv[k]
            r["holm_threshold"] = H[k]["threshold"]
            r["decision"] = "pass" if H[k]["rejected"] else ("fail" if r["ci_high"] < -2.0 else "inconclusive")
            r["decision_ci_only"] = decide_noninferiority(r, -2.0)
            f2.append(r)
    result["family2"] = f2
    verdict = {}
    for ep in FAMILY2:
        ds = [r["decision"] for r in f2 if r["endpoint"] == ep.name]
        verdict[ep.name] = "pass" if all(x == "pass" for x in ds) else ("fail" if any(x == "fail" for x in ds) else "inconclusive")
    result["family2_verdict"] = verdict

    # ---- descriptive: secondary metrics per benchmark (seed 0, both presets) + comparators
    print("descriptive", flush=True)
    desc = []
    for b in PLAN:
        kind = kind_of(b)
        mets = SECONDARY[kind]
        entries = []
        for p in presets:
            for v in method.get(b, {}):
                if f"_{p}_s{seeds[0]}" in v:
                    entries.append((f"BURROW[{v}]", method_eval(b, v)))
        for cv in sorted({c for ep in FAMILY1 + FAMILY2 if ep.benchmark == b for c in ep.comparators} | ({"p2rank_2.5.1", "fpocket_4.2.3+PRANK_2.4.2"} if b == "ligysis_ni" else set())):
            if cv.startswith("published:") or baseline_pred_dir(b, cv).exists():
                entries.append((cv, comparator_eval(b, cv)))
        for name, d in entries:
            sv = summary_values(d)
            for m in mets:
                if m in sv.index:
                    desc.append({"benchmark": b, "method": name, "metric": m, "value": float(sv.loc[m, "value"]),
                                 "ci_low": float(sv.loc[m, "ci_low"]) if pd.notna(sv.loc[m, "ci_low"]) else None,
                                 "ci_high": float(sv.loc[m, "ci_high"]) if pd.notna(sv.loc[m, "ci_high"]) else None})
    result["descriptive"]["secondary"] = desc

    # ---- strata (seed 0, default preset, vs best family comparator of that benchmark)
    strata = []
    lc = None
    for ep in [e for e in FAMILY1[:1]] + FAMILY2:
        A = rows(ep.benchmark, method_eval(ep.benchmark, var_name(ep, "default", seeds[0])))
        cv = next((r["best_comparator"] for r in f2 if r["endpoint"] == ep.name), ep.comparators[-1])
        B = rows(ep.benchmark, comparator_eval(ep.benchmark, cv))
        sid_col = "complex_id" if A.kind == "dimer" else "structure_id"
        fs = fold_strata(ep.benchmark, A.rows[sid_col].unique())
        for r in stratified(A, B, ep.metric if ep.metric != "P1K" else "recall_topn2", A.rows[sid_col].map(fs), a.n_boot):
            strata.append({"endpoint": ep.name, "comparator": cv, "kind": "fold_novelty_maxTM", **r})
        if ep.benchmark == "ligysis":
            lc = lc or ligand_class_map()
            lab = pd.Series([lc.get((x, str(y)), "unknown") for x, y in zip(A.rows.structure_id, A.rows.site_id)], index=A.rows.index)
            for r in stratified(A, B, ep.metric, lab, a.n_boot):
                strata.append({"endpoint": ep.name, "comparator": cv, "kind": "ligand_class", **r})
        if ep.benchmark == "af_set":
            st = pd.read_csv(REGISTRY["af_set"].abspath, sep="\t", usecols=["structure_id", "site_id", "plddt_band"], dtype=str)
            band = dict(zip(zip(st.structure_id, st.site_id), st.plddt_band))
            lab = pd.Series([band.get((x, str(y)), "unknown") for x, y in zip(A.rows.structure_id, A.rows.site_id)], index=A.rows.index)
            for r in stratified(A, B, ep.metric, lab, a.n_boot):
                strata.append({"endpoint": ep.name, "comparator": cv, "kind": "site_plddt_band", **r})
        if ep.benchmark == "dimer":
            for r in stratified(A, B, ep.metric, A.rows.site_class, a.n_boot):
                strata.append({"endpoint": ep.name, "comparator": cv, "kind": "dimer_site_class", **r})
    result["descriptive"]["strata"] = strata

    # ---- controls
    if not a.skip_controls:
        print("controls", flush=True)
        for b in ("ligysis", "cryptobench", "pocketwatch", "af_set", "coach420_mlig", "holo4k_mlig", "dimer"):
            for kind in ("oracle", "centroid", "random", "tiling"):
                pdir = make_control(b, kind, COMMON / "controls" / b / kind / "preds")
                if pdir is None:
                    continue
                d = evaluate(b, pdir, OUT / "controls" / b / kind, min(a.n_boot, 200))
                sv = summary_values(d)
                m = REGISTRY[b].headline[0]
                result["controls"].append({"benchmark": b, "control": kind, "metric": m, "value": float(sv.loc[m, "value"]) if m in sv.index else None,
                                           "P1K": float(sv.loc["P1K", "value"]) if "P1K" in sv.index else None})

    # ---- dry-run sanity: method vs itself (Δ = 0, CI = [0, 0])
    if a.dry_run:
        for ep in FAMILY1 + FAMILY2:
            A = rows(ep.benchmark, method_eval(ep.benchmark, var_name(ep, "default", seeds[0])))
            c = paired(A, A, ep.metric, stratum=ep.stratum, n_boot=200)
            result["sanity"].append({"check": f"self-comparison {ep.name}", "diff": c["diff"], "ci_low": c["ci_low"], "ci_high": c["ci_high"],
                                     "ok": c["diff"] == 0 and c["ci_low"] == 0 and c["ci_high"] == 0})

        # oracle as the method under test: must clear C1 by a wide margin against both comparators
        if not a.skip_controls:
            O = rows("ligysis", OUT / "controls" / "ligysis" / "oracle")
            for cv in FAMILY1[0].comparators:
                c = paired(O, rows("ligysis", comparator_eval("ligysis", cv)), "recall_topn2", n_boot=200)
                result["sanity"].append({"check": f"oracle vs {cv} (C1 metric)", "diff": round(c["diff"], 2), "ci_low": round(c["ci_low"], 2),
                                         "ci_high": round(c["ci_high"], 2), "ok": c["diff"] >= 2.0 and c["ci_low"] > 0 and abs(c["a"] - 100) < 1e-9})

    # ---- provenance + outputs
    result["provenance"] = {"commit": commit, "dirty_tree": dirty, "prereg_sha256": sha256(ROOT / "docs/PREREG.md"), "models": provenance_models,
                            "scoring_code_sha256": {f: sha256(ROOT / f) for f in ("src/pocketdetect/evaluate.py", "src/pocketdetect/ligands.py",
                                                                                   "src/pocketdetect/site_eval.py", "src/pocketdetect/prereg.py",
                                                                                   "src/pocketdetect/benchmarks.py")},
                            "gt_sha256": {b: sha256(REGISTRY[b].abspath) for b in PLAN if REGISTRY[b].abspath.is_file()},
                            "bootstrap": {"n_boot": a.n_boot, "seed": BOOTSTRAP_SEED, "units": "30% sequence clusters (MMseqs2)"},
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0)}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    (OUT / "SUMMARY.md").write_text(render(result))
    print((OUT / "SUMMARY.md").read_text())


def render(R: dict) -> str:
    f = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"  # noqa: E731
    L = [f"# Pre-registered test analysis — {R['version']}{' (DRY RUN: a baseline stands in for BURROW)' if R['dry_run'] else ''}", "",
         "Generated by `scripts/run_prereg_analysis.py` (do not edit). Values in percentage points; Δ = method − comparator; "
         "95 % paired cluster-bootstrap CIs (30 % sequence clusters, 1,000 resamples).", "",
         "## Family 1 — fixed sequence (C1 → C1-strict → C2)", "",
         "| endpoint | status |", "|---|---|"] + [f"| {k} | {v} |" for k, v in R["family1_sequence"].items()] + [
        "", "| endpoint | seed | comparator | method | comparator | Δ [95% CI] | n units | pass | common-subset Δ [CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for r in R["family1"]:
        cs = r.get("common_subset")
        L.append(f"| {r['endpoint']} | {r['seed']} | {r['comparator']} | {f(r['a'])} | {f(r['b'])} | {f(r['diff'])} [{f(r['ci_low'])}, {f(r['ci_high'])}] | "
                 f"{r['n_units']} | {'yes' if r['pass'] else 'no'} | {(f(cs['diff']) + ' [' + f(cs['ci_low']) + ', ' + f(cs['ci_high']) + ']') if cs else ''} |")
    L += ["", "## Family 2 — Holm over O2a/O2b/O3/O4/O5/O6 (non-inferiority margin −2 pts vs best baseline)", "",
          "| endpoint | verdict (all seeds) |", "|---|---|"] + [f"| {k} | {v} |" for k, v in R["family2_verdict"].items()] + [
        "", "| endpoint | seed | metric | method | best comparator | Δ [95% CI] | p (H0 Δ ≤ −2) | Holm thr | decision | superior |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in R["family2"]:
        L.append(f"| {r['endpoint']} | {r['seed']} | {r['metric']} | {f(r['a'])} | {r['best_comparator']} {f(r['b'])} | {f(r['diff'])} [{f(r['ci_low'])}, {f(r['ci_high'])}] | "
                 f"{r['p_noninferiority']:.4f} | {r['holm_threshold']:.4f} | {r['decision']} | {'yes' if r['superior'] else 'no'} |")
    L += ["", "## Descriptive — secondary metrics", "", "| benchmark | method | metric | value [95% CI] |", "|---|---|---|---|"]
    for r in R["descriptive"]["secondary"]:
        sc = 100 if not r["metric"].startswith("TP@") else 1
        ci = f" [{r['ci_low'] * sc:.1f}, {r['ci_high'] * sc:.1f}]" if r["ci_low"] is not None else ""
        L.append(f"| {r['benchmark']} | {r['method']} | {r['metric']} | {r['value'] * sc:.1f}{ci} |")
    L += ["", "## Descriptive — strata (seed 0, default preset; strata < 100 rows flagged *)", "", "| endpoint | stratum kind | stratum | n | method | comparator | Δ [95% CI] |", "|---|---|---|---|---|---|---|"]
    for r in R["descriptive"]["strata"]:
        L.append(f"| {r['endpoint']} | {r['kind']} | {r['stratum']}{' *' if r['small_stratum'] else ''} | {r['n_rows']} | {f(r['a'])} | {r['comparator']} {f(r['b'])} | {f(r['diff'])} [{f(r['ci_low'])}, {f(r['ci_high'])}] |")
    if R["controls"]:
        L += ["", "## Controls", "", "| benchmark | control | headline metric | value | P1K |", "|---|---|---|---|---|"]
        for r in R["controls"]:
            L.append(f"| {r['benchmark']} | {r['control']} | {r['metric']} | {f(100 * r['value']) if r['value'] is not None else '–'} | {f(100 * r['P1K']) if r['P1K'] is not None else '–'} |")
    if R["sanity"]:
        L += ["", "## Dry-run sanity checks", "", "| check | Δ | CI | ok |", "|---|---|---|---|"] + [
            f"| {r['check']} | {r['diff']} | [{r['ci_low']}, {r['ci_high']}] | {r['ok']} |" for r in R["sanity"]]
    p = R["provenance"]
    L += ["", "## Provenance", "", f"- commit `{p['commit']}`{' (dirty tree)' if p['dirty_tree'] else ''}; PREREG sha256 `{p['prereg_sha256']}`",
          f"- models: `{json.dumps(p['models'])}`", f"- bootstrap: {p['bootstrap']}", f"- generated {p['utc']} in {p['seconds']} s"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
