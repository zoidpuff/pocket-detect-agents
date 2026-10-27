"""Benchmark registry + one evaluation entry point shared by scripts/run_test_eval.py (gated, test
sets) and scripts/run_dev_eval.py (dev sets).

Each benchmark is either a *site table* (pocketdetect.sites; LIGYSIS protocol via site_eval) or a
*ligand* benchmark (P2Rank .ds file; pocketdetect.ligands + evaluate). Bootstrap resampling units
are 30 % sequence clusters of the test union (pocketdetect.clusters), UniProt as sensitivity.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .clusters import groups_for_structures
from .seeds import BOOTSTRAP_SEED

ROOT = Path(__file__).resolve().parents[2]

__all__ = ["Benchmark", "REGISTRY", "read_p2rank_ds", "run_benchmark"]


@dataclass(frozen=True)
class Benchmark:
    name: str
    kind: str  # "sites" | "ligands" | "dimer"
    path: str  # site table (.tsv) or P2Rank dataset (.ds), relative to ROOT
    cluster_sources: tuple
    headline: tuple  # metric names reported in the ledger (first = primary endpoint)
    split: str = "test"
    rules: str = "p2rank_strict"  # ligand benchmarks
    criteria: tuple = ("DCA4", "DCC4")
    note: str = ""
    extras: tuple = ()  # e.g. ("hallucination",) for AF models

    @property
    def abspath(self) -> Path:
        return ROOT / self.path


REGISTRY: dict[str, Benchmark] = {b.name: b for b in [
    Benchmark("ligysis", "sites", "data/processed/gt/ligysis_lbs/sites.tsv", ("ligysis", "ligysis_extra"),
              ("recall_topn2", "recall_ldca_topn2", "P1K", "recall_topn", "recall_max", "recall_1to1_topn2", "P1K_1to1"),
              note="PREREG C1 / C1-strict / C2"),
    Benchmark("ligysis_ni", "sites", "data/processed/gt/ligysis_ni/sites.tsv", ("ligysis", "ligysis_extra"),
              ("recall_topn2", "recall_ldca_topn2", "P1K"), note="LIGYSIS_NI (descriptive): ion_prop != 1"),
    Benchmark("coach420_mlig", "ligands", "data/raw/p2rank_datasets/current/coach420(mlig).ds", ("coach420",),
              ("success_DCA4_topn", "success_DCA4_topn2"), note="PREREG O2a"),
    Benchmark("holo4k_mlig", "ligands", "data/raw/p2rank_datasets/current/holo4k(mlig).ds", ("holo4k",),
              ("success_DCA4_topn", "success_DCA4_topn2"), note="PREREG O2b"),
    Benchmark("pocketwatch", "sites", "data/processed/gt/pocketwatch_test/sites.tsv", ("pocketwatch_test",),
              ("recall_topn2", "P1K"), note="PREREG O3"),
    Benchmark("cryptobench", "sites", "data/processed/gt/cryptobench_test/sites.tsv", ("cryptobench",),
              ("recall_topn2", "P1K"), note="PREREG O4"),
    Benchmark("af_set", "sites", "data/processed/gt/af_set/sites.tsv", ("af_set",),
              ("recall_topn2", "recall_ldca_topn2", "P1K", "hallucination_rate_topn2", "hallucination_rate"),
              note="PREREG O5 (scripts/build_gt_af.py from Track C)", extras=("hallucination",)),
    Benchmark("af_set_allsites", "sites", "data/processed/gt/af_set_allsites/sites.tsv", ("af_set",),
              ("recall_topn2", "P1K", "hallucination_rate_topn2"), note="O5 sensitivity: all LIGYSIS sites of the UniProt",
              extras=("hallucination",)),
    Benchmark("dimer_chains", "dimer", "data/processed/dimer", ("dimer",),
              ("recall_DCC12_topn2@interface", "recall_DCA4_topn2@interface", "precision_DCC12_topn2@all", "recall_DCC12_topn2@all"),
              note="descriptive: isolated-chain inputs (eval_dimer --mode chain --chain-set all, as the baselines' dimer_chains)"),
    Benchmark("dimer", "dimer", "data/processed/dimer", ("dimer",),
              ("recall_DCC12_topn2@interface", "recall_DCA4_topn2@interface", "precision_DCC12_topn2@all", "recall_DCC12_topn2@all"),
              note="PREREG O6 via scripts/eval_dimer.py (assembly input, J>=0.5 grouping = amendment A1, label-free collapse)"),
]}


def read_p2rank_ds(ds: str | Path) -> pd.DataFrame:
    """P2Rank .ds → manifest (structure_id, structure_path, ligand_spec)."""
    ds = Path(ds)
    rows, header = [], None
    for line in ds.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("HEADER:"):
            header = line.split(":", 1)[1].split()
            continue
        cols = dict(zip(header or ["protein"], line.split()))
        prot = cols["protein"]
        rows.append({"structure_id": Path(prot).name, "structure_path": str(ds.parent / prot),
                     "ligand_spec": cols.get("ligand_codes", cols.get("ligands", ""))})
    return pd.DataFrame(rows)


def _ligand_gt(man: pd.DataFrame, rules: str, jobs: int):
    from concurrent.futures import ProcessPoolExecutor

    from .ligands import build_ground_truth

    args = [(r.structure_path, r.structure_id, rules, r.ligand_spec or None) for r in man.itertuples()]
    with ProcessPoolExecutor(jobs) as ex:
        return list(ex.map(_gt_one, args, chunksize=8))


def _gt_one(a):
    from .ligands import build_ground_truth

    path, sid, rules, spec = a
    return build_ground_truth(path, structure_id=sid, rules=rules, spec=spec)


def run_benchmark(bench: Benchmark | str, pred_dir: str | Path, out_dir: str | Path, n_boot: int = 1000,
                  seed: int = BOOTSTRAP_SEED, jobs: int = 16, gt_path: str | Path | None = None) -> dict:
    """Evaluate ``pred_dir`` on a benchmark; write outputs to ``out_dir``; return headline dict.
    Writes summary.csv (cluster bootstrap) and summary_uniprot.csv (UniProt-unit sensitivity)."""
    b = REGISTRY[bench] if isinstance(bench, str) else bench
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(gt_path) if gt_path else b.abspath
    if not path.exists():
        raise FileNotFoundError(f"{b.name}: ground truth {path} not found ({b.note})")
    headline = {}
    if b.kind == "dimer":
        s, n_missing = _run_dimer(b, path, pred_dir, out, n_boot, seed)
    elif b.kind == "sites":
        from .site_eval import evaluate_sites, summarize_sites
        from .sites import read_sites, sites_by_structure

        st = read_sites(path)
        by = sites_by_structure(st)
        g30 = groups_for_structures(list(by), b.cluster_sources, "cluster30")
        gup = groups_for_structures(list(by), b.cluster_sources, "uniprot")
        res = evaluate_sites(by, str(pred_dir), groups=g30)
        res.save(out)
        summ = summarize_sites(res, n_boot=n_boot, seed=seed)
        summ.to_csv(out / "summary.csv", index=False)
        res.sites["group"] = res.sites.structure_id.map(gup)
        res.structures["group"] = res.structures.structure_id.map(gup)
        res.pockets["group"] = res.pockets.structure_id.map(gup)
        summarize_sites(res, n_boot=n_boot, seed=seed).to_csv(out / "summary_uniprot.csv", index=False)
        if "hallucination" in b.extras:
            from .af_eval import hallucination_counts, pocket_plddt, summarize_hallucination
            from .evaluate import empty_predictions, read_predictions
            from .structure import load_structure

            rows = []
            for sid, ss in by.items():
                p = Path(pred_dir) / f"{sid}.csv"
                pred = read_predictions(p, sid) if p.exists() else empty_predictions(sid, missing=True)
                pl = pocket_plddt(pred, load_structure(ss[0].meta["structure_path"])) if pred.n else np.zeros(0)
                rows.append({"structure_id": sid, **hallucination_counts(pred, pl, ss)})
            hdf = pd.DataFrame(rows)
            hdf.to_csv(out / "hallucination_per_structure.csv", index=False)
            hs = summarize_hallucination(hdf, g30, n_boot, seed)
            hs.to_csv(out / "hallucination_summary.csv", index=False)
            summ = pd.concat([summ, hs], ignore_index=True)
            summ.to_csv(out / "summary.csv", index=False)
        s = summ.set_index("metric")
        n_missing = int(s.loc["n_pred_missing", "value"])
    else:
        from .evaluate import evaluate

        man = read_p2rank_ds(path)
        gts = _ligand_gt(man, b.rules, jobs)
        g30 = groups_for_structures([g.structure_id for g in gts], b.cluster_sources, "cluster30")
        gup = groups_for_structures([g.structure_id for g in gts], b.cluster_sources, "uniprot")
        ev = evaluate(gts, str(pred_dir), criteria=list(b.criteria), n_boot=n_boot, seed=seed, groups=g30)
        ev.save(out)
        ev_u = evaluate(gts, str(pred_dir), criteria=list(b.criteria), n_boot=n_boot, seed=seed, groups=gup)
        ev_u.summary.to_csv(out / "summary_uniprot.csv", index=False)
        s = ev.summary[ev.summary.average.isin(["ligand", "pooled", "count"])].drop_duplicates("metric").set_index("metric")
        n_missing = ev.info["n_pred_missing"]
    for m in b.headline:
        if m in s.index:
            headline[m] = {"value": float(s.loc[m, "value"]), "ci_low": float(s.loc[m, "ci_low"]), "ci_high": float(s.loc[m, "ci_high"])}
    info = {"benchmark": asdict(b), "gt_path": str(path), "pred_dir": str(pred_dir), "n_boot": n_boot, "seed": seed,
            "n_pred_missing": n_missing, "headline": headline}
    (out / "eval_info.json").write_text(json.dumps(info, indent=2, default=str))
    return info


def _run_dimer(b: Benchmark, data: Path, pred_dir, out: Path, n_boot: int, seed: int):
    """O6 through Track D's evaluator (scripts/eval_dimer.py: assembly input, J≥0.5 unique-site
    grouping, label-free prediction collapse); CIs recomputed here with 30 % sequence-cluster units
    over complexes (their own summary bootstraps complexes)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_dimer", ROOT / "scripts/eval_dimer.py")
    ed = importlib.util.module_from_spec(spec)
    sys.modules["eval_dimer"] = ed  # dataclasses need the module registered
    spec.loader.exec_module(ed)
    gts = ed.load_gts(Path(data))
    mode, chain_set = ("chain", "all") if b.name == "dimer_chains" else ("assembly", "primary")
    sites, units, summ_complex = ed.run(gts, mode, Path(pred_dir), "j50", chain_set, True, False, Path(data), 0)
    sites.to_csv(out / "sites.csv", index=False)
    units.to_csv(out / "units.csv", index=False)
    summ_complex.to_csv(out / "summary_eval_dimer_point.csv", index=False)
    rows = []
    for level, fname in (("cluster30", "summary.csv"), ("uniprot", "summary_uniprot.csv")):
        g = groups_for_structures(sorted(units.complex_id.unique()), b.cluster_sources, level)
        un = sorted(set(g.values()))
        pos = {u: i for i, u in enumerate(un)}
        W = _bootstrap_weights_units(len(un), n_boot, seed)
        si = sites.complex_id.map(lambda c: pos[g[c]]).to_numpy()
        ui = units.complex_id.map(lambda c: pos[g[c]]).to_numpy()
        res = []
        strata = {"all": np.ones(len(sites), bool), "interface": sites.site_class.isin(["homo_interface", "hetero_interface"]).to_numpy(),
                  "homo_interface": (sites.site_class == "homo_interface").to_numpy(), "hetero_interface": (sites.site_class == "hetero_interface").to_numpy()}
        for c in ("DCC12", "DCA4", "LDCC12"):
            for lab in ("topn", "topn2", "all"):
                x = sites[f"success_{c}_{lab}"].to_numpy(float)
                for st, mk in strata.items():
                    res.append(_ratio_row(f"recall_{c}_{lab}@{st}", np.bincount(si[mk], weights=x[mk], minlength=len(un)),
                                          np.bincount(si[mk], minlength=len(un)).astype(float), W))
        res.append(_ratio_row("precision_DCC12_topn2@all", np.bincount(ui, weights=units.tp_topn2.to_numpy(float), minlength=len(un)),
                              np.bincount(ui, weights=(units.tp_topn2 + units.fp_topn2).to_numpy(float), minlength=len(un)), W))
        res.append({"metric": "n_pred_missing", "value": float(units.pred_missing.sum()), "ci_low": np.nan, "ci_high": np.nan, "n": len(units)})
        res.append({"metric": "n_groups", "value": float(len(un)), "ci_low": np.nan, "ci_high": np.nan, "n": len(un)})
        df = pd.DataFrame(res)
        df.to_csv(out / fname, index=False)
        rows.append(df)
    return rows[0].set_index("metric"), int(units.pred_missing.sum())


def _bootstrap_weights_units(n: int, n_boot: int, seed: int):
    from .evaluate import _bootstrap_weights

    return _bootstrap_weights(n, n_boot, seed) if n_boot and n else None


def _ratio_row(name, num, den, W):
    v = num.sum() / den.sum() if den.sum() else np.nan
    lo = hi = np.nan
    if W is not None and den.sum():
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = (W @ num) / (W @ den)
        bs = bs[np.isfinite(bs)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"metric": name, "value": float(v), "ci_low": float(lo), "ci_high": float(hi), "n": int(den.sum())}
