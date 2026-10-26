"""Benchmark registry + one evaluation entry point shared by scripts/run_test_eval.py (gated, test
sets) and scripts/run_dev_eval.py (dev sets).

Each benchmark is either a *site table* (pocketdetect.sites; LIGYSIS protocol via site_eval) or a
*ligand* benchmark (P2Rank .ds file; pocketdetect.ligands + evaluate). Bootstrap resampling units
are 30 % sequence clusters of the test union (pocketdetect.clusters), UniProt as sensitivity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .clusters import groups_for_structures
from .seeds import BOOTSTRAP_SEED

ROOT = Path(__file__).resolve().parents[2]

__all__ = ["Benchmark", "REGISTRY", "read_p2rank_ds", "run_benchmark"]


@dataclass(frozen=True)
class Benchmark:
    name: str
    kind: str  # "sites" | "ligands"
    path: str  # site table (.tsv) or P2Rank dataset (.ds), relative to ROOT
    cluster_sources: tuple
    headline: tuple  # metric names reported in the ledger (first = primary endpoint)
    split: str = "test"
    rules: str = "p2rank_strict"  # ligand benchmarks
    criteria: tuple = ("DCA4", "DCC4")
    note: str = ""

    @property
    def abspath(self) -> Path:
        return ROOT / self.path


REGISTRY: dict[str, Benchmark] = {b.name: b for b in [
    Benchmark("ligysis", "sites", "data/processed/gt/ligysis_lbs/sites.tsv", ("ligysis", "ligysis_extra"),
              ("recall_topn2", "recall_ldca_topn2", "P1K", "recall_topn", "recall_max", "recall_1to1_topn2", "P1K_1to1"),
              note="PREREG C1 / C1-strict / C2"),
    Benchmark("coach420_mlig", "ligands", "data/raw/p2rank_datasets/current/coach420(mlig).ds", ("coach420",),
              ("success_DCA4_topn", "success_DCA4_topn2"), note="PREREG O2a"),
    Benchmark("holo4k_mlig", "ligands", "data/raw/p2rank_datasets/current/holo4k(mlig).ds", ("holo4k",),
              ("success_DCA4_topn", "success_DCA4_topn2"), note="PREREG O2b"),
    Benchmark("pocketwatch", "sites", "data/processed/gt/pocketwatch_test/sites.tsv", ("pocketwatch_test",),
              ("recall_topn2", "P1K"), note="PREREG O3"),
    Benchmark("cryptobench", "sites", "data/processed/gt/cryptobench_test/sites.tsv", ("cryptobench",),
              ("recall_topn2", "P1K"), note="PREREG O4"),
    Benchmark("af_set", "sites", "data/processed/gt/af_set/sites.tsv", ("af_set",), ("recall_topn2", "P1K"),
              note="PREREG O5 (site table to be provided by Track C in pocketdetect.sites format)"),
    Benchmark("dimer", "sites", "data/processed/gt/dimer/sites.tsv", ("dimer",), ("recall_topn2", "P1K"),
              note="PREREG O6 (site table to be provided by Track D; symmetric-copy collapse applied upstream)"),
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
    if b.kind == "sites":
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
