#!/usr/bin/env python
"""Track C2 step 4: descriptive 'AF stress' set for hallucination analysis (no new labels).

From the AF test set (data/processed/af; included proteins of the LBS-comparison subset) take the
N = 300 proteins with the largest fraction of residues at pLDDT < 50 (ties broken by accession).
Labels are the existing AF-test-set sites (use build_af_io.load_af_dataset and filter by
structure_id); this script only lists the proteins and describes them against the rest of the set.

Outputs: data/processed/af/af_stress_set.csv (one row per protein, manifest columns + site stats),
         data/processed/af/af_stress_stats.json
Usage:   .venv/bin/python scripts/data/build_af_stress.py [--n 300]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AF = ROOT / "data/processed/af"


def describe(P: pd.DataFrame, S: pd.DataFrame) -> dict:
    q = lambda x: {str(k): round(float(v), 4) for k, v in x.quantile([0.1, 0.5, 0.9]).items()}  # noqa: E731
    return dict(
        n_proteins=len(P), n_sites=len(S), n_sites_no_ion=int((S.ion_prop != 1).sum()),
        frac_plddt_lt50=q(P.frac_plddt_lt50), frac_prunable=q(P.frac_prunable), length=q(P.length),
        n_pdb_entries=q(P.n_pdb_entries), sites_per_protein=round(len(S) / max(1, len(P)), 3),
        site_plddt_band=S.plddt_band.value_counts().to_dict(),
        frac_sites_near_low_plddt=round(float(S.near_low_plddt.mean()), 4) if len(S) else None,
        frac_sites_with_prunable_residue=round(float((S.frac_res_prunable > 0).mean()), 4) if len(S) else None,
        site_plddt_mean=q(S.plddt_mean),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    P = pd.read_csv(AF / "af_manifest.csv")
    S = pd.read_csv(AF / "af_sites.csv")
    P = P[(P.status == "included") & P.in_lbs_benchmark.astype(bool)].copy()
    S = S[S.include & S.acc.isin(P.acc)]
    P = P.sort_values(["frac_plddt_lt50", "acc"], ascending=[False, True]).reset_index(drop=True)
    stress = P.head(args.n).copy()
    rest = P.iloc[args.n:]
    agg = S.groupby("acc").agg(n_sites=("site_name", "size"), n_sites_bench=("in_lbs_benchmark", "sum"),
                               n_sites_near_low=("near_low_plddt", "sum"), site_plddt_mean_min=("plddt_mean", "min"),
                               n_sites_plddt_lt70=("plddt_mean", lambda x: int((x < 70).sum())))
    stress = stress.merge(agg, left_on="acc", right_index=True, how="left")
    stress.insert(0, "stress_rank", np.arange(1, len(stress) + 1))
    cols = ["stress_rank", "acc", "structure_id", "length", "frac_plddt_lt50", "frac_plddt_lt70", "frac_prunable",
            "plddt_mean", "n_pdb_entries", "n_sites", "n_sites_bench", "n_sites_near_low", "n_sites_plddt_lt70",
            "site_plddt_mean_min", "model_path", "gt_path"]
    stress[cols].to_csv(AF / "af_stress_set.csv", index=False)
    stats = dict(selection=f"top {args.n} included LBS-subset proteins by frac_plddt_lt50 (ties: accession)",
                 threshold_frac_plddt_lt50=float(stress.frac_plddt_lt50.min()),
                 stress=describe(stress, S[S.acc.isin(stress.acc)]), rest=describe(rest, S[S.acc.isin(rest.acc)]))
    (AF / "af_stress_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
