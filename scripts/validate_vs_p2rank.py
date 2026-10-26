#!/usr/bin/env python
"""Cross-check our ground-truth extraction and DCA evaluation against P2Rank's own evaluator.

Run P2Rank first (outputs per-ligand cases):
  prank eval-predict <dataset>.ds -o <p2out> -log_cases 1 -visualizations 0

Then:
  python scripts/validate_vs_p2rank.py --ds data/raw/p2rank_datasets/current/coach420.ds \
      --p2rank-out <p2out> --out results/validation/p2rank_coach420 [--rules p2rank_strict]

Checks: (1) per-structure relevant-ligand counts vs cases/proteins.csv, (2) per-ligand DCA(4)
rank vs cases/ligands.csv ``dca4rank``, (3) DCA(4) top-n and top-(n+2) success rates vs
success_rates.csv. P2Rank predictions are converted to the common prediction format and
evaluated with pocketdetect.evaluate. Mismatches are written to <out>/mismatch_*.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pocketdetect.evaluate import evaluate, write_predictions
from pocketdetect.ligands import build_ground_truth


def read_ds(ds: Path) -> pd.DataFrame:
    """Parse a P2Rank .ds file into (structure_id, structure_path, ligand_spec)."""
    rows, header = [], None
    for line in ds.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("HEADER:"):
            header = line.split(":", 1)[1].split()
            continue
        parts = line.split()
        cols = dict(zip(header or ["protein"], parts))
        prot = cols["protein"]
        spec = cols.get("ligand_codes", cols.get("ligands", ""))
        if "chains" in cols and cols["chains"] != "*":
            raise SystemExit("datasets with chain columns are not supported by this validator")
        rows.append({"structure_id": Path(prot).name, "structure_path": str((ds.parent / prot).resolve()), "ligand_spec": spec})
    return pd.DataFrame(rows)


def convert_p2rank_predictions(pred_csv: Path, out_csv: Path) -> None:
    df = pd.read_csv(pred_csv, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    res = [" ".join(t.replace("_", ":", 1) for t in str(r).split()) if isinstance(r, str) else "" for r in df["residue_ids"]]
    write_predictions(out_csv, df[["center_x", "center_y", "center_z"]].to_numpy(float), df["score"].to_numpy(float), res)


def _gt(args):
    sid, path, spec, rules = args
    try:
        return build_ground_truth(path, structure_id=sid, rules=rules, spec=spec or None), None
    except Exception as e:
        return None, f"{sid}: {type(e).__name__}: {e}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", required=True, type=Path)
    ap.add_argument("--p2rank-out", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rules", default="p2rank_strict")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    man = read_ds(a.ds)

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(_gt, [(r.structure_id, r.structure_path, r.ligand_spec, a.rules) for r in man.itertuples()], chunksize=8))
    gts = [g for g, e in res if g is not None]
    errors = [e for g, e in res if e is not None]

    pred_dir = a.out / "preds"
    pred_dir.mkdir(exist_ok=True)
    for g in gts:
        p = a.p2rank_out / "predictions" / f"{g.structure_id}_predictions.csv"
        if p.exists():
            convert_p2rank_predictions(p, pred_dir / f"{g.structure_id}.csv")
    ev = evaluate(gts, pred_dir, criteria=["DCA4"], n_boot=0)

    # (1) ligand counts
    pp = pd.read_csv(a.p2rank_out / "cases" / "proteins.csv", skipinitialspace=True)
    pp.columns = [c.strip() for c in pp.columns]
    ours = pd.DataFrame({"structure_id": [g.structure_id for g in gts], "n_ours": [g.n_ligands for g in gts],
                         "names_ours": [" ".join(l.name for l in g.ligands) for g in gts]})
    cmp_counts = ours.merge(pp[["name", "#ligands", "ligandNames"]].rename(columns={"name": "structure_id", "#ligands": "n_p2rank"}), on="structure_id", how="outer")
    mism_counts = cmp_counts[cmp_counts.n_ours != cmp_counts.n_p2rank]
    mism_counts.to_csv(a.out / "mismatch_counts.csv", index=False)

    # (2) per-ligand DCA4 rank
    pl = pd.read_csv(a.p2rank_out / "cases" / "ligands.csv", skipinitialspace=True)
    pl.columns = [c.strip() for c in pl.columns]
    # P2Rank reports BioJava-internal chain ids (not author ids) -> match on file, residue names
    # and residue numbers only (duplicates within a file are disambiguated by occurrence order)
    def canon(f, names, codes):
        return f"{f}|{'&'.join(sorted(set(str(names).split('&'))))}|{'&'.join(sorted(str(codes).split('&')))}"

    pl["key"] = [canon(f, n, code) for f, n, code in zip(pl["file"], pl["ligand"], pl["ligCode"].astype(str))]
    pl["key"] = pl["key"] + "#" + pl.groupby("key").cumcount().astype(str)
    L = ev.ligands.copy()

    def key_of(row):
        groups = row.ligand_id.split("|", 1)[1].split("&")
        return canon(row.structure_id, row.ligand_name, "&".join(g.rpartition(":")[2] for g in groups))

    L["key"] = [key_of(r) for r in L.itertuples()]
    L["key"] = L["key"] + "#" + L.groupby("key").cumcount().astype(str)
    m = L[["key", "rank_DCA4", "dca_min"]].merge(pl[["key", "dca4rank", "closestPocketDist"]], on="key", how="outer", indicator=True)
    lig_only_ours = int((m._merge == "left_only").sum())
    lig_only_p2 = int((m._merge == "right_only").sum())
    both = m[m._merge == "both"]
    rank_mism = both[both.rank_DCA4.astype(int) != both.dca4rank.astype(int)]
    m[(m._merge != "both")].to_csv(a.out / "mismatch_ligands_unpaired.csv", index=False)
    rank_mism.to_csv(a.out / "mismatch_ligands_rank.csv", index=False)

    # (3) success rates
    sr = pd.read_csv(a.p2rank_out / "success_rates.csv", header=None)
    hdr = [str(x).strip() for x in sr.iloc[0]]
    row = sr[sr[0].astype(str).str.strip() == "DCA(4.0)"].iloc[0]
    p2_rates = {hdr[i]: float(row[i]) / 100 for i in range(1, len(hdr))}
    s = ev.summary.set_index(["metric", "average"])["value"]
    report = {
        "dataset": str(a.ds),
        "rules": a.rules,
        "n_structures": len(man),
        "gt_errors": errors,
        "n_count_mismatch": int(len(mism_counts)),
        "n_ligands_ours": int(len(L)),
        "n_ligands_p2rank": int(len(pl)),
        "n_ligands_only_ours": lig_only_ours,
        "n_ligands_only_p2rank": lig_only_p2,
        "n_rank_mismatch": int(len(rank_mism)),
        "DCA4_topn_ours": float(s[("success_DCA4_topn", "ligand")]),
        "DCA4_topn_p2rank": p2_rates.get("[0]"),
        "DCA4_topn2_ours": float(s[("success_DCA4_topn2", "ligand")]),
        "DCA4_topn2_p2rank": p2_rates.get("[2]"),
    }
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
