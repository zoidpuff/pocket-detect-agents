#!/usr/bin/env python
"""Step 5c: final leakage pass on the *deployed inputs*. The representative chain's observed sequence
is re-extracted from the structure file (pocketdetect.structure.load_structure, protein residues)
— this can differ slightly from the BioLiP receptor sequence used in step 3 (modified residues,
residues BioLiP omits), which can flip borderline hits. Any entry whose representative has a rule
hit to the test union (or, for dev, to any train representative) is dropped from the manifests and
its npz deleted. Dropped entries are listed in data/processed/train/final_dropped.tsv.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.datasets.leakage import apply_rule, mmseqs_search  # noqa: E402
from pocketdetect.structure import load_structure, pdb_entry_path  # noqa: E402

PROC = ROOT / "data" / "processed"
WORK = PROC / "train" / "work"


def _seq(args):
    eid, pdb, chain = args
    s = load_structure(pdb_entry_path(pdb), chains=[chain], protein_only=True)
    out = []
    for n in s.r_name:
        info = gemmi.find_tabulated_residue(n)
        c = info.one_letter_code.upper() if info is not None else "X"
        out.append(c if c.isalpha() else "X")
    return eid, "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--tmp", default="/tmp/pocketdetect_finalcheck")
    args = ap.parse_args()
    mans = {s: pd.read_csv(PROC / s / "manifest.csv", dtype=str, keep_default_na=False) for s in ("train", "dev")}
    jobs = [(r.entry_id, r.pdb, r.chain) for m in mans.values() for r in m.itertuples()]
    # train pooled source chains (label donors) are dev-vs-train targets too
    import numpy as np
    srcs = set()
    for r in mans["train"].itertuples():
        with np.load(ROOT / r.npz, allow_pickle=False) as z:
            srcs.update(str(x) for x in np.unique(z["lig_source"]))
    jobs += [(f"src|{x}", *x.rsplit("_", 1)) for x in sorted(srcs)]
    with ProcessPoolExecutor(args.procs) as ex:
        seqs = dict(ex.map(_seq, jobs, chunksize=8))
    for s, m in mans.items():
        with open(WORK / f"final_{s}_obs.fasta", "w") as fh:
            for e in m.entry_id:
                if len(seqs[e]) >= 10:
                    fh.write(f">{e}\n{seqs[e]}\n")
    with open(WORK / "final_train_obs.fasta", "a") as fh:
        for k, q in seqs.items():
            if k.startswith("src|") and len(q) >= 10:
                fh.write(f">{k}\n{q}\n")
    allfa = WORK / "final_all_obs.fasta"
    allfa.write_text((WORK / "final_train_obs.fasta").read_text() + (WORK / "final_dev_obs.fasta").read_text())
    h = mmseqs_search(allfa, WORK / "test_union_unique.fasta", WORK / "final_vs_test.m8", Path(args.tmp) / "f1", threads=args.procs)
    bad_test = {q for q in h.loc[apply_rule(h), "query"] if not q.startswith("src|")}
    src_test_hits = sorted({q for q in h.loc[apply_rule(h), "query"] if q.startswith("src|")})
    hd = mmseqs_search(WORK / "final_dev_obs.fasta", WORK / "final_train_obs.fasta", WORK / "final_dev_vs_train.m8",
                       Path(args.tmp) / "f2", threads=args.procs)
    bad_dev = set(hd.loc[apply_rule(hd), "query"])
    rows = []
    for s, m in mans.items():
        drop = m.entry_id.isin(bad_test | (bad_dev if s == "dev" else set()))
        for r in m[drop].itertuples():
            rows.append(dict(entry_id=r.entry_id, split=s, reason="test_union_hit" if r.entry_id in bad_test else "dev_vs_train_hit"))
            (ROOT / r.npz).unlink(missing_ok=True)
        m[~drop].to_csv(PROC / s / "manifest.csv", index=False)
    pd.DataFrame(rows, columns=["entry_id", "split", "reason"]).to_csv(PROC / "train" / "final_dropped.tsv", sep="\t", index=False)
    rep = dict(n_checked=len(jobs), train_source_chains=len(srcs), source_chains_hitting_test=src_test_hits, dropped_test=len(bad_test), dropped_dev_vs_train=len(bad_dev - bad_test),
               remaining={s: int((~m.entry_id.isin(bad_test | (bad_dev if s == "dev" else set()))).sum()) for s, m in mans.items()})
    json.dump(rep, open(PROC / "train" / "finalcheck_counts.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
