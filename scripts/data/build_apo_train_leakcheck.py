#!/usr/bin/env python
"""Track A2 step 3: leakage check of the selected apo chains (DESIGN §9 rule, E < 1e-3, fident >= 0.30,
qcov|tcov >= 0.5 or alnlen >= 80).

Queries per apo input: observed sequence (npz `rep_seq`) AND the deposited SEQRES of that chain
(derived_data/pdb_seqres.txt). Targets: (a) builder test union (data/manifests/test_union.tsv),
(b) audit observed-sequence test set results/audit/test_union.fasta (if present), (c) all
representatives + apo inputs of the OTHER split. Also re-checks PDB IDs against every test PDB ID.
Violating apo inputs are deleted and listed in data/processed/apo_train/leak_dropped.tsv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from build_apo_train_search import test_pdb_ids  # noqa: E402
from pocketdetect.datasets.leakage import apply_rule, mmseqs_search_protein, read_m8  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "apo_train"
W = OUT / "work"
SEQRES = Path("/cluster/work/beltrao/shared/rsyncPDB/derived_data/pdb_seqres.txt")


def write(path, items):
    with open(path, "w") as fh:
        for k, s in items:
            if len(s) >= 10:
                fh.write(f">{k}\n{s}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--tmp", default="/tmp/pocketdetect_apo_leak")
    args = ap.parse_args()
    man = {s: pd.read_csv(OUT / s / "manifest.csv", dtype=str, keep_default_na=False) for s in ("train", "dev")}
    want = {(r.pdb, r.chain) for m in man.values() for r in m.itertuples()}
    seqres, pid = {}, None
    with open(SEQRES) as fh:
        for line in fh:
            if line.startswith(">"):
                h = line[1:].split()[0]
                p, c = h.split("_", 1)
                pid = (p.lower(), c) if (p.lower(), c) in want else None
            elif pid:
                seqres[pid] = line.strip()
    q = {s: [] for s in man}
    for s, m in man.items():
        for r in m.itertuples():
            with np.load(ROOT / r.npz, allow_pickle=False) as z:
                q[s].append((f"{r.entry_id}|obs", str(z["rep_seq"])))
            if (r.pdb, r.chain) in seqres:
                q[s].append((f"{r.entry_id}|seqres", seqres[(r.pdb, r.chain)]))
    write(W / "apo_all.fasta", q["train"] + q["dev"])
    # targets
    tu = pd.read_csv(ROOT / "data" / "manifests" / "test_union.tsv", sep="\t", dtype=str)
    write(W / "tu.fasta", [(f"t{i}", x) for i, x in enumerate(sorted(set(tu.sequence)))])
    bad = {}
    m8 = W / "apo_vs_testunion.m8"
    mmseqs_search_protein(W / "apo_all.fasta", W / "tu.fasta", m8, Path(args.tmp) / "a", threads=args.threads,
                          extra=["-s", "7.5", "--max-seqs", "5000", "-e", "1e-3"])
    h = read_m8(m8)
    for qq in h.loc[apply_rule(h), "query"]:
        bad.setdefault(qq.split("|")[0], set()).add("test_union")
    audit = ROOT / "results" / "audit" / "test_union.fasta"
    if audit.exists():
        m8 = W / "apo_vs_audit.m8"
        mmseqs_search_protein(W / "apo_all.fasta", audit, m8, Path(args.tmp) / "b", threads=args.threads,
                              extra=["-s", "7.5", "--max-seqs", "5000", "-e", "1e-3"])
        h = read_m8(m8)
        for qq in h.loc[apply_rule(h), "query"]:
            bad.setdefault(qq.split("|")[0], set()).add("audit_test_set")
    # other split: representatives (observed) + apo inputs of the other split
    for s, other in (("train", "dev"), ("dev", "train")):
        tm = pd.read_csv(PROC / other / "manifest.csv", dtype=str, keep_default_na=False)
        tgt = []
        for r in tm.itertuples():
            with np.load(ROOT / r.npz, allow_pickle=False) as z:
                tgt.append((f"rep|{r.entry_id}", str(z["rep_seq"])))
        tgt += q[other]
        write(W / f"other_{other}.fasta", tgt)
        write(W / f"apo_{s}.fasta", q[s])
        m8 = W / f"apo_{s}_vs_{other}.m8"
        mmseqs_search_protein(W / f"apo_{s}.fasta", W / f"other_{other}.fasta", m8, Path(args.tmp) / f"c{s}", threads=args.threads,
                              extra=["-s", "7.5", "--max-seqs", "5000", "-e", "1e-3"])
        h = read_m8(m8)
        for qq in h.loc[apply_rule(h), "query"]:
            bad.setdefault(qq.split("|")[0], set()).add(f"other_split_{other}")
    tp = test_pdb_ids()
    rows = []
    for s, m in man.items():
        for r in m.itertuples():
            if r.pdb in tp:
                bad.setdefault(r.entry_id, set()).add("test_pdb_id")
        drop = m.entry_id.isin(bad)
        for r in m[drop].itertuples():
            rows.append(dict(entry_id=r.entry_id, split=s, reasons=",".join(sorted(bad[r.entry_id]))))
            (ROOT / r.npz).unlink(missing_ok=True)
        m[~drop].to_csv(OUT / s / "manifest.csv", index=False)
    pd.DataFrame(rows, columns=["entry_id", "split", "reasons"]).to_csv(OUT / "leak_dropped.tsv", sep="\t", index=False)
    rep = dict(apo_inputs_checked={s: len(m) for s, m in man.items()}, dropped=len(rows),
               dropped_by_reason=pd.Series([x for r in rows for x in r["reasons"].split(",")]).value_counts().to_dict() if rows else {},
               audit_test_set_used=audit.exists())
    json.dump(rep, open(OUT / "leakcheck_counts.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
