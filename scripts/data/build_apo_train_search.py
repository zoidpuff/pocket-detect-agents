#!/usr/bin/env python
"""Track A2 step 1: candidate apo chains for every train/dev entry.

Query  = observed representative sequence (npz `rep_seq`) of every entry in the (audit-filtered)
         data/processed/{train,dev}/manifest.csv.
Target = every protein chain of the local PDB mirror (derived_data/pdb_seqres.txt, mol:protein,
         unique sequences).
MMseqs2 easy-search (-s 5.7, --min-seq-id 0.95, -c 0.9 --cov-mode 2 = query coverage); kept hits:
fident >= 0.95 and qcov >= 0.90. Expanded to (pdb, chain), then filtered: wwPDB method diffraction/EM,
0 < resolution <= 3.0 A (derived_data/index/resolu.idx), PDB ID not in any test set (builder
test_union_pdb_ids.txt + audit idmaps), not the representative's own entry, not one of the entry's
pooled holo source entries.

Output: data/processed/apo_train/candidates.parquet, search_counts.json
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
from pocketdetect.datasets.leakage import mmseqs_search_protein  # noqa: E402

DD = Path("/cluster/work/beltrao/shared/rsyncPDB/derived_data")
PROC = ROOT / "data" / "processed"
OUT = PROC / "apo_train"
W = OUT / "work"


def test_pdb_ids() -> set:
    ids = set((ROOT / "data" / "manifests" / "test_union_pdb_ids.txt").read_text().split())
    idm = ROOT / "results" / "audit" / "idmaps.json"
    if idm.exists():
        d = json.load(open(idm))
        for vals in d.get("test", {}).values():
            for v in vals:
                f = v.split("|")
                if len(f) > 1 and len(f[1]) == 4:
                    ids.add(f[1].lower())
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--tmp", default="/tmp/pocketdetect_apo_search")
    args = ap.parse_args()
    W.mkdir(parents=True, exist_ok=True)
    c = {}
    # ---- queries
    rows = []
    for split in ("train", "dev"):
        m = pd.read_csv(PROC / split / "manifest.csv", dtype=str, keep_default_na=False)
        for r in m.itertuples():
            with np.load(ROOT / r.npz, allow_pickle=False) as z:
                seq = str(z["rep_seq"])
                srcs = sorted({str(x).rsplit("_", 1)[0] for x in np.unique(z["lig_source"])})
            rows.append(dict(entry_id=r.entry_id, split=split, rep_pdb=r.pdb, rep_chain=r.chain, rep_seq=seq,
                             source_pdbs=",".join(srcs)))
    ents = pd.DataFrame(rows)
    c["entries"] = ents.split.value_counts().to_dict()
    with open(W / "queries.fasta", "w") as fh:
        for r in ents.itertuples():
            fh.write(f">{r.entry_id}\n{r.rep_seq}\n")
    # ---- PDB protein chains
    recs, pid = [], None
    with open(DD / "pdb_seqres.txt") as fh:
        for line in fh:
            if line.startswith(">"):
                h = line[1:].split()
                pid = h[0] if len(h) > 1 and h[1] == "mol:protein" else None
            elif pid:
                p, ch = pid.split("_", 1)
                recs.append((p.lower(), ch, line.strip()))
    pdbc = pd.DataFrame(recs, columns=["pdb", "chain", "seq"])
    pdbc = pdbc[pdbc.seq.str.len() >= 30]
    useq = sorted(set(pdbc.seq))
    sid = {s: f"p{i:07d}" for i, s in enumerate(useq)}
    pdbc["seq_id"] = pdbc.seq.map(sid)
    with open(W / "pdb_protein_unique.fasta", "w") as fh:
        for s in useq:
            fh.write(f">{sid[s]}\n{s}\n")
    c["pdb_protein_chains"] = len(pdbc)
    c["pdb_protein_unique_seqs"] = len(useq)
    # ---- search
    cols = "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue"
    mmseqs_search_protein(W / "queries.fasta", W / "pdb_protein_unique.fasta", W / "q_vs_pdb.m8", Path(args.tmp), threads=args.threads,
                          extra=["--min-seq-id", "0.95", "-c", "0.9", "--cov-mode", "2", "--max-seqs", "20000"], columns=cols.split(","))
    h = pd.read_csv(W / "q_vs_pdb.m8", sep="\t", header=None, names=cols.split(","), dtype={"query": str, "target": str})
    h = h[(h.fident >= 0.95) & (h.qcov >= 0.90)]
    c["seq_hits"] = len(h)
    cand = h.merge(pdbc[["pdb", "chain", "seq_id"]], left_on="target", right_on="seq_id").drop(columns=["seq_id"])
    cand = cand.rename(columns={"query": "entry_id"}).merge(ents[["entry_id", "split", "rep_pdb", "source_pdbs"]], on="entry_id")
    c["candidate_chains_raw"] = len(cand)
    # method + resolution
    et = pd.read_csv(DD / "pdb_entry_type.txt", sep="\t", header=None, names=["pdb", "poly", "method"], dtype=str)
    res = {}
    with open(DD / "index" / "resolu.idx") as fh:
        for line in fh:
            f = [x.strip() for x in line.split(";")]
            if len(f) == 2 and len(f[0]) == 4:
                try:
                    res[f[0].lower()] = float(f[1])
                except ValueError:
                    pass
    cand["method"] = cand.pdb.map(dict(zip(et.pdb, et.method))).fillna("unknown")
    cand["resolution"] = cand.pdb.map(res)
    ok = cand.method.isin(["diffraction", "EM"]) & (cand.resolution > 0) & (cand.resolution <= 3.0)
    c["fail_method_or_resolution"] = int((~ok).sum())
    cand = cand[ok]
    tp = test_pdb_ids()
    c["n_test_pdb_ids"] = len(tp)
    is_test = cand.pdb.isin(tp)
    is_self = cand.pdb == cand.rep_pdb
    is_src = [p in s.split(",") for p, s in zip(cand.pdb, cand.source_pdbs)]
    c["excluded_test_pdb"] = int(is_test.sum())
    c["excluded_own_entry"] = int(is_self.sum())
    c["excluded_pooled_source_entry"] = int(np.sum(is_src))
    cand = cand[~is_test & ~is_self & ~np.array(is_src, bool)]
    cand = cand.drop(columns=["source_pdbs"]).sort_values(["entry_id", "resolution", "pdb", "chain"])
    cand.to_parquet(OUT / "candidates.parquet", index=False)
    c["candidate_chains"] = len(cand)
    c["entries_with_candidates"] = cand.groupby("split").entry_id.nunique().to_dict()
    c["candidate_entries_pdb"] = int(cand.pdb.nunique())
    json.dump(c, open(OUT / "search_counts.json", "w"), indent=1)
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
