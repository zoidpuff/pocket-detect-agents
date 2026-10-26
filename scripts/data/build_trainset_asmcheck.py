#!/usr/bin/env python
"""Step 5b: leakage check of assembly-1 *partner* chains (chains in the assembly that are not the
representative's UniProt). They are model input when the assembly is used, so they must obey the
same rule: no MMseqs2 rule hit to the test union, and no rule hit to any chain of the OTHER split.
Entries failing the check get their asm_* arrays removed and assembly_status
'dropped_partner_leak' in the manifest. Idempotent.
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
from pocketdetect.datasets.leakage import apply_rule, mmseqs_search  # noqa: E402

PROC = ROOT / "data" / "processed"
WORK = PROC / "train" / "work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--tmp", default="/tmp/pocketdetect_asmcheck")
    args = ap.parse_args()
    partners = {}
    k = None
    for line in open(WORK / "assembly_partner_seqs.fasta"):
        if line.startswith(">"):
            k = line[1:].strip()
        else:
            partners[k] = line.strip()
    partners = {k: s for k, s in partners.items() if len(s.replace("X", "")) >= 10}
    with open(WORK / "asm_partners.fasta", "w") as fh:
        for k, s in partners.items():
            fh.write(f">{k}\n{s}\n")
    hits = mmseqs_search(WORK / "asm_partners.fasta", WORK / "test_union_unique.fasta", WORK / "asm_partners_vs_test.m8",
                         Path(args.tmp) / "a1", threads=args.threads)
    bad_test = {q.split("|")[0] for q in hits.loc[apply_rule(hits), "query"]}
    # other split: all usable chains of the other split
    sc = pd.read_parquet(PROC / "train" / "split_chains.parquet", columns=["seq_id", "split"])
    seqs = {}
    k = None
    for line in open(WORK / "candidate_seqs.fasta"):
        if line.startswith(">"):
            k = line[1:].strip()
        else:
            seqs[k] = line.strip()
    bad_split = set()
    for split, other in (("train", "dev"), ("dev", "train")):
        ids = sorted(set(sc.seq_id[sc.split == other]))
        fa = WORK / f"asm_other_{other}.fasta"
        with open(fa, "w") as fh:
            for i in ids:
                fh.write(f">{i}\n{seqs[i]}\n")
        man = pd.read_csv(PROC / split / "manifest.csv", dtype=str, keep_default_na=False)
        eids = set(man.entry_id)
        qfa = WORK / f"asm_partners_{split}.fasta"
        with open(qfa, "w") as fh:
            for kk, s in partners.items():
                if kk.split("|")[0] in eids:
                    fh.write(f">{kk}\n{s}\n")
        h = mmseqs_search(qfa, fa, WORK / f"asm_partners_{split}_vs_{other}.m8", Path(args.tmp) / f"a_{split}", threads=args.threads)
        bad_split |= {q.split("|")[0] for q in h.loc[apply_rule(h), "query"]}
    bad = bad_test | bad_split
    n_drop = 0
    for split in ("train", "dev"):
        mp = PROC / split / "manifest.csv"
        man = pd.read_csv(mp, dtype=str, keep_default_na=False)
        for i, r in man.iterrows():
            if r.entry_id in bad and r.assembly_status == "stored":
                p = ROOT / r.npz
                with np.load(p, allow_pickle=False) as z:
                    arr = {kk: z[kk] for kk in z.files if not kk.startswith("asm_")}
                meta = json.loads(str(arr["meta_json"]))
                meta["assembly_status"] = "dropped_partner_leak"
                arr["meta_json"] = np.array(json.dumps(meta))
                np.savez_compressed(p, **arr)
                man.loc[i, "assembly_status"] = "dropped_partner_leak"
                man.loc[i, "n_asm_sites"] = "0"
                man.loc[i, "n_asm_interface_sites"] = "0"
                n_drop += 1
        man.to_csv(mp, index=False)
    rep = dict(n_partner_chains=len(partners), entries_partner_hit_test=len(bad_test), entries_partner_hit_other_split=len(bad_split),
               assemblies_dropped=n_drop)
    json.dump(rep, open(PROC / "train" / "asmcheck_counts.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
