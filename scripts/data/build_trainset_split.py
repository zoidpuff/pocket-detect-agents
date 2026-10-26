#!/usr/bin/env python
"""Step 3+4: leakage removal against the frozen test union, 30% clustering, seeded 90/10
train/dev split by cluster, and train-vs-dev leakage removal. Re-runnable (e.g. after the dimer
set is added to the test union): it always reads data/manifests/test_union.tsv and records its sha256.

Rule (DESIGN §9): remove a training chain if any MMseqs2 hit to the test union has E < 1e-3 AND
identity >= 30% AND (coverage >= 50% of either chain OR >= 80 aligned residues). Additionally remove
chains whose PDB ID or any UniProt accession (BioLiP + SIFTS) occurs in any test entry.
Clustering: mmseqs easy-cluster --min-seq-id 0.3 -c 0.5 (cov-mode 0 = both sequences) on surviving
unique sequences; clusters sharing a UniProt accession are merged (union-find) into 'split groups' so
that no UniProt is split across train/dev. Split groups are shuffled with
numpy.random.default_rng(SEEDS[0]); the first 10% (by number of groups) form dev. Dev chains with a
rule hit to any train chain are then removed from dev.

Outputs (data/processed/train/):
  split_chains.parquet / .tsv   every candidate chain with removal reasons, cluster, group, split
  train_chains.txt, dev_chains.txt  final chain lists (pdb_chain)
  split_counts.json
  work/test_hits.m8, work/dev_vs_train.m8, work/clu30_cluster.tsv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.datasets.leakage import MMSEQS, RULE, apply_rule, mmseqs_search  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402

MAN = ROOT / "data" / "manifests"
OUT = ROOT / "data" / "processed" / "train"
WORK = OUT / "work"
DEV_FRAC = 0.10


def write_fasta(path, items):
    with open(path, "w") as fh:
        for k, s in items:
            fh.write(f">{k}\n{s}\n")


def union_find(n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return find, union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--tmp", default="/tmp/pocketdetect_trainset_mmseqs")
    ap.add_argument("--reuse-test-search", action="store_true", help="reuse work/test_hits.m8 if test union sha unchanged")
    args = ap.parse_args()
    tmp = Path(args.tmp)
    c = {"rule": RULE, "seed": SEEDS[0], "dev_frac": DEV_FRAC}

    tu_path = MAN / "test_union.tsv"
    tu_sha = hashlib.sha256(tu_path.read_bytes()).hexdigest()
    c["test_union_sha256"] = tu_sha
    tu = pd.read_csv(tu_path, sep="\t", dtype=str)
    test_seqs = sorted(set(tu.sequence))
    tsid = {s: f"t{i:06d}" for i, s in enumerate(test_seqs)}
    write_fasta(WORK / "test_union_unique.fasta", ((tsid[s], s) for s in test_seqs))
    test_pdb = set((MAN / "test_union_pdb_ids.txt").read_text().split())
    test_unp = set((MAN / "test_union_uniprots.txt").read_text().split())

    ch = pd.read_parquet(WORK / "candidate_chains.parquet")
    c["n_candidate_chains"] = len(ch)
    c["n_candidate_unique_seqs"] = int(ch.seq_id.nunique())

    # ---- (a) sequence leakage vs test union
    m8p = WORK / "test_hits.m8"
    shap = WORK / "test_hits.sha256"
    if args.reuse_test_search and m8p.exists() and shap.exists() and shap.read_text().strip() == tu_sha:
        from pocketdetect.datasets.leakage import read_m8
        m8 = read_m8(m8p)
    else:
        m8 = mmseqs_search(WORK / "candidate_seqs.fasta", WORK / "test_union_unique.fasta", m8p, tmp / "s1", threads=args.threads)
        shap.write_text(tu_sha + "\n")
    hit = m8[apply_rule(m8)]
    leak_seq = set(hit["query"])
    c["test_search_hits_total"] = len(m8)
    c["test_search_hits_rule"] = len(hit)
    c["leaking_unique_seqs"] = len(leak_seq)
    # which test sources cause the hits (for reporting)
    src_of = tu.assign(t=tu.sequence.map(tsid)).groupby("t").source_set.agg(lambda x: ",".join(sorted(set(x))))
    hs = hit.assign(src=hit.target.map(src_of))
    per_src = {}
    for s in sorted(set(tu.source_set)):
        per_src[s] = int(hs[hs.src.str.contains(s, regex=False)]["query"].nunique())
    c["leaking_unique_seqs_by_test_source"] = per_src

    # canonical (full deposited entity) sequences are searched as well: a hit with either removes
    m8c = mmseqs_search(WORK / "candidate_can_seqs.fasta", WORK / "test_union_unique.fasta", WORK / "test_hits_canonical.m8",
                        tmp / "s1c", threads=args.threads)
    leak_can = set(m8c.loc[apply_rule(m8c), "query"])
    c["leaking_unique_canonical_seqs"] = len(leak_can)
    ch["rm_seq_observed"] = ch.seq_id.isin(leak_seq)
    ch["rm_seq_canonical"] = ch.can_id.isin(leak_can)
    c["removed_only_by_canonical"] = int((ch.rm_seq_canonical & ~ch.rm_seq_observed).sum())
    ch["rm_seq"] = ch.rm_seq_observed | ch.rm_seq_canonical
    ch["rm_pdb"] = ch.pdb.isin(test_pdb)
    ch["rm_uniprot"] = [bool(set(u.split(",")) & test_unp) if u else False for u in ch.uniprot_all]
    ch["removed_test"] = ch.rm_seq | ch.rm_pdb | ch.rm_uniprot
    c["removed_chains"] = dict(seq=int(ch.rm_seq.sum()), pdb=int(ch.rm_pdb.sum()), uniprot=int(ch.rm_uniprot.sum()),
                               any=int(ch.removed_test.sum()), frac_any=float(ch.removed_test.mean()),
                               only_pdb_or_uniprot=int((ch.removed_test & ~ch.rm_seq).sum()))
    c["removed_uniprots_frac"] = float(1 - ch.loc[~ch.removed_test & (ch.uniprot != ""), "uniprot"].nunique()
                                       / max(1, ch.loc[ch.uniprot != "", "uniprot"].nunique()))
    surv = ch[~ch.removed_test].copy()

    # ---- (b) 30% clustering of surviving unique sequences
    useqs = surv.drop_duplicates("seq_id")[["seq_id", "seq"]].sort_values("seq_id")
    write_fasta(WORK / "surv_seqs.fasta", zip(useqs.seq_id, useqs.seq))
    pref = WORK / "clu30"
    (tmp / "c1").mkdir(parents=True, exist_ok=True)
    subprocess.run([str(MMSEQS), "easy-cluster", str(WORK / "surv_seqs.fasta"), str(pref), str(tmp / "c1"),
                    "--min-seq-id", "0.3", "-c", "0.5", "--cov-mode", "0", "-s", "7.5", "--threads", str(args.threads),
                    "-v", "1"], check=True)
    cl = pd.read_csv(f"{pref}_cluster.tsv", sep="\t", header=None, names=["rep", "member"], dtype=str)
    clus = dict(zip(cl.member, cl.rep))
    surv["cluster30"] = surv.seq_id.map(clus)
    assert surv.cluster30.notna().all()
    c["n_clusters30"] = int(surv.cluster30.nunique())
    # clusters at removal level for reporting the removed fraction of clusters: cluster ALL candidates too
    (tmp / "c2").mkdir(parents=True, exist_ok=True)
    subprocess.run([str(MMSEQS), "easy-cluster", str(WORK / "candidate_seqs.fasta"), str(WORK / "clu30_all"), str(tmp / "c2"),
                    "--min-seq-id", "0.3", "-c", "0.5", "--cov-mode", "0", "-s", "7.5", "--threads", str(args.threads),
                    "-v", "1"], check=True)
    cla = pd.read_csv(WORK / "clu30_all_cluster.tsv", sep="\t", header=None, names=["rep", "member"], dtype=str)
    ch["cluster30_all"] = ch.seq_id.map(dict(zip(cla.member, cla.rep)))
    kept_cl = ch.groupby("cluster30_all").removed_test.all()
    c["candidate_clusters30_all"] = int(len(kept_cl))
    c["candidate_clusters30_fully_removed"] = int(kept_cl.sum())
    c["candidate_clusters30_frac_fully_removed"] = float(kept_cl.mean())
    c["candidate_clusters30_frac_touched"] = float(ch.groupby("cluster30_all").removed_test.any().mean())
    if kept_cl.mean() > 0.5:
        print("WARNING: > 50% of clusters removed (DESIGN §9 fallback: proceed and report)")

    # ---- (c) split groups = clusters merged through shared UniProt
    clist = sorted(surv.cluster30.unique())
    cidx = {k: i for i, k in enumerate(clist)}
    find, union = union_find(len(clist))
    first = {}
    for u, k in zip(surv.uniprot, surv.cluster30):
        if not u:
            continue
        if u in first:
            union(first[u], cidx[k])
        else:
            first[u] = cidx[k]
    surv["group"] = [clist[find(cidx[k])] for k in surv.cluster30]
    groups = sorted(surv.group.unique())
    c["n_split_groups"] = len(groups)
    rng = np.random.default_rng(SEEDS[0])
    perm = rng.permutation(len(groups))
    n_dev = int(round(DEV_FRAC * len(groups)))
    dev_groups = {groups[i] for i in perm[:n_dev]}
    surv["split"] = np.where(surv.group.isin(dev_groups), "dev", "train")

    # ---- (d) dev vs train leakage
    dv = surv[surv.split == "dev"].drop_duplicates("seq_id")
    tr = surv[surv.split == "train"].drop_duplicates("seq_id")

    def both(df):  # observed + canonical sequences, ids prefixed so either can be traced back
        items = [(f"o{a}", q) for a, q in zip(df.seq_id, df.seq)]
        cc = df[df.can_id != ""].drop_duplicates("can_id")
        return items + [(f"k{a}", q) for a, q in zip(cc.can_id, cc.seq_can)]
    write_fasta(WORK / "dev_seqs.fasta", both(surv[surv.split == "dev"]))
    write_fasta(WORK / "train_seqs.fasta", both(surv[surv.split == "train"]))
    m8d = mmseqs_search(WORK / "dev_seqs.fasta", WORK / "train_seqs.fasta", WORK / "dev_vs_train.m8", tmp / "s2", threads=args.threads)
    dq = set(m8d.loc[apply_rule(m8d), "query"])
    dleak = {q[1:] for q in dq if q.startswith("o")}
    dleak_can = {q[1:] for q in dq if q.startswith("k")}
    surv["rm_dev_vs_train"] = (surv.split == "dev") & (surv.seq_id.isin(dleak) | surv.can_id.isin(dleak_can))
    c["dev_chains_before_dev_vs_train"] = int((surv.split == "dev").sum())
    c["dev_unique_seqs_before"] = len(dv)
    c["dev_unique_seqs_removed_vs_train"] = len(dleak)
    c["dev_unique_canonical_seqs_removed_vs_train"] = len(dleak_can)
    c["dev_chains_removed_vs_train"] = int(surv.rm_dev_vs_train.sum())
    c["dev_groups_before"] = int(surv.loc[surv.split == "dev", "group"].nunique())
    surv.loc[surv.rm_dev_vs_train, "split"] = "removed_dev_vs_train"
    c["dev_groups_after"] = int(surv.loc[surv.split == "dev", "group"].nunique())

    ch = ch.merge(surv[["pdb", "chain", "cluster30", "group", "split", "rm_dev_vs_train"]], on=["pdb", "chain"], how="left")
    ch["split"] = ch.split.fillna("removed_test")
    ch["rm_dev_vs_train"] = ch.rm_dev_vs_train.fillna(False).astype(bool)
    ch["test_union_sha256"] = tu_sha
    cols = ["chain_id", "pdb", "chain", "resolution", "method", "seq_len", "seq_id", "uniprot", "uniprot_all", "n_lig", "n_ion",
            "rm_seq", "rm_seq_observed", "rm_seq_canonical", "can_id", "rm_pdb", "rm_uniprot", "removed_test", "cluster30_all", "cluster30", "group", "split", "rm_dev_vs_train"]
    ch[cols].to_parquet(OUT / "split_chains.parquet", index=False)
    ch[cols].to_csv(OUT / "split_chains.tsv", sep="\t", index=False)
    for s in ("train", "dev"):
        (OUT / f"{s}_chains.txt").write_text("\n".join(sorted(ch.chain_id[ch.split == s])) + "\n")
        sub = ch[ch.split == s]
        c[f"{s}_chains"] = len(sub)
        c[f"{s}_uniprots"] = int(sub.uniprot[sub.uniprot != ""].nunique())
        c[f"{s}_clusters30"] = int(sub.cluster30.nunique())
        c[f"{s}_groups"] = int(sub.group.nunique())
    json.dump(c, open(OUT / "split_counts.json", "w"), indent=1)
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
