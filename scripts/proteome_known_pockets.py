#!/usr/bin/env python
"""Map HOTPocket's BioLiP2-derived known pockets of the human proteome to UniProt numbering.

HOTPocket ``known_pockets.csv`` lists binding-site residues in BioLiP's "reindexed" numbering, i.e.
1-based positions in the PDB chain sequence stored in ``BioLiP.txt``.  AlphaFold DB models are
numbered by the UniProt canonical sequence, so each chain sequence is aligned (local alignment) to the
UniProt sequence from HOTPocket's UP000005640 FASTA and positions are mapped through the alignment.
Sites are deduplicated per protein (greedy, residue Jaccard >= 0.5 -> same site).

    python scripts/proteome_known_pockets.py --out data/processed/proteome_human/known_pockets_uniprot.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
HP = ROOT / "data/raw/hotpocket"
KNOWN = HP / "extracted/all_pocket_preds/all_pocket_preds/known_pockets.csv"
BIOLIP = HP / "zenodo_v2/BioLiP.txt"
FASTA = HP / "extracted/uniprot/uniprot/UP000005640_9606.fasta"
IONS = {"ZN", "CA", "MG", "MN", "FE", "FE2", "CU", "CU1", "NA", "K", "CL", "CO", "NI", "CD", "HG", "SR", "BA",
        "IOD", "BR", "LI", "RB", "CS", "AG", "AU", "PT", "PB", "YB", "GD", "TB", "SM", "EU", "LA", "F", "3CO", "MN3",
        "FE3", "AL", "GA", "TL", "W", "MO", "V", "CR", "SO4", "PO4"}


def category(lig):
    lig = str(lig)
    if lig in ("rna", "dna"):
        return "nucleic"
    if lig == "peptide":
        return "peptide"
    if lig.upper() in IONS:
        return "ion"
    return "small_molecule"


@njit(cache=True)
def _sw(a, b, match=2, mismatch=-1, gap=-2):
    """Smith-Waterman (linear gaps); returns map a_pos -> b_pos (-1 unaligned)."""
    n, m = len(a), len(b)
    H = np.zeros((n + 1, m + 1), np.int32)
    T = np.zeros((n + 1, m + 1), np.int8)
    best, bi, bj = 0, 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = H[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            u = H[i - 1, j] + gap
            l = H[i, j - 1] + gap
            v = max(0, d, u, l)
            H[i, j] = v
            T[i, j] = 1 if v == d else (2 if v == u else (3 if v == l else 0))
            if v > best:
                best, bi, bj = v, i, j
    mp = np.full(n, -1, np.int64)
    i, j = bi, bj
    while i > 0 and j > 0 and H[i, j] > 0:
        t = T[i, j]
        if t == 1:
            if a[i - 1] == b[j - 1]:
                mp[i - 1] = j - 1
            i -= 1
            j -= 1
        elif t == 2:
            i -= 1
        elif t == 3:
            j -= 1
        else:
            break
    return mp


def read_fasta(path):
    seqs, cur, buf = {}, None, []
    for l in open(path):
        if l.startswith(">"):
            if cur:
                seqs[cur] = "".join(buf)
            cur, buf = l.split("|")[1], []
        else:
            buf.append(l.strip())
    if cur:
        seqs[cur] = "".join(buf)
    return seqs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/proteome_human/known_pockets_uniprot.tsv")
    a = ap.parse_args(argv)
    known = pd.read_csv(KNOWN)
    known["pdb"] = known["pdb"].astype(str).str.lower()
    want = set(zip(known.pdb, known.chain))
    seqs = {}
    with open(BIOLIP) as fh:
        for l in fh:
            f = l.rstrip("\n").split("\t")
            k = (f[0].lower(), f[1])
            if k in want and k not in seqs:
                seqs[k] = f[-1].strip()
    uni = read_fasta(FASTA)
    enc = lambda s: np.frombuffer(s.encode(), np.uint8)
    maps, rows, n_fail = {}, [], 0
    for r in known.itertuples(index=False):
        k = (r.pdb, r.chain)
        u = uni.get(r.uniprot)
        s = seqs.get(k)
        if u is None or s is None:
            n_fail += 1
            continue
        if (k, r.uniprot) not in maps:
            off = u.find(s)
            maps[(k, r.uniprot)] = (np.arange(len(s)) + off) if off >= 0 else _sw(enc(s), enc(u))
        mp = maps[(k, r.uniprot)]
        res = str(r[4]).split()        # "binding site residues, reindexed numbering"
        pos = []
        for t in res:
            try:
                i = int(t[1:]) - 1
            except ValueError:
                continue
            if 0 <= i < len(mp) and mp[i] >= 0 and u[mp[i]] == t[0]:
                pos.append(int(mp[i]) + 1)
        if len(pos) >= max(1, 0.5 * len(res)):
            rows.append((r.uniprot, category(r[5]), str(r[5]), sorted(set(pos)), len(res)))
    # deduplicate per protein
    out = []
    df = pd.DataFrame(rows, columns=["uniprot", "category", "ligand", "pos", "n_res_pdb"])
    for (u, cat), g in df.groupby(["uniprot", "category"], sort=True):
        kept = []
        for p in sorted(g.pos, key=len, reverse=True):
            sp = set(p)
            if all(len(sp & q) / len(sp | q) < 0.5 for q in kept):
                kept.append(sp)
        for i, q in enumerate(kept):
            out.append((u, f"{cat}_{i}", cat, len(q), " ".join(f"A:{x}" for x in sorted(q))))
    o = pd.DataFrame(out, columns=["uniprot", "site", "category", "n_res", "residues"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    o.to_csv(a.out, sep="\t", index=False)
    print(f"known rows {len(known)}, unmapped {n_fail}, mapped rows {len(rows)}, dedup sites {len(o)} "
          f"over {o.uniprot.nunique()} proteins; by category {o.category.value_counts().to_dict()}", file=sys.stderr)


if __name__ == "__main__":
    main()
