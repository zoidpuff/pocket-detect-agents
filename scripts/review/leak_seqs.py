#!/usr/bin/env python
"""Reviewer leakage check: write FASTA (and single-chain PDBs for Foldseek) of
 - dev evaluation inputs (results/dev/biolip/inputs/<id>.cif, 579 audited chains)
 - train representative chains (snapshot manifest structure_path + chain)
 - every protein chain of stored train assembly-1 inputs (npz asm_path)
Outputs to results/review_dev/leak/.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/review_dev/leak"
SNAP = ROOT / "results/dev/cache/trainset_snapshot/audited_d6"


def chain_seqs(path, chain=None, tag="", pdb_dir=None):
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_ligands_and_waters()
    out = []
    for ch in st[0]:
        if chain is not None and ch.name != chain:
            continue
        pol = ch.get_polymer()
        seq = gemmi.one_letter_code([r.name for r in pol])
        if pol.check_polymer_type() not in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD) or len(seq) < 10:
            continue
        out.append((f"{tag}|{ch.name}", seq.upper()))
    if pdb_dir is not None and out:
        st2 = gemmi.Structure()
        st2.add_model(gemmi.Model("1"))
        for ch in st[0]:
            if chain is None or ch.name == chain:
                st2[0].add_chain(ch)
        st2.remove_empty_chains()
        st2.setup_entities()
        try:
            st2.write_pdb(str(Path(pdb_dir) / f"{tag.replace('|','_')}.pdb"))
        except Exception:
            st2.shorten_chain_names() if hasattr(st2, "shorten_chain_names") else None
    return out


def job(args):
    kind, path, chain, tag = args
    try:
        pdb_dir = OUT / f"pdb_{kind}" if kind in ("dev", "train") else None
        return kind, chain_seqs(path, chain, tag, pdb_dir), None
    except Exception as e:
        return kind, [], f"{tag}: {e}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for k in ("dev", "train"):
        (OUT / f"pdb_{k}").mkdir(exist_ok=True)
    dev_ids = [Path(l.strip()).stem for l in open(ROOT / "results/dev/biolip/inputs_audited.list") if l.strip()]
    jobs = [("dev", ROOT / "results/dev/biolip/inputs" / f"{d}.cif", None, d) for d in dev_ids]
    tr = pd.read_csv(SNAP / "train/manifest.csv")
    jobs += [("train", r.structure_path, r.chain, r.entry_id) for r in tr.itertuples()]
    for r in tr[tr.assembly_status == "stored"].itertuples():
        z = np.load(SNAP / "train/entries" / Path(r.npz).name, allow_pickle=True)
        jobs.append(("asm", str(z["asm_path"]), None, r.entry_id + "__asm1"))
    seqs = {"dev": [], "train": [], "asm": []}
    errs = []
    with ProcessPoolExecutor(12) as ex:
        for kind, s, e in ex.map(job, jobs, chunksize=8):
            seqs[kind] += s
            if e:
                errs.append(e)
    for k, v in seqs.items():
        with open(OUT / f"{k}.fasta", "w") as f:
            for n, s in v:
                f.write(f">{n}\n{s}\n")
        print(k, len(v))
    (OUT / "errors.txt").write_text("\n".join(errs))
    print("errors", len(errs))


if __name__ == "__main__":
    main()
