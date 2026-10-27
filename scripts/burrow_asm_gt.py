#!/usr/bin/env python
"""v3 dev signal: dimer-benchmark-format ground truth (scripts/eval_dimer.py) for Track A's stored
assembly-1 inputs (train and dev).  One JSON per input in results/dev/biolip/asm/gt/<split>/<id>.json.

Copies = the entry's assembly ligands (asm_lig_*; BioLiP-relevant, per entry, not pooled);
binding residues = residues with a heavy atom <= 4 Å of a ligand heavy atom; entity = identical
chain sequence (entity-level residue keys 'E<k>:<resnum><icode>'); site class (as Track D):
>= 3 contact residues on each of >= 2 chains -> homo/hetero_interface (same/different entity);
>= 2 contacting chains otherwise -> minor_interface; else single_chain.

  python scripts/burrow_asm_gt.py
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.biolip import ASM, entries  # noqa: E402
from pocketdetect.datasets.trainset import load_entry  # noqa: E402
from pocketdetect.structure import load_structure, residue_id  # noqa: E402

OUT = ROOT / "results/dev/biolip/asm/gt"
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
         "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
         "TYR": "Y", "VAL": "V", "MSE": "M"}


def _job(args):
    split, eid, npz = args
    e = load_entry(npz)
    a = e.arrays
    s = load_structure(str(a["asm_path"]), protein_only=True)
    s = s.select(s.protein_heavy_mask())
    seqs = {}
    for ch in s.chains:
        m = s.r_chain == ch
        seqs[ch] = "".join(THREE.get(r, "X") for r in s.r_name[m])
    ent_of_seq = {q: str(i + 1) for i, q in enumerate(dict.fromkeys(seqs.values()))}
    chain_ent = {c: ent_of_seq[q] for c, q in seqs.items()}
    tree = cKDTree(s.coords)
    ptr = a["asm_lig_ptr"]
    copies = []
    for k in range(len(a["asm_lig_site"])):
        xyz = a["asm_lig_coords"][ptr[k]:ptr[k + 1]].astype(np.float64)
        if len(xyz) == 0:
            continue
        atoms = np.unique(np.concatenate([np.asarray(l, np.int64) for l in tree.query_ball_point(xyz, 4.0)]))
        res = np.unique(s.res_index[atoms]) if len(atoms) else np.zeros(0, np.int64)
        rids = [residue_id(s.r_chain[r], s.r_num[r], s.r_icode[r]) for r in res]
        counts = {}
        for r in res:
            counts[str(s.r_chain[r])] = counts.get(str(s.r_chain[r]), 0) + 1
        big = [c for c, n in counts.items() if n >= 3]
        if len(big) >= 2:
            cls = "homo_interface" if len({chain_ent[c] for c in big}) == 1 else "hetero_interface"
        elif len(counts) >= 2:
            cls = "minor_interface"
        else:
            cls = "single_chain"
        ents = sorted({f"E{chain_ent[str(s.r_chain[r])]}:{s.r_num[r]}{s.r_icode[r]}" for r in res})
        copies.append(dict(ligand_id=f"{a['asm_lig_ccd'][k]}|{k}", ccd=str(a["asm_lig_ccd"][k]), n_atoms=len(xyz),
                           is_ion=bool(a["asm_lig_is_ion"][k]), center=xyz.mean(0).round(3).tolist(),
                           coords=xyz.round(3).tolist(), binding_residues=rids, chain_counts=counts,
                           entity_residues=ents, site_class=cls))
    sid = eid + ASM
    gt = dict(pdb_id=sid, chains=[dict(chain=c, entity_id=chain_ent[c]) for c in s.chains], copies=copies,
              asm_path=str(a["asm_path"]), status="ok")
    d = OUT / split
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}.json").write_text(json.dumps(gt))
    return sid, split, len(copies), sum(c["site_class"] in ("homo_interface", "hetero_interface") for c in copies)


def main():
    tasks = []
    for sp in ("train", "dev"):
        m = entries(sp)
        m = m[m.assembly_status == "stored"]
        tasks += [(sp, e, n) for e, n in zip(m.entry_id, m.npz)]
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(_job, tasks, chunksize=4))
    df = pd.DataFrame(res, columns=["structure_id", "split", "n_copies", "n_interface_copies"])
    df.to_csv(OUT / "index.csv", index=False)
    print(df.groupby("split")[["n_copies", "n_interface_copies"]].sum())


if __name__ == "__main__":
    main()
