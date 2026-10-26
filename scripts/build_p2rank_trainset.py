#!/usr/bin/env python
"""P2Rank training dataset from OUR audit-filtered BioLiP2 train set (data/processed/train/manifest.csv).

Per entry: <out>/data/<entry_id>.pdb = protein heavy atoms of the representative chain (polymer residues, first altloc, as deposited frame)
+ every pooled non-ion site ligand as a HETATM residue (unique residue number, resname LGx, chain Z/Y; ions are NOT written, matching P2Rank's
default rules: ligands with < 5 heavy atoms (min_ligand_atoms) are ignored anyway and ions are excluded for the `small`-equivalent model),
plus <out>/train.ds (liganated-protein list, P2Rank format).  Optional --subset-frac f --seed s (random subset of entries).

    .venv/bin/python scripts/build_p2rank_trainset.py --out baselines/p2rank_ours/data [--split train|dev] [--subset-frac 0.5]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.datasets.trainset import load_entry, read_manifest  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402


def write_one(args):
    npz, out = args
    e = load_entry(npz)
    st = gemmi.read_structure(e.meta["structure_path"])
    st.setup_entities()
    model = st[0]
    for i in range(len(model) - 1, -1, -1):
        if model[i].name != e.chain:
            del model[i]
    if len(model) != 1:
        return e.entry_id, "no_chain", 0
    st.remove_ligands_and_waters()
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    st.remove_empty_chains()
    model[0].name = "A"  # representative chain only; PDB format needs a 1-char id
    used = {c.name for c in model}
    lig_chain = next(c for c in "ZYXWVU" if c not in used)
    # append ligands as a new chain of HETATM residues
    a = e.arrays
    ptr, is_ion = a["lig_ptr"], a["lig_is_ion"]
    coords = a["lig_coords"]
    lc = gemmi.Chain(lig_chain)
    n = 0
    for k in range(len(is_ion)):
        if is_ion[k]:
            continue
        xyz = coords[ptr[k]:ptr[k + 1]]
        if len(xyz) < 5:
            continue
        n += 1
        r = gemmi.Residue()
        r.name = f"L{n % 100:02d}"
        r.seqid = gemmi.SeqId(n, " ")
        r.het_flag = "H"
        for j, p in enumerate(xyz, 1):
            at = gemmi.Atom()
            at.name = f"C{j}"
            at.element = gemmi.Element("C")
            at.pos = gemmi.Position(*map(float, p))
            at.occ = 1.0
            r.add_atom(at)
        lc.add_residue(r)
    if n:
        model.add_chain(lc)
    st.write_pdb(str(Path(out) / f"{e.entry_id}.pdb"))
    return e.entry_id, "ok", n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEEDS[0])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--name", default="train")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = read_manifest(a.split)
    if a.subset_frac < 1:
        man = man.sample(frac=a.subset_frac, random_state=a.seed).sort_index()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(write_one, [(p, str(out)) for p in man.npz], chunksize=8))
    ok = [(i, n) for i, s, n in res if s == "ok" and n > 0]
    print(f"{len(man)} entries, {len(ok)} with >=1 non-ion ligand (>=5 heavy atoms); no_chain={sum(s=='no_chain' for _, s, _ in res)}; ligands total {sum(n for _, n in ok)}")
    (out / f"{a.name}.ds").write_text("# P2Rank training set built by scripts/build_p2rank_trainset.py\n\n" + "".join(f"{(out/i).resolve()}.pdb\n" for i, _ in ok))
    (out / f"{a.name}.entries.txt").write_text("\n".join(i for i, _ in ok) + "\n")


if __name__ == "__main__":
    main()
