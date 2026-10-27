"""Verifier: spot-check C1-strict ligand atoms (sites.tsv) against the mirror mmCIF of the representative
entry for placements that come from the rep entry itself. No pocketdetect imports."""
import sys

import gemmi
import numpy as np
import pandas as pd

ROOT = "/cluster/project/beltrao/gmagnus/pocket_detect"
s = pd.read_csv(f"{ROOT}/data/processed/gt/ligysis_lbs/sites.tsv", sep="\t", keep_default_na=False)
p = pd.read_csv(f"{ROOT}/data/processed/gt/ligysis_lbs/ligand_placements.tsv", sep="\t", low_memory=False)
p["entry"] = p.key.str.split("_").str[0]
p["rep"] = p.structure_id.str.split("_").str[0]
p["repch"] = p.structure_id.str.split("_").str[1]
cand = p[(p.entry == p.rep) & (p.status == "ok") & (p.chain == p.repch)]
cand = cand.merge(s[s.scored == 1][["structure_id", "site_id", "structure_path", "ligand_atoms"]], on=["structure_id", "site_id"])
cand = cand[cand.ligand_atoms != ""]
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
pick = cand.iloc[rng.choice(len(cand), 12, replace=False)]
for r in pick.itertuples():
    _, resn, lch, lnum = r.key.split("_")[:4]
    st = gemmi.read_structure(r.structure_path)
    st.setup_entities()
    atoms = []
    for ch in st[0]:
        for res in ch:
            if res.name == resn and str(res.seqid.num) == lnum and (ch.name == lch or res.subchain == lch):
                atoms += [a.pos.tolist() for a in res if a.element.name not in ("H", "D") and (a.altloc in ("\0", "A"))]
        if atoms:
            break
    L = np.array([[float(v) for v in t.split(",")] for t in r.ligand_atoms.split()])
    if not atoms:
        print(r.structure_id, r.site_id, r.key, "ligand residue not found in mmCIF")
        continue
    A = np.array(atoms)
    d = np.sqrt(((A[:, None] - L[None]) ** 2).sum(-1)).min(1)
    # site-residue proximity of the real ligand
    print(f"{r.structure_id} site {r.site_id} {r.key}: real heavy atoms {len(A)}, matched <=0.01A {int((d <= 0.01).sum())}, "
          f"max nn-dist {d.max():.3f} A; table atoms for site {len(L)}")
