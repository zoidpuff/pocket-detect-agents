#!/usr/bin/env python
"""Sanity check of wrapper outputs against bound ligands: for a few well-known complexes report, per output dir,
the rank of the first pocket whose center lies within 4 A of any ligand heavy atom (DCA<=4) and the DCA of rank 1.

  .venv/bin/python baselines/check_outputs.py OUTDIR [OUTDIR ...]
(outputs must contain <stem>.pockets.csv for the stems below; the runtime set contains all of them)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import gemmi

MIRROR = Path("/cluster/work/beltrao/shared/rsyncPDB/data/structures/divided/mmCIF")
CASES = {"1atp": {"ATP"}, "3ert": {"OHT"}, "1hsg": {"MK1"}, "4hhb": {"HEM"}}


def ligand_xyz(pid, names):
    st = gemmi.read_structure(str(MIRROR / pid[1:3] / f"{pid}.cif.gz"))
    xyz = [[a.pos.x, a.pos.y, a.pos.z] for ch in st[0] for r in ch if r.name in names for a in r if a.element.name != "H"]
    return np.array(xyz)


def validate(d):
    """format validation of every *.pockets.csv / *.points.csv in d; returns list of problems"""
    import csv
    import re
    bad = []
    rx = re.compile(r"^\S+:-?\d+[A-Za-z]?$")
    for f in sorted(Path(d).glob("*.pockets.csv")):
        rows = list(csv.reader(open(f)))
        if rows[0] != C.POCKET_COLS:
            bad.append(f"{f.name}: header {rows[0]}")
            continue
        ranks = [int(r[0]) for r in rows[1:]]
        if ranks != list(range(1, len(ranks) + 1)):
            bad.append(f"{f.name}: ranks not 1..n")
        for r in rows[1:]:
            if not all(np.isfinite(float(x)) for x in r[1:5]):
                bad.append(f"{f.name}: non-finite value")
            if r[5] and not all(rx.match(t) for t in r[5].split()):
                bad.append(f"{f.name}: bad residue token in rank {r[0]}")
        pf = f.with_name(f.name.replace(".pockets.", ".points."))
        if pf.exists():
            prow = list(csv.reader(open(pf)))
            if prow[0] != C.POINT_COLS:
                bad.append(f"{pf.name}: header {prow[0]}")
            elif {int(r[0]) for r in prow[1:]} - set(ranks):
                bad.append(f"{pf.name}: point rank without pocket")
    return bad


def main(dirs):
    for d in dirs:
        for b in validate(d)[:5]:
            print("FORMAT PROBLEM", d, b)
    lig = {p: ligand_xyz(p, n) for p, n in CASES.items()}
    print(f"{'outdir':45s} " + " ".join(f"{p:>16s}" for p in CASES) + "   (first rank with DCA<=4A / DCA of rank1 / #pockets)")
    for d in dirs:
        cells = []
        for p in CASES:
            f = Path(d) / f"{p}.pockets.csv"
            if not f.exists():
                cells.append(f"{'missing':>16s}")
                continue
            pk = C.read_pockets_csv(f)
            if not pk:
                cells.append(f"{'no pockets':>16s}")
                continue
            dca = [np.linalg.norm(lig[p] - np.array(q["center"]), axis=1).min() for q in pk]
            hit = next((q["rank"] for q, x in zip(pk, dca) if x <= 4.0), None)
            cells.append(f"{'r'+str(hit) if hit else 'miss':>4s}/{dca[0]:5.1f}/{len(pk):<4d}   ")
        print(f"{str(d)[-45:]:45s} " + " ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
