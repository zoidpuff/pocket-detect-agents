#!/usr/bin/env python
"""Write harness manifests for the interim BURROW development sets (P2Rank training sets).

CHEN11 (251) and JOINED (537 = ASTEX 78 + B210 176 + BU48 93 + DT198 190) from
data/raw/p2rank_datasets/current/{chen11,joined}.ds.  These are P2Rank's own training /
validation sets, NOT project test sets (docs/PREREG.md).  structure_id = file stem (unique).

Files with a blank chain id get an unused chain letter (otherwise the baseline wrappers fall
back to mmCIF, which fpocket cannot read and P2Rank output naming breaks).  Some (old-format) files carry sequence numbers / segment ids in columns 73-80, which gemmi
rejects ("Wrong format for charge").  Those files (only those) are rewritten with columns 73-80
blanked (element re-inferred from the atom name, as BioJava does) into
results/dev/structures_sanitized/ (git-ignored) and the manifest points to the copy.  The same
file is given to every method.
Output: results/dev/manifests/{chen11,joined,all}.csv / .list
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gemmi
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data/raw/p2rank_datasets/current"
SAN = ROOT / "results/dev/structures_sanitized"


def sanitize(p: Path) -> Path:
    blank_chain = False
    try:
        st = gemmi.read_structure(str(p))
        blank_chain = any(ch.name.strip() == "" for ch in st[0])
        if not blank_chain:
            return p
    except Exception:
        pass
    out = SAN / p.parent.name / p.name
    out.parent.mkdir(parents=True, exist_ok=True)
    text = p.read_text().splitlines()
    used = {ln[21] for ln in text if ln.startswith(("ATOM  ", "HETATM")) and len(ln) > 21}
    free = next(c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in used)
    lines = []
    for ln in text:
        if ln.startswith(("ATOM  ", "HETATM", "ANISOU", "TER")):
            ln = ln[:72]
            if len(ln) > 21 and ln[21] == " ":   # blank chain id -> unused letter
                ln = ln[:21] + free + ln[22:]
        lines.append(ln)
    out.write_text("\n".join(lines) + "\n")
    gemmi.read_structure(str(out))  # must now parse
    return out


def read_ds(ds: Path) -> pd.DataFrame:
    rows = []
    for line in ds.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("PARAM") or line.startswith("HEADER"):
            continue
        prot = line.split()[0]
        p = (ds.parent / prot).resolve()
        q = sanitize(p)
        rows.append({"structure_id": p.stem, "structure_path": str(q), "ligand_spec": "",
                     "subset": Path(prot).parts[1] if Path(prot).parts[0] == "joined" else "chen11",
                     "sanitized": int(q != p), "original_path": str(p)})
    df = pd.DataFrame(rows)
    assert df.structure_id.is_unique, ds
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/dev/manifests")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    SAN.mkdir(parents=True, exist_ok=True)
    (SAN / ".gitignore").write_text("*\n")
    dfs = []
    for name in ("chen11", "joined"):
        df = read_ds(DS / f"{name}.ds")
        df.to_csv(a.out / f"{name}.csv", index=False)
        (a.out / f"{name}.list").write_text("\n".join(df.structure_path) + "\n")
        print(name, len(df), df.subset.value_counts().to_dict(), "sanitized:", int(df.sanitized.sum()))
        dfs.append(df)
    al = pd.concat(dfs, ignore_index=True)
    al.to_csv(a.out / "all.csv", index=False)
    (a.out / "all.list").write_text("\n".join(al.structure_path) + "\n")


if __name__ == "__main__":
    main()
