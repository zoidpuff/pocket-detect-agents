"""BioLiP2 annotation reader and ligand classification (ion / metal / excluded polymer ligands).

BioLiP2 rows = one (receptor chain, ligand instance) interaction; ligand relevance is BioLiP's own
curation (artifact/crystallisation additives are removed by BioLiP unless judged biologically
relevant). Column definitions: data/raw/biolip2/current/readme.txt.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

__all__ = ["BIOLIP_COLUMNS", "read_biolip", "read_ligand_table", "classify_ccd", "POLYMER_LIGANDS", "METALS"]

BIOLIP_COLUMNS = [
    "pdb", "chain", "resolution", "site", "ccd", "lig_chain", "lig_serial", "bres", "bres_renum",
    "cat", "cat_renum", "ec", "go", "aff_manual", "aff_moad", "aff_pdbbind", "aff_bindingdb",
    "uniprot", "pubmed", "lig_resnum", "seq",
]

# BioLiP pseudo-CCD codes for polymer ligands; excluded as labels for now (DESIGN §2.5 / TRAINSET.md)
POLYMER_LIGANDS = frozenset({"peptide", "rna", "dna"})

METALS = frozenset(
    "LI BE NA MG AL K CA SC TI V CR MN FE CO NI CU ZN GA RB SR Y ZR NB MO TC RU RH PD AG CD IN SN "
    "CS BA LA CE PR ND SM EU GD TB DY HO ER TM YB LU HF TA W RE OS IR PT AU HG TL PB BI U PU AM".split()
)


def read_biolip(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=BIOLIP_COLUMNS, dtype=str, keep_default_na=False,
                     quoting=3, engine="c")
    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")
    df["lig_resnum"] = df["lig_resnum"].str.strip()
    df["seq"] = df["seq"].str.strip()
    return df


def read_ligand_table(path: str | Path) -> pd.DataFrame:
    import gzip

    rows = []
    with gzip.open(path, "rt") as fh:  # some rows have extra tab-separated fields: parse by hand
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            rows.append((f[0], f[1] if len(f) > 1 else "", f[5] if len(f) > 5 else ""))
    return pd.DataFrame(rows, columns=["ccd", "formula", "name"])


_TOKEN = re.compile(r"^([A-Za-z]{1,2})(\d*)$")


def classify_ccd(formula: str) -> tuple[int, bool, bool]:
    """From a CCD formula like 'C10 H16 N5 O13 P3' or 'Zn' return
    (heavy_atom_count, is_ion, is_metal). ``is_ion`` = exactly one heavy atom (metal or other
    monoatomic ion, also NH4-like); ``is_metal`` = that single heavy atom is a metal element."""
    heavy, elems = 0, []
    for tok in str(formula).split():
        m = _TOKEN.match(tok.strip())
        if not m:
            continue
        el, n = m.group(1).upper(), int(m.group(2) or 1)
        if el in ("H", "D"):
            continue
        heavy += n
        elems.append(el)
    is_ion = heavy == 1
    return heavy, is_ion, bool(is_ion and elems and elems[0] in METALS)
