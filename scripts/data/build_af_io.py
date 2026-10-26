"""Loader for the AF-mode test set (Track C) -> ``pocketdetect.ligands.GroundTruth`` objects that
``pocketdetect.evaluate.evaluate`` consumes directly (no change to the harness needed).

    import sys; sys.path.insert(0, "scripts/data")
    from build_af_io import load_af_dataset
    gts = load_af_dataset()                                  # defaults below
    res = evaluate(gts, "results/preds/<method>/af", criteria=["LDCC12", "DCA4", "DCC4"])

Every LIGYSIS site becomes one ``Ligand`` (one evaluation target) with
  * ``ligand_id`` = LIGYSIS site name (``<ACC>_<SEG>_<ID>``; join key to af_sites.csv for strata)
  * ``center``   = site centre on the AF model (``center=``; default "ca" = LIGYSIS definition:
                   mean CA of the site residues)
  * ``coords``   = atoms for DCA: transferred ligand heavy atoms (instances with local CA RMSD
                   <= ``lig_rmsd_max``), or -- if a site has none -- the centre as a single point
                   (``missing_ligand="center"``; such sites are listed in ``gt.meta["dca_fallback"]``)
                   or the site is dropped (``missing_ligand="drop"``).
  * ``binding_residues`` = site residues as ``A:<UniProt position>`` (AF chain A, residue number
                   == UniProt position), matching the harness' ``chain:resnum`` convention.
Structure ids are ``AF-<ACC>-F1``; prediction files must be ``<pred_dir>/AF-<ACC>-F1.csv``.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AF_DIR = ROOT / "data/processed/af"

try:
    from pocketdetect.ligands import GroundTruth, Ligand
except ImportError:  # running without the package installed
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from pocketdetect.ligands import GroundTruth, Ligand

CENTERS = {"ca": "center_ca", "heavy": "center_heavy", "sc": "center_sc", "ligand": "center_lig"}


def load_af_manifest(af_dir: str | Path = AF_DIR, included_only: bool = True, benchmark_proteins: bool = True) -> pd.DataFrame:
    """Protein manifest. ``benchmark_proteins``: only UniProts with >= 1 LIGYSIS benchmark site
    (the 2,775-chain LBS-comparison set); ``included_only``: AF model present, not fragmented, >= 1
    valid site."""
    m = pd.read_csv(Path(af_dir) / "af_manifest.csv")
    if included_only:
        m = m[m["status"] == "included"]
    if benchmark_proteins:
        m = m[m["in_lbs_benchmark"]]
    m = m.copy()
    m["structure_path"] = [str(Path(af_dir) / p) for p in m["model_path"]]
    return m.reset_index(drop=True)


def read_gt_json(path: str | Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def af_ground_truth(
    gt: dict | str | Path,
    site_set: str = "all",
    ions: str = "include",
    center: str = "ca",
    atoms: str = "ligand",
    lig_rmsd_max: float = 2.0,
    missing_ligand: str = "center",
    include_flagged: bool = False,
) -> GroundTruth:
    """Build a GroundTruth from one per-protein JSON (see module docstring).

    site_set: "all" = every valid LIGYSIS site of the UniProt (full-length model => full site set);
              "benchmark" = only sites of the LBS-comparison benchmark (6,882-site set).
    ions:     "include" | "exclude" (drop sites with ion_prop == 1, the LIGYSIS_NI convention).
    center:   "ca" | "heavy" | "sc" | "ligand" (sites without transferred ligand are dropped
              for "ligand").
    atoms:    "ligand" | "center" (DCA computed to the centre only).
    """
    if not isinstance(gt, dict):
        gt = read_gt_json(gt)
    if center not in CENTERS:
        raise ValueError(f"center must be one of {sorted(CENTERS)}")
    chain = gt["chain"]
    ligs, dropped, fallback = [], [], []
    for s in gt["sites"]:
        if not (s["include"] or include_flagged):
            continue
        if site_set == "benchmark" and not s["in_lbs_benchmark"]:
            continue
        if ions == "exclude" and s["ion_prop"] is not None and s["ion_prop"] == 1:
            continue
        c = s.get(CENTERS[center])
        if c is None:
            dropped.append((s["site_name"], f"no_{center}_center"))
            continue
        c = np.asarray(c, dtype=float)
        acc_l = [l for l in s["ligands"] if l["rmsd"] <= lig_rmsd_max]
        if atoms == "ligand" and acc_l:
            xyz = np.vstack([np.asarray(l["coords"], dtype=float).reshape(-1, 3) for l in acc_l])
            el = np.array([e for l in acc_l for e in l["elements"]], dtype=object)
            name = "&".join(sorted({l["resname"] for l in acc_l}))
        elif atoms == "ligand" and missing_ligand == "drop":
            dropped.append((s["site_name"], "no_transferred_ligand"))
            continue
        else:
            xyz, el, name = c[None, :], np.array(["X"], dtype=object), "SITE"
            if atoms == "ligand":
                fallback.append(s["site_name"])
        ligs.append(
            Ligand(
                ligand_id=s["site_name"],
                name=name,
                groups=(),
                category="relevant",
                coords=xyz,
                elements=el,
                center=c,
                contact_dist=float("nan"),
                center_dist=float("nan"),
                binding_residues=tuple(f"{chain}:{u}" for u in s["residues"]),
            )
        )
    prot = np.array([f"{chain}:{n}" for n in gt["residue_numbers"]], dtype=object)
    return GroundTruth(
        structure_id=gt["structure_id"],
        rules=f"ligysis_af[{site_set},{ions},{center},{atoms},rmsd<={lig_rmsd_max},{missing_ligand}]",
        ligands=ligs,
        other_ligands=[],
        protein_residues=prot,
        n_protein_atoms=-1,
        path=str(AF_DIR / gt["model_path"]),
        meta=dict(acc=gt["acc"], dropped=dropped, dca_fallback=fallback, af_seq_eq_uniprot=gt["af_seq_eq_uniprot"]),
    )


def load_af_dataset(af_dir: str | Path = AF_DIR, benchmark_proteins: bool = True, drop_empty: bool = True, **kw) -> list[GroundTruth]:
    """All included proteins as GroundTruth objects (``kw`` -> :func:`af_ground_truth`). Proteins
    left without sites under the chosen filters are dropped when ``drop_empty`` (the harness would
    ignore them in all summaries anyway)."""
    m = load_af_manifest(af_dir, included_only=True, benchmark_proteins=benchmark_proteins)
    out = []
    for p in m["gt_path"]:
        g = af_ground_truth(Path(af_dir) / p, **kw)
        if g.ligands or not drop_empty:
            out.append(g)
    return out
