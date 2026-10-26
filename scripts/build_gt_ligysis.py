#!/usr/bin/env python
"""Build the LIGYSIS (LBS-comparison, 2,775 chains) site table in the frame of the wwPDB
asymmetric-unit mmCIF of each representative chain, for scoring new methods.

  python -I scripts/build_gt_ligysis.py [--jobs 32]  ->  data/processed/gt/ligysis_lbs/{sites.tsv,README.md}

* Sites, N, scored (non-redundant) flags: pocketdetect.ligysis.load_reference_sites (identical to
  the set used by the authors; verified by scripts/reproduce_ligysis.py).
* Centre: the authors' site centre (CA centroid of the site residues, TRANS frame) mapped back with
  x = Rᵀ x_T + c0 (their rotation matrix / centroid of the chain). Checked against the CA centroid
  recomputed from the mirror file (``aas`` = label_seq_id on chain auth_asym_id);
  ``center_check_dev`` reports the deviation. If it exceeds 1 Å the recomputed centroid is used and
  ``center_source`` says so.
* residues: the authors' ``aas`` are label_seq_ids of the representative chain; mapped to author
  ids ``<auth_chain>:<num><icode>`` of the mirror file; ``residues_uniprot`` keeps ``U:<n>``. input structure: mirror mmCIF + ``input_chains`` (single representative chain).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketdetect.ligysis import load_reference_sites, load_transforms, to_original_frame  # noqa: E402
from pocketdetect.sites import write_sites  # noqa: E402
from pocketdetect.structure import PROTEIN, load_structure, pdb_entry_path  # noqa: E402

OUT = ROOT / "data/processed/gt/ligysis_lbs"


def chain_job(args):
    """aas are label_seq_ids of the representative chain (verified: 1a52 site residues 43.. = auth 339..).
    Map label_seq -> author residue id and CA via gemmi (first model, first altloc)."""
    import gemmi

    rep, chain, rows = args
    pdb = rep.split("_")[0]
    path = pdb_entry_path(pdb)
    out = []
    try:
        st = gemmi.read_structure(str(path))
        st.remove_alternative_conformations()
        ch = st[0][chain]
        s = load_structure(path, chains=[chain])
    except Exception as e:  # missing entry/chain
        return rep, str(path), None, [(r["site_id"], np.full(3, np.nan), [], [], f"load_error:{type(e).__name__}") for r in rows]
    nres = int((s.r_kind == PROTEIN).sum())
    lab = {}
    for res in ch:
        if res.label_seq is None:
            continue
        ca = res.find_atom("CA", "*")
        ic = res.seqid.icode.strip()
        lab.setdefault(str(res.label_seq), (f"{chain}:{res.seqid.num}{ic}", None if ca is None else np.array([ca.pos.x, ca.pos.y, ca.pos.z])))
    for r in rows:
        toks = r["residues_pdb"].split()
        found = [lab[t] for t in toks if t in lab]
        missing = [t for t in toks if t not in lab]
        xyz = [c for _, c in found if c is not None]
        cen = np.mean(xyz, axis=0) if xyz else np.full(3, np.nan)
        out.append((r["site_id"], cen, missing, [rid for rid, _ in found], "ok"))
    return rep, str(path), nres, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    sites = load_reference_sites()
    T = load_transforms()
    sites["auth_chain"] = sites.structure_id.str.split("_").str[1]
    tasks = [(rep, g.auth_chain.iloc[0], g[["site_id", "residues_pdb"]].to_dict("records")) for rep, g in sites.groupby("structure_id", sort=True)]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(chain_job, tasks, chunksize=8))
    recomputed, paths, nres, missing_n, auth_ids = {}, {}, {}, {}, {}
    for rep, path, nr, out in res:
        paths[rep], nres[rep] = path, nr
        for sid, cen, miss, rids, status in out:
            recomputed[(rep, sid)] = cen
            missing_n[(rep, sid)] = len(miss)
            auth_ids[(rep, sid)] = rids
    rows = []
    for r in sites.itertuples(index=False):
        Rm, c0 = T[r.structure_id]
        inv = to_original_frame(np.array([r.center_x, r.center_y, r.center_z]), Rm, c0)
        rc = recomputed[(r.structure_id, r.site_id)]
        dev = float(np.linalg.norm(inv - rc)) if np.all(np.isfinite(rc)) else np.nan
        use_rc = np.isfinite(dev) and dev > 1.0
        cen = rc if use_rc else inv
        rows.append({
            "structure_id": r.structure_id, "structure_path": paths[r.structure_id], "input_chains": r.auth_chain,
            "site_id": r.site_id, "center_x": cen[0], "center_y": cen[1], "center_z": cen[2],
            "residues": " ".join(auth_ids[(r.structure_id, r.site_id)]),
            "residues_label_seq": r.residues_pdb,
            "n_sites": r.n_sites, "scored": r.scored, "n_residues_structure": nres[r.structure_id] if nres[r.structure_id] is not None else "",
            "residues_uniprot": r.residues, "uniprot": r.uniprot, "site_name": r.site_name, "ion_prop": r.ion_prop,
            "center_source": "recomputed_ca_centroid" if use_rc else "authors_trans_inverted", "center_check_dev": dev,
            "n_residues_unmapped": missing_n[(r.structure_id, r.site_id)],
            "center_trans_x": r.center_x, "center_trans_y": r.center_y, "center_trans_z": r.center_z,
        })
    df = pd.DataFrame(rows)
    lig_file = OUT / "ligand_atoms.tsv"  # from scripts/build_ligysis_ligands.py (C1-strict DCA)
    if lig_file.exists():
        L = pd.read_csv(lig_file, sep="\t", dtype={"structure_id": str, "site_id": str, "ligand_atoms": str}, keep_default_na=False)
        L = L.rename(columns={"status": "ligand_status", "n_atoms": "n_ligand_atoms"})[
            ["structure_id", "site_id", "ligand_atoms", "ligand_status", "n_ligand_instances", "n_ligand_atoms"]]
        df = df.merge(L, on=["structure_id", "site_id"], how="left")
        df["ligand_atoms"] = df["ligand_atoms"].fillna("")
    write_sites(df, OUT / "sites.tsv")
    # LIGYSIS_NI (descriptive, PREREG): sites with ion_prop != 1 (LIGYSIS convention); N recomputed over
    # the remaining sites of the chain, scored flags kept; chains left without sites dropped.
    ni = df[df.ion_prop.astype(float) != 1].copy()
    ni["n_sites"] = ni.groupby("structure_id").site_id.transform("size").astype(int)
    (OUT.parent / "ligysis_ni").mkdir(exist_ok=True)
    write_sites(ni, OUT.parent / "ligysis_ni" / "sites.tsv")
    (OUT.parent / "ligysis_ni" / "README.md").write_text(
        "# LIGYSIS_NI site table\n\nSubset of `../ligysis_lbs/sites.tsv` with `ion_prop != 1` (LIGYSIS non-ion convention); "
        f"N recomputed per chain. {len(ni)} sites, {int(ni.scored.sum())} scored, {ni.structure_id.nunique()} chains. "
        "Built by scripts/build_gt_ligysis.py.\n")
    dev = df.center_check_dev
    summary = {
        "n_sites": len(df), "n_scored": int(df.scored.sum()), "n_chains": df.structure_id.nunique(),
        "center_dev_median": float(dev.median()), "center_dev_p99": float(dev.quantile(0.99)), "n_dev_gt_1A": int((dev > 1).sum()),
        "n_dev_nan": int(dev.isna().sum()), "n_sites_with_unmapped_residues": int((df.n_residues_unmapped > 0).sum()),
        "n_load_errors": int(sum(1 for v in nres.values() if v is None)),
        "n_scored_with_ligand_atoms": int(((df.get("ligand_atoms", pd.Series("", index=df.index)) != "") & df.scored).sum()),
    }
    print(summary)
    (OUT / "README.md").write_text(
        "# LIGYSIS (LBS-comparison) site table, wwPDB asymmetric-unit frame\n\n"
        f"Built by `scripts/build_gt_ligysis.py`. Definitions in the script docstring and docs/EVALUATION.md §10.\n\n"
        f"Summary: `{summary}`\n\n"
        "Inputs: data/raw/ligysis/lbs_comparison_results (ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl, LIGYSIS_sites_DEF_TRANS.pkl, "
        "PDB_rot_matrices_ALL_CLUST.pkl, PDB_orig_centroids_ALL_CLUST.pkl; sha256 in data/manifests/ligysis_lbs_comparison_results.tsv) "
        "and the local wwPDB mirror.\n")


if __name__ == "__main__":
    main()
