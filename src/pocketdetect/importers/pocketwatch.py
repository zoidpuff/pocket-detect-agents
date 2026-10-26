"""PocketWatch (genesisml/PocketWatch on HuggingFace, MIT; Portnoi et al., arXiv 2610.17696) -> site
table for the temporal **test** split.

Definitions taken from the dataset README / ``verify_bundle.py`` / parquet tables:

* **Targets / inputs**: one ESMFold2 sample-0 fold per test protein,
  ``structures/<uniprot>.cif.gz`` (single chain ``A``, residues numbered 1..L against the canonical
  sequence). ``structure_id`` = UniProt accession.
* **Sites**: rows of ``data/pocket_label_sites.parquet`` with ``split == "test"`` (2,301 sites,
  Q-BioLiP contacts at 5 Å clustered across entries; ligands with < 10 heavy atoms do not occur,
  i.e. ions are excluded). ``residues`` = ``canonical_positions`` (all lining positions) as
  ``A:<pos>``; ``label_residues`` = ``single_chain_positions`` (the positions that are positive in
  their per-residue labels for scored sites).
* **Mask**: ``data/site_policy.parquet`` ``decision`` — ``scored`` -> ``scored = 1``; ``masked``
  (``interface_only`` / ``pre_cutoff`` / ``subsumed_by_seen``) -> ``scored = 0``. Masked sites are
  kept as rows (they are real pockets, "never a negative").
* **N** (``n_sites``) = number of *scored* sites of the protein (README "Scoring": "top-N, where N
  is that protein's number of scored pockets, and the more forgiving top-(N+2)"). Masked sites do
  not count toward N; proteins without scored sites get ``n_sites = 0`` and contribute no recall
  target (the paper's 1,279-chain basis).
* **Centre**: the ligand centroid in the fold frame, rebuilt exactly as ``verify_bundle.py``: for
  each member observation (``projection/ligand_centroids.parquet``) Kabsch-superpose its
  entry-frame contact-residue CAs (``projection/contact_residues.parquet``) onto the fold CAs at
  the same canonical positions (positions outside 1..L or without CA dropped; >= 3 required),
  transform the entry-frame ligand centroid, average over members. Checked against the shipped
  ``data/site_centroids.parquet``. ``center_ca_*`` (metadata) = CA centroid of ``residues`` on the
  fold, LIGYSIS-style, for sensitivity analyses.
* ``ligand_atoms``: not available — the release ships only per-observation ligand *centroids*,
  not heavy-atom coordinates, so the column is left empty.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pocketdetect.importers import abspath_nofollow, protein_residue_count, residue_centroids
from pocketdetect.structure import Structure, load_structure, residue_id

__all__ = ["MIN_MATCHED_CA", "kabsch", "project_centroid", "fold_ca_by_position", "build_sites", "resolve_release_dir"]

MIN_MATCHED_CA = 3


def resolve_release_dir(raw_dir: str | os.PathLike) -> Path:
    """``data/raw/pocketwatch/current`` is a symlink to the commit directory; return the commit
    directory (absolute, parent not resolved) so paths in the table are pinned to the revision."""
    p = Path(abspath_nofollow(raw_dir))
    if p.is_symlink():
        target = os.readlink(p)
        p = Path(abspath_nofollow(p.parent / target))
    return p


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising ``|source @ R.T + t - target|`` (reflection
    corrected); same convention as PocketWatch ``verify_bundle.kabsch``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    sc, tc = source.mean(axis=0), target.mean(axis=0)
    cov = (source - sc).T @ (target - tc)
    u, _, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rot, tc - rot @ sc


def project_centroid(source_ca: np.ndarray, fold_ca: np.ndarray, centroid: np.ndarray):
    """Carry one entry-frame ligand centroid into the fold frame. Returns ``(xyz, fit_rmsd)`` or
    ``None`` if fewer than :data:`MIN_MATCHED_CA` CA pairs."""
    if len(source_ca) < MIN_MATCHED_CA:
        return None
    rot, tr = kabsch(source_ca, fold_ca)
    fitted = np.asarray(source_ca, dtype=np.float64) @ rot.T + tr
    rmsd = float(np.sqrt(((fitted - np.asarray(fold_ca, dtype=np.float64)) ** 2).sum(axis=1).mean()))
    return rot @ np.asarray(centroid, dtype=np.float64) + tr, rmsd


def fold_ca_by_position(s: Structure, length: int) -> np.ndarray:
    """(L,3) float32 CA coordinates indexed by canonical position (1..L), NaN where absent. Requires
    a single chain numbered exactly 1..L (as PocketWatch's reader does)."""
    if len(set(s.r_chain.tolist())) != 1:
        raise ValueError(f"{s.id}: expected one chain, found {sorted(set(s.r_chain.tolist()))}")
    if s.r_num.tolist() != list(range(1, length + 1)):
        raise ValueError(f"{s.id}: residues not numbered 1..{length}")
    ca = np.full((length, 3), np.nan, dtype=np.float32)
    for ri in range(s.n_residues):
        a0, a1 = int(s.r_start[ri]), int(s.r_end[ri])
        hit = np.flatnonzero(s.atom_name[a0:a1] == "CA")
        if len(hit):
            ca[int(s.r_num[ri]) - 1] = s.coords[a0 + hit[0]]
    return ca


def _site_centroid(members: pd.DataFrame, contacts: dict, fold_ca: np.ndarray):
    projected, fits = [], []
    for m in members.itertuples(index=False):
        part = contacts.get(m.pocket_observation_id)
        if part is None:
            continue
        pos, src = part
        inside = (pos >= 1) & (pos <= len(fold_ca))
        pos, src = pos[inside], src[inside]
        tgt = fold_ca[pos - 1]
        ok = ~np.isnan(tgt).any(axis=1)
        res = project_centroid(src[ok], tgt[ok], np.array([m.ligand_centroid_x, m.ligand_centroid_y, m.ligand_centroid_z]))
        if res is not None:
            projected.append(res[0])
            fits.append(res[1])
    if not projected:
        return None
    stacked = np.stack(projected)
    centre = stacked.mean(axis=0)
    return {
        "centre": centre,
        "n_members": len(projected),
        "spread": float(np.linalg.norm(stacked - centre, axis=1).max()),
        "median_fit_rmsd": float(np.median(fits)),
    }


def build_sites(raw_dir: str | os.PathLike, split: str = "test", check_tol: float = 1e-3) -> tuple[pd.DataFrame, dict]:
    """Build the PocketWatch site table for ``split`` (default ``test``). Returns ``(df, stats)``.

    Raises if any rebuilt centroid deviates from the shipped ``site_centroids.parquet`` by more
    than ``check_tol`` Å (max abs coordinate difference)."""
    root = resolve_release_dir(raw_dir)
    policy = pd.read_parquet(root / "data/site_policy.parquet")
    sites = pd.read_parquet(root / "data/pocket_label_sites.parquet")
    shipped = pd.read_parquet(root / "data/site_centroids.parquet").set_index("pocket_site_id")
    seqs = pd.read_parquet(root / "data/sequences.parquet").set_index("normalized_uniprot_id")
    manifest = pd.read_parquet(root / "structures/manifest.parquet").set_index("normalized_uniprot_id")
    excluded = pd.read_csv(root / "data/paper_excluded_test_proteins.csv")
    excluded_ids = set(excluded["normalized_uniprot_id"])

    pol = policy[policy["split"] == split]
    st = sites[sites["split"] == split].set_index("pocket_site_id")
    if set(pol["pocket_site_id"]) != set(st.index):
        raise ValueError("site_policy and pocket_label_sites disagree on the site set")
    site_ids = set(pol["pocket_site_id"])

    lig = pd.read_parquet(root / "projection/ligand_centroids.parquet")
    lig = lig[lig["pocket_site_id"].isin(site_ids)]
    cr = pd.read_parquet(root / "projection/contact_residues.parquet")
    cr = cr[cr["pocket_observation_id"].isin(set(lig["pocket_observation_id"]))]
    contacts = {
        oid: (g["canonical_position"].to_numpy(), g[["ca_x", "ca_y", "ca_z"]].to_numpy(dtype=np.float64))
        for oid, g in cr.groupby("pocket_observation_id", sort=False)
    }
    members_by_site = {sid: g for sid, g in lig.groupby("pocket_site_id", sort=False)}

    n_scored = pol.assign(sc=pol["decision"] == "scored").groupby("normalized_uniprot_id")["sc"].sum()
    rows = []
    stats = {"structures": 0, "sites": 0, "scored_sites": 0, "masked_sites": 0, "structures_with_scored": 0,
             "max_centre_dev_vs_shipped": 0.0, "sites_without_centre": 0, "residues_unmapped": 0}
    for uniprot, g in sorted(pol.groupby("normalized_uniprot_id"), key=lambda kv: kv[0]):
        length = int(seqs.loc[uniprot, "sequence_length"])
        path = root / "structures" / f"{uniprot}.cif.gz"
        s = load_structure(path, structure_id=uniprot)
        fold_ca = fold_ca_by_position(s, length)
        chain = s.r_chain[0]
        n_res = protein_residue_count(s)
        stats["structures"] += 1
        stats["structures_with_scored"] += int(n_scored[uniprot] > 0)
        for p in g.sort_values("pocket_site_id").itertuples(index=False):
            info = st.loc[p.pocket_site_id]
            rids = [residue_id(chain, int(x)) for x in sorted(info["canonical_positions"])]
            label_rids = [residue_id(chain, int(x)) for x in sorted(info["single_chain_positions"])]
            c = residue_centroids(s, rids)
            stats["residues_unmapped"] += len(c["missing"])
            mem = members_by_site.get(p.pocket_site_id)
            proj = _site_centroid(mem, contacts, fold_ca) if mem is not None else None
            if proj is None:
                stats["sites_without_centre"] += 1
                centre = np.full(3, np.nan)
            else:
                centre = proj["centre"]
                ref = shipped.loc[p.pocket_site_id, ["ligand_centroid_x", "ligand_centroid_y", "ligand_centroid_z"]].to_numpy(dtype=float)
                dev = float(np.abs(centre - ref).max())
                stats["max_centre_dev_vs_shipped"] = max(stats["max_centre_dev_vs_shipped"], dev)
                if dev > check_tol:
                    raise ValueError(f"{p.pocket_site_id}: rebuilt centroid deviates {dev:.4f} A from shipped")
            scored = p.decision == "scored"
            stats["sites"] += 1
            stats["scored_sites"] += int(scored)
            stats["masked_sites"] += int(not scored)
            comps = sorted({x for v in mem["ligand_component_ids"] for x in v}) if mem is not None else []
            rows.append({
                "structure_id": uniprot,
                "structure_path": abspath_nofollow(path),
                "site_id": p.pocket_site_id,
                "center_x": centre[0],
                "center_y": centre[1],
                "center_z": centre[2],
                "residues": " ".join(rids),
                "n_sites": int(n_scored[uniprot]),
                "scored": int(scored),
                "ligand_atoms": "",
                "n_residues_structure": n_res,
                "uniprot": uniprot,
                "split": split,
                "decision": p.decision,
                "mask_reason": p.reason if isinstance(p.reason, str) else "",
                "chain_annotation": p.chain_annotation,
                "site_first_observed": str(p.site_first_observed),
                "label_residues": " ".join(label_rids),
                "n_residues_unmapped": len(c["missing"]),
                "center_ca_x": c["center_ca"][0],
                "center_ca_y": c["center_ca"][1],
                "center_ca_z": c["center_ca"][2],
                "centroid_member_count": proj["n_members"] if proj else 0,
                "centroid_spread_angstrom": proj["spread"] if proj else np.nan,
                "median_fit_rmsd_angstrom": proj["median_fit_rmsd"] if proj else np.nan,
                "pdb_ids": " ".join(sorted(info["pdb_ids"])),
                "ligand_components": " ".join(comps),
                "median_heavy_atom_count": float(info["median_heavy_atom_count"]),
                "residual_quality": info["residual_quality"],
                "fold_plddt": float(manifest.loc[uniprot, "plddt_0_100"]),
                "fold_ptm": float(manifest.loc[uniprot, "ptm"]),
                "paper_excluded": int(uniprot in excluded_ids),
            })
    df = pd.DataFrame(rows)
    return df, stats
