"""CryptoBench (Škrhák et al., Bioinformatics 2025; OSF 10.17605/OSF.IO/PZ4A9) -> site table.

Source annotation: ``cryptobench-dataset/folds/<split>.json`` — ``{apo_pdb_id: [holo entry, ...]}``.
Each holo entry describes one apo–holo pair (one ligand) with ``apo_chain`` (``"A"`` or, for
multi-chain pockets, dash-joined ``"A-B"``) and ``apo_pocket_selection`` = ``["<auth_asym_id>_
<auth_seq_id>", ...]`` on the apo entry (author numbering per the dataset README; verified: every
test residue exists under auth numbering and apo/holo residue names agree for 97.9 % of residue
pairs under auth numbering vs 34 % under label numbering).

Rules implemented here (see ``data/processed/gt/cryptobench_test/README.md``):

* **Input structure** = the apo entry restricted to its CryptoBench chain(s): the union of the
  ``apo_chain`` values over the apo's holo entries. This equals CryptoBench's own
  ``the_most_chains`` rule (``G-download-.../create-annotations.py``: the entry with the most
  dash-separated chains) for every test apo. ``structure_id`` = ``<apo_pdb>_<chains>`` with the
  chains dash-joined in CryptoBench order (e.g. ``7qoq_A``, ``1fd4_G-H``).
* **Sites** = CryptoBench's own "cryptic pocket" definition (``src/F-statistics/stats.ipynb``,
  Table 2 "No. of cryptic pockets"): the apo pocket selections of all holo entries of one apo are
  merged iteratively whenever the overlap exceeds 75 % of either selection
  (``MAX_POCKET_SIMILARITY = 0.75``); merged residue set = union. :func:`merge_pockets` is a
  line-by-line port (including its order dependence).
* **Centre** = centroid of the CA atoms of the site residues on the apo structure (first altloc);
  ``center_heavy_*`` = centroid of all heavy atoms of those residues.
* ``n_sites`` = number of merged cryptic sites of the apo structure; all rows ``scored = 1``.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from pocketdetect.importers import abspath_nofollow, protein_residue_count, residue_centroids
from pocketdetect.structure import load_structure, residue_id

__all__ = [
    "MAX_POCKET_SIMILARITY",
    "load_split",
    "merge_pockets",
    "parse_selection_residue",
    "input_chains",
    "build_sites",
]

MAX_POCKET_SIMILARITY = 0.75


def load_split(dataset_dir: str | os.PathLike, split: str = "test") -> dict:
    """Load ``folds/<split>.json`` (``split`` = ``test`` or ``train-fold-<k>``) and check its keys
    against ``folds.json``."""
    dataset_dir = Path(dataset_dir)
    fname = "test.json" if split == "test" else f"{split}.json"
    with open(dataset_dir / "folds" / fname) as fh:
        data = json.load(fh, object_pairs_hook=OrderedDict)
    with open(dataset_dir / "folds.json") as fh:
        folds = json.load(fh)
    key = "test" if split == "test" else split.replace("train-fold-", "train-")
    if key in folds and set(folds[key]) != set(data):
        raise ValueError(f"folds.json[{key!r}] disagrees with folds/{fname}")
    return data


def merge_pockets(pockets: list[list[str]], threshold: float = MAX_POCKET_SIMILARITY) -> list[list[str]]:
    """Port of the pocket-merging loop in CryptoBench ``src/F-statistics/stats.ipynb``.

    Two pockets merge when ``|A ∩ B| / |A| > threshold`` or ``|A ∩ B| / |B| > threshold``; the
    merged pocket replaces the earlier one and the loop restarts until the count is stable.
    Residue lists are returned sorted (order is irrelevant to the algorithm, which only uses set
    operations and list lengths)."""
    old = [list(p) for p in pockets]
    new: list[list[str]] = []
    j = 0
    while len(old) != len(new):
        if j != 0:
            old = new
        new = []
        merged: list[int] = []
        for i in range(len(old)):
            if i in merged:
                continue
            for ii in range(i + 1, len(old)):
                fst, snd = (old[i], old[ii]) if len(old[i]) < len(old[ii]) else (old[ii], old[i])
                inter = len(set(fst).intersection(snd))
                if inter / len(fst) > threshold or inter / len(snd) > threshold:
                    merged.append(ii)
                    old[i] = sorted(set(fst + snd))
            new.append(old[i])
        j += 1
    return [sorted(set(p), key=_res_sort_key) for p in old]


def parse_selection_residue(token: str) -> str:
    """``"B_12"`` (auth_asym_id _ auth_seq_id) -> ``"B:12"``. Chain IDs may be multi-letter."""
    chain, _, num = token.rpartition("_")
    if not chain or not num.lstrip("-").isdigit():
        raise ValueError(f"unexpected CryptoBench residue token {token!r}")
    return residue_id(chain, int(num))


def _res_sort_key(token: str):
    chain, _, num = token.rpartition("_")
    return (chain, int(num)) if num.lstrip("-").isdigit() else (chain, 0)


def input_chains(holos: list[dict]) -> list[str]:
    """Chains of the apo input structure: union of ``apo_chain`` over holo entries, in order of
    first appearance (dash-separated). Asserts it matches CryptoBench's ``the_most_chains``."""
    chains: list[str] = []
    most = holos[0]["apo_chain"]
    for h in holos:
        if h["apo_chain"].count("-") > most.count("-"):
            most = h["apo_chain"]
        for c in h["apo_chain"].split("-"):
            if c not in chains:
                chains.append(c)
    if set(most.split("-")) != set(chains):
        raise ValueError(f"chain union {chains} differs from CryptoBench the_most_chains {most}")
    order = most.split("-")
    return order + [c for c in chains if c not in order]


def build_sites(dataset_dir: str | os.PathLike, cif_dir: str | os.PathLike, split: str = "test") -> tuple[pd.DataFrame, dict]:
    """Build the site table for one CryptoBench split. Returns ``(df, stats)``."""
    data = load_split(dataset_dir, split)
    cif_dir = Path(cif_dir)
    rows = []
    stats = {"apo_structures": 0, "holo_entries": 0, "sites": 0, "site_residues": 0, "residues_unmapped": 0,
             "residues_no_ca": 0, "residues_non_protein": 0, "multichain_inputs": 0, "multichain_sites": 0}
    for apo in sorted(data):
        holos = data[apo]
        chains = input_chains(holos)
        sid = f"{apo}_{'-'.join(chains)}"
        path = cif_dir / f"{apo}.cif"
        if not path.exists():
            raise FileNotFoundError(path)
        s = load_structure(path, structure_id=sid, chains=chains)
        n_res = protein_residue_count(s)
        pockets = merge_pockets([h["apo_pocket_selection"] for h in holos])
        stats["apo_structures"] += 1
        stats["holo_entries"] += len(holos)
        stats["multichain_inputs"] += len(chains) > 1
        uniprots = sorted({h["uniprot_id"] for h in holos})
        for k, pocket in enumerate(pockets, start=1):
            pset = set(pocket)
            members = [h for h in holos if set(h["apo_pocket_selection"]) <= pset]
            rids = [parse_selection_residue(t) for t in pocket]
            c = residue_centroids(s, rids)
            stats["sites"] += 1
            stats["site_residues"] += len(rids)
            stats["residues_unmapped"] += len(c["missing"])
            stats["residues_no_ca"] += len(c["no_ca"])
            stats["residues_non_protein"] += len(c["non_protein"])
            site_chains = sorted({r.rpartition(":")[0] for r in rids})
            stats["multichain_sites"] += len(site_chains) > 1
            rows.append({
                "structure_id": sid,
                "structure_path": abspath_nofollow(path),
                "site_id": f"site{k}",
                "center_x": c["center_ca"][0],
                "center_y": c["center_ca"][1],
                "center_z": c["center_ca"][2],
                "residues": " ".join(rids),
                "n_sites": len(pockets),
                "scored": 1,
                "ligand_atoms": "",
                "n_residues_structure": n_res,
                "center_heavy_x": c["center_heavy"][0],
                "center_heavy_y": c["center_heavy"][1],
                "center_heavy_z": c["center_heavy"][2],
                "apo_pdb_id": apo,
                "input_chains": "-".join(chains),
                "site_chains": "-".join(site_chains),
                "n_site_residues": len(rids),
                "n_residues_unmapped": len(c["missing"]),
                "n_residues_no_ca": len(c["no_ca"]),
                "n_holo_entries": len(members),
                "holo_pdb_ids": " ".join(sorted({f"{h['holo_pdb_id']}_{h['holo_chain']}" for h in members})),
                "ligands": " ".join(sorted({f"{h['ligand']}:{h['ligand_chain']}:{h['ligand_index']}@{h['holo_pdb_id']}" for h in members})),
                "ligand_names": " ".join(sorted({h["ligand"] for h in members})),
                "max_prmsd": max((float(h["pRMSD"]) for h in members), default=np.nan),
                "has_main_holo": int(any(h.get("is_main_holo_structure") for h in members)),
                "uniprot": " ".join(sorted({h["uniprot_id"] for h in members}) or uniprots),
                "split": split,
            })
    df = pd.DataFrame(rows)
    return df, stats
