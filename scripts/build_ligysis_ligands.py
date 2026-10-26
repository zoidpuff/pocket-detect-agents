#!/usr/bin/env python
"""Place the ligands that define each LIGYSIS site onto its representative chain (wwPDB
asymmetric-unit frame) — ground truth for the PREREG C1-strict DCA criterion.

  nohup .venv/bin/python -I scripts/build_ligysis_ligands.py --jobs 48 > logs/ligysis_ligands.log 2>&1 &

Stage 1 (resumable cache): for every PDB entry that contributes a site ligand, locate the ligand
instances and the contacted protein chains (pocketdetect.importers.ligysis_ligands.extract_entry);
cache JSON per entry in data/processed/gt/ligysis_lbs/ligand_cache/.
Stage 2: per representative chain, superpose each contacted source chain onto the rep chain (CA of
UniProt-paired site-core residues via SIFTS, rigid-core Kabsch, core RMSD ≤ 2 Å) and transform the
ligand heavy atoms (place_ligand); keep a placed copy only if it has a heavy atom within 4.5 Å of a
site-residue heavy atom of the rep chain (else ``off_site``); pool per site, dropping duplicate copies and non-ion atoms
< 2.2 Å from rep-chain protein heavy atoms (pool_site).
Output: data/processed/gt/ligysis_lbs/ligand_atoms.tsv (structure_id, site_id, ligand_atoms, stats)
+ ligand_placements.tsv (one row per placement attempt) + ligand_README.md.
scripts/build_gt_ligysis.py merges ligand_atoms into sites.tsv.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketdetect.importers.ligysis_ligands import (PARAMS, SIFTS_TSV, extract_entry, label_seq_to_uniprot, load_sifts,  # noqa: E402
                                                    place_ligand, pool_site, read_entry, rep_chain_data)
from pocketdetect.ligysis import LIGYSIS_RESULTS, load_reference_sites, safe_unpickle  # noqa: E402
from pocketdetect.sites import format_atoms  # noqa: E402
from pocketdetect.structure import pdb_entry_path  # noqa: E402

OUT = ROOT / "data/processed/gt/ligysis_lbs"
CACHE = OUT / "ligand_cache"
SITE_CONTACT = 4.5  # A; a placed ligand copy must have a heavy atom within this of a site-residue heavy atom


def stage1(args):
    pdb, requests, sifts_entry = args
    f = CACHE / f"{pdb}.json.gz"
    if f.exists():
        return pdb, "cached"
    path = pdb_entry_path(pdb)
    if not path.exists():
        res = {"pdb": pdb, "status": "missing_in_mirror", "ligands": {}, "chains": {}}
    else:
        res = extract_entry(pdb, path, requests, sifts_entry)
    tmp = f.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(res, fh)
    tmp.rename(f)
    return pdb, res["status"]


_cache: dict = {}


def _load(pdb):
    if pdb not in _cache:
        if len(_cache) > 400:
            _cache.clear()
        with gzip.open(CACHE / f"{pdb}.json.gz", "rt") as fh:
            _cache[pdb] = json.load(fh)
    return _cache[pdb]


def stage2(args):
    rep, sites, sifts_rep = args  # sites: list of dict(site_id, acc, up, ligs={key: chains})
    pdb, chain = rep.split("_", 1)
    out_rows, pl_rows = [], []
    try:
        entry = read_entry(pdb_entry_path(pdb))
        rc = rep_chain_data(entry, chain)
    except Exception as e:  # noqa: BLE001
        rc = None
        err = f"rep_read_error:{type(e).__name__}"
    else:
        err = "rep_chain_missing" if rc is None else ""
    if rc is None:
        for s in sites:
            out_rows.append({"structure_id": rep, "site_id": s["site_id"], "ligand_atoms": "", "status": err})
        return out_rows, pl_rows
    rep_atoms = rc["atoms"]
    rep_tree = cKDTree(rep_atoms)
    for s in sites:
        m = label_seq_to_uniprot(sifts_rep.get(chain, []), s["acc"])
        rep_up_ca = {m[int(ls)]: ca for ls, ca in zip(rc["label_seq"], rc["ca"]) if int(ls) in m and np.isfinite(ca[0])}
        upset = set(s["up"])
        site_idx = [i for i, ls in enumerate(rc["label_seq"]) if m.get(int(ls)) in upset]
        site_mask = np.isin(rc["atom_res"], site_idx)
        site_tree = cKDTree(rc["atoms"][site_mask]) if site_mask.any() else None
        placed = []
        n_far = 0
        for key in s["ligs"]:
            lpdb = key.split("_")[0].lower()
            try:
                ent = _load(lpdb)
            except FileNotFoundError:
                ent = {"ligands": {}, "chains": {}}
            rec = ent["ligands"].get(f"{key}|{s['acc']}")
            rows, pl = place_ligand(key, rec, ent["chains"], s["up"], rep_up_ca)
            for r in rows:
                r.update(structure_id=rep, site_id=s["site_id"])
            base = len(pl_rows)
            # each placed copy must touch the site residues of the rep chain (min heavy-atom distance)
            for ident, xyz, ion, tag in pl:
                d = float(site_tree.query(xyz, k=1)[0].min()) if site_tree is not None else np.inf
                rows[tag]["site_contact_dist"] = d
                if d <= SITE_CONTACT:
                    placed.append((ident, xyz, ion, base + tag))
                else:
                    rows[tag]["status"] = "off_site"
                    n_far += 1
            pl_rows += rows
        pooled = pool_site(placed, rep_atoms, rep_tree=rep_tree)
        A = pooled["atoms"]
        dmin = float(site_tree.query(A, k=1)[0].min()) if len(A) and site_tree is not None else np.nan
        out_rows.append({
            "structure_id": rep, "site_id": s["site_id"], "ligand_atoms": format_atoms(A) if len(A) else "",
            "status": "ok" if len(A) else ("dropped_clash" if pooled["dropped_site"] else "no_placement"),
            "n_ligand_instances": len(s["ligs"]), "n_placed": len(placed) - pooled["n_duplicate"], "n_atoms": len(A),
            "n_atoms_raw": pooled["n_atoms_raw"], "n_clash": pooled["n_clash"], "min_dist_to_site_residues": dmin, "n_off_site": n_far,
        })
    return out_rows, pl_rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="first N representative chains (debug)")
    a = ap.parse_args(argv)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S = load_reference_sites()
    cpl = safe_unpickle(LIGYSIS_RESULTS / "LIGYSIS_human_chains_per_lig_MAY_2024.pkl")
    reps = sorted(S.structure_id.unique())
    if a.limit:
        reps = reps[: a.limit]
        S = S[S.structure_id.isin(reps)]
    per_rep = defaultdict(list)
    per_pdb = defaultdict(list)
    for r in S.itertuples(index=False):
        ligs = cpl.get(r.seg_id, {}).get(int(r.site_id), {})
        up = [int(t[2:]) for t in r.residues.split()]
        per_rep[r.structure_id].append({"site_id": r.site_id, "acc": r.uniprot, "up": up, "ligs": {k: sorted(v) for k, v in ligs.items()}})
        for k, v in ligs.items():
            per_pdb[k.split("_")[0].lower()].append({"key": k, "chains": sorted(v), "acc": r.uniprot})
    pdbs = sorted(set(per_pdb) | {r.split("_")[0] for r in reps})
    sifts = load_sifts(SIFTS_TSV, pdb_ids=set(pdbs))
    print(f"{len(reps)} rep chains, {len(S)} sites, {sum(len(v) for v in per_pdb.values())} ligand requests, {len(pdbs)} entries; "
          f"sifts loaded {time.time() - t0:.0f}s", flush=True)
    todo = [(p, per_pdb.get(p, []), sifts.get(p, {})) for p in pdbs if per_pdb.get(p)]
    status = defaultdict(int)
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, (p, st) in enumerate(ex.map(stage1, todo, chunksize=4)):
            status[st] += 1
            if i % 2000 == 0:
                print(f"stage1 {i}/{len(todo)} {dict(status)} {time.time() - t0:.0f}s", flush=True)
    print("stage1 done", dict(status), flush=True)
    tasks = [(rep, per_rep[rep], sifts.get(rep.split("_")[0], {})) for rep in reps]
    rows, pls = [], []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, (r, p) in enumerate(ex.map(stage2, tasks, chunksize=4)):
            rows += r
            pls += p
            if i % 500 == 0:
                print(f"stage2 {i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ligand_atoms.tsv", sep="\t", index=False, float_format="%.3f")
    pd.DataFrame(pls).to_csv(OUT / "ligand_placements.tsv", sep="\t", index=False, float_format="%.3f")
    summ = {
        "n_sites": len(df), "status": df.status.value_counts().to_dict(),
        "n_scored_with_atoms": int(df.merge(S[["structure_id", "site_id", "scored"]], on=["structure_id", "site_id"]).query("scored and n_atoms > 0").shape[0])
        if "n_atoms" in df else 0,
        "min_dist_to_site_residues_median": float(df.min_dist_to_site_residues.median()),
        "frac_min_dist_le_4.5": float((df.min_dist_to_site_residues <= 4.5).mean()),
        "params": {**PARAMS, "site_contact": SITE_CONTACT}, "n_off_site_copies": int(df.n_off_site.sum()), "seconds": round(time.time() - t0),
    }
    (OUT / "ligand_README.md").write_text("# LIGYSIS site ligands on representative chains\n\nBuilt by `scripts/build_ligysis_ligands.py` "
                                          "(definitions in its docstring and in src/pocketdetect/importers/ligysis_ligands.py).\n\n```\n"
                                          + json.dumps(summ, indent=2, default=str) + "\n```\n")
    print(json.dumps(summ, indent=2, default=str))


if __name__ == "__main__":
    main()
