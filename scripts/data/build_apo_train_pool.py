#!/usr/bin/env python
"""Track A2 step 2: apo-chain selection and label transfer for train/dev entries.

For every entry (audit-filtered manifests) and its candidate chains (build_apo_train_search.py), up
to --max-pdb candidate PDB entries (best resolution first; per entry the chains are tried in order
until one qualifies) are examined:

* apo test: no BioLiP2-relevant ligand (incl. ions; polymer ligands peptide/RNA/DNA included) with a
  heavy atom within 6 A of any heavy atom of the parent's site residues (mapped onto the candidate
  chain by sequence alignment). Relevance = the hetero group is listed in BioLiP2 for that entry
  (any receptor chain). Entries released after the BioLiP2 snapshot (2025-11-12) are not covered by
  BioLiP2; for them every non-water hetero group except a fixed additive list counts as relevant,
  and any nucleic-acid residue counts. Non-relevant hetero groups and protein partner chains near
  the site are allowed but recorded. Ligands elsewhere on the chain are allowed and counted.
* label transfer per parent site: Calpha of parent residues within 10 A of the site's pooled ligand
  atoms, mapped through the alignment (>= 6 pairs, >= 70% of the site residues present), Kabsch onto
  the candidate; site kept if this local Calpha RMSD <= 3.0 A. Recorded: local RMSD, site Calpha
  RMSD, pocket RMSD = heavy-atom RMSD of the matched site-residue atoms after the local superposition;
  cryptic_like = pocket RMSD >= 2.0 A. Ligand atoms are NOT clash-trimmed (cryptic pockets are
  closed in apo); per-atom min distance to the apo protein is stored instead.
* <= 2 apo chains per entry, from different PDB entries: first the chain with the largest max
  pocket RMSD (ties: resolution), second the one whose per-site pocket-RMSD vector differs most
  (L1) from the first.

Output: data/processed/apo_train/{train,dev}/entries/<apo_id>.npz, {train,dev}/manifest.csv,
work/candidate_log.csv, pool_counts.json. npz keys mirror the train-set entries
(pocketdetect.datasets.trainset.load_entry works); extra keys are documented in docs/TRAINSET.md.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.datasets.biolip import METALS  # noqa: E402
from pocketdetect.datasets.pooling import align_map, kabsch, load_entry  # noqa: E402
from pocketdetect.datasets.trainset import load_entry as load_train_entry  # noqa: E402
from pocketdetect.structure import pdb_entry_path  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "apo_train"
BIOLIP = ROOT / "data" / "raw" / "biolip2" / "current" / "BioLiP.txt.gz"
BIOLIP_SNAPSHOT = "2025-11-12"
ADDITIVES = frozenset(
    "HOH DOD WAT UNX UNL GOL EDO PEG PG4 PGE 1PE P6G 2PE PE4 PE8 MPD DMS ACT ACY FMT CL NA K BR IOD NO3 NH4 "
    "SO4 PO4 TRS EPE MES BME IMD IPA EOH MOH BU3 PGO TAR".split()
)
P = dict(apo_contact=6.0, pocket_ca_radius=10.0, min_pairs=6, min_site_coverage=0.7, max_local_rmsd=3.0,
         cryptic_pocket_rmsd=2.0, min_align_identity=0.9, max_apo_per_entry=2, near_dist=4.5)


def _release_date(block) -> str:
    v = block.find_values("_pdbx_audit_revision_history.revision_date")
    ds = sorted(gemmi.cif.as_string(x) for x in v) if v else []
    return ds[0] if ds else ""


def _local(src_ca, mp, dst_ca, ref_xyz):
    ok = ~np.isnan(src_ca[:, 0])
    idx = np.flatnonzero(ok)
    near = set()
    for lst in cKDTree(src_ca[idx]).query_ball_point(ref_xyz, P["pocket_ca_radius"]):
        near.update(idx[j] for j in lst)
    near = np.array(sorted(near), int)
    if len(near) == 0:
        return None
    m = mp[near]
    k = m >= 0
    near, m = near[k], m[k]
    k = ~np.isnan(dst_ca[m, 0])
    near, m = near[k], m[k]
    if len(near) < P["min_pairs"]:
        return None
    R, t, rmsd = kabsch(src_ca[near], dst_ca[m])
    return R, t, rmsd, len(near)


def _res_atoms(cd, i):
    sel = np.flatnonzero(cd.atom_res == i)
    return {cd.atom_names[j]: cd.atoms[j] for j in sel}


def process(task):
    st = Counter()
    try:
        return _process(task, st)
    except Exception as e:  # noqa: BLE001
        return dict(entry_id=task["entry_id"], status=f"error: {type(e).__name__}: {e}", tb=traceback.format_exc()[-1200:]), [], st


def _process(task, st):
    t0 = time.time()
    E = load_train_entry(ROOT / task["npz"])
    a = E.arrays
    R = load_entry(pdb_entry_path(E.meta["pdb"]))
    rc = R.chains[E.chain]
    rkidx = rc.key_index
    nsite = E.n_sites
    sites = []
    ptr = a["lig_ptr"]
    for s in range(nsite):
        li = np.flatnonzero(a["lig_site"] == s)
        xyz = np.vstack([a["lig_coords"][ptr[i]:ptr[i + 1]] for i in li]) if len(li) else np.zeros((0, 3))
        res = [rkidx[k] for k in (str(a["site_residues"][s]).split(";") if a["site_residues"][s] else []) if k in rkidx]
        sites.append(dict(lig=li, xyz=xyz, res=res))
    log, accepted = [], []
    for cand in task["cands"]:
        pdb, chains = cand["pdb"], cand["chains"]
        path = pdb_entry_path(pdb)
        if not path.exists():
            log.append(dict(entry_id=task["entry_id"], pdb=pdb, chain="", outcome="missing_file"))
            continue
        block = gemmi.cif.read(str(path)).sole_block()
        rel = _release_date(block)
        covered = bool(rel) and rel <= BIOLIP_SNAPSHOT
        stc = gemmi.make_structure_from_block(block)
        stc.setup_entities()
        stc.remove_hydrogens()
        stc.remove_alternative_conformations()
        A = load_entry(path, want_ligands=False, structure=stc)
        bl = task["biolip"].get(pdb, {"nonpoly": set(), "polychains": set(), "per_chain": {}})
        for ch in chains:
            rec = dict(entry_id=task["entry_id"], pdb=pdb, chain=ch, resolution=cand["resolution"], release_date=rel,
                       biolip_covered=covered)
            ac = A.chains.get(ch)
            if ac is None:
                log.append({**rec, "outcome": "chain_missing"})
                continue
            mp, ident, nal = align_map(rc.seq, ac.seq)
            rec["align_identity"] = round(ident, 4)
            if ident < P["min_align_identity"]:
                log.append({**rec, "outcome": "low_identity"})
                continue
            # mapped site residues on the candidate
            site_res_apo = sorted({int(mp[i]) for s in sites for i in s["res"] if mp[i] >= 0})
            if not site_res_apo:
                log.append({**rec, "outcome": "no_site_residues"})
                continue
            sel = np.isin(ac.atom_res, site_res_apo)
            tree = cKDTree(ac.atoms[sel])
            relevant, nonrel, partners = [], set(), set()
            for gch in stc[0]:
                for r in gch:
                    if r.is_water():
                        continue
                    info = gemmi.find_tabulated_residue(r.name)
                    is_aa = info is not None and info.is_amino_acid()
                    is_poly = r.entity_type == gemmi.EntityType.Polymer
                    if gch.name == ch and is_poly and is_aa:
                        continue  # the candidate protein itself
                    xyz = np.array([[at.pos.x, at.pos.y, at.pos.z] for at in r])
                    if len(xyz) == 0:
                        continue
                    d, _ = tree.query(xyz, k=1, distance_upper_bound=P["apo_contact"])
                    if not np.isfinite(d).any():
                        continue
                    tag = f"{r.name}|{gch.name}:{r.seqid.num}"
                    if not is_poly:
                        rel_ = ((gch.name, r.name, int(r.seqid.num)) in bl["nonpoly"]) if covered else (r.name not in ADDITIVES)
                    else:
                        if covered:
                            rel_ = gch.name in bl["polychains"]
                        else:
                            rel_ = not is_aa  # nucleic acid
                        if not rel_:
                            partners.add(gch.name)
                            continue
                    (relevant.append(tag) if rel_ else nonrel.add(r.name))
            rec["n_biolip_ligands_on_chain"] = int(bl["per_chain"].get(ch, 0))
            rec["nonrelevant_het_near_site"] = ",".join(sorted(nonrel))
            rec["partner_chains_near_site"] = ",".join(sorted(partners))
            if relevant:
                log.append({**rec, "outcome": "holo_at_site", "relevant_near_site": ",".join(relevant[:10])})
                continue
            # ---- transfer
            per_site = []
            for si, s in enumerate(sites):
                out = dict(site=si, ok=False)
                if len(s["xyz"]) == 0 or not s["res"]:
                    per_site.append(out)
                    continue
                mapped = [i for i in s["res"] if mp[i] >= 0]
                out["coverage"] = len(mapped) / len(s["res"])
                sup = _local(rc.ca, mp, ac.ca, s["xyz"])
                if sup is None or out["coverage"] < P["min_site_coverage"]:
                    per_site.append(out)
                    continue
                Rm, t, lr, npairs = sup
                out.update(local_rmsd=lr, npairs=npairs, R=Rm, t=t)
                ca_p = [(rc.ca[i], ac.ca[mp[i]]) for i in mapped if not (np.isnan(rc.ca[i, 0]) or np.isnan(ac.ca[mp[i], 0]))]
                if ca_p:
                    x = np.array([p for p, _ in ca_p]) @ Rm.T + t
                    y = np.array([q for _, q in ca_p])
                    out["site_ca_rmsd"] = float(np.sqrt(np.mean(np.sum((x - y) ** 2, 1))))
                xs, ys = [], []
                for i in mapped:
                    ra, aa = _res_atoms(rc, i), _res_atoms(ac, int(mp[i]))
                    for nm in ra.keys() & aa.keys():
                        xs.append(ra[nm])
                        ys.append(aa[nm])
                if xs:
                    x = np.array(xs, float) @ Rm.T + t
                    out["pocket_rmsd"] = float(np.sqrt(np.mean(np.sum((x - np.array(ys, float)) ** 2, 1))))
                    out["pocket_atoms"] = len(xs)
                out["ok"] = lr <= P["max_local_rmsd"] and "pocket_rmsd" in out
                per_site.append(out)
            n_ok = sum(o["ok"] for o in per_site)
            rec["sites_transferred"] = n_ok
            rec["sites_parent"] = nsite
            if n_ok == 0:
                log.append({**rec, "outcome": "no_site_transferred"})
                continue
            prm = [o.get("pocket_rmsd", np.nan) if o["ok"] else np.nan for o in per_site]
            rec["max_pocket_rmsd"] = float(np.nanmax(prm))
            log.append({**rec, "outcome": "accepted"})
            accepted.append(dict(rec=rec, ac=ac, mp=mp, per_site=per_site, prm=np.array(prm)))
            break  # one chain per PDB entry
        st["candidates_examined"] += 1
    # ---- choose <= 2
    chosen = []
    if accepted:
        acc = sorted(accepted, key=lambda x: (-x["rec"]["max_pocket_rmsd"], x["rec"]["resolution"], x["rec"]["pdb"]))
        chosen.append(acc[0])
        rest = [x for x in acc[1:] if x["rec"]["pdb"] != acc[0]["rec"]["pdb"]]
        if rest and P["max_apo_per_entry"] > 1:
            v0 = np.nan_to_num(acc[0]["prm"], nan=0.0)
            rest.sort(key=lambda x: (-float(np.abs(np.nan_to_num(x["prm"], nan=0.0) - v0).sum()), x["rec"]["resolution"], x["rec"]["pdb"]))
            chosen.append(rest[0])
    rows = []
    for k, ch in enumerate(chosen):
        r = _save(task, E, rc, sites, ch, k)
        if r is not None:
            rows.append(r)
        else:
            st["chosen_but_no_site_after_near_filter"] += 1
    st["accepted"] += len(accepted)
    st["chosen"] += len(chosen)
    summ = dict(entry_id=task["entry_id"], split=task["split"], status="ok", n_candidates=len(task["cands"]),
                n_accepted=len(accepted), n_chosen=len(rows), seconds=round(time.time() - t0, 1))
    return summ, rows, st, log


def _save(task, E, rc, sites, ch, k):
    a = E.arrays
    ac, mp, per_site, rec = ch["ac"], ch["mp"], ch["per_site"], ch["rec"]
    akeys = ac.res_keys
    atree = cKDTree(ac.atoms)
    ptr = a["lig_ptr"]
    coords, el, lsite, lidx, lmin, amind = [], [], [], [], [], []
    # ligands farther than near_dist from the apo chain are implausible placements: dropped (counted);
    # a site left without ligands is treated as untransferred
    far = 0
    for o in per_site:
        if not o["ok"]:
            continue
        keep_l = []
        for i in sites[o["site"]]["lig"]:
            x = a["lig_coords"][ptr[i]:ptr[i + 1]].astype(float) @ o["R"].T + o["t"]
            if atree.query(x, k=1)[0].min() <= P["near_dist"]:
                keep_l.append(i)
            else:
                far += 1
        o["lig_keep"] = keep_l
        if not keep_l:
            o["ok"] = False
    keep_sites = [o for o in per_site if o["ok"]]
    if not keep_sites:
        return None
    new_site = {o["site"]: j for j, o in enumerate(keep_sites)}
    for o in keep_sites:
        for i in o["lig_keep"]:
            x = a["lig_coords"][ptr[i]:ptr[i + 1]].astype(float) @ o["R"].T + o["t"]
            d, _ = atree.query(x, k=1)
            coords.append(x)
            el.extend(a["lig_elements"][ptr[i]:ptr[i + 1]].tolist())
            lsite.append(new_site[o["site"]])
            lidx.append(i)
            lmin.append(float(d.min()))
            amind.append(d)
    arr = {}
    arr["lig_coords"] = (np.vstack(coords) if coords else np.zeros((0, 3))).astype(np.float32)
    arr["lig_elements"] = np.array(el, dtype="U2")
    arr["lig_ptr"] = np.concatenate([[0], np.cumsum([len(c) for c in coords])]).astype(np.int32)
    arr["lig_site"] = np.array(lsite, np.int32)
    lidx = np.array(lidx, int)
    for key in ("lig_ccd", "lig_id", "lig_source", "lig_is_ion", "lig_is_metal"):
        arr[key] = a[key][lidx] if len(lidx) else a[key][:0]
    arr["lig_parent_rmsd"] = a["lig_rmsd"][lidx] if len(lidx) else a["lig_rmsd"][:0]
    arr["lig_rmsd"] = np.array([per_site[int(a["lig_site"][i])]["local_rmsd"] for i in lidx], np.float32)
    arr["lig_min_dist"] = np.array(lmin, np.float32)
    arr["lig_atom_min_dist"] = (np.concatenate(amind) if amind else np.zeros(0)).astype(np.float32)
    arr["lig_parent_index"] = lidx.astype(np.int32)

    def mapkeys(res_idx):
        return [akeys[int(mp[i])] for i in res_idx if mp[i] >= 0]
    arr["site_residues"] = np.array([";".join(mapkeys(sites[o["site"]]["res"])) for o in keep_sites], dtype="U4000")
    for key in ("site_is_ion", "site_has_ion", "site_n_ligands", "site_n_sources"):
        arr[key] = a[key][[o["site"] for o in keep_sites]]
    arr["site_center"] = np.array([np.vstack([coords[j] for j in range(len(coords)) if lsite[j] == new_site[o["site"]]]).mean(0)
                                   for o in keep_sites], np.float32).reshape(-1, 3)
    arr["site_parent_index"] = np.array([o["site"] for o in keep_sites], np.int32)
    arr["site_local_rmsd"] = np.array([o["local_rmsd"] for o in keep_sites], np.float32)
    arr["site_ca_rmsd"] = np.array([o.get("site_ca_rmsd", np.nan) for o in keep_sites], np.float32)
    arr["site_pocket_rmsd"] = np.array([o["pocket_rmsd"] for o in keep_sites], np.float32)
    arr["site_npairs"] = np.array([o["npairs"] for o in keep_sites], np.int32)
    arr["site_coverage"] = np.array([o["coverage"] for o in keep_sites], np.float32)
    arr["site_cryptic_like"] = arr["site_pocket_rmsd"] >= P["cryptic_pocket_rmsd"]
    arr["site_clash_frac"] = np.array([float(np.mean(np.concatenate([amind[j] for j in range(len(amind)) if lsite[j] == new_site[o["site"]]]) < 2.2))
                                       for o in keep_sites], np.float32)
    arr["rep_res_keys"] = np.array(akeys, dtype="U8")
    arr["rep_seq"] = np.array(ac.seq)
    # masks: parent polymer-ligand residues + residues of parent sites that could not be transferred
    rk = {kk: i for i, kk in enumerate(rc.res_keys)}
    masks, kinds = [], []
    for kind, v in zip(a["mask_kinds"], a["mask_residues"]):
        kinds.append(str(kind))
        masks.append(";".join(mapkeys([rk[x] for x in str(v).split(";") if x in rk])) if v else "")
    untr = [o["site"] for o in per_site if not o["ok"]]
    kinds.append("untransferred_site")
    masks.append(";".join(sorted({x for s in untr for x in mapkeys(sites[s]["res"])})))
    arr["mask_kinds"] = np.array(kinds)
    arr["mask_residues"] = np.array(masks, dtype="U8000")
    apo_id = f'{task["entry_id"]}__apo_{rec["pdb"]}_{rec["chain"]}'
    meta = dict(entry_id=apo_id, parent_entry_id=task["entry_id"], split=task["split"], uniprot=E.meta.get("uniprot"),
                pdb=rec["pdb"], chain=rec["chain"], structure_path=str(pdb_entry_path(rec["pdb"])), resolution=rec["resolution"],
                method=task["method_of"].get(rec["pdb"], ""), release_date=rec["release_date"], biolip_covered=rec["biolip_covered"],
                align_identity=rec["align_identity"], n_biolip_ligands_on_chain=rec["n_biolip_ligands_on_chain"],
                nonrelevant_het_near_site=rec["nonrelevant_het_near_site"], partner_chains_near_site=rec["partner_chains_near_site"],
                sites_parent=rec["sites_parent"], sites_transferred=len(keep_sites), ligands_dropped_far=far, pick_rank=k,
                parent_pdb=E.meta["pdb"], parent_chain=E.chain, test_union_sha256=E.meta.get("test_union_sha256"), params=P)
    arr["meta_json"] = np.array(json.dumps(meta))
    outp = OUT / task["split"] / "entries" / f"{apo_id}.npz"
    np.savez_compressed(outp, **arr)
    return dict(entry_id=apo_id, parent_entry_id=task["entry_id"], split=task["split"], pdb=rec["pdb"], chain=rec["chain"],
                structure_path=meta["structure_path"], resolution=rec["resolution"], method=meta["method"], release_date=rec["release_date"],
                biolip_covered=rec["biolip_covered"], align_identity=rec["align_identity"], n_res=len(akeys), pick_rank=k,
                sites_parent=rec["sites_parent"], n_sites=int(len(keep_sites)), n_sites_cryptic_like=int(arr["site_cryptic_like"].sum()),
                max_pocket_rmsd=float(arr["site_pocket_rmsd"].max()), max_local_rmsd=float(arr["site_local_rmsd"].max()),
                n_ligands=int(len(lidx)), n_ligands_dropped_far=far, n_biolip_ligands_on_chain=rec["n_biolip_ligands_on_chain"],
                partner_chains_near_site=rec["partner_chains_near_site"], npz=str(outp.relative_to(ROOT)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--max-pdb", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cand = pd.read_parquet(OUT / "candidates.parquet")
    method_of = dict(zip(cand.pdb, cand.method))
    # BioLiP2 rows (all, any resolution, incl. polymer ligands) for candidate entries
    need = set(cand.pdb)
    bl = defaultdict(lambda: {"nonpoly": set(), "polychains": set(), "per_chain": Counter()})
    with gzip.open(BIOLIP, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[0] not in need:
                continue
            d = bl[f[0]]
            d["per_chain"][f[1]] += 1
            if f[4] in ("peptide", "rna", "dna"):
                d["polychains"].add(f[5])
            else:
                try:
                    d["nonpoly"].add((f[5], f[4], int(f[19].strip().split()[0])))
                except (ValueError, IndexError):
                    pass
    man = {s: pd.read_csv(PROC / s / "manifest.csv", dtype=str, keep_default_na=False) for s in ("train", "dev")}
    tasks = []
    for s, m in man.items():
        (OUT / s / "entries").mkdir(parents=True, exist_ok=True)
        cs = cand[cand.split == s]
        g = dict(tuple(cs.groupby("entry_id")))
        mm = m if not args.limit else m.iloc[:args.limit]
        for r in mm.itertuples():
            c = g.get(r.entry_id)
            if c is None:
                continue
            per_pdb = []
            for pdb, gg in c.groupby("pdb", sort=False):
                per_pdb.append(dict(pdb=pdb, resolution=float(gg.resolution.iloc[0]), chains=list(gg.chain)))
            per_pdb.sort(key=lambda x: (x["resolution"], x["pdb"]))
            per_pdb = per_pdb[:args.max_pdb]
            tasks.append(dict(entry_id=r.entry_id, split=s, npz=r.npz, cands=per_pdb,
                              biolip={p["pdb"]: {k: (dict(v) if k == "per_chain" else v) for k, v in bl[p["pdb"]].items()} for p in per_pdb},
                              method_of={p["pdb"]: method_of.get(p["pdb"], "") for p in per_pdb}))
    print(f"tasks {len(tasks)}", flush=True)
    summ, rows, logs, stats = [], [], [], Counter()
    t0 = time.time()
    with Pool(args.procs, maxtasksperchild=100) as pool:
        for i, res in enumerate(pool.imap_unordered(process, tasks, chunksize=1)):
            if len(res) == 3:
                summ.append(res[0])
                stats.update(res[2])
            else:
                s_, r_, st_, lg = res
                summ.append(s_)
                rows.extend(r_)
                stats.update(st_)
                logs.extend(lg)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    rows = pd.DataFrame(rows)
    for s in ("train", "dev"):
        (rows[rows.split == s] if len(rows) else rows).sort_values("entry_id").to_csv(OUT / s / "manifest.csv", index=False)
    pd.DataFrame(logs).to_csv(OUT / "work" / "candidate_log.csv", index=False)
    sm = pd.DataFrame(summ)
    sm.to_csv(OUT / "work" / "entry_summary.csv", index=False)
    lg = pd.DataFrame(logs)
    counts = dict(params=P, max_pdb=args.max_pdb, tasks=len(tasks), errors=int((sm.status != "ok").sum()),
                  error_examples=sm.loc[sm.status != "ok", "status"].head(5).tolist(),
                  candidate_outcomes=lg.outcome.value_counts().to_dict() if len(lg) else {},
                  per_split={s: dict(entries_total=int(len(man[s])), entries_with_candidates=int((sm.split == s).sum()),
                                     entries_with_apo=int(((sm.split == s) & (sm.get("n_chosen", 0) > 0)).sum()) if len(sm) else 0,
                                     apo_inputs=int((rows.split == s).sum()) if len(rows) else 0,
                                     sites=int(rows.n_sites[rows.split == s].sum()) if len(rows) else 0,
                                     cryptic_like_sites=int(rows.n_sites_cryptic_like[rows.split == s].sum()) if len(rows) else 0)
                             for s in ("train", "dev")},
                  stats=dict(stats), wall_s=round(time.time() - t0, 1))
    json.dump(counts, open(OUT / "pool_counts.json", "w"), indent=1, default=str)
    print(json.dumps(counts, indent=1, default=str))


if __name__ == "__main__":
    main()
