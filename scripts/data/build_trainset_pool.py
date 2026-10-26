#!/usr/bin/env python
"""Step 5+6: representatives, LIGYSIS-style ligand pooling, site clustering, assembly-1 inputs.

Per selected UniProt (per split; chains without UniProt are grouped per 30% cluster):
  * representative chain = among chains with observed length >= 0.9 x the longest chain of that
    UniProt, the best resolution (ties: more BioLiP ligands, then PDB ID);
  * up to 20 OTHER PDB entries of the same UniProt are pooled (one chain per entry, the one with most
    BioLiP ligands), chosen greedily to maximise new ligand CCD codes, then by resolution;
  * every BioLiP-relevant ligand (incl. ions; polymer ligands excluded) of a pooled chain is
    superposed onto the representative by sequence-anchored local superposition: Calpha of source
    residues within 10 A of any ligand heavy atom, mapped by global alignment (identity >= 0.9 over
    aligned columns required), >= 6 pairs, kept if Calpha RMSD <= 2.0 A (recorded);
  * plausibility: ligands > 4.5 A from every representative protein heavy atom are dropped;
    non-ion ligand atoms < 2.2 A from a representative protein heavy atom are dropped (DESIGN §9),
    sites losing > 50% of their (pre-clash) ligand heavy atoms are dropped;
  * sites = LIGYSIS clustering of ligand residue fingerprints (BioLiP binding residues of the source
    chain mapped onto the representative; relative intersection, average linkage, cut 0.5);
  * polymer-ligand (peptide/RNA/DNA) binding residues of all chains of the UniProt are mapped onto
    the representative and stored as `mask_residues` (not labels; for optional negative masking);
  * for a seeded ~30% of entries with an assembly-1 file: pooled ligands are mapped onto every copy
    of the UniProt in assembly 1 (local superposition again) plus the own BioLiP ligands of other
    entities' chains (exact superposition original->assembly chain), and re-clustered with
    contact fingerprints (4 A, assembly residues) -> interface sites exposed.

Selection (step 6): every 30% cluster contributes its richest UniProt (most holo entries); further
UniProts (up to --max-per-cluster) from different 50%-identity clusters are added, richest first.

Outputs: data/processed/{train,dev}/entries/<entry_id>.npz, data/processed/{train,dev}/manifest.csv,
data/processed/train/pool_counts.json, data/processed/train/work/assembly_partner_seqs.fasta
"""
from __future__ import annotations

import argparse
import re
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.datasets.biolip import METALS  # noqa: E402
from pocketdetect.datasets.leakage import MMSEQS  # noqa: E402
from pocketdetect.datasets.pooling import align_map, cluster_sites, kabsch, load_entry, parse_bres  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402
from pocketdetect.structure import pdb_assembly_path, pdb_entry_path  # noqa: E402

PROC = ROOT / "data" / "processed"
WORK = PROC / "train" / "work"
SIFTS = PROC / "af" / "external" / "pdb_chain_uniprot.tsv.gz"

P = dict(max_sources=20, pocket_ca_radius=10.0, min_pairs=6, max_rmsd=2.0, min_align_identity=0.9,
         near_dist=4.5, clash_dist=2.2, site_max_atom_loss=0.5, site_cut=0.5, assembly_frac=0.30,
         assembly_contact=4.0, interface_min_res=3, max_assembly_chains=60, max_assembly_atoms=150_000)


# ----------------------------------------------------------------------------------------- worker
_CACHE: dict = {}


def _entry(pdb: str):
    if pdb in _CACHE:
        return _CACHE[pdb]
    p = pdb_entry_path(pdb)
    ed = load_entry(p) if p.exists() else None
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[pdb] = ed
    return ed


def _lig_instance(ed, lc, ccd, num):
    m = re.match(r"^\s*(-?\d+)", str(num))  # BioLiP residue numbers may carry an insertion code ('500A')
    if m is None:
        return None, None
    inst = ed.ligands.get((lc, ccd, int(m.group(1))))
    if not inst:
        return None, None
    xyz = np.vstack([x for x, _ in inst])
    el = sum([e for _, e in inst], [])
    return xyz, el


def _local_superpose(src_ca, map_sr, dst_ca, lig_xyz):
    """Superpose source pocket Calpha (within radius of ligand) onto destination; returns (R,t,rmsd,npairs)."""
    ok = ~np.isnan(src_ca[:, 0])
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return None
    tree = cKDTree(src_ca[idx])
    near = set()
    for lst in tree.query_ball_point(lig_xyz, P["pocket_ca_radius"]):
        near.update(idx[j] for j in lst)
    near = np.array(sorted(near), int)
    if len(near) == 0:
        return None
    m = map_sr[near]
    keep = m >= 0
    near, m = near[keep], m[keep]
    keep = ~np.isnan(dst_ca[m, 0])
    near, m = near[keep], m[keep]
    if len(near) < P["min_pairs"]:
        return ("few_pairs", len(near))
    R, t, rmsd = kabsch(src_ca[near], dst_ca[m])
    return (R, t, rmsd, len(near))


def process(task):
    t0 = time.time()
    st = Counter()
    try:
        out = _process(task, st)
    except Exception as e:  # noqa: BLE001
        return dict(entry_id=task["entry_id"], status=f"error: {type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:]), st, []
    out[0]["seconds"] = round(time.time() - t0, 2)
    return out


def _process(task, st):
    eid = task["entry_id"]
    rp, rc_name = task["rep"]
    R = _entry(rp)
    if R is None or rc_name not in R.chains:
        return dict(entry_id=eid, status="rep_missing"), st, []
    rc = R.chains[rc_name]
    rkeys = rc.res_keys
    rtree = cKDTree(rc.atoms)
    rkidx = rc.key_index

    ligs = []  # dicts: coords, el, ccd, source, rmsd, npairs, fp(set of rep res keys), is_ion, is_metal

    def add_lig(xyz, el, row, source, rmsd, npairs, fp):
        heavy = len(xyz)
        ligs.append(dict(xyz=np.asarray(xyz, float), el=list(el), ccd=row["ccd"], source=source, rmsd=rmsd,
                         npairs=npairs, fp=fp, is_ion=bool(heavy == 1 or row["is_ion"]),
                         is_metal=bool((heavy == 1 and el[0] in METALS) or row["is_metal"]),
                         lig_id=f'{row["ccd"]}|{row["lig_chain"]}:{row["lig_resnum"]}'))

    # own ligands
    for row in task["rep_ligs"]:
        xyz, el = _lig_instance(R, row["lig_chain"], row["ccd"], row["lig_resnum"])
        st["lig_rows"] += 1
        if xyz is None:
            st["lig_not_found"] += 1
            continue
        fp = {k for k in parse_bres(row["bres"]) if k in rkidx}
        add_lig(xyz, el, row, f"{rp}_{rc_name}", 0.0, 0, fp)
    # pooled ligands
    n_src_used = 0
    for (sp, sc_name), rows in task["sources"]:
        S = _entry(sp)
        st["src_entries"] += 1
        if S is None or sc_name not in S.chains:
            st["src_missing"] += 1
            continue
        sc = S.chains[sc_name]
        mp, ident, nal = align_map(sc.seq, rc.seq)
        if ident < P["min_align_identity"] or nal < 20:
            st["src_low_identity"] += 1
            continue
        skidx = sc.key_index
        used = False
        for row in rows:
            st["lig_rows"] += 1
            xyz, el = _lig_instance(S, row["lig_chain"], row["ccd"], row["lig_resnum"])
            if xyz is None:
                st["lig_not_found"] += 1
                continue
            sup = _local_superpose(sc.ca, mp, rc.ca, xyz)
            if sup is None or isinstance(sup[0], str):
                st["lig_few_pairs"] += 1
                continue
            Rm, t, rmsd, npairs = sup
            if rmsd > P["max_rmsd"]:
                st["lig_rmsd_gt2"] += 1
                continue
            fp = set()
            for k in parse_bres(row["bres"]):
                i = skidx.get(k)
                if i is not None and mp[i] >= 0:
                    fp.add(rkeys[mp[i]])
            add_lig(xyz @ Rm.T + t, el, row, f"{sp}_{sc_name}", rmsd, npairs, fp)
            used = True
        n_src_used += used

    # plausibility + clash filtering (representative chain only = deployed input)
    kept = []
    for L in ligs:
        d, _ = rtree.query(L["xyz"], k=1)
        L["min_dist"] = float(d.min())
        if L["min_dist"] > P["near_dist"]:
            st["lig_far"] += 1
            continue
        if not L["fp"]:
            st["lig_no_fingerprint"] += 1
            continue
        L["n_atoms_orig"] = len(L["xyz"])
        if L["is_ion"]:
            L["keep_atoms"] = np.ones(len(L["xyz"]), bool)
        else:
            L["keep_atoms"] = d >= P["clash_dist"]
        st["atoms_total"] += len(d)
        st["atoms_clash_dropped"] += int((~L["keep_atoms"]).sum())
        kept.append(L)
    st["lig_candidates_after_near"] += len(kept)

    labels = cluster_sites([L["fp"] for L in kept], cut=P["site_cut"]) if kept else np.zeros(0, int)
    sites = []
    for s in range(int(labels.max()) + 1 if len(labels) else 0):
        mem = [kept[i] for i in np.flatnonzero(labels == s)]
        orig = sum(L["n_atoms_orig"] for L in mem)
        rem = sum(int(L["keep_atoms"].sum()) for L in mem)
        if rem < (1 - P["site_max_atom_loss"]) * orig:
            st["sites_dropped_clash"] += 1
            st["lig_in_dropped_sites"] += len(mem)
            continue
        mem = [L for L in mem if L["keep_atoms"].any()]
        sites.append(mem)
    st["sites_kept"] += len(sites)

    # masked polymer-ligand residues (peptide / RNA / DNA) mapped onto the representative
    mask = defaultdict(set)
    for seq, bres_renum, kind in task["masked"]:
        mp, ident, nal = align_map(seq, rc.seq)
        if ident < P["min_align_identity"]:
            continue
        for tok in str(bres_renum).split():
            try:
                i = int(tok[1:]) - 1
            except ValueError:
                continue
            if 0 <= i < len(seq) and mp[i] >= 0:
                mask[kind].add(rkeys[mp[i]])

    # ---------------------------------------------------------------- assemble arrays
    arr = {}
    L_all = [L for mem in sites for L in mem]
    site_of = [si for si, mem in enumerate(sites) for _ in mem]
    coords = [L["xyz"][L["keep_atoms"]] for L in L_all]
    ptr = np.concatenate([[0], np.cumsum([len(c) for c in coords])]).astype(np.int32)
    arr["lig_coords"] = (np.vstack(coords) if coords else np.zeros((0, 3))).astype(np.float32)
    arr["lig_elements"] = np.array(sum([[e for e, k in zip(L["el"], L["keep_atoms"]) if k] for L in L_all], []), dtype="U2")
    arr["lig_ptr"] = ptr
    arr["lig_site"] = np.array(site_of, np.int32)
    arr["lig_ccd"] = np.array([L["ccd"] for L in L_all], dtype="U8")
    arr["lig_id"] = np.array([L["lig_id"] for L in L_all], dtype="U32")
    arr["lig_source"] = np.array([L["source"] for L in L_all], dtype="U16")
    arr["lig_rmsd"] = np.array([L["rmsd"] for L in L_all], np.float32)
    arr["lig_npairs"] = np.array([L["npairs"] for L in L_all], np.int32)
    arr["lig_is_ion"] = np.array([L["is_ion"] for L in L_all], bool)
    arr["lig_is_metal"] = np.array([L["is_metal"] for L in L_all], bool)
    arr["lig_min_dist"] = np.array([L["min_dist"] for L in L_all], np.float32)
    arr["lig_n_atoms_orig"] = np.array([L["n_atoms_orig"] for L in L_all], np.int32)
    arr["site_residues"] = np.array([";".join(sorted(set().union(*[L["fp"] for L in mem]), key=lambda k: rkidx[k])) for mem in sites], dtype="U4000")
    arr["site_is_ion"] = np.array([all(L["is_ion"] for L in mem) for mem in sites], bool)
    arr["site_has_ion"] = np.array([any(L["is_ion"] for L in mem) for mem in sites], bool)
    arr["site_n_ligands"] = np.array([len(mem) for mem in sites], np.int32)
    arr["site_n_sources"] = np.array([len({L["source"] for L in mem}) for mem in sites], np.int32)
    arr["site_center"] = np.array([np.vstack([L["xyz"][L["keep_atoms"]] for L in mem]).mean(0) for mem in sites], np.float32).reshape(-1, 3)
    arr["rep_res_keys"] = np.array(rkeys, dtype="U8")
    arr["rep_seq"] = np.array(rc.seq)
    arr["mask_residues"] = np.array([";".join(sorted(mask[k], key=lambda x: rkidx[x])) for k in ("peptide", "rna", "dna")], dtype="U8000")
    arr["mask_kinds"] = np.array(["peptide", "rna", "dna"])

    # ---------------------------------------------------------------- assembly 1
    partner_seqs = []
    asm_status = "not_selected"
    if task["do_assembly"]:
        asm_status, partner_seqs = _assembly(task, R, rc, sites, arr, st)
    meta = dict(entry_id=eid, split=task["split"], uniprot=task["uniprot"], pdb=rp, chain=rc_name,
                structure_path=str(pdb_entry_path(rp)), resolution=task["resolution"], method=task["method"],
                cluster30=task["cluster30"], group=task["group"], test_union_sha256=task["tu_sha"],
                n_sources_considered=len(task["sources"]), n_sources_used=n_src_used, params=P,
                assembly_status=asm_status)
    arr["meta_json"] = np.array(json.dumps(meta))
    outp = PROC / task["split"] / "entries" / f"{eid}.npz"
    np.savez_compressed(outp, **arr)
    row = dict(entry_id=eid, split=task["split"], status="ok", uniprot=task["uniprot"], pdb=rp, chain=rc_name,
               structure_path=str(pdb_entry_path(rp)), resolution=task["resolution"], method=task["method"],
               n_res=len(rkeys), cluster30=task["cluster30"], group=task["group"], rank_in_cluster=task["rank_in_cluster"],
               n_uniprot_entries=task["n_uniprot_entries"], n_sites=len(sites),
               n_sites_ion=int(arr["site_is_ion"].sum()), n_ligands=len(L_all), n_ligands_ion=int(arr["lig_is_ion"].sum()),
               n_sources_used=n_src_used, max_lig_rmsd=float(arr["lig_rmsd"].max()) if len(L_all) else 0.0,
               n_mask_residues=sum(len(v) for v in mask.values()), has_assembly1=task["has_assembly1"],
               assembly_status=asm_status, n_asm_sites=int(len(arr.get("asm_site_residues", []))),
               n_asm_interface_sites=int(arr["asm_site_is_interface"].sum()) if "asm_site_is_interface" in arr else 0,
               npz=str(outp.relative_to(ROOT)))
    return row, st, partner_seqs


def _assembly(task, R, rc, sites, arr, st):
    from pocketdetect.datasets.pooling import load_entry as _le
    ap = pdb_assembly_path(task["rep"][0], 1)
    if not ap.exists():
        return "no_file", []
    A = _le(ap, want_ligands=False)
    if len(A.chains) > P["max_assembly_chains"] or sum(len(c.atoms) for c in A.chains.values()) > P["max_assembly_atoms"]:
        return "too_large", []
    rc_name = task["rep"][1]
    unp_of = task["entry_chain_uniprot"]  # original chain -> uniprot (rep entry)
    ligs = []
    partner = []
    for cname, ac in A.chains.items():
        base = cname.split("-")[0] if cname not in R.chains else cname
        if base == rc_name or (unp_of.get(base) and unp_of.get(base) == task["uniprot"]):
            # pooled sites of the representative mapped by local superposition
            mp, ident, nal = align_map(rc.seq, ac.seq)
            if ident < P["min_align_identity"]:
                continue
            for si, mem in enumerate(sites):
                for L in mem:
                    xyz = L["xyz"]
                    sup = _local_superpose(rc.ca, mp, ac.ca, xyz)
                    if sup is None or isinstance(sup[0], str) or sup[2] > P["max_rmsd"]:
                        st["asm_lig_map_fail"] += 1
                        continue
                    Rm, t, rmsd, _ = sup
                    ligs.append(dict(xyz=xyz @ Rm.T + t, el=L["el"], ccd=L["ccd"], is_ion=L["is_ion"], is_metal=L["is_metal"],
                                     source=L["source"], rmsd=max(rmsd, L["rmsd"]), on_chain=cname, origin="pooled"))
        else:
            partner.append((f'{task["entry_id"]}|{cname}', ac.seq))
            oc = R.chains.get(base)
            rows = task["entry_other_ligs"].get(base, [])
            if oc is None or not rows:
                continue
            mp, ident, nal = align_map(oc.seq, ac.seq)
            pairs = np.flatnonzero((mp >= 0) & ~np.isnan(oc.ca[:, 0]))
            pairs = pairs[~np.isnan(ac.ca[mp[pairs], 0])]
            if len(pairs) < 3:
                continue
            Rm, t, rmsd = kabsch(oc.ca[pairs], ac.ca[mp[pairs]])
            if rmsd > 0.5:  # must be an exact copy
                continue
            for row in rows:
                xyz, el = _lig_instance(R, row["lig_chain"], row["ccd"], row["lig_resnum"])
                if xyz is None:
                    continue
                ligs.append(dict(xyz=xyz @ Rm.T + t, el=el, ccd=row["ccd"], is_ion=bool(len(xyz) == 1 or row["is_ion"]),
                                 is_metal=bool(row["is_metal"]), source=f'{task["rep"][0]}_{base}', rmsd=0.0,
                                 on_chain=cname, origin="own_other_entity"))
    if not ligs:
        return "no_ligands_mapped", partner
    # contact fingerprints in assembly frame, clash filtering vs all assembly protein atoms
    names = list(A.chains)
    atoms = np.vstack([A.chains[c].atoms for c in names])
    a_chain = np.concatenate([[i] * len(A.chains[c].atoms) for i, c in enumerate(names)])
    a_res = np.concatenate([A.chains[c].atom_res for c in names])
    tree = cKDTree(atoms)
    kept = []
    for L in ligs:
        d, _ = tree.query(L["xyz"], k=1)
        if d.min() > P["near_dist"]:
            continue
        L["keep"] = np.ones(len(d), bool) if L["is_ion"] else d >= P["clash_dist"]
        nb = set()
        for lst in tree.query_ball_point(L["xyz"], P["assembly_contact"]):
            nb.update(lst)
        L["fp"] = {f"{names[a_chain[j]]}:{A.chains[names[a_chain[j]]].res_keys[a_res[j]]}" for j in nb}
        if L["fp"]:
            kept.append(L)
    if not kept:
        return "no_ligands_mapped", partner
    lab = cluster_sites([L["fp"] for L in kept], cut=P["site_cut"])
    asites = []
    for s in range(int(lab.max()) + 1):
        mem = [kept[i] for i in np.flatnonzero(lab == s)]
        if sum(int(L["keep"].sum()) for L in mem) < 0.5 * sum(len(L["xyz"]) for L in mem):
            st["asm_sites_dropped_clash"] += 1
            continue
        asites.append([L for L in mem if L["keep"].any()])
    L_all = [L for mem in asites for L in mem]
    coords = [L["xyz"][L["keep"]] for L in L_all]
    arr["asm_path"] = np.array(str(ap))
    arr["asm_lig_coords"] = (np.vstack(coords) if coords else np.zeros((0, 3))).astype(np.float32)
    arr["asm_lig_ptr"] = np.concatenate([[0], np.cumsum([len(c) for c in coords])]).astype(np.int32)
    arr["asm_lig_site"] = np.array([si for si, mem in enumerate(asites) for _ in mem], np.int32)
    arr["asm_lig_ccd"] = np.array([L["ccd"] for L in L_all], dtype="U8")
    arr["asm_lig_is_ion"] = np.array([L["is_ion"] for L in L_all], bool)
    arr["asm_lig_is_metal"] = np.array([L["is_metal"] for L in L_all], bool)
    arr["asm_lig_rmsd"] = np.array([L["rmsd"] for L in L_all], np.float32)
    arr["asm_lig_on_chain"] = np.array([L["on_chain"] for L in L_all], dtype="U12")
    arr["asm_lig_origin"] = np.array([L["origin"] for L in L_all], dtype="U16")
    sres, schains, sint = [], [], []
    for mem in asites:
        fp = set().union(*[L["fp"] for L in mem])
        cnt = Counter(k.rsplit(":", 1)[0] for k in fp)
        chs = sorted(c for c, n in cnt.items() if n >= P["interface_min_res"])
        sres.append(";".join(sorted(fp)))
        schains.append(",".join(sorted(cnt)))
        sint.append(len(chs) >= 2)
    arr["asm_site_residues"] = np.array(sres, dtype="U8000")
    arr["asm_site_chains"] = np.array(schains, dtype="U200")
    arr["asm_site_is_interface"] = np.array(sint, bool)
    arr["asm_site_is_ion"] = np.array([all(L["is_ion"] for L in mem) for mem in asites], bool)
    arr["asm_chains"] = np.array(names, dtype="U12")
    st["asm_stored"] += 1
    return "stored", partner


# ----------------------------------------------------------------------------------------- main
def choose_rep(chs: pd.DataFrame) -> pd.Series:
    mx = chs.seq_len.max()
    c = chs[chs.seq_len >= 0.9 * mx].sort_values(["resolution", "n_lig", "seq_len", "pdb", "chain"],
                                                  ascending=[True, False, False, True, True])
    return c.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--max-per-cluster", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N entries per split")
    ap.add_argument("--tmp", default="/tmp/pocketdetect_trainset_mmseqs")
    args = ap.parse_args()

    sc = pd.read_parquet(PROC / "train" / "split_chains.parquet")
    cc = pd.read_parquet(WORK / "candidate_chains.parquet", columns=["pdb", "chain", "seq"])
    sc = sc.merge(cc, on=["pdb", "chain"], how="left")
    lig = pd.read_parquet(WORK / "candidate_ligands.parquet", columns=["pdb", "chain", "ccd", "lig_chain", "lig_resnum", "bres", "is_ion", "is_metal"])
    masked = pd.read_parquet(WORK / "masked_ligands.parquet", columns=["pdb", "chain", "ccd", "bres_renum"])
    tu_sha = json.load(open(PROC / "train" / "split_counts.json"))["test_union_sha256"]
    assert tu_sha == hashlib.sha256((ROOT / "data/manifests/test_union.tsv").read_bytes()).hexdigest(), "split built on a different test union"

    use = sc[sc.split.isin(["train", "dev"])].copy()
    use["ukey"] = np.where(use.uniprot != "", use.uniprot, "nounp-" + use.cluster30.astype(str))

    # 50% clusters for diversity among UniProts of a 30% cluster
    useqs = use.drop_duplicates("seq_id")
    fa = WORK / "use_seqs.fasta"
    with open(fa, "w") as fh:
        for a, s in zip(useqs.seq_id, useqs.seq):
            fh.write(f">{a}\n{s}\n")
    (Path(args.tmp) / "c50").mkdir(parents=True, exist_ok=True)
    c50f = WORK / "clu50_cluster.tsv"
    if not (c50f.exists() and c50f.stat().st_mtime > (PROC / "train" / "split_chains.parquet").stat().st_mtime):
      subprocess.run([str(MMSEQS), "easy-cluster", str(fa), str(WORK / "clu50"), str(Path(args.tmp) / "c50"), "--min-seq-id", "0.5",
                    "-c", "0.5", "--cov-mode", "0", "--threads", str(args.procs), "-v", "1"], check=True)
    c50 = pd.read_csv(WORK / "clu50_cluster.tsv", sep="\t", header=None, names=["rep", "member"], dtype=str)
    use["cluster50"] = use.seq_id.map(dict(zip(c50.member, c50.rep)))

    keyset = pd.MultiIndex.from_arrays([use.pdb, use.chain])
    lig = lig[pd.MultiIndex.from_arrays([lig.pdb, lig.chain]).isin(keyset)]
    masked = masked[pd.MultiIndex.from_arrays([masked.pdb, masked.chain]).isin(keyset)]
    lig_by_chain = {k: g.drop(columns=["pdb", "chain"]).to_dict("records") for k, g in lig.groupby(["pdb", "chain"], sort=False)}
    ccds_by_chain = {k: set(r["ccd"] for r in v) for k, v in lig_by_chain.items()}
    masked_by_chain = defaultdict(list)
    for r in masked.itertuples():
        masked_by_chain[(r.pdb, r.chain)].append((r.ccd, r.bres_renum))
    seq_of = dict(zip(zip(sc.pdb, sc.chain), sc.seq))
    # uniprot of every chain of an entry (BioLiP + SIFTS) for assembly mapping
    chain_unp = dict(zip(zip(sc.pdb, sc.chain), sc.uniprot))
    sifts_unp = {}
    if SIFTS.exists():
        s = pd.read_csv(SIFTS, sep="\t", comment="#", dtype=str, keep_default_na=False, usecols=["PDB", "CHAIN", "SP_PRIMARY"])
        for p, ch, u in zip(s.PDB, s.CHAIN, s.SP_PRIMARY):
            sifts_unp.setdefault((p.lower(), ch), u)
    ok_chains = set(zip(use.pdb, use.chain))  # chains allowed as label sources (not test-removed, same split)

    # ---- selection
    sel_rows = []
    for split in ("train", "dev"):
        u = use[use.split == split]
        rich = u.groupby("ukey").agg(n_entries=("pdb", "nunique"), cluster30=("cluster30", lambda x: x.mode().iloc[0]),
                                     cluster50=("cluster50", lambda x: x.mode().iloc[0]), group=("group", "first"),
                                     has_unp=("uniprot", lambda x: (x != "").any())).reset_index()
        rich = rich.sort_values(["cluster30", "has_unp", "n_entries", "ukey"], ascending=[True, False, False, True])
        for cl, g in rich.groupby("cluster30", sort=True):
            seen50, k = set(), 0
            for r in g.itertuples():
                if k >= args.max_per_cluster:
                    break
                if r.cluster50 in seen50:
                    continue
                seen50.add(r.cluster50)
                sel_rows.append(dict(split=split, ukey=r.ukey, cluster30=cl, group=r.group, n_entries=r.n_entries, rank_in_cluster=k))
                k += 1
    sel = pd.DataFrame(sel_rows)
    rng = np.random.default_rng([SEEDS[0], 30])
    tasks = []
    for split in ("train", "dev"):
        (PROC / split / "entries").mkdir(parents=True, exist_ok=True)
        s_sel = sel[sel.split == split].sort_values("ukey").reset_index(drop=True)
        if args.limit:
            s_sel = s_sel.iloc[:args.limit]
        asm_draw = rng.random(len(s_sel)) < P["assembly_frac"]
        u = use[use.split == split]
        ug = dict(tuple(u.groupby("ukey")))
        for i, r in s_sel.iterrows():
            chs = ug[r.ukey]
            rep = choose_rep(chs)
            others = chs[chs.pdb != rep.pdb].copy()
            others = others.sort_values(["n_lig", "resolution", "chain"], ascending=[False, True, True]).drop_duplicates("pdb")
            # greedy: new CCDs first, then resolution
            seen = set(ccds_by_chain.get((rep.pdb, rep.chain), set()))
            pool, cand = [], list(others.itertuples())
            while cand and len(pool) < P["max_sources"]:
                best = max(cand, key=lambda o: (len(ccds_by_chain.get((o.pdb, o.chain), set()) - seen), -o.resolution, o.pdb))
                cand.remove(best)
                pool.append(best)
                seen |= ccds_by_chain.get((best.pdb, best.chain), set())
            safe = lambda s: "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)
            eid = f"{safe(r.ukey)}__{rep.pdb}_{safe(rep.chain)}"
            m_list = []
            for o in chs.itertuples():
                for ccd, br in masked_by_chain.get((o.pdb, o.chain), [])[:200]:
                    m_list.append((o.seq, br, ccd))
            has_a1 = pdb_assembly_path(rep.pdb, 1).exists()
            tasks.append(dict(entry_id=eid, split=split, uniprot=r.ukey, rep=(rep.pdb, rep.chain), resolution=float(rep.resolution),
                              method=rep.method, cluster30=r.cluster30, group=r.group, tu_sha=tu_sha, rank_in_cluster=int(r.rank_in_cluster),
                              n_uniprot_entries=int(r.n_entries),
                              rep_ligs=lig_by_chain.get((rep.pdb, rep.chain), []),
                              sources=[((o.pdb, o.chain), lig_by_chain.get((o.pdb, o.chain), [])) for o in pool],
                              masked=m_list, do_assembly=bool(asm_draw[i] and has_a1), has_assembly1=has_a1,
                              entry_chain_uniprot={}, entry_other_ligs={}))
    # fill entry-level chain info for assembly tasks: UniProt of every chain of the entry (BioLiP,
    # else SIFTS); own ligands of other chains only if that chain is a usable chain of the SAME split
    split_chains = defaultdict(list)
    for p, c, s in zip(use.pdb, use.chain, use.split):
        split_chains[(p, s)].append(c)
    sifts_by_pdb = defaultdict(dict)
    for (p, c), u in sifts_unp.items():
        sifts_by_pdb[p][c] = u
    for t in tasks:
        if not t["do_assembly"]:
            continue
        p = t["rep"][0]
        cu = dict(sifts_by_pdb.get(p, {}))
        cu.update({c: u for (pp, c), u in chain_unp.items() if pp == p and u})
        t["entry_chain_uniprot"] = cu
        t["entry_other_ligs"] = {c: lig_by_chain.get((p, c), []) for c in split_chains.get((p, t["split"]), []) if c != t["rep"][1]}
    print(f"tasks: {len(tasks)} ({Counter(t['split'] for t in tasks)}); assembly-selected {sum(t['do_assembly'] for t in tasks)}", flush=True)

    rows, stats, partners = [], Counter(), []
    t0 = time.time()
    with Pool(args.procs, maxtasksperchild=200) as pool:
        for i, (row, st, part) in enumerate(pool.imap_unordered(process, sorted(tasks, key=lambda t: -len(t["sources"])), chunksize=1)):
            rows.append(row)
            stats.update(st)
            partners.extend(part)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    man = pd.DataFrame(rows)
    sel_info = sel.assign(entry_key=sel.ukey)
    for split in ("train", "dev"):
        m = man[man.split == split] if "split" in man else man.iloc[:0]
        m.sort_values("entry_id").to_csv(PROC / split / "manifest.csv", index=False)
    errs = man[~man.status.eq("ok")]
    errs.to_csv(WORK / "pool_failures.csv", index=False)
    with open(WORK / "assembly_partner_seqs.fasta", "w") as fh:
        for k, s in partners:
            fh.write(f">{k}\n{s}\n")
    ok = man[man.status.eq("ok")]
    counts = dict(params=P, max_per_cluster=args.max_per_cluster, test_union_sha256=tu_sha,
                  n_tasks=len(tasks), n_ok=len(ok), n_failed=len(errs), failures=errs.status.value_counts().to_dict(),
                  per_split={s: dict(entries=int((ok.split == s).sum()), entries_with_sites=int(((ok.split == s) & (ok.n_sites > 0)).sum()),
                                     sites=int(ok.n_sites[ok.split == s].sum()), ion_sites=int(ok.n_sites_ion[ok.split == s].sum()),
                                     ligands=int(ok.n_ligands[ok.split == s].sum()),
                                     assembly_stored=int(((ok.split == s) & (ok.assembly_status == "stored")).sum()),
                                     has_assembly1=int(((ok.split == s) & ok.has_assembly1).sum()))
                             for s in ("train", "dev")},
                  stats=dict(stats), wall_s=round(time.time() - t0, 1))
    json.dump(counts, open(PROC / "train" / "pool_counts.json", "w"), indent=1, default=str)
    print(json.dumps(counts, indent=1, default=str))


if __name__ == "__main__":
    main()
