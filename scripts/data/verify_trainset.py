#!/usr/bin/env python
"""Independent verification of the training/dev sets (docs/TRAINSET.md §Verification).

Checks (all recomputed from the final manifests / npz files, NOT from intermediate build tables):
 1. counts: entries, sites, ligands, ion fractions, sites-per-protein distribution, assemblies.
 2. test overlap: sequences of every representative chain (observed residues, re-extracted with
    pocketdetect.structure.load_structure) and of every pooled source chain and assembly partner
    chain are searched with MMseqs2 against data/manifests/test_union.tsv (sha256 must match the one
    recorded in the entries); the leakage rule must give 0 hits. Canonical full-entity sequences
    (mmCIF _entity_poly) of the representatives are checked as well (reported separately).
    PDB-ID and UniProt (mmCIF _struct_ref + SIFTS) overlap with the test union must be empty.
    Train-vs-dev: dev representatives vs train representatives+sources, same rule, 0 hits expected.
 3. spot checks on a seeded random sample: pooled ligand heavy atoms vs the representative chain
    loaded independently: non-ion min distance >= 2.2 A, every ligand within 4.5 A of the protein,
    site residues exist in the chain, own ligands (RMSD 0) coincide with the deposited HETATM atoms.
    Same distance checks for stored assembly inputs.
Writes data/processed/train/verify_report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.datasets.leakage import apply_rule, mmseqs_search  # noqa: E402
from pocketdetect.datasets.pdbinfo import entry_info  # noqa: E402
from pocketdetect.datasets.trainset import load_entry, read_manifest  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402
from pocketdetect.structure import load_structure, pdb_entry_path, residue_id  # noqa: E402

MAN = ROOT / "data" / "manifests"
PROC = ROOT / "data" / "processed"
SIFTS = PROC / "af" / "external" / "pdb_chain_uniprot.tsv.gz"


def _one(names):
    out = []
    for n in names:
        info = gemmi.find_tabulated_residue(n)
        c = info.one_letter_code.upper() if info is not None else "X"
        out.append(c if c.isalpha() and c != "?" else "X")
    return "".join(out)


def observed_seq(args):
    pdb, chain = args
    try:
        s = load_structure(pdb_entry_path(pdb), chains=[chain], protein_only=True)
        return pdb, chain, _one(list(s.r_name)), None
    except Exception as e:  # noqa: BLE001
        return pdb, chain, "", str(e)


def entity_info_safe(pdb):
    try:
        return pdb, entry_info(pdb)
    except Exception:  # noqa: BLE001
        return pdb, None


def spot_check(npz):
    e = load_entry(npz)
    a = e.arrays
    r = dict(entry=e.entry_id, n_lig=len(a["lig_ccd"]))
    s = load_structure(e.meta["structure_path"], chains=[e.chain])
    prot = s.protein_heavy_mask()
    tree = cKDTree(s.coords[prot])
    rids = set(s.residue_ids[np.unique(s.res_index[prot])])
    ptr = a["lig_ptr"]
    mind, near_ok, clash_ok, own_dev = [], 0, 0, []
    sf = load_structure(e.meta["structure_path"])  # whole entry: own ligands may sit in another author chain
    lig_mask = sf.is_ligand_candidate & ~sf.is_hydrogen
    ltree = cKDTree(sf.coords[lig_mask]) if lig_mask.any() else None
    for i in range(len(a["lig_ccd"])):
        xyz = a["lig_coords"][ptr[i]:ptr[i + 1]]
        d, _ = tree.query(xyz, k=1)
        mind.append(float(d.min()))
        near_ok += bool(d.min() <= 4.5)
        clash_ok += bool(a["lig_is_ion"][i] or d.min() >= 2.2 - 1e-3)
        if a["lig_rmsd"][i] == 0 and a["lig_source"][i] == f'{e.meta["pdb"]}_{e.chain}' and ltree is not None:
            dd, _ = ltree.query(xyz, k=1)
            own_dev.append(float(dd.max()))
    missing_res = sum(1 for site in e.site_residues() for k in site if f"{e.chain}:{k}" not in rids)
    r.update(near_ok=near_ok, clash_ok=clash_ok, min_dists=mind, own_max_dev=own_dev, missing_site_res=missing_res,
             n_site_res=sum(len(x) for x in e.site_residues()))
    if e.has_assembly:
        sa = load_structure(str(a["asm_path"]), protein_only=True)
        t2 = cKDTree(sa.coords[~sa.is_hydrogen])
        p2 = a["asm_lig_ptr"]
        an, ac = 0, 0
        for i in range(len(a["asm_lig_ccd"])):
            d, _ = t2.query(a["asm_lig_coords"][p2[i]:p2[i + 1]], k=1)
            an += bool(d.min() <= 4.5)
            ac += bool(a["asm_lig_is_ion"][i] or d.min() >= 2.2 - 1e-3)
        r.update(asm_n_lig=len(a["asm_lig_ccd"]), asm_near_ok=an, asm_clash_ok=ac)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--n-spot", type=int, default=400)
    ap.add_argument("--tmp", default="/tmp/pocketdetect_verify_mmseqs")
    args = ap.parse_args()
    rep = {}
    W = PROC / "train" / "verify_work"
    W.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ 1. counts
    mans = {s: read_manifest(s) for s in ("train", "dev")}
    tu_sha = hashlib.sha256((MAN / "test_union.tsv").read_bytes()).hexdigest()
    rep["test_union_sha256"] = tu_sha
    allsrc, entries, asm_entries = {}, {}, set()
    for s, m in mans.items():
        sps, lig_ion, lig_n, site_ion, metal, asm, asm_int, asm_sites, shas = [], 0, 0, 0, 0, 0, 0, 0, set()
        srcs = set()
        for p in m.npz:
            e = load_entry(p)
            a = e.arrays
            shas.add(e.meta["test_union_sha256"])
            sps.append(e.n_sites)
            lig_n += len(a["lig_ccd"])
            lig_ion += int(a["lig_is_ion"].sum())
            metal += int(a["lig_is_metal"].sum())
            site_ion += int(a["site_is_ion"].sum())
            srcs.update(str(x) for x in np.unique(a["lig_source"]))
            if e.has_assembly:
                asm_entries.add(e.entry_id)
                asm += 1
                asm_sites += len(a["asm_site_residues"])
                asm_int += int(a["asm_site_is_interface"].sum())
        sps = np.array(sps)
        allsrc[s] = srcs
        rep[s] = dict(entries=len(m), entries_per_cluster30=float(len(m) / max(1, m.cluster30.nunique())),
                      clusters30=int(m.cluster30.nunique()), uniprots=int(m.uniprot.nunique()),
                      rank0_entries=int((m.rank_in_cluster == 0).sum()),
                      entries_without_sites=int((sps == 0).sum()), sites=int(sps.sum()),
                      sites_per_protein=dict(mean=float(sps.mean()), median=float(np.median(sps)), p90=float(np.percentile(sps, 90)),
                                             max=int(sps.max()), hist={str(k): int(v) for k, v in sorted(Counter(np.minimum(sps, 15)).items())}),
                      ion_site_fraction=float(site_ion / max(1, sps.sum())), ligands=lig_n,
                      ion_ligand_fraction=float(lig_ion / max(1, lig_n)), metal_ligand_fraction=float(metal / max(1, lig_n)),
                      assembly_inputs=asm, assembly_fraction=float(asm / max(1, len(m))), assembly_sites=asm_sites,
                      assembly_interface_sites=asm_int, pooled_source_chains=len(srcs),
                      test_union_sha_in_entries=sorted(shas), test_union_sha_ok=shas == {tu_sha})
        print(s, json.dumps(rep[s]), flush=True)

    # ------------------------------------------------------------------ 2. overlap re-check
    reps = {s: list(zip(m.pdb, m.chain)) for s, m in mans.items()}
    src_chains = {s: sorted({tuple(x.rsplit("_", 1)) for x in allsrc[s]}) for s in mans}
    jobs = sorted(set(reps["train"]) | set(reps["dev"]) | set(src_chains["train"]) | set(src_chains["dev"]))
    with ProcessPoolExecutor(args.procs) as ex:
        obs = {(p, c): (sq, err) for p, c, sq, err in ex.map(observed_seq, jobs, chunksize=8)}
    rep["observed_seq_failures"] = sum(1 for v in obs.values() if v[1])
    seqs = {f"{p}_{c}": sq for (p, c), (sq, err) in obs.items() if sq}
    # assembly partner chains (written by the builder) are inputs too
    partner_fa = PROC / "train" / "work" / "assembly_partner_seqs.fasta"
    partners = {}
    if partner_fa.exists():
        k = None
        for line in open(partner_fa):
            if line.startswith(">"):
                k = line[1:].strip()
            elif k.split("|")[0] in asm_entries:  # only assemblies still stored after the partner check
                partners[k] = line.strip()
    with open(W / "final_chains.fasta", "w") as fh:
        for k, sq in seqs.items():
            fh.write(f">{k}\n{sq}\n")
        for k, sq in partners.items():
            if len(sq) >= 10:
                fh.write(f">asm|{k}\n{sq}\n")
    tu = pd.read_csv(MAN / "test_union.tsv", sep="\t", dtype=str)
    with open(W / "test_union.fasta", "w") as fh:
        for i, sq in enumerate(sorted(set(tu.sequence))):
            fh.write(f">t{i}\n{sq}\n")
    m8 = mmseqs_search(W / "final_chains.fasta", W / "test_union.fasta", W / "final_vs_test.m8", Path(args.tmp) / "v1", threads=args.procs)
    hits = m8[apply_rule(m8)]
    rep_ids = {f"{p}_{c}" for s in reps for p, c in reps[s]}
    rep["test_overlap_observed_seq"] = dict(
        queries=len(seqs) + len(partners), rule_hits_total=len(hits),
        representatives_hit=sorted(set(hits["query"]) & rep_ids)[:50], n_representatives_hit=len(set(hits["query"]) & rep_ids),
        n_source_chains_hit=len({q for q in hits["query"] if not q.startswith("asm|")} - rep_ids),
        n_assembly_partner_chains_hit=len({q for q in hits["query"] if q.startswith("asm|")}),
        assembly_entries_with_partner_hit=sorted({q.split("|")[1] for q in hits["query"] if q.startswith("asm|")}))
    # canonical full entity sequences of representatives + PDB/UniProt overlap
    pdbs = sorted({p for s in reps for p, _ in reps[s]} | {p for s in src_chains for p, _ in src_chains[s]})
    with ProcessPoolExecutor(args.procs) as ex:
        infos = dict(ex.map(entity_info_safe, pdbs, chunksize=8))
    can, unp = {}, {}
    for s in reps:
        for p, c in reps[s] + src_chains[s]:
            inf = infos.get(p)
            if inf is None:
                continue
            for e in inf["entities"]:
                if c in e["chains"]:
                    can[f"{p}_{c}"] = e["seq"]
                    unp[f"{p}_{c}"] = set(e["uniprot"])
    if SIFTS.exists():
        sf = pd.read_csv(SIFTS, sep="\t", comment="#", dtype=str, keep_default_na=False, usecols=["PDB", "CHAIN", "SP_PRIMARY"])
        for p, c, u in zip(sf.PDB, sf.CHAIN, sf.SP_PRIMARY):
            k = f"{p.lower()}_{c}"
            if k in unp or k in seqs:
                unp.setdefault(k, set()).add(u)
    with open(W / "rep_canonical.fasta", "w") as fh:
        for k in sorted(rep_ids):
            if can.get(k):
                fh.write(f">{k}\n{can[k]}\n")
    m8c = mmseqs_search(W / "rep_canonical.fasta", W / "test_union.fasta", W / "repcan_vs_test.m8", Path(args.tmp) / "v2", threads=args.procs)
    hc = m8c[apply_rule(m8c)]
    rep["test_overlap_canonical_entity_seq_reps"] = dict(rule_hits=len(hc), n_representatives_hit=int(hc["query"].nunique()),
                                                         examples=sorted(hc["query"].unique())[:30])
    test_pdb = set((MAN / "test_union_pdb_ids.txt").read_text().split())
    test_unp = set((MAN / "test_union_uniprots.txt").read_text().split())
    rep["pdb_id_overlap"] = sorted({p for s in reps for p, _ in reps[s] + src_chains[s]} & test_pdb)
    rep["uniprot_overlap"] = sorted({u for k, us in unp.items() for u in us} & test_unp)[:100]
    rep["uniprot_overlap_n"] = len({u for k, us in unp.items() for u in us} & test_unp)
    # train vs dev
    with open(W / "dev.fasta", "w") as fh:
        for p, c in reps["dev"]:
            if seqs.get(f"{p}_{c}"):
                fh.write(f">{p}_{c}\n{seqs[f'{p}_{c}']}\n")
    with open(W / "train.fasta", "w") as fh:
        for p, c in sorted(set(reps["train"]) | set(src_chains["train"])):
            if seqs.get(f"{p}_{c}"):
                fh.write(f">{p}_{c}\n{seqs[f'{p}_{c}']}\n")
    m8d = mmseqs_search(W / "dev.fasta", W / "train.fasta", W / "dev_vs_train.m8", Path(args.tmp) / "v3", threads=args.procs)
    hd = m8d[apply_rule(m8d)]
    rep["dev_vs_train"] = dict(rule_hits=len(hd), n_dev_reps_hit=int(hd["query"].nunique()), examples=sorted(hd["query"].unique())[:20])
    rep["uniprot_shared_train_dev"] = len(set(mans["train"].uniprot) & set(mans["dev"].uniprot))
    print(json.dumps({k: rep[k] for k in ("test_overlap_observed_seq", "test_overlap_canonical_entity_seq_reps", "uniprot_overlap_n", "dev_vs_train")}, default=str), flush=True)

    # ------------------------------------------------------------------ 3. spot checks
    rng = np.random.default_rng(SEEDS[0])
    allm = pd.concat(mans.values())
    sample = list(allm.npz.iloc[rng.choice(len(allm), size=min(args.n_spot, len(allm)), replace=False)])
    asm_m = allm[allm.assembly_status == "stored"]
    sample += list(asm_m.npz.iloc[rng.choice(len(asm_m), size=min(100, len(asm_m)), replace=False)])
    sample = sorted(set(sample))
    with ProcessPoolExecutor(args.procs) as ex:
        res = list(ex.map(spot_check, sample, chunksize=2))
    nl = sum(r["n_lig"] for r in res)
    md = np.concatenate([r["min_dists"] for r in res]) if nl else np.zeros(0)
    own = np.concatenate([r["own_max_dev"] for r in res if r["own_max_dev"]]) if any(r["own_max_dev"] for r in res) else np.zeros(0)
    an = sum(r.get("asm_n_lig", 0) for r in res)
    rep["spot_checks"] = dict(
        n_entries=len(res), n_ligands=nl, frac_near_4p5=float(sum(r["near_ok"] for r in res) / max(1, nl)),
        frac_nonclash_2p2_nonion=float(sum(r["clash_ok"] for r in res) / max(1, nl)),
        min_dist_quantiles={q: float(np.quantile(md, q)) for q in (0.0, 0.01, 0.1, 0.5, 0.9, 1.0)} if nl else {},
        own_ligands_checked=int(len(own)), own_ligand_max_dev_A=float(own.max()) if len(own) else None,
        own_ligands_dev_gt_0p01=int((own > 0.01).sum()),
        site_residues_missing=int(sum(r["missing_site_res"] for r in res)), site_residues_checked=int(sum(r["n_site_res"] for r in res)),
        assembly_entries=int(sum(1 for r in res if "asm_n_lig" in r)), assembly_ligands=an,
        assembly_frac_near=float(sum(r.get("asm_near_ok", 0) for r in res) / max(1, an)),
        assembly_frac_nonclash=float(sum(r.get("asm_clash_ok", 0) for r in res) / max(1, an)))
    print(json.dumps(rep["spot_checks"]), flush=True)
    ok = (rep["test_overlap_observed_seq"]["n_representatives_hit"] == 0 and rep["test_overlap_observed_seq"]["n_assembly_partner_chains_hit"] == 0
          and rep["test_overlap_canonical_entity_seq_reps"]["n_representatives_hit"] == 0
          and not rep["pdb_id_overlap"] and rep["uniprot_overlap_n"] == 0
          and rep["dev_vs_train"]["rule_hits"] == 0 and all(rep[s]["test_union_sha_ok"] for s in ("train", "dev"))
          and rep["spot_checks"]["frac_near_4p5"] == 1.0 and rep["spot_checks"]["frac_nonclash_2p2_nonion"] == 1.0)
    rep["PASS"] = bool(ok)
    json.dump(rep, open(PROC / "train" / "verify_report.json", "w"), indent=1, default=str)
    print("PASS" if ok else "FAIL (see verify_report.json)")


if __name__ == "__main__":
    main()
