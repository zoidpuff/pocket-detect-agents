#!/usr/bin/env python
"""Independent verification of the apo training/dev inputs (Track A2) -> data/processed/apo_train/verify_report.json.

1. counts: apo inputs, parents with >= 1 apo chain, sites, cryptic-like fraction, pocket-RMSD quantiles.
2. split integrity: every apo input's parent is in the SAME split's manifest; no apo (pdb, chain) used in
   both splits; apo chain != parent chain.
3. leakage: observed sequence re-extracted with pocketdetect.structure.load_structure, searched against
   the builder test union and the audit test set (results/audit/test_union.fasta) and against the other
   split's representatives: 0 rule hits required. PDB ID not in any test set; explicitly not a CryptoBench
   apo/holo entry nor a PocketMiner entry.
4. apo status (seeded sample): BioLiP2 rows of the apo entry re-read from BioLiP.txt.gz; no listed ligand
   residue (any chain) with a heavy atom within 6 A of the site residues (BioLiP-covered entries).
5. labels: every transferred ligand within 4.5 A of the apo chain; clash statistics reported.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.datasets.apotrain import load_apo_entry, read_apo_manifest  # noqa: E402
from pocketdetect.datasets.leakage import apply_rule, mmseqs_search_protein, read_m8  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402
from pocketdetect.structure import load_structure  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "apo_train"
W = OUT / "verify_work"
RAW = ROOT / "data" / "raw"


def obs_seq(args):
    eid, path, chain = args
    s = load_structure(path, chains=[chain], protein_only=True)
    out = []
    for n in s.r_name:
        info = gemmi.find_tabulated_residue(n)
        c = info.one_letter_code.upper() if info is not None else "X"
        out.append(c if c.isalpha() else "X")
    return eid, "".join(out)


def apo_check(args):
    npz, bl_rows = args
    e = load_apo_entry(npz)
    s = load_structure(e.meta["structure_path"])
    rid = s.atom_residue_ids
    site_keys = {f"{e.chain}:{k}" for site in e.site_residues() for k in site}
    sel = np.isin(rid, list(site_keys)) & s.is_protein & ~s.is_hydrogen
    tree = cKDTree(s.coords[sel])
    lig_keys = {(c, n, int(num)) for c, n, num in bl_rows["nonpoly"]}
    near = []
    for i in np.flatnonzero(~sel & ~s.is_water & ~s.is_hydrogen):
        r = s.res_index[i]
        key = (s.r_chain[r], s.r_name[r], int(s.r_num[r]))
        poly_lig = s.r_chain[r] in bl_rows["polychains"] and s.r_chain[r] != e.chain
        if key in lig_keys or poly_lig:
            d, _ = tree.query(s.coords[i], k=1)
            if d <= 6.0:
                near.append(f"{key[1]}|{key[0]}:{key[2]}")
    # transferred ligands near the apo chain
    ch = s.select(s.is_protein & ~s.is_hydrogen & (s.chain == e.chain))
    t2 = cKDTree(ch.coords)
    a = e.arrays
    ptr = a["lig_ptr"]
    mind = [float(t2.query(a["lig_coords"][ptr[i]:ptr[i + 1]], k=1)[0].min()) for i in range(len(a["lig_ccd"]))]
    return dict(entry=e.entry_id, covered=e.meta["biolip_covered"], relevant_near=sorted(set(near)), lig_min_dist=mind,
                n_lig=len(mind))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--n-spot", type=int, default=300)
    ap.add_argument("--tmp", default="/tmp/pocketdetect_apo_verify")
    args = ap.parse_args()
    W.mkdir(parents=True, exist_ok=True)
    rep = {}
    man = {s: read_apo_manifest(s) for s in ("train", "dev")}
    par = {s: pd.read_csv(PROC / s / "manifest.csv", dtype=str, keep_default_na=False) for s in ("train", "dev")}
    # 1. counts
    for s, m in man.items():
        prm, cry, nsites = [], 0, 0
        for p in m.npz:
            e = load_apo_entry(p)
            prm.extend(e.arrays["site_pocket_rmsd"].tolist())
            cry += int(e.arrays["site_cryptic_like"].sum())
            nsites += e.n_sites
        prm = np.array(prm)
        rep[s] = dict(apo_inputs=len(m), parents_total=len(par[s]), parents_with_apo=int(m.parent_entry_id.nunique()),
                      parents_with_2_apo=int((m.groupby("parent_entry_id").size() >= 2).sum()),
                      frac_parents_with_apo=float(m.parent_entry_id.nunique() / max(1, len(par[s]))),
                      sites=nsites, cryptic_like_sites=cry, cryptic_like_site_fraction=float(cry / max(1, nsites)),
                      apo_inputs_with_cryptic_like_site=int((pd.to_numeric(m.n_sites_cryptic_like) > 0).sum()),
                      pocket_rmsd_quantiles={q: float(np.quantile(prm, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)} if len(prm) else {})
        print(s, json.dumps(rep[s]), flush=True)
    # 2. split integrity
    rep["parent_in_same_split"] = all(set(man[s].parent_entry_id) <= set(par[s].entry_id) for s in man)
    rep["apo_chain_in_both_splits"] = len(set(zip(man["train"].pdb, man["train"].chain)) & set(zip(man["dev"].pdb, man["dev"].chain)))
    pc = {s: dict(zip(par[s].entry_id, zip(par[s].pdb, par[s].chain))) for s in par}
    rep["apo_equals_parent_chain"] = int(sum((r.pdb, r.chain) == pc[s].get(r.parent_entry_id) for s, m in man.items() for r in m.itertuples()))
    rep["apo_pdb_equals_parent_pdb"] = int(sum(r.pdb == pc[s].get(r.parent_entry_id, ("", ""))[0] for s, m in man.items() for r in m.itertuples()))
    # 3. leakage
    jobs = [(r.entry_id, r.structure_path, r.chain) for m in man.values() for r in m.itertuples()]
    with ProcessPoolExecutor(args.procs) as ex:
        seqs = dict(ex.map(obs_seq, jobs, chunksize=8))
    with open(W / "apo_obs.fasta", "w") as fh:
        for k, q in seqs.items():
            if len(q) >= 10:
                fh.write(f">{k}\n{q}\n")
    tu = pd.read_csv(ROOT / "data" / "manifests" / "test_union.tsv", sep="\t", dtype=str)
    with open(W / "tu.fasta", "w") as fh:
        for i, q in enumerate(sorted(set(tu.sequence))):
            fh.write(f">t{i}\n{q}\n")
    targets = {"builder_test_union": W / "tu.fasta"}
    audit = ROOT / "results" / "audit" / "test_union.fasta"
    if audit.exists():
        targets["audit_test_set"] = audit
    rep["leakage"] = {}
    for name, t in targets.items():
        m8 = W / f"apo_vs_{name}.m8"
        mmseqs_search_protein(W / "apo_obs.fasta", t, m8, Path(args.tmp) / name, threads=args.procs, extra=["-s", "7.5", "--max-seqs", "5000", "-e", "1e-3"])
        h = read_m8(m8)
        rep["leakage"][name] = dict(rule_hits=int(apply_rule(h).sum()), inputs_hit=sorted(set(h.loc[apply_rule(h), "query"]))[:20])
    for s, other in (("train", "dev"), ("dev", "train")):
        with open(W / f"rep_{other}.fasta", "w") as fh:
            for r in par[other].itertuples():
                with np.load(ROOT / r.npz, allow_pickle=False) as z:
                    fh.write(f">{r.entry_id}\n{z['rep_seq']}\n")
        with open(W / f"apo_{s}.fasta", "w") as fh:
            for r in man[s].itertuples():
                if len(seqs[r.entry_id]) >= 10:
                    fh.write(f">{r.entry_id}\n{seqs[r.entry_id]}\n")
        m8 = W / f"apo_{s}_vs_rep_{other}.m8"
        mmseqs_search_protein(W / f"apo_{s}.fasta", W / f"rep_{other}.fasta", m8, Path(args.tmp) / f"x{s}", threads=args.procs,
                              extra=["-s", "7.5", "--max-seqs", "5000", "-e", "1e-3"])
        h = read_m8(m8)
        rep["leakage"][f"{s}_apo_vs_{other}_reps"] = dict(rule_hits=int(apply_rule(h).sum()))
    test_pdb = set((ROOT / "data" / "manifests" / "test_union_pdb_ids.txt").read_text().split())
    cb = json.load(open(RAW / "cryptobench" / "osf" / "cryptobench" / "cryptobench-dataset" / "dataset.json"))
    cb_ids = set(cb) | {h["holo_pdb_id"] for v in cb.values() for h in v}
    pm_ids = {p.name[:4].lower() for p in (RAW / "pocketminer" / "data" / "pm-dataset").rglob("*.pdb")}
    apo_pdbs = {r.pdb for m in man.values() for r in m.itertuples()}
    rep["pdb_overlap_test_union"] = sorted(apo_pdbs & test_pdb)
    rep["pdb_overlap_cryptobench"] = sorted(apo_pdbs & {x.lower() for x in cb_ids})
    rep["pdb_overlap_pocketminer"] = sorted(apo_pdbs & pm_ids)
    print(json.dumps({k: rep[k] for k in ("leakage", "pdb_overlap_test_union", "pdb_overlap_cryptobench", "pdb_overlap_pocketminer")}), flush=True)
    # 4+5. apo status + labels on a seeded sample
    allm = pd.concat(man.values())
    rng = np.random.default_rng(SEEDS[0])
    smp = allm.iloc[rng.choice(len(allm), size=min(args.n_spot, len(allm)), replace=False)]
    need = set(smp.pdb)
    bl = {p: {"nonpoly": set(), "polychains": set()} for p in need}
    with gzip.open(RAW / "biolip2" / "current" / "BioLiP.txt.gz", "rt") as fh:
        for line in fh:
            f = line.split("\t")
            if f[0] in need:
                if f[4] in ("peptide", "rna", "dna"):
                    bl[f[0]]["polychains"].add(f[5])
                else:
                    try:
                        bl[f[0]]["nonpoly"].add((f[5], f[4], int(f[19].split()[0])))
                    except (ValueError, IndexError):
                        pass
    with ProcessPoolExecutor(args.procs) as ex:
        res = list(ex.map(apo_check, [(ROOT / r.npz, bl[r.pdb]) for r in smp.itertuples()], chunksize=2))
    md = np.concatenate([r["lig_min_dist"] for r in res]) if res else np.zeros(0)
    cov = [r for r in res if r["covered"]]
    rep["spot"] = dict(n=len(res), n_biolip_covered=len(cov), covered_with_relevant_ligand_near_site=sum(1 for r in cov if r["relevant_near"]),
                       examples=[(r["entry"], r["relevant_near"][:3]) for r in cov if r["relevant_near"]][:10],
                       n_ligands=int(len(md)), frac_ligands_within_4p5=float(np.mean(md <= 4.5)) if len(md) else None,
                       ligand_min_dist_quantiles={q: float(np.quantile(md, q)) for q in (0.0, 0.05, 0.5, 0.95, 1.0)} if len(md) else {})
    print(json.dumps(rep["spot"]), flush=True)
    ok = (rep["parent_in_same_split"] and rep["apo_chain_in_both_splits"] == 0 and rep["apo_equals_parent_chain"] == 0
          and all(v["rule_hits"] == 0 for v in rep["leakage"].values()) and not rep["pdb_overlap_test_union"]
          and not rep["pdb_overlap_cryptobench"] and not rep["pdb_overlap_pocketminer"]
          and rep["spot"]["covered_with_relevant_ligand_near_site"] == 0 and (rep["spot"]["frac_ligands_within_4p5"] or 0) == 1.0)
    rep["PASS"] = bool(ok)
    json.dump(rep, open(OUT / "verify_report.json", "w"), indent=1, default=str)
    print("PASS" if ok else "FAIL (see verify_report.json)")


if __name__ == "__main__":
    main()
