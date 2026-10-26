#!/usr/bin/env python
"""Track C steps 2-4: map LIGYSIS sites onto AFDB v6 models and write the AF-mode test set.

Inputs (all local, see docs/AF_BENCHMARK.md):
  data/processed/af/models/*.cif.gz, extract_index.csv      (scripts/data/build_af_extract.py)
  data/raw/ligysis/lbs_comparison_results/*.pkl              (LIGYSIS sites, benchmark rows, ligands)
  data/raw/ligysis/lbs_comparison_code/.../trans_rep_chains  (LIGYSIS representative chains)
  data/processed/af/external/uniprot_human_reviewed.tsv.gz   (UniProt canonical sequences)
  data/processed/af/external/pdb_chain_uniprot.tsv.gz        (SIFTS chain -> UniProt segments)
  PDB mirror mmCIF (ligand transfer) + derived_data/index/resolu.idx

Per protein (one AF model, chain A, residue number == UniProt position):
  * AF model sequence vs UniProt canonical; per-site residue-type check against the LIGYSIS
    representative PDB chain (else the SIFTS-mapped PDB chains the site ligands come from, else the
    UniProt sequence); sites with positions outside the model, residue-type identity
    < MIN_RESTYPE_ID, or no usable reference are kept in the table but flagged include=False.
  * Site centres on the model: Calpha centroid (= LIGYSIS definition, benchmark notebook
    7.0_CONCAT_data_LIGYSIS: mean of site-residue CA), heavy-atom centroid, side-chain centroid,
    and centroid of transferred ligand atoms.
  * Ligand transfer: up to MAX_LIG_PER_SITE ligand instances per site (LIGYSIS ligand list,
    representative entry first, then by resolution) are located in the asymmetric-unit mmCIF,
    the site residues' CA of the contacting chain (SIFTS-mapped to UniProt numbering) are
    superposed onto the AF CA (Kabsch) and the ligand heavy atoms transformed; local RMSD stored.
  * Annotations: site pLDDT mean/min/band, distance to low-pLDDT (prunable) segments, protein
    pLDDT<50 fraction, length, #PDB entries (SIFTS), #ligand entries, local CA (and heavy-atom)
    RMSD AF vs representative PDB chain at the site.

Outputs (data/processed/af/):
  af_manifest.csv   one row per target UniProt (included or not, with reason)
  af_sites.csv      one row per LIGYSIS site of those UniProts (+ .parquet)
  af_ligands.csv    one row per attempted ligand transfer (no coordinates)
  af_sequences.fasta AF model sequences (all extracted proteins; for leakage filtering)
  gt/<structure_id>.json.gz  per-protein ground truth (sites + transferred ligand atoms)
  build_info.json   parameters, input checksums, counts

Usage: .venv/bin/python scripts/data/build_af_set.py [--jobs 48] [--limit N] [--accs P03372,...]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_pickle import safe_load  # noqa: E402

OUT = ROOT / "data/processed/af"
LIG_RES = ROOT / "data/raw/ligysis/lbs_comparison_results"
LIG_CODE = next((ROOT / "data/raw/ligysis/lbs_comparison_code").glob("LBS-comparison-*/bartongroup-*"))
TRANS_DIR = LIG_CODE / "DATA" / "trans_rep_chains"
PDB_MIRROR = Path("/cluster/work/beltrao/shared/rsyncPDB")
MMCIF_DIR = PDB_MIRROR / "data/structures/divided/mmCIF"

# ---- frozen parameters (documented in docs/AF_BENCHMARK.md; never tuned on results)
PARAMS = dict(
    MIN_RESTYPE_ID=0.75,  # site kept if >= 75% of site residues have the reference residue type
    # (tolerates engineered point mutations in the PDB reference; numbering shifts give ~0-0.3)
    MAX_LIG_PER_SITE=5,  # ligand instances transferred per site
    LIG_CONTACT=4.5,  # A, ligand heavy atom -> site residue heavy atom to confirm the instance
    MIN_SUPERPOSE_CA=3,  # minimum common site CA for the local superposition
    ENV_SUPERPOSE=8.0,  # A, fallback superposition set (residues near the ligand) for sites < 3 CA
    LIG_RMSD_ACCEPT=2.0,  # A, default acceptance of a transferred ligand (loader can change)
    PLDDT_LOW=50.0,  # pLDDT "very low"
    PRUNE_INTERNAL_MIN=7,  # internal low-pLDDT stretch length that the AF2BIND rule prunes
    NEAR_LOW=8.0,  # A, "site near a low-pLDDT segment"
    PLDDT_BANDS=(50.0, 70.0, 90.0),
)
BB = {"N", "CA", "C", "O", "OXT"}


# ============================================================================ helpers
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def one_letter(resname: str) -> str:
    r = gemmi.find_tabulated_residue(resname)
    if r is None:
        return "X"
    c = r.one_letter_code.upper()
    return c if c.isalpha() and c != " " else "X"


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation R, translation t minimising |P @ R.T + t - Q|; returns (R, t, rmsd)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = qc - pc @ R.T
    rmsd = float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, axis=1))))
    return R, t, rmsd


def plddt_band(x: float) -> str:
    a, b, c = PARAMS["PLDDT_BANDS"]
    if not np.isfinite(x):
        return ""
    return "<50" if x < a else "50-70" if x < b else "70-90" if x < c else ">=90"


def prunable_mask(plddt: np.ndarray) -> np.ndarray:
    """DESIGN §2.6 / AF2BIND rule: terminal runs with pLDDT < 50 and internal runs of >= 7
    consecutive residues with pLDDT < 50."""
    low = plddt < PARAMS["PLDDT_LOW"]
    out = np.zeros_like(low)
    n = len(low)
    i = 0
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            if i == 0 or j == n or (j - i) >= PARAMS["PRUNE_INTERNAL_MIN"]:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


class Chain:
    """Residue-level view of one polymer chain: per residue name, number, atoms."""

    def __init__(self, ch: gemmi.Chain, label_seq: bool = False):
        self.names, self.nums, self.lseq, self.icodes = [], [], [], []
        self.atom_xyz, self.atom_names, self.atom_res, self.atom_el = [], [], [], []
        self.ca = {}
        self.bfac_ca = []
        for r in ch:
            if r.het_flag == "H" and gemmi.find_tabulated_residue(r.name) is None:
                continue
            tab = gemmi.find_tabulated_residue(r.name)
            if tab is None or not tab.is_amino_acid():
                continue
            k = len(self.names)
            self.names.append(r.name)
            self.nums.append(r.seqid.num)
            self.icodes.append(r.seqid.icode.strip())
            self.lseq.append(r.label_seq if r.label_seq is not None else -1)
            b = np.nan
            for a in r:
                if a.is_hydrogen():
                    continue
                self.atom_xyz.append((a.pos.x, a.pos.y, a.pos.z))
                self.atom_names.append(a.name)
                self.atom_res.append(k)
                self.atom_el.append(a.element.name)
                if a.name == "CA":
                    self.ca[k] = np.array([a.pos.x, a.pos.y, a.pos.z])
                    b = a.b_iso
            self.bfac_ca.append(b)
        self.atom_xyz = np.asarray(self.atom_xyz, dtype=float).reshape(-1, 3)
        self.atom_res = np.asarray(self.atom_res, dtype=int)
        self.atom_names = np.asarray(self.atom_names, dtype=object)
        self.bfac_ca = np.asarray(self.bfac_ca, dtype=float)
        self.seq = "".join(one_letter(n) for n in self.names)
        self.res_atoms = defaultdict(list)
        for i, k in enumerate(self.atom_res):
            self.res_atoms[int(k)].append(i)


def read_first_model(path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    if len(st) > 1:
        for i in range(len(st) - 1, 0, -1):
            del st[i]
    return st


# ============================================================================ global tables
G: dict = {}


def load_tables(accs_filter=None) -> None:
    hs = safe_load(LIG_RES / "LIGYSIS_human_sites_APRIL_2024.pkl")
    defs = safe_load(LIG_RES / "LIGYSIS_sites_DEF_TRANS.pkl")
    preds = safe_load(LIG_RES / "ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl")
    bench = preds[preds["origin"] == "LIGYSIS"][["rep_chain", "ID", "monomer_prop"]].copy()
    bench = bench.merge(defs[["rep_chain", "ID", "SITE_NAME"]], on=["rep_chain", "ID"], how="left")
    assert bench["SITE_NAME"].notna().all() and len(bench) == 6882
    ionp = safe_load(LIG_RES / "LIGYSIS_human_sites_ion_prop_APRIL_2024.pkl")
    cpl = safe_load(LIG_RES / "LIGYSIS_human_chains_per_lig_MAY_2024.pkl")

    sites = hs.copy()
    sites["SEG"] = sites["SEG"].astype(str)
    sites["ion_prop"] = [ionp.get((a, int(s), int(i)), np.nan) for a, s, i in zip(sites.ACC, sites.SEG, sites.ID)]
    d = defs.set_index("SITE_NAME")
    sites["rep_chain"] = sites["SITE_NAME"].map(d["rep_chain"])
    sites["rep_aas"] = sites["SITE_NAME"].map(d["aas"])
    sites["ligysis_centre_pdb"] = sites["SITE_NAME"].map(d["centre"])
    b = bench.set_index("SITE_NAME")
    sites["in_lbs_benchmark"] = sites["SITE_NAME"].isin(b.index)
    sites["monomer_prop"] = sites["SITE_NAME"].map(b["monomer_prop"])
    sites["ligands"] = [list(cpl.get(sid, {}).get(int(i), {}).items()) for sid, i in zip(sites.SEG_ID, sites.ID)]
    G["sites_by_acc"] = {a: g.to_dict("records") for a, g in sites.groupby("ACC")}
    G["bench_accs"] = set(sites.loc[sites.in_lbs_benchmark, "ACC"])
    G["bench_rep_chains"] = set(bench["rep_chain"])

    up = pd.read_csv(OUT / "external/uniprot_human_reviewed.tsv.gz", sep="\t")
    G["uniprot"] = dict(zip(up["Entry"], up["Sequence"]))
    G["uniprot_ver"] = dict(zip(up["Entry"], up["Sequence version"]))

    sifts = pd.read_csv(OUT / "external/pdb_chain_uniprot.tsv.gz", sep="\t", comment="#", dtype=str)
    G["n_pdb_entries"] = sifts.groupby("SP_PRIMARY")["PDB"].nunique().to_dict()
    G["n_pdb_chains"] = sifts.groupby("SP_PRIMARY").size().to_dict()
    need = set(G["sites_by_acc"]) if accs_filter is None else set(accs_filter)
    s2 = sifts[sifts["SP_PRIMARY"].isin(need)]
    smap: dict = defaultdict(list)
    for r in s2.itertuples(index=False):
        try:
            smap[(r.PDB, r.SP_PRIMARY)].append((r.CHAIN, int(r.RES_BEG), int(r.RES_END), int(r.SP_BEG), int(r.SP_END)))
        except ValueError:
            continue
    G["sifts"] = dict(smap)

    res = {}
    with open(PDB_MIRROR / "derived_data/index/resolu.idx") as f:
        for line in f:
            p = line.split(";")
            if len(p) == 2 and len(p[0].strip()) == 4:
                try:
                    res[p[0].strip().lower()] = float(p[1])
                except ValueError:
                    pass
    G["resolution"] = res
    G["extract"] = pd.read_csv(OUT / "extract_index.csv", dtype=str, keep_default_na=False).set_index("acc")


# ============================================================================ per protein
def mmcif_path(pdb: str) -> Path:
    return MMCIF_DIR / pdb[1:3] / f"{pdb}.cif.gz"


def _entry_chains(pdb: str, acc: str, entry_cache: dict):
    """Per entry (cached): model and, for every chain SIFTS-mapped to ``acc``, a dict
    UniProt position -> (heavy-atom xyz, CA xyz or None, one-letter type)."""
    key = (pdb, acc)
    if key in entry_cache:
        return entry_cache[key]
    res = None
    p = mmcif_path(pdb)
    segs = G["sifts"].get((pdb, acc))
    if segs and p.exists():
        try:
            st = read_first_model(p)
        except Exception:  # noqa: BLE001
            st = None
        if st is not None:
            maps: dict = defaultdict(dict)
            for ch_name, rb, re_, sb, se in segs:
                for ls in range(rb, re_ + 1):
                    maps[ch_name][ls] = sb + (ls - rb)
            chains = {}
            for ch in st[0]:
                m = maps.get(ch.name)
                if not m:
                    continue
                d = {}
                for r in ch:
                    ls = r.label_seq
                    if ls is None or ls not in m:
                        continue
                    xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r if not a.is_hydrogen()]).reshape(-1, 3)
                    ca = r.find_atom("CA", "*")
                    d[m[ls]] = (xyz, None if ca is None else np.array([ca.pos.x, ca.pos.y, ca.pos.z]), one_letter(r.name))
                chains[ch.name] = d
            res = (st, chains)
    entry_cache[key] = res
    return res


def transfer_instance(key, site_up, af, af_idx, entry_cache, acc):
    """Locate one LIGYSIS ligand instance (key = pdb_RESNAME_labelasym_authseq, from the PDBe
    assembly used by LIGYSIS) in the wwPDB asymmetric-unit entry and transfer it onto the AF model.

    Candidate copies = all residues named RESNAME. Each is scored against every chain mapped to the
    UniProt (SIFTS) by the number of site residues within LIG_CONTACT; ties prefer exact identifiers
    (auth seq num and label_asym id). Contacted site residues required: exact match (both ids) >= 1,
    one id matching >= 2, any other copy (renumbered entries / assembly-only copies)
    >= min(5, max(2, ceil(20% of the site))). The site residues' CA of the
    chosen chain are superposed onto the AF CA (Kabsch) and the ligand heavy atoms transformed."""
    parts = key.split("_")
    pdb, resname, lchain, lnum = parts[0], parts[1], parts[2], parts[3]
    row = dict(key=key, pdb=pdb, resname=resname, status="", chain="", match="", n_contact_res=0, n_ca=0,
               rmsd=np.nan, resolution=G["resolution"].get(pdb, np.nan), coords=None, elements=None)
    if not G["sifts"].get((pdb, acc)):
        row["status"] = "no_sifts"
        return row
    ent = _entry_chains(pdb, acc, entry_cache)
    if ent is None:
        row["status"] = "no_entry"
        return row
    st, chains = ent
    site_set = set(site_up)
    n_site = len(site_set)
    cands = []
    for ch in st[0]:
        for r in ch:
            if r.name != resname:
                continue
            xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r if not a.is_hydrogen()]).reshape(-1, 3)
            if not len(xyz):
                continue
            exact = int(str(r.seqid.num) + r.seqid.icode.strip() == lnum) + int(r.subchain == lchain)
            cands.append((exact, xyz, [a.element.name for a in r if not a.is_hydrogen()]))
    if not cands:
        row["status"] = "ligand_not_found"
        return row
    best = None
    for exact, xyz, el in cands:
        lt = cKDTree(xyz)
        for ch_name, d in chains.items():
            contact = [u for u in site_set if u in d and len(d[u][0])
                       and lt.query(d[u][0], k=1, distance_upper_bound=PARAMS["LIG_CONTACT"] + 1e-6)[0].min() <= PARAMS["LIG_CONTACT"]]
            score = (len(contact), exact)
            if best is None or score > best[0]:
                best = (score, ch_name, xyz, el, contact, exact)
    (nc, exact), ch_name, xyz, el, contact, exact = best
    need = {2: 1, 1: 2}.get(exact, min(5, max(2, math.ceil(0.2 * n_site))))
    need = min(need, n_site)
    if nc < need:
        row.update(status="no_contact", n_contact_res=nc)
        return row
    d = chains[ch_name]
    pairs = [(u, d[u][1], d[u][2]) for u in sorted(site_set) if u in d and d[u][1] is not None
             and u in af_idx and af_idx[u] in af.ca]
    row.update(chain=ch_name, match=["any_copy", "partial_id", "exact_id"][exact], n_contact_res=nc, n_ca=len(pairs),
               sup_set="site_ca")
    if pairs:
        row["restype_id"] = float(np.mean([aa == af.seq[af_idx[u]] for u, _, aa in pairs]))
        row["restype_n"] = len(pairs)
    if len(pairs) < PARAMS["MIN_SUPERPOSE_CA"]:
        # small sites (1-2 residues, mostly ions): superpose on the CA of all mapped residues with a
        # heavy atom within ENV_SUPERPOSE of the ligand instead
        lt = cKDTree(xyz)
        pairs = [(u, v[1], v[2]) for u, v in sorted(d.items()) if v[1] is not None and len(v[0])
                 and u in af_idx and af_idx[u] in af.ca
                 and lt.query(v[0], k=1, distance_upper_bound=PARAMS["ENV_SUPERPOSE"] + 1e-6)[0].min() <= PARAMS["ENV_SUPERPOSE"]]
        row.update(n_ca=len(pairs), sup_set=f"env{PARAMS['ENV_SUPERPOSE']:g}_ca")
        if len(pairs) < PARAMS["MIN_SUPERPOSE_CA"]:
            row["status"] = "too_few_ca"
            return row
    P = np.array([c for _, c, _ in pairs])
    Q = np.array([af.ca[af_idx[u]] for u, _, _ in pairs])
    R, t, rmsd = kabsch(P, Q)
    row.update(status="ok", rmsd=rmsd, coords=(xyz @ R.T + t), elements=el)
    return row


def process_protein(acc: str):
    try:
        return _process_protein(acc)
    except Exception as e:  # report, never silently drop
        return dict(acc=acc, status="error", reason=f"{type(e).__name__}: {e}", tb=traceback.format_exc()), [], [], None


def _process_protein(acc: str):
    sites = G["sites_by_acc"][acc]
    ex = G["extract"].loc[acc] if acc in G["extract"].index else None
    sid = f"AF-{acc}-F1"
    prot = dict(acc=acc, structure_id=sid, in_lbs_benchmark=acc in G["bench_accs"], n_sites_all=len(sites),
                n_sites_benchmark=int(sum(s["in_lbs_benchmark"] for s in sites)),
                n_pdb_entries=int(G["n_pdb_entries"].get(acc, 0)), n_pdb_chains=int(G["n_pdb_chains"].get(acc, 0)),
                uniprot_seq_version=G["uniprot_ver"].get(acc, np.nan))
    if ex is None or ex["status"] != "extracted":
        st = "absent" if ex is None else ex["status"]
        prot.update(status="excluded", reason=f"af_model_{st}", n_fragments=int(ex["n_fragments"]) if ex is not None else 0)
        return prot, [], [], None
    mpath = OUT / "models" / os.path.basename(ex["member"])
    st = gemmi.read_structure(str(mpath))
    assert len(st[0]) == 1, f"{acc}: AF model has {len(st[0])} chains"
    ch = st[0][0]
    af = Chain(ch)
    L = len(af.names)
    nums = np.array(af.nums)
    prot["_seq"] = None  # filled below; written to af_sequences.fasta, not to the manifest
    prot.update(model_path=str(mpath.relative_to(OUT)), model_sha256=ex["sha256"], af_source=ex["source"],
                length=L, chain=ch.name, numbering_1_to_L=bool(np.array_equal(nums, np.arange(1, L + 1))))
    prot["_seq"] = af.seq
    up_seq = G["uniprot"].get(acc)
    prot["uniprot_in_release"] = up_seq is not None
    prot["af_seq_eq_uniprot"] = bool(up_seq == af.seq) if up_seq is not None else False
    if up_seq is not None and up_seq != af.seq:
        same_len = len(up_seq) == L
        prot["af_vs_uniprot_identity"] = float(np.mean([a == b for a, b in zip(up_seq, af.seq)])) if same_len else np.nan
        prot["uniprot_length"] = len(up_seq)
    plddt = af.bfac_ca
    prot["plddt_mean"] = float(np.nanmean(plddt))
    prot["frac_plddt_lt50"] = float(np.mean(plddt < 50))
    prot["frac_plddt_lt70"] = float(np.mean(plddt < 70))
    prune = prunable_mask(plddt)
    prot["frac_prunable"] = float(prune.mean())
    af_idx = {int(n): k for k, n in enumerate(af.nums)}  # UniProt position -> residue index

    low_atoms = np.flatnonzero(plddt[af.atom_res] < PARAMS["PLDDT_LOW"])
    prune_atoms = np.flatnonzero(prune[af.atom_res])
    t_low = cKDTree(af.atom_xyz[low_atoms]) if len(low_atoms) else None
    t_prune = cKDTree(af.atom_xyz[prune_atoms]) if len(prune_atoms) else None

    rep_cache: dict = {}
    entry_cache: dict = {}
    site_rows, lig_rows, gt_sites = [], [], []
    seen_sets: dict = {}
    for s in sites:
        up = [int(x) for x in s["up_aas"]]
        row = dict(acc=acc, structure_id=sid, site_name=s["SITE_NAME"], seg=s["SEG"], site_id=int(s["ID"]),
                   in_lbs_benchmark=bool(s["in_lbs_benchmark"]), ion_prop=s["ion_prop"], monomer_prop=s["monomer_prop"],
                   rep_chain=s["rep_chain"] if isinstance(s["rep_chain"], str) else "", n_res=len(up),
                   n_ligand_instances=len(s["ligands"]),
                   n_ligand_entries=len({k.split("_")[0] for k, _ in s["ligands"]}))
        flags = []
        missing = [u for u in up if u not in af_idx]
        row["n_res_missing_in_model"] = len(missing)
        if missing:
            flags.append("pos_out_of_model")
        present = [u for u in up if u in af_idx]
        # ---- residue-type check: reference = LIGYSIS representative PDB chain (else UniProt)
        ref_types = None
        rc = row["rep_chain"]
        rep = None
        if rc and isinstance(s["rep_aas"], (list, tuple)):
            if rc not in rep_cache:
                p = TRANS_DIR / f"{rc}.trans.pdb"
                rep_cache[rc] = None
                if p.exists():
                    rst = gemmi.read_structure(str(p))
                    rst.remove_hydrogens()
                    rst.remove_alternative_conformations()
                    rch = rst[0][0]
                    rep_cache[rc] = Chain(rch)
            rep = rep_cache[rc]
        if rep is not None:
            key2k = {f"{n}{i}": k for k, (n, i) in enumerate(zip(rep.nums, rep.icodes))}
            pairs = [(u, key2k.get(str(a))) for u, a in zip(up, s["rep_aas"])]
            ref_types = {u: one_letter(rep.names[k]) for u, k in pairs if k is not None}
            row["restype_ref"] = "pdb_rep_chain"
            # verify our reading of the representative chain against LIGYSIS's stored centre
            cas = [rep.ca[k] for u, k in pairs if k is not None and k in rep.ca]
            if len(cas) == len(up) and isinstance(s["ligysis_centre_pdb"], tuple):
                row["ligysis_centre_check"] = float(np.linalg.norm(np.mean(cas, 0) - np.array(s["ligysis_centre_pdb"])))
        elif up_seq is not None:
            ref_types = {u: up_seq[u - 1] for u in up if 1 <= u <= len(up_seq)}
            row["restype_ref"] = "uniprot"
        else:
            row["restype_ref"] = "none"
        comp = [ref_types[u] == af.seq[af_idx[u]] for u in present if u in ref_types] if ref_types else []
        row["restype_n_compared"] = len(comp)
        row["restype_id"] = float(np.mean(comp)) if comp else np.nan
        # vs current UniProt sequence (informative for every site)
        if up_seq is not None:
            cmpu = [up_seq[u - 1] == af.seq[af_idx[u]] for u in present if 1 <= u <= len(up_seq)]
            row["restype_id_uniprot"] = float(np.mean(cmpu)) if cmpu else np.nan
        # duplicate residue sets across segments of the same protein
        fs = frozenset(up)
        if fs in seen_sets:
            flags.append(f"duplicate_of:{seen_sets[fs]}")
        else:
            seen_sets[fs] = s["SITE_NAME"]
        if not present:
            row.update(flags=";".join(flags), include=False)
            site_rows.append(row)
            continue
        ks = [af_idx[u] for u in present]
        # ---- centres
        ca = np.array([af.ca[k] for k in ks if k in af.ca])
        row["center_ca"] = ca.mean(0).round(3).tolist()
        aidx = np.concatenate([af.res_atoms[k] for k in ks])
        row["center_heavy"] = af.atom_xyz[aidx].mean(0).round(3).tolist()
        sc = [i for k in ks for i in af.res_atoms[k] if af.atom_names[i] not in BB]
        sc += [i for k in ks if af.names[k] == "GLY" for i in af.res_atoms[k] if af.atom_names[i] == "CA"]
        row["center_sc"] = af.atom_xyz[sc].mean(0).round(3).tolist()
        row["rog_ca"] = float(np.sqrt(np.mean(np.sum((ca - ca.mean(0)) ** 2, 1))))
        # ---- pLDDT
        pl = plddt[ks]
        row["plddt_mean"] = float(pl.mean())
        row["plddt_min"] = float(pl.min())
        row["plddt_band"] = plddt_band(row["plddt_mean"])
        row["frac_res_plddt_lt50"] = float(np.mean(pl < 50))
        row["frac_res_prunable"] = float(np.mean(prune[ks]))
        sxyz = af.atom_xyz[aidx]
        row["dist_to_low_plddt"] = float(t_low.query(sxyz, k=1)[0].min()) if t_low is not None else np.inf
        row["dist_to_prunable"] = float(t_prune.query(sxyz, k=1)[0].min()) if t_prune is not None else np.inf
        row["near_low_plddt"] = bool(row["dist_to_prunable"] <= PARAMS["NEAR_LOW"])
        # ---- local RMSD AF vs representative PDB chain at the site
        row["rep_resolution"] = G["resolution"].get(rc.split("_")[0], np.nan) if rc else np.nan
        if rep is not None:
            pr = [(u, k) for u, k in pairs if k is not None and k in rep.ca and u in af_idx and af_idx[u] in af.ca]
            row["rep_n_ca"] = len(pr)
            if len(pr) >= PARAMS["MIN_SUPERPOSE_CA"]:
                P = np.array([rep.ca[k] for _, k in pr])
                Q = np.array([af.ca[af_idx[u]] for u, _ in pr])
                R, t, rmsd = kabsch(P, Q)
                row["rmsd_ca_af_vs_rep"] = rmsd
                # heavy-atom RMSD over identically named atoms of residues with identical type
                dd = []
                for u, k in pr:
                    ka = af_idx[u]
                    if one_letter(rep.names[k]) != af.seq[ka]:
                        continue
                    an = {af.atom_names[i]: i for i in af.res_atoms[ka]}
                    for i in rep.res_atoms[k]:
                        j = an.get(rep.atom_names[i])
                        if j is not None:
                            dd.append(np.sum((rep.atom_xyz[i] @ R.T + t - af.atom_xyz[j]) ** 2))
                row["rmsd_heavy_af_vs_rep"] = float(np.sqrt(np.mean(dd))) if dd else np.nan
        # ---- ligand transfer
        # order: representative entry first if it has a resolution (X-ray/EM), then by resolution
        # (entries without resolution, e.g. NMR, last), then by key (deterministic)
        rep_pdb = rc.split("_")[0] if rc else ""

        def _res(pdb):
            r = G["resolution"].get(pdb, -1.0)
            return r if r > 0 else 99.0

        insts = sorted(s["ligands"], key=lambda kv: (not (kv[0].split("_")[0] == rep_pdb and _res(rep_pdb) < 99.0),
                                                    _res(kv[0].split("_")[0]), kv[0]))
        site_ligs = []
        n_tried = 0
        for key, _chains in insts:
            if sum(1 for l in site_ligs if l["status"] == "ok") >= PARAMS["MAX_LIG_PER_SITE"]:
                break
            if n_tried >= 4 * PARAMS["MAX_LIG_PER_SITE"]:
                break
            n_tried += 1
            lr = transfer_instance(key, up, af, af_idx, entry_cache, acc)
            site_ligs.append(lr)
            lig_rows.append({k: v for k, v in lr.items() if k not in ("coords", "elements")} | dict(acc=acc, site_name=s["SITE_NAME"]))
        ok = [l for l in site_ligs if l["status"] == "ok"]
        acc_ok = [l for l in ok if l["rmsd"] <= PARAMS["LIG_RMSD_ACCEPT"]]
        row["n_lig_transferred"] = len(ok)
        row["n_lig_accepted"] = len(acc_ok)
        row["lig_rmsd_min"] = float(min(l["rmsd"] for l in ok)) if ok else np.nan
        row["lig_rmsd_median"] = float(np.median([l["rmsd"] for l in ok])) if ok else np.nan
        row["lig_resnames"] = "&".join(sorted({l["resname"] for l in acc_ok}))
        row["lig_from_rep_entry"] = bool(any(l["pdb"] == rep_pdb for l in ok))
        if acc_ok:
            allx = np.vstack([l["coords"] for l in acc_ok])
            row["center_lig"] = allx.mean(0).round(3).tolist()
            row["dist_center_lig_vs_ca"] = float(np.linalg.norm(allx.mean(0) - np.array(row["center_ca"])))
            # sanity: transferred ligand must sit on the site (min distance to site atoms)
            row["lig_min_dist_to_site"] = float(cKDTree(sxyz).query(allx, k=1)[0].min())
        # residue-type decision. Reference: representative PDB chain; else the SIFTS-mapped PDB chains
        # the ligands were transferred from (best identity); else the current UniProt sequence.
        if row["restype_ref"] != "pdb_rep_chain":
            lr = [l for l in site_ligs if np.isfinite(l.get("restype_id", np.nan))]
            if lr:
                bl = max(lr, key=lambda l: (l["restype_id"], l.get("restype_n", 0)))
                row.update(restype_ref="pdb_ligand_entry", restype_id=bl["restype_id"], restype_n_compared=bl.get("restype_n", 0))
        if not np.isfinite(row["restype_id"]):
            flags.append("restype_unchecked")
        elif row["restype_id"] < PARAMS["MIN_RESTYPE_ID"]:
            flags.append("restype_mismatch")
        elif row["restype_id"] < 1:
            flags.append("restype_partial")
        if row["restype_ref"] == "uniprot":
            # no PDB reference: the comparison is only meaningful if AF != UniProt sequence
            flags.append("restype_unchecked" if not prot["af_seq_eq_uniprot"] else "restype_uniprot_only")
        if "restype_mismatch" in flags or "restype_unchecked" in flags or "pos_out_of_model" in flags or any(f.startswith("duplicate_of") for f in flags):
            row["include"] = False
        else:
            row["include"] = True
        row["flags"] = ";".join(flags)
        site_rows.append(row)
        gt_sites.append(dict(
            site_name=s["SITE_NAME"], include=row["include"], flags=row["flags"], in_lbs_benchmark=row["in_lbs_benchmark"],
            ion_prop=None if not np.isfinite(row["ion_prop"]) else float(row["ion_prop"]),
            residues=up, center_ca=row["center_ca"], center_heavy=row["center_heavy"], center_sc=row["center_sc"],
            center_lig=row.get("center_lig"), plddt_mean=row["plddt_mean"],
            ligands=[dict(key=l["key"], resname=l["resname"], rmsd=round(l["rmsd"], 4), n_ca=l["n_ca"], match=l["match"],
                          coords=np.round(l["coords"], 3).tolist(), elements=l["elements"]) for l in ok],
        ))
    n_inc = sum(r["include"] for r in site_rows)
    prot["n_sites_included"] = int(n_inc)
    prot["n_sites_benchmark_included"] = int(sum(r["include"] and r["in_lbs_benchmark"] for r in site_rows))
    if n_inc == 0:
        prot.update(status="excluded", reason="no_valid_site")
    else:
        prot.update(status="included", reason="")
    gt = dict(structure_id=sid, acc=acc, model_path=prot["model_path"], chain=ch.name, length=L,
              residue_numbers=[int(x) for x in af.nums], plddt=[round(float(x), 2) for x in plddt],
              af_seq_eq_uniprot=prot["af_seq_eq_uniprot"], params=PARAMS, sites=gt_sites)
    return prot, site_rows, lig_rows, gt


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N accessions (debug)")
    ap.add_argument("--accs", default="", help="comma-separated accessions (debug)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    t0 = time.time()
    accs_f = [a for a in args.accs.split(",") if a] or None
    load_tables(accs_f)
    accs = sorted(G["sites_by_acc"])
    if accs_f:
        accs = [a for a in accs if a in set(accs_f)]
    if args.limit:
        accs = accs[: args.limit]
    print(f"tables loaded in {time.time()-t0:.0f}s; {len(accs)} accessions", flush=True)
    gt_dir = out / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    prots, sites, ligs, errors = [], [], [], []
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(args.jobs, mp_context=ctx, initializer=_init_worker) as exe:
        for i, (p, s, l, gt) in enumerate(exe.map(process_protein, accs, chunksize=1)):
            prots.append(p)
            sites += s
            ligs += l
            if p.get("status") == "error":
                errors.append(p)
                print("ERROR", p["acc"], p["reason"], flush=True)
            if gt is not None:
                gp = gt_dir / f"{gt['structure_id']}.json.gz"
                with gzip.open(gp, "wt") as f:
                    json.dump(gt, f, separators=(",", ":"))
                p["gt_path"] = str(gp.relative_to(out))
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(accs)} proteins, {time.time()-t0:.0f}s", flush=True)
    with open(out / f"af_sequences{'' if not (args.limit or accs_f) else '_debug'}.fasta", "w") as f:
        for p in prots:
            if p.get("_seq"):
                f.write(f">{p['structure_id']} acc={p['acc']} status={p['status']} in_lbs_benchmark={p['in_lbs_benchmark']}\n{p['_seq']}\n")
    for p in prots:
        p.pop("_seq", None)
    P = pd.DataFrame(prots)
    S = pd.DataFrame(sites)
    Lg = pd.DataFrame(ligs)
    sfx = "" if not (args.limit or accs_f) else "_debug"
    P.drop(columns=[c for c in ("tb",) if c in P.columns]).to_csv(out / f"af_manifest{sfx}.csv", index=False)
    S.to_csv(out / f"af_sites{sfx}.csv", index=False)
    S2 = S.copy()
    for c in ("center_ca", "center_heavy", "center_sc", "center_lig"):
        if c in S2.columns:
            S2[c] = S2[c].apply(lambda v: json.dumps(v) if isinstance(v, list) else "")
    S2.to_parquet(out / f"af_sites{sfx}.parquet", index=False)
    Lg.to_csv(out / f"af_ligands{sfx}.csv", index=False)
    inputs = {
        "LIGYSIS_human_sites_APRIL_2024.pkl": sha256_file(LIG_RES / "LIGYSIS_human_sites_APRIL_2024.pkl"),
        "LIGYSIS_sites_DEF_TRANS.pkl": sha256_file(LIG_RES / "LIGYSIS_sites_DEF_TRANS.pkl"),
        "ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl": sha256_file(LIG_RES / "ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl"),
        "LIGYSIS_human_chains_per_lig_MAY_2024.pkl": sha256_file(LIG_RES / "LIGYSIS_human_chains_per_lig_MAY_2024.pkl"),
        "uniprot_human_reviewed.tsv.gz": sha256_file(out / "external/uniprot_human_reviewed.tsv.gz"),
        "pdb_chain_uniprot.tsv.gz": sha256_file(out / "external/pdb_chain_uniprot.tsv.gz"),
        "extract_index.csv": sha256_file(out / "extract_index.csv"),
    }
    info = dict(params=PARAMS, inputs_sha256=inputs, n_accessions=len(accs), n_errors=len(errors),
                protein_status=P["status"].value_counts().to_dict(),
                protein_reason=P["reason"].value_counts().to_dict(),
                n_sites=len(S), n_sites_included=int(S["include"].sum()) if len(S) else 0,
                wall_s=round(time.time() - t0, 1), script_sha256=sha256_file(Path(__file__)))
    (out / f"build_info{sfx}.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
