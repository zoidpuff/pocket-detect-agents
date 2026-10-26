#!/usr/bin/env python
"""Step 2: candidate training chains + ligands from the BioLiP2 snapshot (2025-11-12).

Filters (DESIGN §2.5; docs/TRAINSET.md):
  * experimental method X-ray (wwPDB 'diffraction') or EM (derived_data/pdb_entry_type.txt),
  * 0 < resolution <= 3.0 A (BioLiP resolution column; -1 = NMR/none),
  * receptor chain >= 50 residues (BioLiP receptor sequence = residues with coordinates),
  * ligands = BioLiP relevant ligands incl. metal ions; polymer ligands (peptide/rna/dna) excluded
    as labels (kept in a separate table to *mask* such sites later),
  * chain must keep >= 1 non-polymer ligand.

Outputs (data/processed/train/work/):
  candidate_ligands.parquet   one row per (chain, ligand instance), with ion/metal flags
  masked_ligands.parquet      polymer-ligand rows (peptide/rna/dna) of the same chains
  candidate_chains.parquet    one row per chain (pdb, chain, resolution, method, uniprot, seq, seq_id)
  candidate_seqs.fasta        unique observed (BioLiP receptor) sequences (id = seq_id)
  candidate_can_seqs.fasta    unique canonical mmCIF entity sequences (id = can_id), searched too
  entry_info.parquet          cached mmCIF entity info (canonical sequence, UNP refs) per chain
  candidates_counts.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.datasets.pdbinfo import entry_info  # noqa: E402
from pocketdetect.datasets.biolip import POLYMER_LIGANDS, classify_ccd, read_biolip, read_ligand_table  # noqa: E402

BL = ROOT / "data" / "raw" / "biolip2" / "current"
ENTRY_TYPE = Path("/cluster/work/beltrao/shared/rsyncPDB/derived_data/pdb_entry_type.txt")
SIFTS = ROOT / "data" / "processed" / "af" / "external" / "pdb_chain_uniprot.tsv.gz"
WORK = ROOT / "data" / "processed" / "train" / "work"
MAX_RES = 3.0
MIN_LEN = 50


def _info(pid):
    try:
        return pid, entry_info(pid)
    except Exception:  # noqa: BLE001
        return pid, None


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    c = {}
    df = read_biolip(BL / "BioLiP.txt.gz")
    c["biolip_rows"] = len(df)
    c["biolip_chains"] = int(df.groupby(["pdb", "chain"]).ngroups)
    et = pd.read_csv(ENTRY_TYPE, sep="\t", header=None, names=["pdb", "poly", "method"], dtype=str)
    meth = dict(zip(et.pdb, et.method))
    df["method"] = df.pdb.map(meth).fillna("unknown")
    m_method = df.method.isin(["diffraction", "EM"])
    m_res = (df.resolution > 0) & (df.resolution <= MAX_RES)
    m_len = df.seq.str.len() >= MIN_LEN
    c["rows_fail_method"] = int((~m_method).sum())
    c["rows_fail_resolution"] = int((m_method & ~m_res).sum())
    c["rows_fail_length"] = int((m_method & m_res & ~m_len).sum())
    df = df[m_method & m_res & m_len].copy()
    poly = df.ccd.isin(POLYMER_LIGANDS)
    masked = df[poly].copy()
    lig = df[~poly].copy()
    c["rows_polymer_ligand_excluded"] = int(poly.sum())
    c["rows_polymer_ligand_by_type"] = masked.ccd.value_counts().to_dict()

    lt = read_ligand_table(BL / "ligand.tsv.gz")
    cls = {r.ccd: classify_ccd(r.formula) for r in lt.itertuples()}
    lig["heavy_atoms_ccd"] = lig.ccd.map(lambda x: cls.get(x, (-1, False, False))[0])
    lig["is_ion"] = lig.ccd.map(lambda x: cls.get(x, (-1, False, False))[1])
    lig["is_metal"] = lig.ccd.map(lambda x: cls.get(x, (-1, False, False))[2])
    c["ligand_ccd_without_formula"] = sorted(set(lig.ccd[lig.heavy_atoms_ccd < 0]))[:50]
    # multi-residue ligand instances (BioLiP 'lig_resnum' like '401 402')? keep as written
    c["ligand_rows_multi_resnum"] = int(lig.lig_resnum.str.contains(r"\s|~").sum())

    # UniProt: BioLiP column, completed from SIFTS (pdb_chain_uniprot, Track C download)
    sifts = defaultdict(set)
    if SIFTS.exists():
        s = pd.read_csv(SIFTS, sep="\t", comment="#", dtype=str, keep_default_na=False, usecols=["PDB", "CHAIN", "SP_PRIMARY"])
        for p, ch, u in zip(s.PDB, s.CHAIN, s.SP_PRIMARY):
            sifts[(p.lower(), ch)].add(u)
    ch = (lig.groupby(["pdb", "chain"], sort=True)
          .agg(resolution=("resolution", "first"), method=("method", "first"), seq=("seq", "first"),
               uniprot_biolip=("uniprot", "first"), n_lig=("ccd", "size"), n_ion=("is_ion", "sum"))
          .reset_index())
    # canonical entity sequence + mmCIF _struct_ref UNP accessions (cached)
    ei_path = WORK / "entry_info.parquet"
    pdbs = sorted(set(ch.pdb))
    ei = pd.read_parquet(ei_path) if ei_path.exists() else pd.DataFrame(columns=["pdb", "chain", "seq_can", "uniprot_mmcif"])
    todo = sorted(set(pdbs) - set(ei.pdb))
    if todo:
        rows = []
        with ProcessPoolExecutor(32) as ex:
            for pid, info in ex.map(_info, todo, chunksize=32):
                if info is None:
                    rows.append(dict(pdb=pid, chain="", seq_can="", uniprot_mmcif=""))
                    continue
                for e in info["entities"]:
                    for cc in e["chains"]:
                        rows.append(dict(pdb=pid, chain=cc, seq_can=e["seq"], uniprot_mmcif=",".join(e["uniprot"])))
        ei = pd.concat([ei, pd.DataFrame(rows)], ignore_index=True)
        ei.to_parquet(ei_path, index=False)
    ch = ch.merge(ei.drop_duplicates(["pdb", "chain"]), on=["pdb", "chain"], how="left")
    ch["seq_can"] = ch.seq_can.fillna("")
    ch["uniprot_mmcif"] = ch.uniprot_mmcif.fillna("")
    c["chains_without_canonical_seq"] = int((ch.seq_can == "").sum())
    ch["uniprot_all"] = [",".join(sorted((set(u.split(",")) if u else set()) | sifts.get((p, cc), set())
                                         | (set(m.split(",")) if m else set())))
                         for p, cc, u, m in zip(ch.pdb, ch.chain, ch.uniprot_biolip, ch.uniprot_mmcif)]
    ch["uniprot"] = [u if u else (a.split(",")[0] if a else "") for u, a in zip(ch.uniprot_biolip, ch.uniprot_all)]
    ch["seq_len"] = ch.seq.str.len()
    seqs = sorted(set(ch.seq))
    sid = {s: f"s{i:07d}" for i, s in enumerate(seqs)}
    ch["seq_id"] = ch.seq.map(sid)
    ch["chain_id"] = ch.pdb + "_" + ch.chain
    with open(WORK / "candidate_seqs.fasta", "w") as fh:
        for s in seqs:
            fh.write(f">{sid[s]}\n{s}\n")
    cseqs = sorted(set(ch.seq_can) - {""})
    cid = {s: f"c{i:07d}" for i, s in enumerate(cseqs)}
    ch["can_id"] = ch.seq_can.map(cid).fillna("")
    with open(WORK / "candidate_can_seqs.fasta", "w") as fh:
        for s in cseqs:
            fh.write(f">{cid[s]}\n{s}\n")
    lig.drop(columns=["seq"]).to_parquet(WORK / "candidate_ligands.parquet", index=False)
    masked = masked[masked.set_index(["pdb", "chain"]).index.isin(ch.set_index(["pdb", "chain"]).index)]
    masked.drop(columns=["seq"]).to_parquet(WORK / "masked_ligands.parquet", index=False)
    ch.to_parquet(WORK / "candidate_chains.parquet", index=False)
    c.update(
        candidate_ligand_rows=len(lig), candidate_ion_rows=int(lig.is_ion.sum()), candidate_metal_rows=int(lig.is_metal.sum()),
        candidate_chains=len(ch), candidate_entries=int(ch.pdb.nunique()), candidate_unique_seqs=len(seqs),
        chains_without_uniprot=int((ch.uniprot == "").sum()), candidate_uniprots=int(ch.uniprot[ch.uniprot != ""].nunique()),
        chains_by_method=ch.method.value_counts().to_dict(),
        biolip_sha256=hashlib.sha256((BL / "BioLiP.txt.gz").read_bytes()).hexdigest(),
        sifts_sha256=hashlib.sha256(SIFTS.read_bytes()).hexdigest() if SIFTS.exists() else None,
    )
    json.dump(c, open(WORK / "candidates_counts.json", "w"), indent=1)
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
