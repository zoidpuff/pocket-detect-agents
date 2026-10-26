#!/usr/bin/env python
"""Track C2 steps 2-3: transfer Track A's pooled train/dev site ligands onto AFDB v6 models.

For every entry of data/processed/{train,dev}/manifest.csv (Track A; split kept exactly):
  1. AF model(s) of the entry's UniProt accession(s) (scripts/data/build_af_train_index.py);
     fragmented / absent models -> status, no entry.
  2. Residue mapping representative chain -> AF model by overlap (semi-global, free end gaps)
     affine-gap alignment of the representative's observed sequence (``rep_seq``, one letter per
     ``rep_res_keys`` residue) to the AF sequence (Gotoh, numba; match +3, mismatch -2, gap open -8,
     extend -1). Entries with several accessions (chimeras) use the accession whose alignment maps
     the most site residues to identical residue types.
  3. Per site: Kabsch superposition representative -> AF on the CA of the mapped site residues
     (fallback: CA of mapped residues within 8 A of the site's ligand atoms if < 3 site CA); the
     site's pooled ligand heavy atoms are transformed with it. Local RMSD, residue-type identity
     (representative vs AF at the mapped site residues) and mapped fraction are stored.
     Sites are DROPPED (and counted) if: fraction of site residues mapped < 0.5 ("unmapped"),
     < 3 superposition pairs ("too_few_pairs"), identity < 0.75 ("identity"), RMSD > 3 A ("rmsd").

Outputs (data/processed/af_train/):
  {train,dev}/manifest.csv     one row per Track A entry (status ok / no_af_model / fragmented /
                               no_alignment / no_valid_site), per-entry counts
  {train,dev}/entries/<entry_id>.npz   AF-frame labels (no pickles), read with
                               scripts/data/build_af_train_io.py::load_af_entry
  sites.csv                    one row per Track A site (kept or drop reason, RMSD, identity, pLDDT)
  build_info.json              parameters, counts per split

Usage: nohup .venv/bin/python scripts/data/build_af_train_set.py --jobs 16 > log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numba
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.datasets.trainset import load_entry, read_manifest  # noqa: E402

OUT = ROOT / "data/processed/af_train"
PARAMS = dict(
    MAX_RMSD=3.0,  # A, local CA RMSD threshold (PI spec)
    MIN_IDENTITY=0.75,  # residue-type identity at mapped site residues (PI spec)
    MIN_FRAC_MAPPED=0.5,  # fraction of site residues that must map onto the AF model
    MIN_PAIRS=3,  # CA pairs for the superposition
    ENV_SUPERPOSE=8.0,  # A, fallback superposition set
    ALIGN=dict(match=3, mismatch=-2, gap_open=-8, gap_extend=-1),
    PLDDT_LOW=50.0,
)
AA = "ACDEFGHIKLMNPQRSTVWY"
_CODE = np.full(256, 20, dtype=np.uint8)
for _i, _c in enumerate(AA):
    _CODE[ord(_c)] = _i


def encode(seq: str) -> np.ndarray:
    return _CODE[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]


@numba.njit(cache=True)
def overlap_align(q, t, match, mismatch, gapo, gape):
    """Gotoh overlap alignment (end gaps free in both sequences). Returns qmap: index in t of every
    q position, -1 if gapped. Code 20 (X / non-standard) never scores as a match."""
    n, m = q.shape[0], t.shape[0]
    NEG = -(10 ** 9)
    H = np.zeros((n + 1, m + 1), np.int32)
    E = np.full((n + 1, m + 1), NEG, np.int32)
    F = np.full((n + 1, m + 1), NEG, np.int32)
    tH = np.zeros((n + 1, m + 1), np.int8)
    tE = np.zeros((n + 1, m + 1), np.int8)
    tF = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a = H[i, j - 1] + gapo
            b = E[i, j - 1] + gape
            if a >= b:
                E[i, j] = a
                tE[i, j] = 0
            else:
                E[i, j] = b
                tE[i, j] = 1
            a = H[i - 1, j] + gapo
            b = F[i - 1, j] + gape
            if a >= b:
                F[i, j] = a
                tF[i, j] = 0
            else:
                F[i, j] = b
                tF[i, j] = 1
            s = match if (q[i - 1] == t[j - 1] and q[i - 1] < 20) else mismatch
            d = H[i - 1, j - 1] + s
            best, tb = d, 0
            if E[i, j] > best:
                best, tb = E[i, j], 1
            if F[i, j] > best:
                best, tb = F[i, j], 2
            H[i, j] = best
            tH[i, j] = tb
    # best end cell on the last row or column
    bi, bj, bs = n, 0, NEG
    for j in range(m + 1):
        if H[n, j] > bs:
            bs, bi, bj = H[n, j], n, j
    for i in range(n + 1):
        if H[i, m] > bs:
            bs, bi, bj = H[i, m], i, m
    qmap = np.full(n, -1, np.int64)
    i, j, state = bi, bj, 0
    while i > 0 and j > 0:
        if state == 0:
            p = tH[i, j]
            if p == 0:
                qmap[i - 1] = j - 1
                i -= 1
                j -= 1
            else:
                state = p
        elif state == 1:
            p = tE[i, j]
            j -= 1
            state = 0 if p == 0 else 1
        else:
            p = tF[i, j]
            i -= 1
            state = 0 if p == 0 else 2
    return qmap


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = qc - pc @ R.T
    return R, t, float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, axis=1))))


def one_letter(name: str) -> str:
    r = gemmi.find_tabulated_residue(name)
    c = r.one_letter_code.upper() if r is not None else "X"
    return c if c.isalpha() else "X"


def read_chain(path: str, chain: str | None):
    """Per residue key 'num[icode]': (CA xyz or None, heavy-atom xyz, one-letter). First altloc,
    first model, amino-acid residues only."""
    st = gemmi.read_structure(path)
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    model = st[0]
    ch = model[chain] if chain is not None else model[0]
    keys, cas, atoms, seq = [], [], [], []
    for r in ch:
        tab = gemmi.find_tabulated_residue(r.name)
        if tab is None or not tab.is_amino_acid():
            continue
        keys.append(f"{r.seqid.num}{r.seqid.icode.strip()}")
        ca = r.find_atom("CA", "*")
        cas.append(None if ca is None else np.array([ca.pos.x, ca.pos.y, ca.pos.z]))
        atoms.append(np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r]).reshape(-1, 3))
        seq.append(one_letter(r.name))
    return keys, cas, atoms, "".join(seq), [r.seqid.num for r in ch]


def read_af(path: Path):
    st = gemmi.read_structure(str(path))
    ch = st[0][0]
    nums, cas, seq, pl, atoms = [], [], [], [], []
    for r in ch:
        ca = r.find_atom("CA", "*")
        nums.append(r.seqid.num)
        cas.append(np.array([ca.pos.x, ca.pos.y, ca.pos.z]))
        pl.append(ca.b_iso)
        seq.append(one_letter(r.name))
    return np.array(nums), np.array(cas), "".join(seq), np.array(pl, dtype=np.float32), ch.name


G: dict = {}


def process(args):
    split, row = args
    try:
        return _process(split, row)
    except Exception as e:  # report, never drop silently
        return dict(entry_id=row["entry_id"], split=split, status="error", reason=f"{type(e).__name__}: {e}",
                    tb=traceback.format_exc()), []


def _process(split: str, row: dict):
    out = OUT / split / "entries"
    accs = [a for a in row["uniprot"].split(",") if a]
    rec = dict(entry_id=row["entry_id"], split=split, uniprot=row["uniprot"], pdb=row["pdb"], chain=row["chain"],
               cluster30=row["cluster30"], group=row["group"], rank_in_cluster=row["rank_in_cluster"])
    mi = G["model_index"]
    stats = [mi.get(a, {"status": "absent"})["status"] for a in accs]
    usable = [a for a, s in zip(accs, stats) if s == "extracted"]
    rec["af_model_status"] = ";".join(stats)
    if not usable:
        rec.update(status="fragmented" if "fragmented" in stats else "no_af_model", n_sites_src=int(row["n_sites"]))
        return rec, []
    e = load_entry(row["npz"])
    a = e.arrays
    rkeys = [str(k) for k in a["rep_res_keys"]]
    rseq = str(a["rep_seq"])
    keys, cas, atoms, sseq, _ = read_chain(e.meta["structure_path"], e.chain)
    kidx = {k: i for i, k in enumerate(keys)}
    if len(rseq) != len(rkeys):  # fall back to the structure-derived sequence
        rseq = "".join(sseq[kidx[k]] if k in kidx else "X" for k in rkeys)
    rk_index = {k: i for i, k in enumerate(rkeys)}
    sites = e.site_residues()
    n_site = len(sites)
    rec["n_sites_src"] = n_site
    # ---- choose accession / alignment
    best = None
    for acc in usable:
        nums, af_ca, af_seq, plddt, af_chain = read_af(OUT / "models" / f"AF-{acc}-F1-model_v6.cif.gz")
        qmap = overlap_align(encode(rseq), encode(af_seq), *PARAMS["ALIGN"].values())
        aligned = qmap >= 0
        ident = np.array([aligned[i] and rseq[i] == af_seq[qmap[i]] for i in range(len(rseq))])
        site_ok = sum(1 for s in sites for k in s if k in rk_index and ident[rk_index[k]])
        score = (site_ok, int(ident.sum()))
        if best is None or score > best[0]:
            best = (score, acc, nums, af_ca, af_seq, plddt, af_chain, qmap, ident)
    _, acc, nums, af_ca, af_seq, plddt, af_chain, qmap, ident = best
    n_al = int((qmap >= 0).sum())
    rec.update(acc=acc, model_path=f"models/AF-{acc}-F1-model_v6.cif.gz", af_length=len(af_seq),
               rep_length=len(rseq), align_coverage_rep=n_al / max(1, len(rseq)),
               align_identity=float(ident.sum() / max(1, n_al)), plddt_mean=float(plddt.mean()),
               frac_plddt_lt50=float((plddt < PARAMS["PLDDT_LOW"]).mean()))
    if n_al < PARAMS["MIN_PAIRS"]:
        rec.update(status="no_alignment")
        return rec, []
    # ---- per-site transfer
    ptr, lsite, lis_ion = a["lig_ptr"], a["lig_site"], a["lig_is_ion"]
    site_rows, kept = [], []
    out_coords, out_el, out_ptr, out_site, out_ion, out_ccd, out_id = [], [], [0], [], [], [], []
    for si, sres in enumerate(sites):
        sr = dict(entry_id=row["entry_id"], split=split, acc=acc, site=si, n_res=len(sres),
                  is_ion=bool(a["site_is_ion"][si]))
        ligs = np.flatnonzero(lsite == si)
        lxyz = np.vstack([a["lig_coords"][ptr[l]:ptr[l + 1]] for l in ligs]).astype(float) if len(ligs) else np.zeros((0, 3))
        mapped = []  # (rep key, af index)
        for k in sres:
            ri = rk_index.get(k)
            if ri is not None and qmap[ri] >= 0:
                mapped.append((k, int(qmap[ri]), rseq[ri]))
        sr["frac_mapped"] = len(mapped) / max(1, len(sres))
        sr["identity"] = float(np.mean([aa == af_seq[j] for _, j, aa in mapped])) if mapped else np.nan
        pairs = [(cas[kidx[k]], af_ca[j]) for k, j, _ in mapped if k in kidx and cas[kidx[k]] is not None]
        sr["sup_set"] = "site_ca"
        if len(pairs) < PARAMS["MIN_PAIRS"] and len(lxyz):
            lt = cKDTree(lxyz)
            pairs = []
            for ri, k in enumerate(rkeys):
                if qmap[ri] < 0 or k not in kidx or cas[kidx[k]] is None:
                    continue
                if lt.query(atoms[kidx[k]], k=1)[0].min() <= PARAMS["ENV_SUPERPOSE"]:
                    pairs.append((cas[kidx[k]], af_ca[qmap[ri]]))
            sr["sup_set"] = f"env{PARAMS['ENV_SUPERPOSE']:g}_ca"
        sr["n_pairs"] = len(pairs)
        reason = ""
        if sr["frac_mapped"] < PARAMS["MIN_FRAC_MAPPED"]:
            reason = "unmapped"
        elif len(pairs) < PARAMS["MIN_PAIRS"]:
            reason = "too_few_pairs"
        elif not (sr["identity"] >= PARAMS["MIN_IDENTITY"]):
            reason = "identity"
        if not reason:
            R, t, rmsd = kabsch(np.array([p for p, _ in pairs]), np.array([q for _, q in pairs]))
            sr["rmsd"] = rmsd
            if rmsd > PARAMS["MAX_RMSD"]:
                reason = "rmsd"
        if not reason and len(ligs) == 0:
            reason = "no_ligand_atoms"
        sr["drop_reason"] = reason
        sr["kept"] = not reason
        if not reason:
            new_si = len(kept)
            af_idx = sorted({j for _, j, _ in mapped})
            sr["plddt_mean"] = float(plddt[af_idx].mean())
            sr["plddt_min"] = float(plddt[af_idx].min())
            txyz = lxyz @ R.T + t
            for l in ligs:
                c = a["lig_coords"][ptr[l]:ptr[l + 1]].astype(float) @ R.T + t
                out_coords.append(c.astype(np.float32))
                out_el += list(a["lig_elements"][ptr[l]:ptr[l + 1]])
                out_ptr.append(out_ptr[-1] + len(c))
                out_site.append(new_si)
                out_ion.append(bool(lis_ion[l]))
                out_ccd.append(str(a["lig_ccd"][l]))
                out_id.append(str(a["lig_id"][l]))
            kept.append(dict(src_site=si, residues=";".join(str(int(nums[j])) for j in af_idx), rmsd=rmsd,
                             identity=sr["identity"], frac_mapped=sr["frac_mapped"], n_pairs=len(pairs),
                             sup_set=sr["sup_set"], is_ion=sr["is_ion"], plddt_mean=sr["plddt_mean"],
                             plddt_min=sr["plddt_min"], center=txyz.mean(0)))
        site_rows.append(sr)
    rec["n_sites_kept"] = len(kept)
    for r_ in ("unmapped", "too_few_pairs", "identity", "rmsd", "no_ligand_atoms"):
        rec[f"n_drop_{r_}"] = sum(1 for s in site_rows if s["drop_reason"] == r_)
    if not kept:
        rec["status"] = "no_valid_site"
        return rec, site_rows
    meta = dict(entry_id=row["entry_id"], split=split, acc=acc, uniprot=row["uniprot"], af_chain=af_chain,
                model_path=str(OUT / rec["model_path"]), trainset_npz=row["npz"], rep_pdb=row["pdb"],
                rep_chain=row["chain"], cluster30=row["cluster30"], group=row["group"],
                test_union_sha256=e.meta.get("test_union_sha256", ""), align_identity=rec["align_identity"],
                align_coverage_rep=rec["align_coverage_rep"], params=PARAMS)
    arrays = dict(
        meta_json=np.array(json.dumps(meta)),
        af_res_nums=nums.astype(np.int32), af_seq=np.array(af_seq), plddt=plddt,
        lig_coords=np.vstack(out_coords).astype(np.float32), lig_elements=np.array(out_el, dtype="<U2"),
        lig_ptr=np.array(out_ptr, dtype=np.int32), lig_site=np.array(out_site, dtype=np.int32),
        lig_is_ion=np.array(out_ion, dtype=bool), lig_ccd=np.array(out_ccd), lig_id=np.array(out_id),
        site_residues=np.array([k["residues"] for k in kept]), site_src_index=np.array([k["src_site"] for k in kept], np.int32),
        site_rmsd=np.array([k["rmsd"] for k in kept], np.float32), site_identity=np.array([k["identity"] for k in kept], np.float32),
        site_frac_mapped=np.array([k["frac_mapped"] for k in kept], np.float32),
        site_n_pairs=np.array([k["n_pairs"] for k in kept], np.int32), site_sup_set=np.array([k["sup_set"] for k in kept]),
        site_is_ion=np.array([k["is_ion"] for k in kept], bool), site_plddt_mean=np.array([k["plddt_mean"] for k in kept], np.float32),
        site_plddt_min=np.array([k["plddt_min"] for k in kept], np.float32),
        site_center=np.array([k["center"] for k in kept], np.float32),
    )
    path = out / f"{row['entry_id']}.npz"
    np.savez_compressed(path, **arrays)
    rec.update(status="ok", npz=str(path.relative_to(ROOT)))
    return rec, site_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.jobs > 16:
        sys.exit("at most 16 processes (PI constraint)")
    t0 = time.time()
    mi = pd.read_csv(OUT / "model_index.csv", dtype=str, keep_default_na=False)
    G["model_index"] = {r["acc"]: r for r in mi.to_dict("records")}
    overlap_align(encode("ACDE"), encode("ACDE"), 3, -2, -8, -1)  # compile once before fork
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    info = dict(params=PARAMS, splits={})
    all_sites = []
    for split in ("train", "dev"):
        man = read_manifest(split)
        if args.limit:
            man = man.head(args.limit)
        edir = OUT / split / "entries"
        if not args.limit and edir.exists():  # never keep entries of a previous (e.g. pre-audit) manifest
            for f in edir.glob("*.npz"):
                f.unlink()
        edir.mkdir(parents=True, exist_ok=True)
        tasks = [(split, r) for r in man.to_dict("records")]
        recs = []
        with ProcessPoolExecutor(args.jobs, mp_context=ctx) as ex:
            for i, (rec, srows) in enumerate(ex.map(process, tasks, chunksize=4)):
                recs.append(rec)
                all_sites += srows
                if rec.get("status") == "error":
                    print("ERROR", rec["entry_id"], rec["reason"], flush=True)
                if (i + 1) % 1000 == 0:
                    print(f"  {split}: {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
        R = pd.DataFrame(recs)
        sfx = "_debug" if args.limit else ""
        R.drop(columns=[c for c in ("tb",) if c in R]).to_csv(OUT / split / f"manifest{sfx}.csv", index=False)
        S = pd.DataFrame([s for s in all_sites if s["split"] == split])
        info["splits"][split] = dict(
            n_entries=len(R), status=R.status.value_counts().to_dict(),
            coverage_af_model=float(R.af_model_status.str.split(";").apply(lambda x: "extracted" in x).mean()),
            coverage_ok=float((R.status == "ok").mean()),
            n_sites_src=int(R.n_sites_src.sum()),
            n_sites_considered=len(S), n_sites_kept=int(S.kept.sum()) if len(S) else 0,
            drop_reasons=S.drop_reason[S.drop_reason != ""].value_counts().to_dict() if len(S) else {},
        )
        print(split, json.dumps(info["splits"][split]), flush=True)
    S = pd.DataFrame(all_sites)
    S.to_csv(OUT / f"sites{'_debug' if args.limit else ''}.csv", index=False)
    info["wall_s"] = round(time.time() - t0, 1)
    info["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    info["trainset_manifest_sha256"] = {sp: hashlib.sha256((ROOT / f"data/processed/{sp}/manifest.csv").read_bytes()).hexdigest()
                                        for sp in ("train", "dev")}
    info["model_index_sha256"] = hashlib.sha256((OUT / "model_index.csv").read_bytes()).hexdigest()
    (OUT / f"build_info{'_debug' if args.limit else ''}.json").write_text(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
