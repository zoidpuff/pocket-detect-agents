#!/usr/bin/env python3
"""Sanity checks for every raw dataset in data/raw/. Run with the project venv:

    .venv/bin/python -I scripts/data/verify_datasets.py [dataset ...]

datasets: p2rank ligysis cryptobench pocketminer hotpocket af2bind biolip2 scpdb pocketwatch tools
Prints a summary per dataset and writes data/raw/VERIFY_REPORT.json. Exit code 1 if any hard
check fails. Third-party pickles are read via a restricted unpickler (scripts/data/safe_pickle.py).
"""
import gzip
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(HERE))
from safe_pickle import SafeUnpickler, safe_load  # noqa: E402

import gemmi  # noqa: E402
import numpy as np  # noqa: E402

random.seed(0)
REPORT = {}
FAILS = []

# P2Rank defaults (rdk/p2rank Params.groovy): ignore_het_groups, min_ligand_atoms=5,
# ligand_protein_contact_distance=4, ligc_prot_dist=5.5 (ligand centre to nearest protein atom)
P2R_IGNORE = {"HOH", "DOD", "WAT", "NAG", "MAN", "UNK", "GLC", "ABA", "MPD", "GOL", "SO4", "PO4"}


def check(name, cond, msg):
    if not cond:
        FAILS.append(f"{name}: {msg}")
        print(f"  FAIL: {msg}")
    return cond


def read_structure(path, sanitize_on_error=False, return_flag=False):
    """gemmi read; if sanitize_on_error, retry with ATOM/HETATM lines cut to 78 columns (some
    legacy PDB files, e.g. metaPocket2 sets in P2Rank JOINED, carry junk in the charge columns)."""
    try:
        st = gemmi.read_structure(str(path))
        sanitized = False
    except RuntimeError:
        if not sanitize_on_error:
            raise
        op = gzip.open if str(path).endswith(".gz") else open
        lines = [l.rstrip("\n") for l in op(path, "rt", errors="replace")]
        lines = [l[:78] if l.startswith(("ATOM", "HETATM")) else l for l in lines]
        st = gemmi.read_pdb_string("\n".join(lines))
        sanitized = True
    st.setup_entities()
    return (st, sanitized) if return_flag else st


def ligand_groups(st, ignore=P2R_IGNORE, min_atoms=5, contact=4.0, centre_dist=5.5):
    """Approximation of P2Rank 'relevant ligands': HETATM residues (not water / ignore list),
    >= min_atoms heavy atoms, some atom within `contact` A of a protein (ATOM-record) atom and
    ligand centre within `centre_dist` A of a protein atom. Grouped per residue (P2Rank additionally
    merges HETATM groups closer than 1.7 A, so multi-residue ligands may count >1 here)."""
    model = st[0]
    ns = gemmi.NeighborSearch(model, gemmi.UnitCell(), 6).populate()  # no crystal symmetry mates

    def min_prot_dist(pos, radius):
        d = 99.0
        for m in ns.find_atoms(pos, "\0", radius=radius):
            cra = m.to_cra(model)
            if cra.residue.het_flag == "A":
                d = min(d, cra.atom.pos.dist(pos))
        return d

    out = []
    for ch in model:
        for r in ch:
            if r.het_flag != "H" or r.name in ignore or r.is_water() :
                continue
            tab = gemmi.find_tabulated_residue(r.name)
            if tab is not None and (tab.is_amino_acid() or tab.is_nucleic_acid()):
                continue  # modified residues (MSE, SEP, LLP, ...) are not ligands (BioJava/P2Rank treat them as polymer)
            heavy = [a for a in r if not a.is_hydrogen()]
            if len(heavy) < min_atoms:
                continue
            if min(min_prot_dist(a.pos, contact) for a in heavy) > contact:
                continue
            c = gemmi.Position(*np.mean([[a.pos.x, a.pos.y, a.pos.z] for a in heavy], axis=0))
            if min_prot_dist(c, centre_dist) > centre_dist:
                continue
            out.append(r.name)
    return out


def _p2rank_one(args):
    path, codes = args
    try:
        st, sanitized = read_structure(path, sanitize_on_error=True, return_flag=True)
        n_prot = sum(1 for ch in st[0] for r in ch if r.het_flag == "A")
        ligs = ligand_groups(st)
        het_names = {r.name for ch in st[0] for r in ch if r.het_flag == "H"}
        missing = [c for c in codes if c not in het_names] if codes else []
        return str(path), n_prot, ligs, missing, None, sanitized
    except Exception as e:  # noqa: BLE001
        return str(path), 0, [], [], repr(e), False


def parse_ds(ds_path):
    """Parse a P2Rank .ds file -> list of (relpath, [ligand codes] or None)."""
    items, header = [], None
    for line in open(ds_path):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("PARAM."):
            continue
        if line.startswith("HEADER:"):
            header = line.split(":", 1)[1].split()
            continue
        parts = line.split()
        if header and "ligand_codes" in header and len(parts) > header.index("ligand_codes"):
            codes = parts[header.index("ligand_codes")].split(",")
        else:
            codes = None
        items.append((parts[0], codes))
    return items


def verify_p2rank():
    base = RAW / "p2rank_datasets" / "current"
    rep = {}
    main = ["coach420.ds", "coach420(mlig).ds", "holo4k.ds", "holo4k(mlig).ds", "chen11.ds",
            "joined.ds", "joined(mlig).ds", "fptrain.ds", "ah4h.holoraw.ds", "ah4h.holoraw(mlig).ds"]
    for ds in main:
        p = base / ds
        if not check("p2rank", p.exists(), f"{ds} missing"):
            continue
        items = parse_ds(p)
        paths = [(base / rel, codes) for rel, codes in items]
        missing_files = [str(x) for x, _ in paths if not x.exists()]
        check("p2rank", not missing_files, f"{ds}: {len(missing_files)} referenced files missing, e.g. {missing_files[:3]}")
        with ProcessPoolExecutor(32) as ex:
            res = list(ex.map(_p2rank_one, [x for x in paths if x[0].exists()], chunksize=8))
        errs = [r for r in res if r[4]]
        no_lig = [r[0] for r in res if not r[2] and not r[4]]
        mlig_missing = [(r[0], r[3]) for r in res if r[3]]
        n_ligs = sum(len(r[2]) for r in res)
        n_sanitized = sum(1 for r in res if r[5])
        lig_counter = Counter(l for r in res for l in r[2])
        rep[ds] = dict(entries=len(items), files_missing=len(missing_files), parse_errors=len(errs),
                       relevant_ligands_p2rank_like=n_ligs, proteins_without_relevant_ligand=len(no_lig),
                       mlig_codes_not_found=len(mlig_missing), files_needing_charge_column_sanitizing=n_sanitized,
                       examples_no_ligand_paths=[str(Path(x).relative_to(base)) for x in no_lig[:40]], top_ligands=lig_counter.most_common(8),
                       examples_no_ligand=[os.path.basename(x) for x in no_lig[:5]],
                       examples_mlig_missing=[(os.path.basename(a), b) for a, b in mlig_missing[:5]],
                       parse_error_examples=[(os.path.basename(r[0]), r[4]) for r in errs[:3]])
        print(f"  {ds}: {len(items)} entries, {n_ligs} P2Rank-like relevant ligands, "
              f"{len(no_lig)} w/o ligand, {len(errs)} parse errors, {len(mlig_missing)} mlig-code misses, {n_sanitized} sanitized")
        check("p2rank", not errs, f"{ds}: {len(errs)} parse errors {errs[:2]}")
    # directory-level file counts (dirs contain more files than the .ds lists)
    rep["dir_file_counts"] = {d.name: sum(1 for _ in d.rglob("*") if _.is_file()) for d in base.iterdir() if d.is_dir()}
    return rep


def verify_ligysis():
    base = RAW / "ligysis"
    res_dir = base / "lbs_comparison_results"
    code = next((base / "lbs_comparison_code").glob("LBS-comparison-*/bartongroup-*"))
    chains = safe_load(res_dir / "LIGYSIS_3448_chains.pkl")
    sites = safe_load(res_dir / "LIGYSIS_sites_DEF_TRANS.pkl")
    labels = safe_load(res_dir / "LIGYSIS_ress_binary_labels.pkl")
    preds = safe_load(res_dir / "ALL_PREDS_COMBINED_RIGHT_EXTENDED_DEF.pkl")
    trans = sorted((code / "DATA" / "trans_rep_chains").glob("*.trans.pdb"))
    trans_ids = {p.name.replace(".trans.pdb", "") for p in trans}
    site_chains = set(sites["rep_chain"])
    rep = dict(n_chains_list=len(chains), n_sites=len(sites), n_site_chains=len(site_chains),
               n_trans_rep_chain_files=len(trans), site_chains_without_file=len(site_chains - trans_ids),
               n_residue_label_rows=len(labels), n_label_chains=labels["rep_chain"].nunique(),
               label_methods=[c for c in labels.columns if c not in ("rep_chain", "chain", "resnum", "resname")],
               preds_rows=len(preds), preds_by_origin=preds["origin"].value_counts().to_dict(),
               preds_chains_by_origin=preds.groupby("origin")["rep_chain"].nunique().to_dict())
    check("ligysis", len(chains) == 3448, f"expected 3448 chains, got {len(chains)}")
    check("ligysis", not (site_chains - trans_ids), f"{len(site_chains - trans_ids)} site chains lack structure")
    # residue-level consistency: site residues (PDB numbering, 'aas') must exist in the structure
    sample = random.sample(sorted(site_chains & trans_ids), 25)
    bad = 0
    for rc in sample:
        st = read_structure(code / "DATA" / "trans_rep_chains" / f"{rc}.trans.pdb")
        nums = {r.seqid.num for ch in st[0] for r in ch}
        for aas in sites.loc[sites.rep_chain == rc, "aas"]:
            miss = [a for a in aas if int(a) not in nums]
            bad += bool(miss)
        hets = sum(1 for ch in st[0] for r in ch if r.het_flag == "H")
        check("ligysis", hets == 0, f"{rc}: has {hets} HETATM residues (expected ligand-stripped)")
    rep["sample_sites_with_residues_missing_in_structure"] = bad
    check("ligysis", bad == 0, f"{bad} sampled sites have residues absent from structure")
    print(f"  {len(chains)} chains listed; {len(sites)} sites on {len(site_chains)} chains; "
          f"{len(trans)} structure files; preds rows {len(preds)}")
    return rep


def _cb_apo(args):
    path, sel = args
    st = gemmi.read_structure(path)
    present = {f"{ch.name}_{r.seqid.num}{r.seqid.icode.strip()}" for ch in st[0] for r in ch}
    return [x for x in sel if x not in present]


def _cb_holo(args):
    path, ligs = args
    st = gemmi.read_structure(path)
    have = {(r.name, ch.name, str(r.seqid.num)) for ch in st[0] for r in ch}
    return [l for l in ligs if tuple(l) not in have]


def verify_cryptobench():
    base = RAW / "cryptobench"
    ds_dir = base / "osf" / "cryptobench" / "cryptobench-dataset"
    data = json.load(open(ds_dir / "dataset.json"))
    cif_dir = base / "cif-files" / "cif-files"
    cifs = {p.stem.lower() for p in cif_dir.glob("*.cif")}
    apo = set(data)
    holo = {h["holo_pdb_id"] for v in data.values() for h in v}
    pairs = sum(len(v) for v in data.values())
    ligs = Counter(h["ligand"] for v in data.values() for h in v)
    folds = {}
    for f in sorted((ds_dir / "folds").glob("*.json")):
        folds[f.stem] = len(json.load(open(f)))
    noncryptic = json.load(open(ds_dir / "auxiliary-data" / "non-cryptic-pockets" / "noncryptic-pockets.json"))
    rep = dict(n_apo=len(apo), n_holo_unique=len(holo), n_pairs=pairs, n_cif=len(cifs),
               apo_missing_cif=len(apo - cifs), holo_missing_cif=len(holo - cifs), folds=folds,
               fold_sum=sum(folds.values()), noncryptic_apo_entries=len(noncryptic),
               top_ligands=ligs.most_common(10))
    check("cryptobench", not (apo - cifs), f"{len(apo - cifs)} apo cif missing")
    check("cryptobench", not (holo - cifs), f"{len(holo - cifs)} holo cif missing")
    # exhaustive: every holo ligand (code, chain, auth index) present; every apo pocket residue present
    jobs = {}
    for a, hs in data.items():
        jobs.setdefault(a, set()).update(hs_apo for h in hs for hs_apo in h["apo_pocket_selection"])
    holo_jobs = {}
    for a, hs in data.items():
        for h in hs:
            holo_jobs.setdefault(h["holo_pdb_id"], []).append((h["ligand"], h["ligand_chain"], str(h["ligand_index"])))
    with ProcessPoolExecutor(32) as ex:
        apo_res = dict(zip(jobs, ex.map(_cb_apo, [(str(cif_dir / f"{a}.cif"), sorted(v)) for a, v in jobs.items()], chunksize=8)))
        holo_res = dict(zip(holo_jobs, ex.map(_cb_holo, [(str(cif_dir / f"{h}.cif"), v) for h, v in holo_jobs.items()], chunksize=8)))
    bad_apo = {a: m for a, m in apo_res.items() if m}
    bad_lig = {h: m for h, m in holo_res.items() if m}
    rep["holo_ligands_not_found"] = dict(n_files=len(bad_lig), n_ligs=sum(len(v) for v in bad_lig.values()), examples=dict(list(bad_lig.items())[:5]))
    rep["apo_pocket_residues_missing"] = dict(n_apo=len(bad_apo), n_res=sum(len(v) for v in bad_apo.values()), examples=dict(list(bad_apo.items())[:5]))
    print(f"  {len(apo)} apo, {len(holo)} holo, {pairs} pairs, {len(cifs)} cifs, folds {folds}; "
          f"holo files w/ ligand miss {len(bad_lig)}, apo w/ pocket-res miss {len(bad_apo)}")
    return rep


def _load_npy_safe(path):
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        hdr = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = hdr(f)
        if dtype.hasobject:
            return SafeUnpickler(f).load()
        f.seek(0)
    return np.load(path, allow_pickle=False)


def verify_pocketminer():
    base = RAW / "pocketminer" / "data" / "pm-dataset"
    rep = {}
    for split in ("val", "test"):
        ids = _load_npy_safe(base / f"{split}_apo_ids_with_chainids.npy")
        lab = _load_npy_safe(base / f"{split}_label_dictionary.npy")
        lab = lab.item() if hasattr(lab, "item") and getattr(lab, "shape", None) == () else lab
        npos = sum(int(np.sum(np.asarray(v) == 1)) for v in lab.values())
        nneg = sum(int(np.sum(np.asarray(v) == 0)) for v in lab.values())
        rep[split] = dict(n_ids=len(ids), n_label_entries=len(lab), pos_residues=npos, neg_residues=nneg,
                          example_ids=[str(x) for x in list(ids)[:3]])
    rep["n_apo_structures"] = len(list((base / "apo-structures").glob("*.pdb")))
    rep["n_all_structures"] = len(list((base / "all-structures").glob("*.pdb")))
    holo_lig = list((base / "all-structures").glob("*ligand*.pdb"))
    with_het = sum(1 for p in holo_lig if any(l.startswith("HETATM") for l in open(p)))
    rep["holo_ligand_files"] = len(holo_lig)
    rep["holo_ligand_files_with_HETATM"] = with_het
    print(f"  {rep}")
    return rep


def verify_hotpocket():
    import pandas as pd
    base = RAW / "hotpocket" / "extracted"
    rep = {}
    for p in sorted(x for x in base.rglob("*.csv") if x.is_file()):
        n = sum(1 for _ in open(p)) - 1
        cols = pd.read_csv(p, nrows=2).columns.tolist()
        rep[p.name] = dict(rows=n, columns=cols)
        print(f"  {p.name}: {n} rows")
    kp = pd.read_csv(base / "all_pocket_preds" / "all_pocket_preds" / "known_pockets.csv")
    rep["known_pockets_unique_uniprot"] = int(kp["uniprot"].nunique())
    rep["known_pockets_unique_pdb"] = int(kp["pdb"].nunique())
    return rep


def verify_af2bind():
    base = next((RAW / "af2bind").glob("af2bind-*/"))
    lab = safe_load(base / "training" / "new_labels_2k_addseq.pkl")
    asg = safe_load(base / "training" / "assignments_2k.pkl")
    npos = sum(int(np.sum(np.asarray(v["bind_alt"]) > 0)) for v in lab.values())
    preds = list((base / "preds").rglob("*.csv"))
    rep = dict(n_label_chains=len(lab), label_keys=list(next(iter(lab.values())).keys()),
               n_assignment_lists=len(asg), assignment_value_counts=[dict(Counter(d.values())) for d in asg[:2]],
               binding_residues_bind_alt=npos, n_proteome_pred_csv=len(preds))
    print(f"  {len(lab)} labelled chains, {len(asg)} split assignments, {len(preds)} pred csvs")
    return rep


def verify_biolip2():
    base = RAW / "biolip2" / "current"
    rep = {}
    for f in ("BioLiP.txt.gz", "BioLiP_nr.txt.gz"):
        n, pdbs, ncols, ligs = 0, set(), Counter(), Counter()
        with gzip.open(base / f, "rt") as fh:
            for line in fh:
                c = line.rstrip("\n").split("\t")
                n += 1
                pdbs.add(c[0])
                ncols[len(c)] += 1
                ligs[c[4]] += 1
        rep[f] = dict(rows=n, unique_pdb=len(pdbs), column_counts=dict(ncols), unique_ligand_codes=len(ligs),
                      top_ligands=ligs.most_common(10))
        check("biolip2", set(ncols) == {21}, f"{f}: unexpected column counts {dict(ncols)}")
        print(f"  {f}: {n} rows, {len(pdbs)} PDB entries, {len(ligs)} ligand codes")
    for f in ("protein.fasta.gz", "protein_nr.fasta.gz"):
        with gzip.open(base / f, "rt") as fh:
            rep[f] = sum(1 for l in fh if l.startswith(">"))
    with gzip.open(base / "ligand.tsv.gz", "rt") as fh:
        rep["ligand.tsv.gz_rows"] = sum(1 for _ in fh) - 1
    # cross-check a few BioLiP rows against the local PDB mirror
    mirror = Path("/cluster/work/beltrao/shared/rsyncPDB/data/structures/divided/mmCIF")
    with gzip.open(base / "BioLiP_nr.txt.gz", "rt") as fh:
        rows = [l.rstrip("\n").split("\t") for _, l in zip(range(200000), fh)]
    ok = miss = nofile = 0
    for c in random.sample(rows, 30):
        p = mirror / c[0][1:3] / f"{c[0]}.cif.gz"
        if not p.exists():
            nofile += 1
            continue
        st = gemmi.read_structure(str(p))
        hit = any(r.name == c[4] for ch in st[0] for r in ch) if c[4] not in ("peptide", "rna", "dna") else True
        ok += hit
        miss += not hit
    rep["mirror_crosscheck_30"] = dict(ligand_found=ok, ligand_not_found=miss, entry_not_in_mirror=nofile)
    print(f"  mirror cross-check: {rep['mirror_crosscheck_30']}")
    return rep


def _mol2_atoms(path):
    n, on = 0, False
    for line in open(path, errors="replace"):
        if line.startswith("@<TRIPOS>"):
            on = line.strip() == "@<TRIPOS>ATOM"
            continue
        if on and line.strip():
            n += 1
    return n


def verify_scpdb():
    base = RAW / "scpdb" / "extracted" / "scPDB"
    entries = sorted(p for p in base.iterdir() if p.is_dir())
    need = ("protein.mol2", "ligand.mol2", "site.mol2", "cavity6.mol2")
    incomplete = [p.name for p in entries if not all((p / f).exists() for f in need)]
    pdb_ids = {p.name.split("_")[0] for p in entries}
    sample = random.sample(entries, 50)
    lig_atoms = [_mol2_atoms(p / "ligand.mol2") for p in sample]
    prot_atoms = [_mol2_atoms(p / "protein.mol2") for p in sample]
    rep = dict(n_entries=len(entries), n_unique_pdb=len(pdb_ids), incomplete_entries=len(incomplete),
               incomplete_examples=incomplete[:5], sample50_ligand_atoms_min_median_max=[min(lig_atoms), int(np.median(lig_atoms)), max(lig_atoms)],
               sample50_protein_atoms_median=int(np.median(prot_atoms)))
    rep["note"] = "website states 16034 entries for release 2017; archive count may differ"
    check("scpdb", min(lig_atoms) > 0, "empty ligand mol2 in sample")
    print(f"  {rep}")
    return rep


def verify_tools():
    bin_ = ROOT / "tools" / "bin"
    rep = {}
    for t in ("mmseqs", "foldseek"):
        rep[t] = subprocess.run([str(bin_ / t), "version"], capture_output=True, text=True).stdout.strip()
    # functional toy test with known answer: a sequence must hit itself with 100% identity
    pm = RAW / "pocketminer" / "data" / "pm-dataset" / "apo-structures"
    pdbs = sorted(pm.glob("*.pdb"))[:3]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fa = td / "q.fasta"
        with open(fa, "w") as f:
            for p in pdbs:
                st = gemmi.read_structure(str(p))
                seq = gemmi.one_letter_code([r.name for r in st[0][0]])
                f.write(f">{p.stem}\n{seq}\n")
        subprocess.run([str(bin_ / "mmseqs"), "easy-search", str(fa), str(fa), str(td / "m.m8"), str(td / "tmpm"), "-v", "1"], check=True)
        m8 = [l.split("\t") for l in open(td / "m.m8")]
        self_hits = sum(1 for r in m8 if r[0] == r[1] and float(r[2]) >= 0.99)
        rep["mmseqs_self_hits"] = f"{self_hits}/{len(pdbs)}"
        sd = td / "s"
        sd.mkdir()
        for p in pdbs:
            os.symlink(p, sd / p.name)
        subprocess.run([str(bin_ / "foldseek"), "easy-search", str(sd), str(sd), str(td / "f.m8"), str(td / "tmpf"), "-v", "1",
                        "--exhaustive-search", "1"], check=True)
        f8 = [l.split("\t") for l in open(td / "f.m8")]
        fs_self = sum(1 for r in f8 if r[0] == r[1])
        rep["foldseek_self_hits"] = f"{fs_self}/{len(pdbs)}"
    check("tools", rep["mmseqs_self_hits"] == f"{len(pdbs)}/{len(pdbs)}", "mmseqs self-hit test failed")
    check("tools", rep["foldseek_self_hits"] == f"{len(pdbs)}/{len(pdbs)}", "foldseek self-hit test failed")
    print(f"  {rep}")
    return rep


def verify_pocketwatch():
    import pandas as pd
    base = RAW / "pocketwatch" / "current"
    sites = pd.read_parquet(base / "data" / "pocket_label_sites.parquet")
    pol = pd.read_parquet(base / "data" / "site_policy.parquet")
    lab = pd.read_parquet(base / "data" / "pocket_labels.parquet")
    cen = pd.read_parquet(base / "data" / "site_centroids.parquet")
    man = json.load(open(base / "labels_manifest.json"))
    rep = dict(proteins_by_split=lab["split"].value_counts().to_dict(),
               sites_by_split=sites["split"].value_counts().to_dict(),
               scored_sites_by_split=pol[pol["decision"] != "masked"]["split"].value_counts().to_dict(),
               policy_reasons=pol["reason"].value_counts().to_dict(),
               n_site_centroids=len(cen),
               n_structures_valtest=len(list((base / "structures").glob("*.cif.gz"))),
               n_structures_train=len(list((base / "structures_train").glob("*.cif.gz"))),
               test_cutoff=man["test_cutoff"], validation_cutoff=man["validation_cutoff"])
    for split in ("train", "validation", "test"):
        check("pocketwatch", man["splits"][split]["proteins"] == rep["proteins_by_split"].get(split),
              f"{split}: protein count disagrees with labels_manifest.json")
    # temporal sanity: every *scored* test site first observed on/after the cutoff (pre-cutoff ones are masked)
    t = sites[sites["split"] == "test"]
    scored = set(pol.loc[pol["decision"] != "masked", "pocket_site_id"])
    ts = t[t["pocket_site_id"].isin(scored)]
    rep["test_scored_min_site_first_observed"] = str(ts["site_first_observed"].min())
    rep["max_site_first_observed"] = str(sites["site_first_observed"].max())
    check("pocketwatch", str(ts["site_first_observed"].min()) >= man["test_cutoff"], "scored test site before cutoff")
    # are test-site PDB entries present in our local PDB mirror?
    mirror = Path("/cluster/work/beltrao/shared/rsyncPDB/data/structures/divided/mmCIF")
    ids = sorted({i for l in t["pdb_ids"] for i in l})
    have = sum((mirror / i[1:3] / f"{i}.cif.gz").exists() for i in ids)
    rep["test_pdb_ids"] = len(ids)
    rep["test_pdb_ids_in_local_mirror"] = have
    # label vectors match sequence lengths
    bad = int(sum(len(l) != n for l, n in zip(lab["label"], lab["sequence_length"])))
    rep["label_length_mismatches"] = bad
    check("pocketwatch", bad == 0, f"{bad} label vectors != sequence length")
    print(f"  {rep}")
    return rep


ALL = dict(p2rank=verify_p2rank, ligysis=verify_ligysis, cryptobench=verify_cryptobench,
           pocketminer=verify_pocketminer, hotpocket=verify_hotpocket, af2bind=verify_af2bind,
           biolip2=verify_biolip2, scpdb=verify_scpdb, pocketwatch=verify_pocketwatch, tools=verify_tools)


def main():
    which = sys.argv[1:] or list(ALL)
    out = RAW / "VERIFY_REPORT.json"
    prev = json.load(open(out)) if out.exists() else {}
    for k in which:
        print(f"== {k}", flush=True)
        try:
            prev[k] = ALL[k]()
        except Exception as e:  # noqa: BLE001
            FAILS.append(f"{k}: exception {e!r}")
            print(f"  EXCEPTION {e!r}")
    prev["_fails_last_run"] = FAILS
    json.dump(prev, open(out, "w"), indent=1, default=str)
    print("FAILS:" if FAILS else "ALL CHECKS PASSED", *FAILS, sep="\n  ")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
