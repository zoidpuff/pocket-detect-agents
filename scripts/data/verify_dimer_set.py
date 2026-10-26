#!/usr/bin/env python
"""Verify the dimer benchmark (data/processed/dimer). Independent re-computations where possible.

Checks (FAIL = non-zero exit; WARN = reported):
  A. entry/assembly rules: method, resolution <= 3.0, assembly details, 2-8 protein chains,
     chains >= 50 residues, one complex per LIGYSIS UniProt (no ACC in two complexes).
  B. inputs: assembly input = protein atoms only, chain ids = chain_map; isolated-chain inputs
     concatenate exactly to the assembly input; no HETATM ligand / water left.
  C. ground truth reproducible: build_ground_truth(holo, spec of gt_manifest_assembly.csv) gives
     the same ligand ids and centroids as gt/<pdb>.json (what scripts/evaluate.py will see).
  D. contacts recomputed independently (cKDTree on the polymer-only input, heavy atoms <= 4 A):
     same binding residues and same site class for every copy; >= 1 contact per copy.
  E. clashes: inter-chain protein heavy-atom pairs < 1.5 A (FAIL if > 10 in a complex);
     ligand non-metal atoms < 2.2 A from protein (counted; covalent ligands), < 1.0 A (WARN,
     listed), < 0.8 A (FAIL).
  F. symmetric copies: for unique sites (j50) with copies on two chains of the same entity,
     superpose chain X onto chain Y (Kabsch on common CA atoms) and map copy X's centroid; it
     should land near copy Y's centroid (reports distribution; WARN if median > 2 A).
  G. oracle ceilings via scripts/eval_dimer.py: j50 must be 1.0 (assembly and primary chain).
  H. manifest_chains.tsv covers every chain with a non-empty sequence.
Also prints counts per class and a few examples. Writes data/processed/dimer/VERIFY_REPORT.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gemmi  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import eval_dimer as ed  # noqa: E402
from build_dimer_common import classify  # noqa: E402
from build_dimer_scan import ligysis_accessions  # noqa: E402
from pocketdetect.ligands import build_ground_truth  # noqa: E402
from pocketdetect.structure import PROTEIN, load_structure, residue_id  # noqa: E402

D = ROOT / "data/processed/dimer"
FAIL: list[str] = []
WARN: list[str] = []


def fail(msg):
    FAIL.append(msg)


def kabsch(P, Q):
    """R, t minimising |R P + t - Q|."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def main():
    cx = pd.read_csv(D / "complexes.tsv", sep="\t", dtype={"complex_id": str, "pdb_id": str})
    cm = pd.read_csv(D / "chain_map.tsv", sep="\t", dtype=str, keep_default_na=False)
    man = pd.read_csv(D / "gt_manifest_assembly.csv", dtype=str, keep_default_na=False)
    mchains = pd.read_csv(D / "manifest_chains.tsv", sep="\t", dtype=str, keep_default_na=False)
    gts = ed.load_gts(D)
    acc = ligysis_accessions()
    rep: dict = {}

    # ---------------- A
    if (cx.resolution > 3.0).any():
        fail("resolution > 3.0")
    if not cx.method.str.split(";").apply(lambda m: set(m) <= {"X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"}).all():
        fail("method")
    if (cx.assembly_details != "author_and_software_defined_assembly").any():
        fail("assembly details")
    if not cx.n_chains.between(2, 8).all():
        fail("chain count")
    if (cm.n_observed_residues.astype(int) < 50).any():
        fail("chain < 50 residues")
    seen = Counter(a for u in cx.uniprots.fillna("") for a in set(u.split(";")) & acc)
    dup = [a for a, n in seen.items() if n > 1]
    if dup:
        fail(f"LIGYSIS UniProt in >1 complex: {dup[:5]}")
    rep["n_ligysis_uniprots_covered"] = len(seen)

    # ---------------- B-F per complex
    contact_mismatch, class_mismatch, gt_mismatch, no_contact = [], [], [], []
    lig_close_warn, lig_close_fail, pp_clash = 0, [], {}
    sym_dev = []
    n_copies = 0
    for pid, gt in gts.items():
        chains = [c["chain"] for c in gt["chains"]]
        chain_entity = {c["chain"]: c["entity_id"] for c in gt["chains"]}
        s = load_structure(str(D / "inputs/assembly" / f"{pid}.pdb"))
        if not (s.kind == PROTEIN).all():
            fail(f"{pid}: non-protein atoms in assembly input")
        if sorted(set(s.chain.tolist())) != sorted(chains) or sorted(chains) != sorted(cm[cm.complex_id == pid].chain):
            fail(f"{pid}: chain ids differ from chain_map")
        parts = [load_structure(str(D / "inputs/chains" / f"{pid}_{c}.pdb")) for c in chains]
        cat = np.concatenate([p.coords for p in parts])
        ref = np.concatenate([s.coords[s.chain == c] for c in chains])
        if cat.shape != ref.shape or not np.array_equal(cat, ref):
            fail(f"{pid}: chain inputs do not reproduce assembly input")
        # C
        row = man[man.structure_id == pid].iloc[0]
        g2 = build_ground_truth(str(D / row.structure_path), structure_id=pid, rules="p2rank", spec=row.ligand_spec)
        a = {l.ligand_id: l.center for l in g2.ligands}
        b = {c["ligand_id"]: np.asarray(c["center"]) for c in gt["copies"]}
        if set(a) != set(b) or any(np.abs(a[k] - b[k]).max() > 1e-3 for k in b):
            gt_mismatch.append(pid)
        # D/E
        tree = cKDTree(s.coords.astype(np.float64))
        rids = np.array([residue_id(c, n, i) for c, n, i in zip(s.chain, s.res_num, s.ins_code)], dtype=object)
        pairs = tree.query_pairs(1.5, output_type="ndarray")
        npp = int((s.chain[pairs[:, 0]] != s.chain[pairs[:, 1]]).sum()) if len(pairs) else 0
        pp_clash[pid] = npp
        if npp > 10:
            fail(f"{pid}: {npp} inter-chain clashes")
        for c in gt["copies"]:
            n_copies += 1
            X = np.asarray(c["coords"])
            hits = tree.query_ball_point(X, r=4.0)
            res = sorted(set(rids[np.concatenate([np.asarray(h, int) for h in hits])].tolist())) if any(len(h) for h in hits) else []
            if res != sorted(c["binding_residues"]):
                contact_mismatch.append((pid, c["ligand_id"]))
            if not res:
                no_contact.append((pid, c["ligand_id"]))
            counts = Counter(r.split(":")[0] for r in res)
            if classify(dict(counts), chain_entity) != c["site_class"]:
                class_mismatch.append((pid, c["ligand_id"]))
            d, _ = tree.query(X, k=1)
            nonmetal = np.array([not gemmi.Element(e).is_metal for e in c["elements"]])
            if ((d < 2.2) & nonmetal).any():
                lig_close_warn += 1
            if (d < 0.8).any():  # physically impossible even for a (distorted) covalent bond
                lig_close_fail.append((pid, c["ligand_id"], float(d.min())))
            elif (d < 1.0).any():
                WARN.append(f"{pid} {c['ligand_id']}: ligand atom {d.min():.2f} A from protein (check covalent link)")
        # F: symmetric copies of single-chain sites on different chains of one entity
        ca = {}
        for ch in chains:
            m = (s.chain == ch) & (s.atom_name == "CA")
            ca[ch] = dict(zip(zip(s.res_num[m].tolist(), s.ins_code[m].tolist()), s.coords[m].astype(np.float64)))
        by = defaultdict(list)
        for c, gidx in zip(gt["copies"], ed.site_gt(gt, "j50").copy_site):
            if c["site_class"] == "single_chain":
                by[int(gidx)].append(c)
        for cs in by.values():
            if len(cs) < 2:
                continue
            c0 = cs[0]
            x = max(c0["chain_counts"], key=c0["chain_counts"].get)
            for c1 in cs[1:]:
                y = max(c1["chain_counts"], key=c1["chain_counts"].get)
                if x == y or chain_entity[x] != chain_entity[y]:
                    continue
                common = sorted(set(ca[x]) & set(ca[y]))
                if len(common) < 10:
                    continue
                R, t = kabsch(np.array([ca[x][k] for k in common]), np.array([ca[y][k] for k in common]))
                sym_dev.append(float(np.linalg.norm(R @ np.asarray(c0["center"]) + t - np.asarray(c1["center"]))))
    if gt_mismatch:
        fail(f"GT not reproducible for {len(gt_mismatch)} complexes: {gt_mismatch[:5]}")
    if contact_mismatch:
        fail(f"independent contacts differ for {len(contact_mismatch)} copies: {contact_mismatch[:5]}")
    if class_mismatch:
        fail(f"class differs for {len(class_mismatch)} copies: {class_mismatch[:5]}")
    if no_contact:
        fail(f"{len(no_contact)} copies without contact")
    if lig_close_fail:
        fail(f"{len(lig_close_fail)} ligand copies with an atom < 0.8 A from protein: {lig_close_fail[:5]}")
    sym = np.asarray(sym_dev)
    rep["symmetric_copy_check"] = {
        "n_pairs": int(sym.size),
        "median_A": float(np.median(sym)) if sym.size else None,
        "frac_lt_2A": float((sym < 2).mean()) if sym.size else None,
        "frac_lt_4A": float((sym < 4).mean()) if sym.size else None,
        "max_A": float(sym.max()) if sym.size else None,
    }
    if sym.size and np.median(sym) > 2:
        WARN.append("symmetric copies: median superposition deviation > 2 A")
    rep["n_copies"] = n_copies
    rep["copies_with_nonmetal_atom_lt_2.2A"] = lig_close_warn
    rep["max_interchain_clash_pairs"] = int(max(pp_clash.values()))
    rep["complexes_with_interchain_clash_pairs"] = int(sum(v > 0 for v in pp_clash.values()))

    # ---------------- G
    ceil = {}
    for grouping in ("j50", "strict"):
        for mode in ("assembly", "chain"):
            sites, units, summ = ed.run(gts, mode, None, grouping, "primary", True, True, D, n_boot=0)
            v = summ[(summ.metric == "recall_DCC12_topn2")].set_index("stratum")["value"]
            ceil[f"{grouping}_{mode}"] = {k: round(float(v[k]), 4) for k in ("all", "interface", "homo_interface", "hetero_interface", "single_chain")}
            if grouping == "j50" and abs(v["all"] - 1.0) > 1e-9:
                fail(f"oracle ceiling j50 {mode} = {v['all']:.4f} != 1")
    rep["oracle_ceiling_recall_DCC12_topn2"] = ceil

    # ---------------- H
    k1 = set(zip(cm.complex_id, cm.chain))
    k2 = set(zip(mchains.pdb_id, mchains.chain))
    if k1 != k2 or (mchains.sequence.str.len() == 0).any():
        fail("manifest_chains.tsv incomplete")
    rep["n_manifest_chains"] = len(mchains)
    rep["n_unique_sequences"] = int(mchains.sequence.nunique())

    # ---------------- counts
    us = pd.read_csv(D / "unique_sites.tsv", sep="\t", dtype={"complex_id": str})
    sc = pd.read_csv(D / "site_copies.tsv", sep="\t", dtype={"complex_id": str})
    rep["n_complexes"] = len(cx)
    rep["complex_type"] = cx.complex_type.value_counts().to_dict()
    rep["n_chains_distribution"] = {int(k): int(v) for k, v in cx.n_chains.value_counts().sort_index().items()}
    rep["complexes_with_interface_site"] = int(((cx.n_homo_interface_j50 + cx.n_hetero_interface_j50) > 0).sum())
    rep["copies_per_class"] = sc.site_class.value_counts().to_dict()
    for grouping in ("j50", "strict"):
        u = us[us.grouping == grouping]
        rep[f"unique_sites_{grouping}"] = {
            "total": len(u),
            **u.site_class.value_counts().to_dict(),
            "ions": int(u.is_ion.sum()),
            "multi_copy": int((u.n_copies > 1).sum()),
            "class_inconsistent": int((~u.class_consistent).sum()),
        }
    rep["examples"] = {}
    for cls in ("homo_interface", "hetero_interface", "single_chain"):
        ex = sc[sc.site_class == cls].head(2)
        rep["examples"][cls] = ex[["complex_id", "ligand_id", "chain_counts", "binding_residues"]].to_dict("records")
    rep["FAIL"] = FAIL
    rep["WARN"] = WARN
    (D / "VERIFY_REPORT.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    print("VERIFY:", "FAIL" if FAIL else "PASS", f"({len(FAIL)} failures, {len(WARN)} warnings)")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
