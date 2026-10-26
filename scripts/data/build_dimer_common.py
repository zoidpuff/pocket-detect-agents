"""Shared code of the dimer benchmark builder (Track D; spec: DESIGN.md §2.7 + §9, docs/PREREG.md,
docs/DIMER_BENCHMARK.md).

``process_entry`` turns one PDB entry into a benchmark candidate:

1. read the asymmetric-unit mmCIF of the local wwPDB mirror (first model, hydrogens, waters and
   alternative conformations beyond the first removed);
2. entry filters: method X-ray and/or EM only, resolution <= 3.0 A, assembly 1 with
   ``_pdbx_struct_assembly.details == author_and_software_defined_assembly``;
3. expand assembly 1 ourselves (gemmi operators) so every new chain is traceable to
   (author chain, operator); new chain ids are single characters (original id kept when possible);
4. assembly filters: 2-8 protein chains, every protein chain >= 50 observed residues, no
   nucleic-acid / other polymer chains, entity-consistent author numbering, <= 99,999 atoms;
5. relevant ligands = BioLiP2 annotations of *this entry* (ligand chain, CCD, author resnum; not
   pooled across entries), every assembly copy of such a ligand residue is a ligand copy;
6. ground truth via ``pocketdetect.ligands.build_ground_truth`` (explicit P2Rank-style spec on
   the written holo assembly file, so the evaluator sees identical ligands), contacts = protein
   heavy atoms <= 4 A of ligand heavy atoms;
7. per copy: residues per chain, class (single_chain / homo_interface / hetero_interface /
   minor_interface), entity-level residue set (entity id + author resnum/icode), ion flag;
8. symmetric copies grouped (strict: identical entity-level sets = PREREG rule; j50: connected
   components of entity-level Jaccard >= 0.5, recorded as a sensitivity grouping).
"""

from __future__ import annotations

import gzip
import json
import os
import string
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from pocketdetect.ligands import build_ground_truth
from pocketdetect.structure import PROTEIN, load_structure, parse_residue_id, pdb_entry_path

MAX_RESOLUTION = 3.0
ALLOWED_METHODS = {"X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"}
ASSEMBLY_DETAILS = "author_and_software_defined_assembly"
MIN_CHAINS, MAX_CHAINS = 2, 8
MIN_CHAIN_RESIDUES = 50
CONTACT_DIST = 4.0  # == LigandRules.binding_residue_dist of the 'p2rank' rule set (asserted)
INTERFACE_MIN_RES = 3
MAX_ATOMS = 99_999  # PDB format atom serial limit
PP_CLASH_DIST = 1.5  # inter-chain protein heavy-atom pairs closer than this = bad assembly
MAX_PP_CLASH_PAIRS = 10
LIG_CLASH_DIST = 2.2  # ligand heavy atom -> protein heavy atom (non-ion; reported, covalent ligands hit)
BIOLIP_POLYMER_LIGANDS = {"peptide", "rna", "dna"}
CHAIN_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits
SITE_CLASSES = ("single_chain", "homo_interface", "hetero_interface", "minor_interface")


# ----------------------------------------------------------------------------------- helpers
def residue_key(chain: str, num: int, icode: str) -> str:
    icode = "" if icode in ("\x00", " ", "?", ".") else icode
    return f"{chain}:{int(num)}{icode}"


def entity_residue(entity: str, rid: str) -> str:
    """Entity-level residue id 'E<entity>:<resnum><icode>' from a 'chain:resnum[icode]' id."""
    _, num, icode = parse_residue_id(rid)
    return f"E{entity}:{num}{icode}"


def jaccard(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def group_strict(sets: list[frozenset]) -> list[int]:
    """Group index per item: identical sets share a group (groups numbered by first occurrence)."""
    seen: dict[frozenset, int] = {}
    return [seen.setdefault(s, len(seen)) for s in sets]


def group_jaccard(sets: list[frozenset], thr: float = 0.5) -> list[int]:
    """Connected components of the graph 'Jaccard >= thr' (numbered by first occurrence)."""
    n = len(sets)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(sets[i], sets[j]) >= thr:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    order: dict[int, int] = {}
    return [order.setdefault(r, len(order)) for r in roots]


def classify(chain_counts: dict[str, int], chain_entity: dict[str, str]) -> str:
    qual = [c for c, k in chain_counts.items() if k >= INTERFACE_MIN_RES]
    if len(qual) >= 2:
        return "hetero_interface" if len({chain_entity[c] for c in qual}) >= 2 else "homo_interface"
    contacted = [c for c, k in chain_counts.items() if k > 0]
    return "minor_interface" if len(contacted) >= 2 else "single_chain"


def _cif_table(block, prefix: str, tags: list[str]) -> list[list[str]]:
    t = block.find(prefix, tags)
    return [[gemmi.cif.as_string(r[i]) for i in range(len(tags))] for r in t]


# ----------------------------------------------------------------------------------- BioLiP2
def load_biolip(path: str | Path, pdb_ids: set | None = None) -> dict[str, set]:
    """pdb_id -> {(ligand_chain, CCD, resnum)} of BioLiP2 non-polymer ligands (ions included)."""
    out: dict[str, set] = defaultdict(set)
    with gzip.open(path, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            pid = p[0].lower()
            if pdb_ids is not None and pid not in pdb_ids:
                continue
            if p[4] in BIOLIP_POLYMER_LIGANDS:
                continue
            try:
                num = int(p[19].strip())
            except ValueError:
                continue
            out[pid].add((p[5], p[4], num))
    return out


# ----------------------------------------------------------------------------------- entry
class Reject(Exception):
    pass


def _assembly1_details(block) -> str:
    for aid, det in _cif_table(block, "_pdbx_struct_assembly.", ["id", "details"]):
        if aid == "1":
            return det
    return ""


def _entity_uniprot(block) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for ent, db, acc in _cif_table(block, "_struct_ref.", ["entity_id", "db_name", "pdbx_db_accession"]):
        if db.upper() == "UNP" and acc not in out[ent]:
            out[ent].append(acc)
    return dict(out)


def expand_assembly(pdb_id: str, biolip: set, ligysis_acc: set):
    """Read entry, apply entry filters, expand assembly 1. Returns (new gemmi.Structure, info)."""
    path = pdb_entry_path(pdb_id)
    if not path.exists():
        raise Reject("no_au_file")
    block = gemmi.cif.read(str(path)).sole_block()
    methods = {gemmi.cif.as_string(m).upper() for m in block.find_values("_exptl.method")}
    if not methods or not methods <= ALLOWED_METHODS:
        raise Reject("method")
    details = _assembly1_details(block)
    if details != ASSEMBLY_DETAILS:
        raise Reject("assembly_details")
    st = gemmi.make_structure_from_block(block)
    if not (0 < st.resolution <= MAX_RESOLUTION):
        raise Reject("resolution")
    st.setup_entities()
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    st.remove_waters()
    asm = next((a for a in st.assemblies if a.name == "1"), None)
    if asm is None:
        raise Reject("no_assembly1")
    model = st[0]
    sub_entity, ent_info = {}, {}
    unp = _entity_uniprot(block)
    for e in st.entities:
        for sc in e.subchains:
            sub_entity[sc] = e.name
        ent_info[e.name] = {
            "type": e.entity_type.name,
            "polymer_type": e.polymer_type.name,
            "sequence": gemmi.one_letter_code(list(e.full_sequence)) if e.entity_type == gemmi.EntityType.Polymer else "",
            "uniprot": unp.get(e.name, []),
        }
    # residues per subchain (with their author chain)
    sub_res: dict[str, list] = defaultdict(list)
    for ch in model:
        for r in ch:
            sub_res[r.subchain].append((ch.name, r))

    # expand: key (auth chain, operator name) -> residues (transformed copies)
    keys: list[tuple] = []
    key_res: dict[tuple, list] = {}
    key_kind: dict[tuple, set] = defaultdict(set)
    for gen in asm.generators:
        for op in gen.operators:
            tr = op.transform
            for sc in gen.subchains:
                ent = sub_entity.get(sc)
                for auth, r in sub_res.get(sc, []):
                    k = (auth, op.name)
                    if k not in key_res:
                        keys.append(k)
                        key_res[k] = []
                    r2 = r.clone()
                    for a in r2:
                        a.pos = gemmi.Position(tr.apply(a.pos))
                    key_res[k].append((sc, ent, r2))
                    if ent is not None and ent_info[ent]["type"] == "Polymer":
                        key_kind[k].add(ent_info[ent]["polymer_type"])
    if not keys:
        raise Reject("empty_assembly")
    # polymer composition
    for k in keys:
        if any(pt != "PeptideL" for pt in key_kind[k]):
            raise Reject("non_protein_polymer")
    prot_keys = [k for k in keys if key_kind[k]]
    # assign new single-character chain ids: protein chains first, keep author id if possible
    new_id: dict[tuple, str] = {}
    used: set = set()
    first_op = {}
    for k in keys:
        first_op.setdefault(k[0], k[1])
    ordered = prot_keys + [k for k in keys if k not in prot_keys]
    for k in ordered:
        if len(k[0]) == 1 and k[0] not in used and first_op[k[0]] == k[1]:
            new_id[k] = k[0]
            used.add(k[0])
    for k in ordered:
        if k in new_id:
            continue
        free = [c for c in CHAIN_ALPHABET if c not in used]
        if not free:
            raise Reject("too_many_chains_for_pdb")
        new_id[k] = free[0]
        used.add(free[0])

    new = gemmi.Structure()
    new.name = pdb_id
    new.cell = gemmi.UnitCell()
    nm = gemmi.Model("1")
    chain_rows = []
    lig_specs = []
    biolip_found = set()
    n_atoms = 0
    for k in ordered:
        cid = new_id[k]
        ch = gemmi.Chain(cid)
        poly_ent = None
        n_obs = 0
        for sc, ent, r in key_res[k]:
            is_poly = ent is not None and ent_info[ent]["type"] == "Polymer"
            if is_poly:
                poly_ent = ent
                n_obs += 1
            else:
                bk = (k[0], r.name, r.seqid.num)
                if bk in biolip:
                    biolip_found.add(bk)
                    lig_specs.append(f"{r.name}[{cid}_{r.seqid.num}{'' if r.seqid.icode == ' ' else r.seqid.icode}]")
            n_atoms += len(r)
            ch.add_residue(r)
        nm.add_chain(ch)
        if k in prot_keys:
            chain_rows.append(
                {
                    "chain": cid,
                    "orig_auth_chain": k[0],
                    "operator": k[1],
                    "label_asym_ids": sorted({sc for sc, ent, _ in key_res[k] if ent == poly_ent}),
                    "entity_id": poly_ent,
                    "n_observed_residues": n_obs,
                }
            )
    new.add_model(nm)
    if not (MIN_CHAINS <= len(chain_rows) <= MAX_CHAINS):
        raise Reject("n_protein_chains")
    if min(c["n_observed_residues"] for c in chain_rows) < MIN_CHAIN_RESIDUES:
        raise Reject("short_protein_chain")
    if n_atoms > MAX_ATOMS:
        raise Reject("too_many_atoms")
    if not any(set(ent_info[c["entity_id"]]["uniprot"]) & ligysis_acc for c in chain_rows):
        raise Reject("no_ligysis_uniprot_in_assembly")
    if not lig_specs:
        raise Reject("no_biolip_ligand_in_assembly")
    # entity-consistent author numbering (entity-level residue ids use author numbering)
    num_map: dict[str, dict] = defaultdict(dict)
    for k in prot_keys:
        for sc, ent, r in key_res[k]:
            if ent is None or ent_info[ent]["type"] != "Polymer" or r.label_seq is None:
                continue
            key = (r.seqid.num, r.seqid.icode)
            prev = num_map[ent].setdefault(r.label_seq, key)
            if prev != key:
                raise Reject("inconsistent_author_numbering")
    info = {
        "pdb_id": pdb_id,
        "method": ";".join(sorted(methods)),
        "resolution": round(float(st.resolution), 3),
        "assembly_details": details,
        "oligomeric_details": asm.oligomeric_details,
        "chains": chain_rows,
        "entities": {e: ent_info[e] for e in sorted({c["entity_id"] for c in chain_rows})},
        "ligand_specs": lig_specs,
        "n_biolip_ligands_entry": len(biolip),
        "n_biolip_ligands_in_assembly": len(biolip_found),
    }
    return new, info


def write_pdb(st: gemmi.Structure, path: str | Path) -> None:
    st.setup_entities()
    opt = gemmi.PdbWriteOptions(minimal=True)
    opt.ter_records = True
    opt.cryst1_record = False  # no (dummy) unit cell: inputs must never be symmetry-expanded
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(path), opt)


def analyse_sites(pdb_id: str, holo_path: str | Path, info: dict) -> dict:
    """Ground truth + site classification on the written holo assembly file."""
    gt = build_ground_truth(str(holo_path), structure_id=pdb_id, rules="p2rank", spec=",".join(info["ligand_specs"]))
    from pocketdetect.ligands import get_rules

    assert get_rules("p2rank").binding_residue_dist == CONTACT_DIST
    unmatched = gt.meta.get("unmatched_spec", [])
    chain_entity = {c["chain"]: c["entity_id"] for c in info["chains"]}
    s = load_structure(str(holo_path), structure_id=pdb_id)
    prot = s.kind == PROTEIN
    pcoords = s.coords[prot].astype(np.float64)
    pchain = s.chain[prot]
    ptree = cKDTree(pcoords)
    # inter-chain protein clash pairs
    pairs = ptree.query_pairs(PP_CLASH_DIST, output_type="ndarray")
    n_pp_clash = int((pchain[pairs[:, 0]] != pchain[pairs[:, 1]]).sum()) if len(pairs) else 0
    # protein chain set according to structure.py must equal our protein chains
    prot_chains = sorted(set(pchain.tolist()))
    copies = []
    for lig in gt.ligands:
        res = list(lig.binding_residues)
        counts: dict[str, int] = defaultdict(int)
        for rid in res:
            c, _, _ = parse_residue_id(rid)
            counts[c] += 1
        unknown = [c for c in counts if c not in chain_entity]
        if unknown:
            raise RuntimeError(f"{pdb_id}: contact on non-protein chain {unknown}")
        ent_set = sorted({entity_residue(chain_entity[parse_residue_id(r)[0]], r) for r in res})
        d, _ = ptree.query(lig.coords, k=1)
        nonmetal = np.array([not gemmi.Element(str(e)).is_metal for e in lig.elements], dtype=bool)
        copies.append(
            {
                "ligand_id": lig.ligand_id,
                "ccd": lig.name,
                "n_atoms": lig.n_atoms,
                "is_ion": bool(lig.n_atoms == 1),
                "center": [round(float(x), 3) for x in lig.center],
                "coords": np.round(lig.coords, 3).tolist(),
                "elements": [str(e) for e in lig.elements],
                "binding_residues": res,
                "chain_counts": dict(sorted(counts.items())),
                "entity_residues": ent_set,
                "site_class": classify(counts, chain_entity) if res else "no_contact",
                "min_dist_protein": round(float(d.min()), 3),
                "n_atoms_clash": int(((d < LIG_CLASH_DIST) & nonmetal).sum()),  # metals excluded
            }
        )
    # drop copies without any protein contact (ligand subchain in assembly but receptor not)
    n_no_contact = sum(c["site_class"] == "no_contact" for c in copies)
    copies = [c for c in copies if c["site_class"] != "no_contact"]
    sets = [frozenset(c["entity_residues"]) for c in copies]
    for c, g, gj in zip(copies, group_strict(sets), group_jaccard(sets, 0.5)):
        c["site_strict"] = g
        c["site_j50"] = gj
    # per-chain contact totals (isolated-chain input = chain with most site contacts)
    chain_contacts = {c["chain"]: 0 for c in info["chains"]}
    for c in copies:
        for ch, k in c["chain_counts"].items():
            chain_contacts[ch] += k
    primary = max(chain_contacts, key=lambda ch: (chain_contacts[ch], -list(chain_contacts).index(ch)))
    return {
        "copies": copies,
        "n_unmatched_spec": len(unmatched),
        "unmatched_spec": unmatched,
        "n_copies_no_contact": n_no_contact,
        "n_pp_clash_pairs": n_pp_clash,
        "protein_chains_structure_py": prot_chains,
        "chain_contacts": chain_contacts,
        "primary_chain": primary,
        "n_protein_atoms": int(prot.sum()),
    }


def unique_sites(copies: list[dict], key: str = "site_strict") -> list[dict]:
    """Unique-site summary (class of a unique site = most 'interface' class among its copies)."""
    rank = {"hetero_interface": 3, "homo_interface": 2, "minor_interface": 1, "single_chain": 0}
    by: dict[int, list] = defaultdict(list)
    for c in copies:
        by[c[key]].append(c)
    out = []
    for g in sorted(by):
        cs = by[g]
        classes = {c["site_class"] for c in cs}
        cls = max(classes, key=lambda x: rank[x])
        out.append(
            {
                "site": g,
                "site_class": cls,
                "class_consistent": len(classes) == 1,
                "n_copies": len(cs),
                "is_ion": all(c["is_ion"] for c in cs),
                "ccd": "&".join(sorted({c["ccd"] for c in cs})),
            }
        )
    return out


def process_entry(pdb_id: str, biolip: set, ligysis_acc: set, holo_path: str | Path) -> dict:
    """Full per-entry processing; writes the holo assembly PDB to ``holo_path``."""
    try:
        st, info = expand_assembly(pdb_id, biolip, ligysis_acc)
        write_pdb(st, holo_path)
        sites = analyse_sites(pdb_id, holo_path, info)
        info.update(sites)
        if sorted(c["chain"] for c in info["chains"]) != sites["protein_chains_structure_py"]:
            raise Reject("protein_chain_mismatch_structure_py")
        if sites["n_pp_clash_pairs"] > MAX_PP_CLASH_PAIRS:
            raise Reject("interchain_clash")
        if not info["copies"]:
            raise Reject("no_contacting_ligand")
        info["status"] = "ok"
        return info
    except Reject as e:
        return {"pdb_id": pdb_id, "status": f"reject:{e}"}
    except Exception as e:  # reported, never silently dropped
        return {"pdb_id": pdb_id, "status": f"error:{type(e).__name__}: {e}"}


def json_dump(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)
