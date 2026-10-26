import numpy as np
import pytest
from conftest import PLUS, R, A, backbone, build, ligand_res, write

from pocketdetect.ligands import (
    RULE_SETS,
    build_ground_truth,
    extract_ligands,
    get_rules,
    parse_ligand_spec,
)
from pocketdetect.structure import load_structure

C0 = np.array([10.0, 0.0, 3.5])  # 3.5 A above the backbone C atom of A:3 at (10.1, 0, 0)


def protein():
    # chain A along +x in the z=0 plane; chain B parallel at z=7 so a ligand at z=3.5 touches both
    return backbone("A", 1, 10) + backbone("B", 1, 10, origin=(0, 0, 7.0))


def gt_of(tmp_path, extra, rules="p2rank", spec=None, fmt=".cif", chains=None):
    p = write(build(protein() + extra), tmp_path / f"t{fmt}")
    return build_ground_truth(p, rules=rules, spec=spec, chains=chains)


def cats(gt):
    return {l.ligand_id: l.category for l in gt.ligands + gt.other_ligands}


@pytest.mark.parametrize("fmt", [".cif", ".pdb"])
def test_relevant_ligand_basic(tmp_path, fmt):
    gt = gt_of(tmp_path, [ligand_res("A", "LIG", 900, C0 + PLUS)], fmt=fmt)
    assert [l.ligand_id for l in gt.ligands] == ["LIG|A:900"]
    l = gt.ligands[0]
    assert l.n_atoms == 5
    assert np.allclose(l.center, C0, atol=1e-4)
    assert l.contact_dist == pytest.approx(3.5, abs=0.02)
    # binding residues: both chains (B atoms at z=7 -> 3.5 A away)
    assert any(r.startswith("A:") for r in l.binding_residues)
    assert any(r.startswith("B:") for r in l.binding_residues)
    assert "A:3" in l.binding_residues
    assert "A:9" not in l.binding_residues
    assert len(gt.protein_residues) == 20


def test_small_ignored_distant(tmp_path):
    extra = [
        ligand_res("A", "SML", 901, (C0 + PLUS)[:4] + [20, 0, 0]),  # 4 atoms -> small
        ligand_res("A", "GOL", 902, C0 + PLUS + [-8, 0, 0]),  # ignored het group
        ligand_res("A", "FAR", 903, C0 + PLUS + [0, 30, 0]),  # no contact -> distant
        # contact ok (3.5 A) but mass centre 6 A above the protein plane... and 6.5 below chain B? put it
        # on the -z side: atoms at z=-3.5 ... -15.5 -> centre z=-9.5 -> distant by ligc_prot_dist
        ligand_res("A", "LNG", 904, [[25, 0, -3.5 - 3 * i] for i in range(5)]),
    ]
    gt = gt_of(tmp_path, extra)
    c = cats(gt)
    assert c == {"SML|A:901": "small", "GOL|A:902": "ignored", "FAR|A:903": "distant", "LNG|A:904": "distant"}
    assert gt.n_ligands == 0
    lng = [l for l in gt.other_ligands if l.name == "LNG"][0]
    assert lng.contact_dist <= 4.0 and lng.center_dist > 5.5


def test_hydrogens_count_only_in_strict(tmp_path):
    heavy = C0 + PLUS[:4]
    hyd = np.array([[10, 0, 4.5], [10.5, 0.5, 4.5]])
    res = ligand_res("A", "LGH", 905, np.vstack([heavy, hyd]), ["C"] * 4 + ["H", "H"])
    gt = gt_of(tmp_path, [res], rules="p2rank")
    assert cats(gt) == {"LGH|A:905": "small"}
    gt2 = gt_of(tmp_path, [res], rules="p2rank_strict")
    assert cats(gt2) == {"LGH|A:905": "relevant"}
    assert gt2.ligands[0].n_atoms == 6


def test_clustering_merges_bonded_groups(tmp_path):
    g1 = ligand_res("A", "LIG", 900, C0 + [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    g2 = ligand_res("A", "LIH", 901, C0 + [[3.5, 0, 0], [4.5, 0, 0], [5.5, 0, 0]])  # 1.5 A from g1
    gt = gt_of(tmp_path, [g1, g2])
    assert [l.ligand_id for l in gt.ligands] == ["LIG&LIH|A:900&A:901"]
    assert gt.ligands[0].n_atoms == 6  # each group alone would be "small"
    g2far = ligand_res("A", "LIH", 901, C0 + [[5.0, 0, 0], [6, 0, 0], [7, 0, 0]])  # 3 A apart
    gt = gt_of(tmp_path, [g1, g2far])
    assert gt.n_ligands == 0 and sorted(cats(gt).values()) == ["small", "small"]


def test_ignored_groups_not_clustered(tmp_path):
    lig = ligand_res("A", "LIG", 900, C0 + PLUS)
    nag = ligand_res("A", "NAG", 901, C0 + PLUS + [3.5, 0, 0])  # 1.5 A from LIG atom
    gt = gt_of(tmp_path, [lig, nag])
    assert cats(gt) == {"LIG|A:900": "relevant", "NAG|A:901": "ignored"}


def test_explicit_spec(tmp_path):
    extra = [
        ligand_res("A", "LIG", 900, C0 + PLUS),
        ligand_res("A", "LIG", 901, C0 + PLUS + [12, 0, 0]),
        ligand_res("A", "MG", 950, [C0 + [0, 0, 2.2]], ["MG"]),
    ]
    gt = gt_of(tmp_path, extra, spec="LIG")
    assert [l.ligand_id for l in gt.ligands] == ["LIG|A:900", "LIG|A:901"]
    assert cats(gt)["MG|A:950"] == "ignored"
    gt = gt_of(tmp_path, extra, spec="LIG[A_901],MG")
    assert sorted(l.ligand_id for l in gt.ligands) == ["LIG|A:901", "MG|A:950"]  # explicit -> no size filter
    gt = gt_of(tmp_path, extra, spec="ATP,LIG[A_900],<CONFLICTS>")  # P2Rank: unmatched -> reported only
    assert [l.ligand_id for l in gt.ligands] == ["LIG|A:900"]
    assert gt.meta["unmatched_spec"] == ["<CONFLICTS>", "ATP"]
    p = write(build(protein() + extra), tmp_path / "strict.cif")
    with pytest.raises(ValueError):
        build_ground_truth(p, spec="ATP", strict_spec=True)
    defs = parse_ligand_spec("GTP[group_id:B_1173A],MG[atom_id:5027],ATP")
    assert defs[0].chain == "B" and defs[0].resnum == 1173 and defs[0].icode == "A"
    assert defs[1].atom_serial == 5027 and defs[2].chain is None
    assert parse_ligand_spec("") is None and parse_ligand_spec(None) is None


def test_insertion_code_and_chain_selection(tmp_path):
    lig = ligand_res("B", "LIG", 900, C0 + PLUS)
    lig.icode = "B"
    gt = gt_of(tmp_path, [lig])
    assert gt.ligands[0].ligand_id == "LIG|B:900B"
    gt = gt_of(tmp_path, [lig], chains=["A"])  # ligand belongs to chain B -> removed
    assert gt.n_ligands == 0 and all(r.startswith("A:") for r in gt.protein_residues)


def test_modified_residue_not_ligand(tmp_path):
    prot = protein()
    prot[2].name, prot[2].het = "MSE", True
    prot[2].atoms.append(A("SE", "SE", (9.0, -3.5, 0)))
    for fmt in (".cif", ".pdb"):
        p = write(build(prot), tmp_path / f"m{fmt}")
        s = load_structure(p, remove_hydrogens=False)
        assert extract_ligands(s, "p2rank") == []


def test_center_of_mass_vs_centroid(tmp_path):
    import dataclasses

    from pocketdetect.ligands import register_rules

    coords = C0 + PLUS
    res = ligand_res("A", "LBR", 900, coords, ["BR", "C", "C", "C", "C"])
    l = gt_of(tmp_path, [res]).ligands[0]
    assert np.allclose(l.center, coords.mean(0), atol=1e-4)  # default: geometric centroid
    register_rules(dataclasses.replace(get_rules("p2rank"), name="test_com", center_method="com"))
    l = gt_of(tmp_path, [res], rules="test_com").ligands[0]
    w = np.array([79.904, 12.011, 12.011, 12.011, 12.011])
    assert np.allclose(l.center, (coords * w[:, None]).sum(0) / w.sum(), atol=1e-3)


def test_rules_registry():
    assert {"p2rank", "p2rank_strict", "all_contacting"} <= set(RULE_SETS)
    r = get_rules("p2rank")
    assert r.min_atoms == 5 and r.contact_dist == 4 and r.center_dist == 5.5 and r.cluster_dist == 1.7
    assert r.ignore_het_groups == {"HOH", "DOD", "WAT", "NAG", "MAN", "UNK", "GLC", "ABA", "MPD", "GOL", "SO4", "PO4"}
    with pytest.raises(KeyError):
        get_rules("nope")


def test_all_hetatm_peptide_is_ligand(tmp_path):
    """Peptidic inhibitor deposited as HETATM residues (cf. 1nlu IVA-PHI-TYB): one merged ligand,
    as in P2Rank; a modified residue bonded to ATOM residues stays protein."""
    pep = backbone("C", 1, 3, origin=(6.0, 0, 3.6))  # parallel to chain A, 3.6 A above it
    for r, nm in zip(pep, ["IVA", "PHI", "TYB"]):  # all non-standard names
        r.name, r.het = nm, True
    for fmt in (".pdb", ".cif"):
        gt = gt_of(tmp_path, pep, fmt=fmt)
        assert [l.ligand_id for l in gt.ligands] == ["IVA&PHI&TYB|C:1&C:2&C:3"], fmt
        assert gt.ligands[0].n_atoms == 15
    # same peptide but the middle residue is an ATOM record -> whole fragment is protein
    pep[1].het = False
    gt = gt_of(tmp_path, pep)
    assert gt.ligands == [] and "C:2" in set(gt.protein_residues)


def test_standard_residue_names_never_ligands_in_p2rank(tmp_path):
    """P2Rank ignores free standard amino acids / nucleotides (e.g. TRP in 1zt9);
    the permissive rule set keeps them."""
    trp = ligand_res("A", "TRP", 900, C0 + PLUS)
    gt = gt_of(tmp_path, [trp])
    assert gt.ligands == [] and gt.other_ligands == []
    gt = gt_of(tmp_path, [trp], rules="all_contacting")
    assert [l.ligand_id for l in gt.ligands] == ["TRP|A:900"]


def test_p2rank_strict_quirks(tmp_path):
    from conftest import A as At

    # (a) ligand touching only an atom named NH1 (Arg-like): P2Rank's name-based H filter drops it
    prot = protein()
    prot[2].atoms.append(At("NH1", "N", (9.0, 6.0, 3.5)))
    lig = ligand_res("A", "LIG", 900, np.array([9.0, 9.5, 3.5]) + PLUS)  # 3.5 A from NH1, >4 A from rest
    p = write(build(prot + [lig]), tmp_path / "q.cif")
    assert build_ground_truth(p, rules="p2rank").ligands[0].ligand_id == "LIG|A:900"
    assert build_ground_truth(p, rules="p2rank_strict").other_ligands[0].category == "distant"
    # (b) C-terminal modified residue (PTR) bonded to the chain: protein in p2rank, ligand in strict
    prot = protein()
    prot[9].name, prot[9].het = "PTR", True
    p = write(build(prot), tmp_path / "ptr.cif")
    assert build_ground_truth(p, rules="p2rank").ligands == []
    assert [l.ligand_id for l in build_ground_truth(p, rules="p2rank_strict").ligands] == ["PTR|A:10"]
    # (c) ligand altlocs: atom present only in conformer B is kept by strict ("by_name")
    atoms = [At(f"C{i}", "C", tuple(C0 + d)) for i, d in enumerate(PLUS)]
    atoms[1] = At("C1", "C", tuple(C0 + PLUS[1]), altloc="A", occ=0.5)
    atoms.append(At("C1", "C", tuple(C0 + [1, 0, 1.0]), altloc="B", occ=0.5))
    atoms.append(At("C9", "C", tuple(C0 + [0, 0, 3.0]), altloc="B", occ=0.5))
    p = write(build(protein() + [R("A", "LIG", 900, atoms, het=True)]), tmp_path / "alt.cif")
    assert build_ground_truth(p, rules="p2rank").ligands[0].n_atoms == 5
    assert build_ground_truth(p, rules="p2rank_strict").ligands[0].n_atoms == 6
