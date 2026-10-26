"""Ligand extraction on real PDB entries from the local mirror (skipped if not mounted)."""

import pytest

from pocketdetect.ligands import build_ground_truth
from pocketdetect.structure import load_structure, pdb_entry_path


def test_egfr_erlotinib_1m17(mirror_available):
    """1M17: EGFR kinase domain with erlotinib (AQ4, 29 heavy atoms) bound at the hinge (Met769)."""
    gt = build_ground_truth(pdb_entry_path("1m17"), structure_id="1m17")
    assert [l.ligand_id for l in gt.ligands] == ["AQ4|A:999"]
    l = gt.ligands[0]
    assert l.n_atoms == 29
    assert l.contact_dist < 4.0
    for r in ("A:769", "A:721", "A:694"):  # hinge Met769, Lys721, Leu694
        assert r in l.binding_residues
    assert gt.other_ligands == []  # only waters otherwise


def test_a2a_zm241385_3eml(mirror_available):
    """3EML: A2A receptor-T4L with ZM241385 (ZMA); stearic acids (STE) relevant, SO4 ignored."""
    p = pdb_entry_path("3eml")
    if not p.exists():
        pytest.skip("3eml not in mirror")
    gt = build_ground_truth(p, structure_id="3eml")
    names = [l.name for l in gt.ligands]
    assert "ZMA" in names and "SO4" not in names
    assert {l.category for l in gt.other_ligands if l.name == "SO4"} == {"ignored"}


def test_gtp_mg_2w83_explicit_spec(mirror_available):
    """Mirrors P2Rank's own test (test_data/specified-ligands-2.ds): 2W83 'GTP,MG' -> 6 ligands,
    'GTP[B_1173]' -> 1, 'GTP' -> 3."""
    p = pdb_entry_path("2w83")
    if not p.exists():
        pytest.skip("2w83 not in mirror")
    assert build_ground_truth(p, spec="GTP,MG").n_ligands == 6
    assert build_ground_truth(p, spec="GTP").n_ligands == 3
    assert build_ground_truth(p, spec="GTP[B_1173]").n_ligands == 1
    assert build_ground_truth(p, spec="GTP[B_1173],MG").n_ligands == 4
    # default rules: MG is "small", GTP relevant
    gt = build_ground_truth(p)
    assert sum(l.name == "GTP" for l in gt.ligands) == 3
    assert all(l.category == "small" for l in gt.other_ligands if l.name == "MG")


def test_selenomethionine_is_protein(mirror_available):
    """An MSE-containing entry: MSE residues must be protein, never ligand candidates."""
    p = pdb_entry_path("1hu3")
    s = load_structure(p)
    if "MSE" not in set(s.r_name):
        pytest.skip("1hu3 has no MSE")
    gt = build_ground_truth(p)
    assert all("MSE" not in l.name for l in gt.ligands + gt.other_ligands)
