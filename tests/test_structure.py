import gzip
import shutil

import numpy as np
import pytest
from conftest import A, R, backbone, build, ligand_res, write

from pocketdetect.structure import (
    LIGAND,
    OTHER_POLYMER,
    PROTEIN,
    WATER,
    load_assembly,
    load_structure,
    parse_residue_id,
    pdb_assembly_path,
    residue_id,
)


def toy_residues():
    res = backbone("A", 1, 3)
    # residue 2 -> MSE as HETATM (modified residue inside the chain)
    res[1].name, res[1].het = "MSE", True
    res[1].atoms.append(A("SE", "SE", (5.0, 3.0, 0.0)))
    res += backbone("B", 10, 2, origin=(0, 10, 0), icodes=["", "A"])
    res.append(ligand_res("A", "ATP", 501, [[0, 0, 8], [1, 0, 8], [2, 0, 8], [3, 0, 8], [4, 0, 8]]))
    res.append(ligand_res("A", "HOH", 601, [[20, 20, 20]], ["O"]))
    # free amino acid ligand (HETATM, not bonded) -> ligand
    res.append(R("A", "GLU", 700, [A("N", "N", (40, 0, 0)), A("CA", "C", (41.2, 0, 0)), A("C", "C", (42.5, 0, 0))], het=True))
    # hydrogen on residue A:1
    res[0].atoms.append(A("H", "H", (-0.5, -0.8, 0)))
    return res


@pytest.mark.parametrize("ext", [".pdb", ".cif", ".pdb.gz", ".cif.gz"])
def test_roundtrip_formats(tmp_path, ext):
    st = build(toy_residues())
    base = ext.replace(".gz", "")
    p = write(st, tmp_path / f"toy{base}")
    if ext.endswith(".gz"):
        with open(p, "rb") as fi, gzip.open(str(p) + ".gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        p = tmp_path / f"toy{ext}"
    s = load_structure(p, remove_hydrogens=False)
    assert s.id == "toy"
    assert s.chains == ["A", "B"]
    ids = list(s.residue_ids)
    assert "B:11A" in ids and "A:2" in ids
    kinds = dict(zip(ids, s.r_kind))
    assert kinds["A:1"] == PROTEIN and kinds["A:2"] == PROTEIN  # MSE in-chain
    assert kinds["A:501"] == LIGAND
    assert kinds["A:601"] == WATER
    assert kinds["A:700"] == LIGAND  # free GLU
    assert kinds["B:11A"] == PROTEIN
    se = np.flatnonzero(s.element == "SE")
    assert len(se) == 1 and np.allclose(s.coords[se[0]], [5.0, 3.0, 0.0])
    assert s.coords.dtype == np.float32
    assert s.is_hetero[se[0]] and not s.is_hetero[0]
    assert np.isclose(s.b_factor[0], 30.0)
    assert s.is_hydrogen.sum() == 1
    # residue <-> atom tables consistent
    for r in range(s.n_residues):
        assert np.all(s.res_index[s.r_start[r] : s.r_end[r]] == r)


def test_filters(tmp_path):
    p = write(build(toy_residues()), tmp_path / "toy.cif")
    s = load_structure(p)
    assert s.is_hydrogen.sum() == 0
    assert set(load_structure(p, protein_only=True).r_kind) == {PROTEIN}
    assert WATER not in set(load_structure(p, remove_water=True).r_kind)
    assert set(load_structure(p, remove_hetero=True).r_name) == {"ALA", "MSE"}
    sb = load_structure(p, chains=["B"])
    assert sb.chains == ["B"] and sb.n_residues == 2
    assert load_structure(p).select_chains(["B"]).n_atoms == sb.n_atoms


def test_first_model_only(tmp_path):
    st = build(backbone("A", 1, 2), models=2, model_shift=(100, 0, 0))
    s = load_structure(write(st, tmp_path / "m.pdb"))
    assert s.meta["n_models"] == 2
    assert s.n_atoms == 10 and s.coords[:, 0].max() < 50


def test_altloc(tmp_path):
    res = backbone("A", 1, 2)
    cb = res[0].atoms.pop()  # replace CB with two altlocs
    res[0].atoms.append(A("CB", "C", (1, 1, 1), altloc="A", occ=0.4))
    res[0].atoms.append(A("CB", "C", (2, 2, 2), altloc="B", occ=0.6))
    for ext in (".pdb", ".cif"):
        p = write(build(res), tmp_path / f"alt{ext}")
        s1 = load_structure(p)
        cbs = np.flatnonzero((s1.atom_name == "CB") & (s1.res_num == 1))
        assert len(cbs) == 1 and np.allclose(s1.coords[cbs[0]], [1, 1, 1])
        s2 = load_structure(p, altloc="occupancy")
        cbs = np.flatnonzero((s2.atom_name == "CB") & (s2.res_num == 1))
        assert len(cbs) == 1 and np.allclose(s2.coords[cbs[0]], [2, 2, 2])
        assert s1.n_atoms == 10


def test_residue_id_roundtrip():
    for rid in ["A:12", "A:52A", "B-2:-3", "AA:1000", "1-60:5B"]:
        c, n, i = parse_residue_id(rid)
        assert residue_id(c, n, i) == rid
    with pytest.raises(ValueError):
        parse_residue_id("A:")


def test_assembly_from_mirror(mirror_available):
    p = pdb_assembly_path("1m17", 1)
    if not p.exists():
        pytest.skip("assembly file missing")
    s = load_assembly("1m17", 1)
    assert s.id == "1m17-assembly1"
    assert "AQ4" in set(s.r_name)
    # a symmetric homo-oligomer assembly: copies get "-n" suffixed chain names
    p2 = pdb_assembly_path("1m11", 1)
    if p2.exists():
        s2 = load_assembly("1m11", 1)
        assert any("-" in c for c in s2.chains)
