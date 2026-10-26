"""Synthetic structure builders (gemmi) with known geometry for harness tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np
import pytest

from pocketdetect.ligands import GroundTruth, Ligand
from pocketdetect.structure import PDB_MIRROR


@dataclass
class A:  # atom spec
    name: str
    element: str
    xyz: tuple
    altloc: str = ""
    occ: float = 1.0
    b: float = 30.0


@dataclass
class R:  # residue spec
    chain: str
    name: str
    num: int
    atoms: list
    het: bool = False
    icode: str = ""


def backbone(chain: str, start: int, n: int, origin=(0.0, 0.0, 0.0), name="ALA", icodes=None, step=(3.8, 0, 0)) -> list[R]:
    """Linear peptide: N, CA, C, O, CB per residue; C(i)-N(i+1) = 1.3 A (peptide-bonded)."""
    o = np.asarray(origin, float)
    st = np.asarray(step, float)
    u = st / np.linalg.norm(st)
    perp = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.9 else np.cross(u, [1.0, 0, 0])
    perp /= np.linalg.norm(perp)
    out = []
    for i in range(n):
        b = o + i * st
        atoms = [
            A("N", "N", tuple(b)),
            A("CA", "C", tuple(b + 1.2 * u + 0.8 * perp)),
            A("C", "C", tuple(b + 2.5 * u)),
            A("O", "O", tuple(b + 2.5 * u - 1.2 * perp)),
            A("CB", "C", tuple(b + 1.2 * u + 2.3 * perp)),
        ]
        ic = icodes[i] if icodes else ""
        out.append(R(chain, name, start + i, atoms, icode=ic))
    return out


def ligand_res(chain: str, name: str, num: int, coords, elements=None, het=True) -> R:
    coords = np.asarray(coords, float)
    elements = elements or ["C"] * len(coords)
    atoms = [A(f"{e}{i+1}", e, tuple(c)) for i, (c, e) in enumerate(zip(coords, elements))]
    return R(chain, name, num, atoms, het=het)


def build(residues: list[R], models: int = 1, model_shift=(0.0, 0.0, 0.0)) -> gemmi.Structure:
    st = gemmi.Structure()
    st.name = "TOY"
    for mi in range(models):
        m = gemmi.Model(mi + 1)
        shift = np.asarray(model_shift) * mi
        chains: dict[str, gemmi.Chain] = {}
        order: list[str] = []
        for r in residues:
            if r.chain not in chains:
                chains[r.chain] = gemmi.Chain(r.chain)
                order.append(r.chain)
            res = gemmi.Residue()
            res.name = r.name
            res.seqid = gemmi.SeqId(r.num, r.icode or " ")
            res.het_flag = "H" if r.het else "A"
            for a in r.atoms:
                at = gemmi.Atom()
                at.name = a.name
                at.element = gemmi.Element(a.element)
                p = np.asarray(a.xyz, float) + shift
                at.pos = gemmi.Position(*p)
                at.occ = a.occ
                at.b_iso = a.b
                if a.altloc:
                    at.altloc = a.altloc
                res.add_atom(at)
            chains[r.chain].add_residue(res)
        for c in order:
            m.add_chain(chains[c])
        st.add_model(m)
    st.setup_entities()
    return st


def write(st: gemmi.Structure, path: Path) -> Path:
    path = Path(path)
    if path.suffix == ".pdb":
        st.write_pdb(str(path))
    else:
        st.make_mmcif_document().write_file(str(path))
    return path


def make_ligand(coords, ligand_id="LIG|A:1", residues=(), elements=None, center=None) -> Ligand:
    coords = np.asarray(coords, float).reshape(-1, 3)
    return Ligand(
        ligand_id=ligand_id,
        name=ligand_id.split("|")[0],
        groups=(ligand_id.split("|")[1],),
        category="relevant",
        coords=coords,
        elements=np.array(elements or ["C"] * len(coords)),
        center=coords.mean(0) if center is None else np.asarray(center, float),
        contact_dist=3.5,
        center_dist=4.0,
        binding_residues=tuple(residues),
    )


def make_gt(sid: str, ligands: list[Ligand], protein_residues=None) -> GroundTruth:
    pr = protein_residues if protein_residues is not None else [f"A:{i}" for i in range(1, 101)]
    return GroundTruth(sid, "test", ligands, [], np.array(pr, dtype=object), 1000)


PLUS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)  # planar 5-atom ligand


@pytest.fixture
def mirror_available():
    p = PDB_MIRROR / "structures" / "divided" / "mmCIF"
    if not p.exists():
        pytest.skip("PDB mirror not mounted")
    return PDB_MIRROR
