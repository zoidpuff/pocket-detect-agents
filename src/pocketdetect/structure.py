"""Fast structure loading (PDB / mmCIF, optionally gzipped) into a compact struct-of-arrays.

Conventions (see docs/EVALUATION.md, section "Structure loading"):

* Only the **first model** is read.
* Alternate locations: ``altloc="first"`` (default; matches BioJava/P2Rank and gemmi's
  ``remove_alternative_conformations``) keeps, per residue position (chain, seqnum, icode), the
  first altloc letter that appears in the file; ``altloc="occupancy"`` keeps the altloc with the
  highest summed occupancy (ties -> first). Atoms without altloc are always kept.
* Chain IDs are *author* chain IDs (``auth_asym_id``); residue numbers are *author* sequence
  numbers plus insertion code. Residue IDs are written ``chain:resnum[icode]`` (e.g. ``A:52A``).
* Residue classes (``Structure.r_kind``):

  - ``PROTEIN``: amino-acid-like residue (tabulated amino acid in gemmi's CCD subset, e.g. MSE,
    SEP, TPO, ..., or having N, CA and C atoms) whose peptide-bonded fragment (consecutive
    residues of a chain linked by C(i)-N(i+1) <= 2.0 A) contains at least one ATOM-record
    residue. Hence modified residues inside a chain are protein, while a free amino acid or an
    all-HETATM peptidic inhibitor is a *ligand* (matches P2Rank on e.g. 1nlu).
  - ``WATER``: HOH/DOD/WAT/H2O.
  - ``LIGAND``: any other residue that is HETATM or belongs to a non-polymer / branched entity
    (candidate ligand; filtered later by ``pocketdetect.ligands`` rules).
  - ``OTHER_POLYMER``: remaining polymer residues (nucleic acids, ...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import gemmi
import numpy as np

__all__ = [
    "Structure",
    "load_structure",
    "load_assembly",
    "pdb_entry_path",
    "pdb_assembly_path",
    "residue_id",
    "parse_residue_id",
    "PROTEIN",
    "LIGAND",
    "WATER",
    "OTHER_POLYMER",
    "PDB_MIRROR",
]

PROTEIN, LIGAND, WATER, OTHER_POLYMER = 0, 1, 2, 3
KIND_NAMES = {PROTEIN: "protein", LIGAND: "ligand", WATER: "water", OTHER_POLYMER: "other_polymer"}

PDB_MIRROR = Path(os.environ.get("POCKETDETECT_PDB_MIRROR", "/cluster/work/beltrao/shared/rsyncPDB/data"))

WATER_NAMES = frozenset({"HOH", "DOD", "WAT", "H2O", "OH2", "TIP", "TIP3", "SOL"})
PEPTIDE_BOND_MAX = 2.0  # A; C(i)-N(i+1) distance to call two residues peptide-bonded


def residue_id(chain: str, resnum: int, icode: str = "") -> str:
    """Canonical residue identifier ``chain:resnum[icode]``."""
    icode = "" if icode in ("\x00", " ", "?", ".") else icode
    return f"{chain}:{int(resnum)}{icode}"


def parse_residue_id(rid: str) -> tuple[str, int, str]:
    """Inverse of :func:`residue_id`. Chain may itself contain ``-`` (assembly copies)."""
    chain, _, rest = rid.rpartition(":")
    if not chain and not rest:
        raise ValueError(f"bad residue id {rid!r}")
    i = len(rest)
    while i > 0 and not (rest[i - 1].isdigit()):
        i -= 1
    num, icode = rest[:i], rest[i:]
    if not num or not (num.lstrip("-").isdigit()):
        raise ValueError(f"bad residue id {rid!r}")
    return chain, int(num), icode


ATOM_FIELDS = (
    "coords", "element", "atom_name", "altloc", "occupancy", "b_factor", "serial", "res_index",
    "chain", "res_name", "res_num", "ins_code", "is_hetero",
)
RES_FIELDS = ("r_chain", "r_name", "r_num", "r_icode", "r_kind", "r_hetero", "r_entity")


@dataclass
class Structure:
    """Struct-of-arrays representation of one model.

    Per-atom arrays (length ``n_atoms``): see ``ATOM_FIELDS``. Per-residue arrays (prefix ``r_``,
    length ``n_residues``): see ``RES_FIELDS`` plus ``r_start``/``r_end`` atom-index bounds
    (atoms of one residue are contiguous). ``res_index`` maps atoms to residues."""

    id: str
    coords: np.ndarray  # (N,3) float32
    element: np.ndarray  # (N,) str, upper case ("C", "SE", "H")
    atom_name: np.ndarray  # (N,) str
    altloc: np.ndarray  # (N,) str ('' if none)
    occupancy: np.ndarray  # (N,) float32
    b_factor: np.ndarray  # (N,) float32 (pLDDT for AF models)
    serial: np.ndarray  # (N,) int32
    res_index: np.ndarray  # (N,) int32 index into residue arrays
    chain: np.ndarray  # (N,) author chain id
    res_name: np.ndarray  # (N,)
    res_num: np.ndarray  # (N,) int32 author seq number
    ins_code: np.ndarray  # (N,) str ('' if none)
    is_hetero: np.ndarray  # (N,) bool: HETATM record
    r_chain: np.ndarray
    r_name: np.ndarray
    r_num: np.ndarray
    r_icode: np.ndarray
    r_kind: np.ndarray  # int8, PROTEIN/LIGAND/WATER/OTHER_POLYMER
    r_hetero: np.ndarray  # bool
    r_entity: np.ndarray  # str: gemmi entity type name
    r_start: np.ndarray | None = None  # int32
    r_end: np.ndarray | None = None  # int32 (exclusive)
    path: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.r_start is None or self.r_end is None:
            counts = np.bincount(self.res_index, minlength=len(self.r_chain)) if len(self.res_index) else np.zeros(len(self.r_chain), int)
            self.r_end = np.cumsum(counts).astype(np.int32)
            self.r_start = (self.r_end - counts).astype(np.int32)

    # ------------------------------------------------------------------ basic properties
    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_residues(self) -> int:
        return int(len(self.r_chain))

    @property
    def is_hydrogen(self) -> np.ndarray:
        return (self.element == "H") | (self.element == "D")

    @property
    def kind(self) -> np.ndarray:
        """Per-atom residue class."""
        return self.r_kind[self.res_index]

    @property
    def is_protein(self) -> np.ndarray:
        return self.kind == PROTEIN

    @property
    def is_water(self) -> np.ndarray:
        return self.kind == WATER

    @property
    def is_ligand_candidate(self) -> np.ndarray:
        return self.kind == LIGAND

    @property
    def residue_ids(self) -> np.ndarray:
        """Per-residue canonical IDs ``chain:resnum[icode]``."""
        return np.array(
            [residue_id(c, n, i) for c, n, i in zip(self.r_chain, self.r_num, self.r_icode)],
            dtype=object,
        )

    @property
    def atom_residue_ids(self) -> np.ndarray:
        return self.residue_ids[self.res_index]

    @property
    def chains(self) -> list[str]:
        return list(dict.fromkeys(self.r_chain.tolist()))

    def protein_heavy_mask(self) -> np.ndarray:
        return self.is_protein & ~self.is_hydrogen

    # ------------------------------------------------------------------ selection
    def select(self, mask: np.ndarray) -> "Structure":
        """Return a new Structure with only atoms where ``mask`` is True (residue table rebuilt;
        atom order preserved)."""
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        kept_res, new_res_index = np.unique(self.res_index[idx], return_inverse=True)
        kw = {f: getattr(self, f)[idx] for f in ATOM_FIELDS}
        kw["res_index"] = new_res_index.astype(np.int32).reshape(-1)
        kw.update({f: getattr(self, f)[kept_res] for f in RES_FIELDS})
        return Structure(id=self.id, path=self.path, meta=dict(self.meta), **kw)

    def select_chains(self, chains: Iterable[str]) -> "Structure":
        return self.select(np.isin(self.chain, list(chains)))


# ---------------------------------------------------------------------------- loading
_AA_CACHE: dict[str, bool] = {}


def _is_tabulated_aa(name: str) -> bool:
    v = _AA_CACHE.get(name)
    if v is None:
        info = gemmi.find_tabulated_residue(name)
        v = bool(info is not None and info.is_amino_acid())
        _AA_CACHE[name] = v
    return v


def _read_gemmi(path: str | os.PathLike) -> gemmi.Structure:
    st = gemmi.read_structure(str(path), merge_chain_parts=True)
    st.setup_entities()
    return st


def _strs(values: list, dtype=None) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=dtype or "<U1")
    return np.array(values, dtype=dtype) if dtype else np.array(values).astype(str)


def load_structure(
    path: str | os.PathLike,
    *,
    structure_id: str | None = None,
    chains: Sequence[str] | None = None,
    protein_only: bool = False,
    remove_water: bool = False,
    remove_hetero: bool = False,
    remove_hydrogens: bool = True,
    altloc: str = "first",
) -> Structure:
    """Load a PDB/mmCIF file (``.gz`` ok) into a :class:`Structure`.

    Parameters
    ----------
    chains: keep only these author chain IDs (``None`` = all chains).
    protein_only: keep only ``PROTEIN`` residues (incl. modified residues such as MSE).
    remove_water: drop waters.
    remove_hetero: drop everything that is not protein (ligands, waters, nucleic acids).
    remove_hydrogens: drop H/D atoms (default True).
    altloc: ``"first"`` or ``"occupancy"`` (see module docstring).
    """
    if altloc not in ("first", "occupancy"):
        raise ValueError("altloc must be 'first' or 'occupancy'")
    path = str(path)
    st = _read_gemmi(path)
    if len(st) == 0:
        raise ValueError(f"no models in {path}")
    model = st[0]
    sid = structure_id or _default_id(path)

    xyz: list[tuple[float, float, float]] = []
    elem: list[str] = []
    aname: list[str] = []
    alt: list[str] = []
    occ: list[float] = []
    bfac: list[float] = []
    serial: list[int] = []
    rindex: list[int] = []
    rc, rn, rnum, ric, rhet, rent, rhint = [], [], [], [], [], [], []
    rbb: list[tuple] = []  # (N xyz or None, C xyz or None)

    for ch in model:
        cname = ch.name
        for res in ch:
            n0 = len(xyz)
            n_xyz = c_xyz = None
            has_ca = False
            for a in res:
                p = a.pos
                xyz.append((p.x, p.y, p.z))
                elem.append(a.element.name.upper())
                aname.append(a.name)
                alt.append(a.altloc if a.altloc != "\x00" else "")
                occ.append(a.occ)
                bfac.append(a.b_iso)
                serial.append(a.serial)
                if a.name == "N" and n_xyz is None:
                    n_xyz = (p.x, p.y, p.z)
                elif a.name == "C" and c_xyz is None:
                    c_xyz = (p.x, p.y, p.z)
                elif a.name == "CA":
                    has_ca = True
            if len(xyz) == n0:
                continue
            rindex.extend([len(rc)] * (len(xyz) - n0))
            rc.append(cname)
            rn.append(res.name)
            rnum.append(res.seqid.num)
            ic = res.seqid.icode
            ric.append("" if ic in ("\x00", " ") else ic)
            rhet.append(res.het_flag == "H")
            rent.append(res.entity_type.name)
            name = res.name
            if res.is_water() or name in WATER_NAMES:
                hint = WATER
            elif _is_tabulated_aa(name) or (n_xyz is not None and c_xyz is not None and has_ca):
                hint = -1  # amino-acid-like, decided below
            elif res.entity_type == gemmi.EntityType.Polymer and res.het_flag != "H":
                hint = OTHER_POLYMER
            else:
                hint = LIGAND
            rhint.append(hint)
            rbb.append((n_xyz, c_xyz))

    n_res = len(rc)
    r_kind = np.array(rhint, dtype=np.int8)
    r_entity = _strs(rent)
    # Amino-acid-like residues: split into peptide-bonded fragments (consecutive residues of one
    # chain with C(i)-N(i+1) <= PEPTIDE_BOND_MAX). A fragment is protein iff it contains at least
    # one ATOM-record residue; all-HETATM fragments (free amino acids, peptidic inhibitors such as
    # IVA-PHI-TYB in 1nlu) are ligands. Modified residues (MSE, ...) inside a chain are protein.
    aa_like = r_kind == -1
    frag = np.arange(n_res)
    for i in range(n_res - 1):
        if rc[i] != rc[i + 1] or not (aa_like[i] and aa_like[i + 1]):
            continue
        c_prev, n_next = rbb[i][1], rbb[i + 1][0]
        if c_prev is None or n_next is None:
            continue
        d2 = sum((c_prev[k] - n_next[k]) ** 2 for k in range(3))
        if d2 <= PEPTIDE_BOND_MAX**2:
            frag[i + 1] = frag[i]
    atom_record = aa_like & ~np.asarray(rhet, dtype=bool)
    frag_has_atom = np.zeros(n_res, dtype=bool)
    np.logical_or.at(frag_has_atom, frag, atom_record)
    is_prot = aa_like & frag_has_atom[frag]
    r_kind[is_prot] = PROTEIN
    r_kind[aa_like & ~is_prot] = LIGAND

    res_index = np.asarray(rindex, dtype=np.int32)
    r_chain = _strs(rc)
    r_name = _strs(rn)
    r_num = np.asarray(rnum, dtype=np.int32)
    r_icode = _strs(ric, "<U1")
    r_hetero = np.asarray(rhet, dtype=bool)
    s = Structure(
        id=sid,
        coords=np.asarray(xyz, dtype=np.float32).reshape(-1, 3),
        element=_strs(elem, "<U2"),
        atom_name=_strs(aname, "<U4"),
        altloc=_strs(alt, "<U1"),
        occupancy=np.asarray(occ, dtype=np.float32),
        b_factor=np.asarray(bfac, dtype=np.float32),
        serial=np.asarray(serial, dtype=np.int32),
        res_index=res_index,
        chain=r_chain[res_index],
        res_name=r_name[res_index],
        res_num=r_num[res_index],
        ins_code=r_icode[res_index],
        is_hetero=r_hetero[res_index],
        r_chain=r_chain,
        r_name=r_name,
        r_num=r_num,
        r_icode=r_icode,
        r_kind=r_kind,
        r_hetero=r_hetero,
        r_entity=r_entity,
        path=path,
        meta={
            "resolution": float(st.resolution) if st.resolution else float("nan"),
            "n_models": len(st),
            "method": st.info["_exptl.method"] if "_exptl.method" in st.info else "",
        },
    )

    mask = _altloc_mask(s, altloc)
    if chains is not None:
        mask &= np.isin(s.chain, list(chains))
    if remove_hydrogens:
        mask &= ~s.is_hydrogen
    kind = s.kind
    if protein_only or remove_hetero:
        mask &= kind == PROTEIN
    if remove_water:
        mask &= kind != WATER
    if mask.all():
        return s
    return s.select(mask)


def _altloc_mask(s: Structure, mode: str) -> np.ndarray:
    has_alt = s.altloc != ""
    mask = np.ones(s.n_atoms, dtype=bool)
    if not has_alt.any():
        return mask
    idx = np.flatnonzero(has_alt)
    # residue *position* key (microheterogeneity = two residue names at one position)
    keys = [(s.chain[i], int(s.res_num[i]), s.ins_code[i]) for i in idx]
    chosen: dict[tuple, str] = {}
    if mode == "first":
        for k, i in zip(keys, idx):
            chosen.setdefault(k, s.altloc[i])
    else:
        tot: dict[tuple, dict[str, float]] = {}
        order: dict[tuple, list[str]] = {}
        for k, i in zip(keys, idx):
            a = s.altloc[i]
            d = tot.setdefault(k, {})
            if a not in d:
                order.setdefault(k, []).append(a)
                d[a] = 0.0
            d[a] += float(s.occupancy[i])
        for k, d in tot.items():
            best = max(order[k], key=lambda a: (d[a], -order[k].index(a)))
            chosen[k] = best
    keep = np.array([s.altloc[i] == chosen[k] for k, i in zip(keys, idx)], dtype=bool)
    mask[idx] = keep
    return mask


def _default_id(path: str) -> str:
    name = Path(path).name
    for ext in (".gz", ".cif", ".pdb", ".ent", ".mmcif"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
    return name


# ---------------------------------------------------------------------------- PDB mirror
def pdb_entry_path(pdb_id: str, mirror: Path = PDB_MIRROR) -> Path:
    """Path of the asymmetric-unit mmCIF in the local wwPDB mirror (divided layout)."""
    pid = pdb_id.lower()
    return Path(mirror) / "structures" / "divided" / "mmCIF" / pid[1:3] / f"{pid}.cif.gz"


def pdb_assembly_path(pdb_id: str, assembly_id: int | str = 1, mirror: Path = PDB_MIRROR) -> Path:
    """Path of a biological-assembly mmCIF in the mirror:
    ``assemblies/mmCIF/divided/<id[1:3]>/<id>-assembly<N>.cif.gz``. Symmetry copies have chain
    names suffixed ``-<n>`` (e.g. ``A-2``), all in model 1."""
    pid = pdb_id.lower()
    return Path(mirror) / "assemblies" / "mmCIF" / "divided" / pid[1:3] / f"{pid}-assembly{assembly_id}.cif.gz"


def load_assembly(pdb_id: str, assembly_id: int | str = 1, mirror: Path = PDB_MIRROR, **kwargs) -> Structure:
    """Load a biological assembly from the PDB mirror (pre-generated wwPDB assembly files)."""
    p = pdb_assembly_path(pdb_id, assembly_id, mirror)
    if not p.exists():
        raise FileNotFoundError(f"assembly file not found: {p}")
    kwargs.setdefault("structure_id", f"{pdb_id.lower()}-assembly{assembly_id}")
    return load_structure(p, **kwargs)
