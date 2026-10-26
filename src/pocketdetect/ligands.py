"""Ground-truth ligand extraction under named, configurable rule sets.

The default rule set ``p2rank`` re-implements the ligand categorisation of P2Rank
(rdk/p2rank, ``cz.siret.prank.domain.Ligands``; parameters from ``Params.groovy``), verified
against the source at commit b9b54b7e49513b665c913cc90fbfa68ec4133521 (develop, 2026-10):

1. Candidate groups: every non-water residue that is not a protein residue and is either a HETATM
   record or belongs to a non-polymer entity (P2Rank ``Struct.getLigandGroups``). Modified amino
   acids inside protein chains (MSE, SEP, ...) are protein, not ligands. Groups named like a
   *standard* amino acid or nucleotide (ALA..VAL, A/C/G/U/DA/DC/DG/DT/...) are never candidates:
   P2Rank (BioJava, reduced chemical-component provider) types them AMINOACID/NUCLEOTIDE and
   ``isLigandCandidateGroup`` requires GroupType HETATM, so free amino acids (e.g. TRP in 1zt9) and
   the standard residues of peptidic ligands are not ligands in P2Rank's ground truth
   (verified empirically on COACH420; option ``standard_residue_ligands``).
2. Groups whose residue name is in ``ignore_het_groups`` =
   [HOH, DOD, WAT, NAG, MAN, UNK, GLC, ABA, MPD, GOL, SO4, PO4] are *ignored* (before clustering).
3. Remaining groups are merged by single-linkage clustering of their atoms with cutoff
   ``ligand_clustering_distance`` = 1.7 A (covalently connected multi-residue ligands become one).
4. Each merged ligand is categorised, in this order:
   * *small*   if atom count < ``min_ligand_atoms`` = 5,
   * *distant* if min distance ligand atom -> protein heavy atom > ``ligand_protein_contact_distance`` = 4 A,
   * *distant* if distance ligand centre (mass-weighted) -> nearest protein heavy atom > ``ligc_prot_dist`` = 5.5 A,
   * otherwise *relevant*. Only relevant ligands are evaluation targets.
5. If an explicit ligand specification is given (P2Rank dataset ``ligands`` column syntax:
   ``NAME``, ``NAME[C_123A]``, ``NAME[group_id:C_123A]``, ``NAME[atom_id:1234]``, comma separated),
   matching groups are clustered as above and are **all** relevant (no size/distance filter, as in
   P2Rank); all other candidate groups become *ignored*.

Difference to the P2Rank code (documented in docs/EVALUATION.md): P2Rank keeps hydrogen atoms in
ligand atom sets (atom count, DCA, centre of mass). Rule set ``p2rank`` uses heavy atoms only (as
the P2Rank documentation "min. heavy atom count" states); ``p2rank_strict`` keeps hydrogens to
reproduce the code exactly. For X-ray files without H the two are identical.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Sequence

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from .structure import LIGAND, PROTEIN, Structure, load_structure, residue_id

__all__ = [
    "LigandRules",
    "Ligand",
    "GroundTruth",
    "RULE_SETS",
    "get_rules",
    "register_rules",
    "parse_ligand_spec",
    "extract_ligands",
    "build_ground_truth",
    "binding_residues",
]

P2RANK_SOURCE = (
    "rdk/p2rank@b9b54b7e (develop): Params.groovy defaults ignore_het_groups, min_ligand_atoms=5, "
    "ligand_protein_contact_distance=4, ligc_prot_dist=5.5, ligand_clustering_distance=1.7; "
    "Ligands.groovy categorizeLigand(); Struct.getLigandGroups()"
)


@dataclass(frozen=True)
class LigandRules:
    """A named ligand-relevance convention. All distances in Angstrom."""

    name: str
    source: str
    ignore_het_groups: frozenset = frozenset()
    min_atoms: int = 5
    ligand_hydrogens: bool = False  # include H/D atoms in ligand atom sets
    contact_dist: float = 4.0  # max (min ligand atom -> protein heavy atom distance)
    center_dist: float = 5.5  # max (ligand centre -> nearest protein heavy atom)
    cluster_dist: float = 1.7  # single-linkage merge cutoff between candidate groups
    center_method: str = "com"  # "com" (mass-weighted, P2Rank) or "centroid"
    binding_residue_dist: float = 4.0  # protein residue within this of a ligand atom = binding residue
    explicit_bypasses_filters: bool = True  # P2Rank: explicitly specified ligands are all relevant
    standard_residue_ligands: bool = False  # False = P2Rank: standard aa/nucleotide names never ligands

    def as_dict(self) -> dict:
        """Deterministic (hash-seed independent) description, e.g. for cache keys and run logs."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["ignore_het_groups"] = sorted(self.ignore_het_groups)
        return d


STANDARD_AA = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
STANDARD_NUC = frozenset("A C G U I N DA DC DG DT DU DI DN".split())

P2RANK_IGNORED = frozenset(["HOH", "DOD", "WAT", "NAG", "MAN", "UNK", "GLC", "ABA", "MPD", "GOL", "SO4", "PO4"])

RULE_SETS: dict[str, LigandRules] = {}


def register_rules(rules: LigandRules) -> LigandRules:
    RULE_SETS[rules.name] = rules
    return rules


def get_rules(name: str | LigandRules) -> LigandRules:
    if isinstance(name, LigandRules):
        return name
    try:
        return RULE_SETS[name]
    except KeyError:
        raise KeyError(f"unknown ligand rule set {name!r}; available: {sorted(RULE_SETS)}") from None


register_rules(LigandRules(name="p2rank", source=P2RANK_SOURCE + "; heavy atoms only", ignore_het_groups=P2RANK_IGNORED))
register_rules(
    LigandRules(
        name="p2rank_strict",
        source=P2RANK_SOURCE + "; hydrogens kept in ligand atom sets exactly as the P2Rank code does",
        ignore_het_groups=P2RANK_IGNORED,
        ligand_hydrogens=True,
    )
)
register_rules(
    LigandRules(
        name="all_contacting",
        source="project convention: every non-water hetero group (any size) within 4 A of protein; "
        "used for 'any ligand' sanity analyses, not for headline metrics",
        ignore_het_groups=frozenset(["HOH", "DOD", "WAT"]),
        min_atoms=1,
        center_dist=math.inf,
        standard_residue_ligands=True,
    )
)


# --------------------------------------------------------------------------- data classes
@dataclass
class Ligand:
    ligand_id: str  # "<names>|<group ids>", e.g. "AQ4|A:999" or "BMA&FUC|A:501&A:502"
    name: str  # "&"-joined sorted unique residue names
    groups: tuple  # residue ids of member groups
    category: str  # relevant / small / distant / ignored
    coords: np.ndarray  # (k,3) float64 ligand atoms (heavy unless rules.ligand_hydrogens)
    elements: np.ndarray
    center: np.ndarray  # (3,)
    contact_dist: float
    center_dist: float
    binding_residues: tuple = ()

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])

    @property
    def chains(self) -> str:
        return "&".join(sorted({g.rpartition(":")[0] for g in self.groups}))


@dataclass
class GroundTruth:
    structure_id: str
    rules: str
    ligands: list  # relevant ligands = evaluation targets
    other_ligands: list  # small / distant / ignored (reported, not evaluated)
    protein_residues: np.ndarray  # all protein residue ids (for residue-level TN counts)
    n_protein_atoms: int
    path: str = ""
    spec: str = ""
    chains: tuple | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_ligands(self) -> int:
        return len(self.ligands)


# --------------------------------------------------------------------------- spec parsing
_SPEC_RE = re.compile(r"^([A-Za-z0-9]{1,5})(?:\[(?:(group_id|atom_id|contact_res_ids):)?([^\]]+)\])?$")


@dataclass(frozen=True)
class LigandDefinition:
    name: str
    chain: str | None = None
    resnum: int | None = None
    icode: str = ""
    atom_serial: int | None = None


def parse_ligand_spec(spec: str | None, bad: list | None = None) -> list[LigandDefinition] | None:
    """Parse P2Rank-style ligand definitions (comma separated). Empty/None -> None (use rules).
    Unparseable tokens raise ValueError, or are appended to ``bad`` if a list is given."""
    if spec is None:
        return None
    spec = str(spec).strip()
    if spec in ("", "*", "nan"):
        return None
    out = []
    for tok in _split_top_level(spec):
        tok = tok.strip()
        m = _SPEC_RE.match(tok)
        if not m:
            if bad is not None:
                bad.append(tok)
                continue
            raise ValueError(f"cannot parse ligand definition {tok!r}")
        name, kind, arg = m.group(1).upper(), m.group(2) or ("group_id" if m.group(3) else None), m.group(3)
        if kind is None:
            out.append(LigandDefinition(name))
        elif kind == "group_id":
            chain, _, rest = arg.rpartition("_")
            mm = re.match(r"^(-?\d+)([A-Za-z]?)$", rest)
            if not chain or not mm:
                raise ValueError(f"bad group_id in {tok!r} (expected CHAIN_RESNUM[ICODE])")
            out.append(LigandDefinition(name, chain=chain, resnum=int(mm.group(1)), icode=mm.group(2)))
        elif kind == "atom_id":
            out.append(LigandDefinition(name, atom_serial=int(arg)))
        else:
            raise NotImplementedError("contact_res_ids ligand specifiers are not supported")
    return out


def _split_top_level(s: str) -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [t for t in out if t.strip()]


def _def_str(d: LigandDefinition) -> str:
    if d.chain is not None:
        return f"{d.name}[{d.chain}_{d.resnum}{d.icode}]"
    if d.atom_serial is not None:
        return f"{d.name}[atom_id:{d.atom_serial}]"
    return d.name


def _matches(defn: LigandDefinition, s: Structure, r: int) -> bool:
    if s.r_name[r] != defn.name:
        return False
    if defn.chain is not None:
        return s.r_chain[r] == defn.chain and int(s.r_num[r]) == defn.resnum and s.r_icode[r] == defn.icode
    if defn.atom_serial is not None:
        return bool(np.any(s.serial[s.r_start[r] : s.r_end[r]] == defn.atom_serial))
    return True


# --------------------------------------------------------------------------- extraction
_MASS_CACHE: dict[str, float] = {}


def _masses(elements: np.ndarray) -> np.ndarray:
    out = np.empty(len(elements), dtype=np.float64)
    for i, e in enumerate(elements):
        m = _MASS_CACHE.get(e)
        if m is None:
            m = gemmi.Element(e).weight if e else 0.0
            if not m:
                m = 12.011  # unknown element: carbon mass
            _MASS_CACHE[e] = m
        out[i] = m
    return out


def ligand_center(coords: np.ndarray, elements: np.ndarray, method: str) -> np.ndarray:
    if method == "centroid":
        return coords.mean(axis=0)
    if method == "com":
        w = _masses(elements)
        return (coords * w[:, None]).sum(axis=0) / w.sum()
    raise ValueError(f"unknown center method {method!r}")


def _union_find_clusters(n: int, pairs: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def binding_residues(s: Structure, coords: np.ndarray, cutoff: float, protein_tree=None, protein_idx=None) -> tuple:
    """Protein residues with any heavy atom within ``cutoff`` of any of ``coords``."""
    if protein_idx is None:
        protein_idx = np.flatnonzero(s.protein_heavy_mask())
    if len(protein_idx) == 0 or len(coords) == 0:
        return ()
    if protein_tree is None:
        protein_tree = cKDTree(s.coords[protein_idx])
    hits = protein_tree.query_ball_point(coords, r=cutoff)
    atoms = np.unique(np.concatenate([np.asarray(h, dtype=np.int64) for h in hits])) if len(hits) else np.zeros(0, int)
    if atoms.size == 0:
        return ()
    res = np.unique(s.res_index[protein_idx[atoms]])
    return tuple(residue_id(s.r_chain[r], s.r_num[r], s.r_icode[r]) for r in res)


def extract_ligands(
    s: Structure,
    rules: str | LigandRules = "p2rank",
    spec: str | None = None,
    strict_spec: bool = False,
    report: dict | None = None,
) -> list[Ligand]:
    """Extract and categorise all candidate ligands of ``s`` (load it with ``remove_hydrogens=False``
    if the rule set uses hydrogens). Returns all ligands; filter on ``category == 'relevant'``.

    With an explicit ``spec``, definitions that match no group (or cannot be parsed) are, as in
    P2Rank, reported but not fatal unless ``strict_spec``; they are listed in
    ``report['unmatched_spec']`` if a dict is passed."""
    rules = get_rules(rules)
    bad: list = []
    defs = parse_ligand_spec(spec, None if strict_spec else bad)
    prot_idx = np.flatnonzero(s.protein_heavy_mask())
    prot_tree = cKDTree(s.coords[prot_idx]) if len(prot_idx) else None

    cand = np.flatnonzero(s.r_kind == LIGAND)
    if not rules.standard_residue_ligands:
        std = STANDARD_AA | STANDARD_NUC
        cand = np.array([r for r in cand if s.r_name[r] not in std], dtype=np.int64)
    atom_ok = np.ones(s.n_atoms, dtype=bool) if rules.ligand_hydrogens else ~s.is_hydrogen

    selected: list[int] = []
    ignored: list[int] = []
    for r in cand:
        if defs is not None:
            (selected if any(_matches(d, s, r) for d in defs) else ignored).append(r)
        else:
            (ignored if s.r_name[r] in rules.ignore_het_groups else selected).append(r)
    if defs is not None:
        unmatched = [d for d in defs if not any(_matches(d, s, r) for r in selected)]
        if unmatched and strict_spec:
            raise ValueError(f"{s.id}: ligand definitions match no group: {unmatched}")
        if report is not None:
            report["unmatched_spec"] = bad + [_def_str(d) for d in unmatched]

    def atoms_of(r):
        a = np.arange(s.r_start[r], s.r_end[r])
        return a[atom_ok[a]]

    # single-linkage clustering of selected groups
    groups = [(r, atoms_of(r)) for r in selected]
    groups = [(r, a) for r, a in groups if a.size]
    clusters: list[list[int]] = []
    if groups:
        atom_group = np.concatenate([np.full(a.size, gi) for gi, (_, a) in enumerate(groups)])
        all_atoms = np.concatenate([a for _, a in groups])
        pairs = cKDTree(s.coords[all_atoms]).query_pairs(r=rules.cluster_dist, output_type="ndarray")
        gp = atom_group[pairs] if len(pairs) else np.zeros((0, 2), int)
        gp = gp[gp[:, 0] != gp[:, 1]]
        labels = _union_find_clusters(len(groups), gp)
        for lab in range(labels.max() + 1):
            clusters.append([gi for gi in range(len(groups)) if labels[gi] == lab])

    out: list[Ligand] = []

    def make(member_res: list[int], atom_idx: np.ndarray, category: str | None) -> Ligand:
        coords = s.coords[atom_idx].astype(np.float64)
        elements = s.element[atom_idx]
        center = ligand_center(coords, elements, rules.center_method)
        if prot_tree is not None:
            contact = float(prot_tree.query(coords, k=1)[0].min())
            cdist = float(prot_tree.query(center, k=1)[0])
        else:
            contact = cdist = math.inf
        if category is None:
            if defs is not None and rules.explicit_bypasses_filters:
                category = "relevant"
            elif len(atom_idx) < rules.min_atoms:
                category = "small"
            elif contact > rules.contact_dist:
                category = "distant"
            elif cdist > rules.center_dist:
                category = "distant"
            else:
                category = "relevant"
        member_res = sorted(member_res, key=lambda r: (s.r_chain[r], int(s.r_num[r]), s.r_icode[r]))
        gids = tuple(residue_id(s.r_chain[r], s.r_num[r], s.r_icode[r]) for r in member_res)
        name = "&".join(sorted({str(s.r_name[r]) for r in member_res}))
        bres = binding_residues(s, coords, rules.binding_residue_dist, prot_tree, prot_idx) if prot_tree is not None else ()
        return Ligand(
            ligand_id=f"{name}|{'&'.join(gids)}",
            name=name,
            groups=gids,
            category=category,
            coords=coords,
            elements=elements,
            center=center,
            contact_dist=contact,
            center_dist=cdist,
            binding_residues=bres,
        )

    for cl in clusters:
        atom_idx = np.concatenate([groups[gi][1] for gi in cl])
        out.append(make([groups[gi][0] for gi in cl], atom_idx, None))
    for r in ignored:
        a = atoms_of(r)
        if a.size:
            out.append(make([r], a, "ignored"))
    out.sort(key=lambda l: (l.category != "relevant", l.name, l.groups))
    return out


def build_ground_truth(
    path: str,
    structure_id: str | None = None,
    rules: str | LigandRules = "p2rank",
    spec: str | None = None,
    chains: Sequence[str] | None = None,
    altloc: str = "first",
    strict_spec: bool = False,
) -> GroundTruth:
    """Load ``path`` and extract the relevant ligands (evaluation targets) under ``rules``."""
    rules = get_rules(rules)
    s = load_structure(path, structure_id=structure_id, chains=chains, remove_hydrogens=False, altloc=altloc)
    report: dict = {}
    ligs = extract_ligands(s, rules, spec, strict_spec=strict_spec, report=report)
    prot_res = np.unique(s.res_index[s.protein_heavy_mask()])
    prot_ids = np.array([residue_id(s.r_chain[r], s.r_num[r], s.r_icode[r]) for r in prot_res], dtype=object)
    return GroundTruth(
        structure_id=s.id,
        rules=rules.name,
        ligands=[l for l in ligs if l.category == "relevant"],
        other_ligands=[l for l in ligs if l.category != "relevant"],
        protein_residues=prot_ids,
        n_protein_atoms=int(s.protein_heavy_mask().sum()),
        path=str(path),
        spec="" if spec is None else str(spec),
        chains=tuple(chains) if chains else None,
        meta={**s.meta, **report},
    )
