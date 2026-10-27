"""Fast structure -> :class:`burrow.prep.Protein` loader (standard mode).

``load_protein(path, structure_id)`` returns exactly
``protein_from_structure(load_structure(path, structure_id=structure_id))`` (same values and
dtypes in every field; checked by ``tests/test_burrow_fast_io.py`` and
``scripts/verify_burrow_fast_io.py``) in about a third of the time:

* one pass over the gemmi model that reads only what the protein view needs (no occupancy /
  serial / entity columns, no intermediate :class:`Structure` and no two ``select`` copies);
* the reference classification rules are applied unchanged (water / amino-acid-like residues,
  peptide-bonded fragments with an ATOM-record residue, altloc "first" per residue position,
  hydrogens removed);
* the per-atom chemistry tables (atom classes, radii, residue groups, metal-coordination and
  first-hit flags) depend only on (element, atom name, residue name): they are computed with the
  reference functions once per distinct triple and cached per process;
* the weighted contact number uses the reference ``_wcn``.
"""
from __future__ import annotations

import gemmi
import numpy as np

from ...geometry.atomtypes import AROMATIC, NEGATIVE, POSITIVE, atom_class_matrix, vdw_radii
from ...structure import PEPTIDE_BOND_MAX, WATER_NAMES, _is_tabulated_aa, residue_id
from ..prep import _COORD_SC, _RG, BACKBONE_CB, RES_GROUPS, Protein, _wcn

_NM4: dict = {}       # raw atom name -> reference upper-cased '<U4' value
_EL2: dict = {}       # raw element name -> reference upper-cased '<U2' value
_UP: dict = {}        # residue name -> upper case (numpy semantics)
_TAB: dict = {}       # (el, nm, rn) -> row index into the tables below
_ROWS: list = []      # per triple: (class row, radius, rg index, coord_sc, coord, hit class, bb)
_CLS_NAMES = None


def _nm4(raw):
    v = _NM4.get(raw)
    if v is None:
        v = _NM4[raw] = str(np.char.upper(np.array([raw], "<U4"))[0])
    return v


def _el2(raw):
    v = _EL2.get(raw)
    if v is None:
        v = _EL2[raw] = str(np.char.upper(np.array([raw.upper()], "<U2"))[0])
    return v


def _up(raw):
    v = _UP.get(raw)
    if v is None:
        v = _UP[raw] = str(np.char.upper(np.array([raw]).astype(str))[0])
    return v


def _row(key):
    """Reference per-atom chemistry for one (element, atom name, residue name) triple."""
    global _CLS_NAMES
    i = _TAB.get(key)
    if i is not None:
        return i
    el, nm, rn = key
    e1, n1, r1 = np.array([el]), np.array([nm]), np.array([rn])
    cls, names = atom_class_matrix(e1, n1, r1)
    _CLS_NAMES = list(names)
    rad = vdw_radii(e1)[0]
    coord_sc = nm in _COORD_SC.get(rn, ())
    coord = coord_sc or nm == "O"
    charged = (nm in POSITIVE.get(rn, ()) or nm in NEGATIVE.get(rn, ())) or nm == "OXT"
    arom = nm in AROMATIC.get(rn, ())
    polar = el in ("N", "O")
    hc = 2 if charged else (3 if arom else (1 if polar else 0))
    _ROWS.append((cls[0], rad, _RG.get(rn, 6), coord_sc, coord, hc, nm in BACKBONE_CB))
    i = _TAB[key] = len(_ROWS) - 1
    return i


def load_protein(path, structure_id: str) -> Protein:
    st = gemmi.read_structure(str(path), merge_chain_parts=True)
    st.setup_entities()
    if len(st) == 0:
        raise ValueError(f"no models in {path}")
    try:
        P = _from_flat(st, structure_id)
    except (UnicodeError, ValueError):
        P = None
    return P if P is not None else _from_model(st, structure_id)


def _from_model(st, structure_id: str) -> Protein:
    """Per-atom loop over the gemmi model (general fallback of :func:`_from_flat`)."""
    model = st[0]
    xyz, el, nm, alt, bf, rix = [], [], [], [], [], []
    rc, rn, rnum, ric, rhet, aa, rbb = [], [], [], [], [], [], []
    for ch in model:
        cname = ch.name
        for res in ch:
            n0 = len(el)
            n_xyz = c_xyz = None
            has_ca = False
            for a in res:
                p = a.pos
                t = (p.x, p.y, p.z)
                name = a.name
                xyz.append(t)
                el.append(a.element.name)
                nm.append(name)
                alt.append(a.altloc)
                bf.append(a.b_iso)
                if name == "N" and n_xyz is None:
                    n_xyz = t
                elif name == "C" and c_xyz is None:
                    c_xyz = t
                elif name == "CA":
                    has_ca = True
            k = len(el) - n0
            if k == 0:
                continue
            rix.extend([len(rc)] * k)
            rc.append(cname)
            name = res.name
            rn.append(name)
            rnum.append(res.seqid.num)
            ic = res.seqid.icode
            ric.append("" if ic in ("\x00", " ") else ic)
            rhet.append(res.het_flag == "H")
            if res.is_water() or name in WATER_NAMES:
                aa.append(False)
            else:
                aa.append(bool(_is_tabulated_aa(name) or (n_xyz is not None and c_xyz is not None and has_ca)))
            rbb.append((n_xyz, c_xyz))
    n_res = len(rc)
    # ---- protein residues: peptide-bonded fragments with >= 1 ATOM-record residue (as load_structure)
    aa_like = np.array(aa, bool)
    frag = np.arange(n_res)
    for i in range(n_res - 1):
        if rc[i] != rc[i + 1] or not (aa_like[i] and aa_like[i + 1]):
            continue
        c_prev, n_next = rbb[i][1], rbb[i + 1][0]
        if c_prev is None or n_next is None:
            continue
        d2 = sum((c_prev[k] - n_next[k]) ** 2 for k in range(3))
        if d2 <= PEPTIDE_BOND_MAX ** 2:
            frag[i + 1] = frag[i]
    atom_record = aa_like & ~np.asarray(rhet, dtype=bool)
    frag_has_atom = np.zeros(n_res, dtype=bool)
    np.logical_or.at(frag_has_atom, frag, atom_record)
    is_prot = aa_like & frag_has_atom[frag]
    # ---- atoms kept: altloc "first" per residue position (over all atoms), no H/D, protein residues
    n = len(el)
    rix_a = np.asarray(rix, np.int64)
    keep = is_prot[rix_a] if n else np.zeros(0, bool)
    el2 = [_el2(e) for e in el]
    for i, e in enumerate(el2):
        if e == "H" or e == "D":
            keep[i] = False
    if any(a != "\x00" for a in alt):
        chosen = {}
        for i, a in enumerate(alt):
            if a == "\x00":
                continue
            r = rix[i]
            key = (rc[r], rnum[r], ric[r])
            c = chosen.setdefault(key, a)
            if c != a:
                keep[i] = False
    idx = np.flatnonzero(keep)
    # residue table of the kept atoms (order preserved)
    kept_res, res_index = np.unique(rix_a[idx], return_inverse=True)
    res_index = res_index.astype(np.int32).reshape(-1)
    Lr = max((len(x) for x in rn), default=1)
    Lc = max((len(x) for x in rc), default=1)
    il = idx.tolist()
    el_k = [el2[i] for i in il]
    nm_k = [_nm4(nm[i]) for i in il]
    rn_u = [_up(rn[r]) for r in kept_res.tolist()]
    rn_k = [rn_u[j] for j in res_index.tolist()]
    rows = np.array([_row(k) for k in zip(el_k, nm_k, rn_k)], np.int64)
    coords = np.asarray(xyz, np.float32).reshape(-1, 3)[idx] if n else np.zeros((0, 3), np.float32)
    element = np.array(el_k, "<U2")
    atom_name = np.array(nm_k, "<U4")
    res_name = np.array(rn_k, f"<U{Lr}")
    chain = np.array([rc[r] for r in kept_res.tolist()], f"<U{Lc}")[res_index]
    residue_ids = np.array([residue_id(np.str_(rc[r]), np.int32(rnum[r]), np.str_(ric[r][:1]))
                            for r in kept_res.tolist()], dtype=object)
    plddt = np.asarray(bf, np.float32)[idx] if n else np.zeros(0, np.float32)
    return _finish(structure_id, coords, element, atom_name, res_name, res_index, residue_ids, chain, plddt, rows)


def _tables():
    """numpy views of the per-triple chemistry rows (rebuilt when new triples were added)."""
    global _TABLES
    if _TABLES is None or len(_TABLES[0]) != len(_ROWS):
        _TABLES = (np.stack([r[0] for r in _ROWS]).astype(np.float32), np.array([r[1] for r in _ROWS], np.float32),
                   np.array([r[2] for r in _ROWS], np.int64), np.array([r[3] for r in _ROWS], bool),
                   np.array([r[4] for r in _ROWS], bool), np.array([r[5] for r in _ROWS], np.int8),
                   np.array([r[6] for r in _ROWS], bool))
    return _TABLES


_TABLES = None


def _finish(structure_id, coords, element, atom_name, res_name, res_index, residue_ids, chain, plddt, rows):
    m = len(rows)
    if m:
        tc, tr, tg, tsc, tco, thc, tbb = _tables()
        cls = tc[rows]
        radii = tr[rows]
        rg = np.zeros((m, len(RES_GROUPS)), np.float32)
        rg[np.arange(m), tg[rows]] = 1.0
        coord_sc, coord, hc, bb = tsc[rows], tco[rows], thc[rows], tbb[rows]
        names = list(_CLS_NAMES)
    else:   # mirror the reference on an empty protein
        cls, names = atom_class_matrix(element, atom_name, res_name)
        radii = vdw_radii(element)
        rg = np.zeros((0, len(RES_GROUPS)), np.float32)
        coord_sc = coord = bb = np.zeros(0, bool)
        hc = np.zeros(0, np.int8)
    wcn = _wcn(coords, res_index, atom_name)
    props = np.concatenate([cls, rg, wcn[:, None], coord[:, None].astype(np.float32)], 1).astype(np.float32)
    prop_names = list(names) + [f"rg_{g}" for g in RES_GROUPS] + ["wcn", "coord"]
    return Protein(
        id=structure_id, coords=np.ascontiguousarray(coords, np.float32), radii=radii, element=element,
        atom_name=atom_name, res_name=res_name, res_index=res_index, residue_ids=residue_ids, chain=chain,
        plddt=plddt, props=props, prop_names=prop_names, hit_class=hc, bb_mask=bb, coord_mask=coord,
        coord_sc_mask=coord_sc)


def _u64(name: str) -> np.uint64:
    return np.frombuffer(name.encode("ascii").ljust(8, b"\0"), np.uint64)[0]


_WATER64 = np.array([_u64(w) for w in sorted(WATER_NAMES) if len(w) <= 8], np.uint64)
_N64, _C64, _CA64 = _u64("N"), _u64("C"), _u64("CA")
_TOP = np.uint64(0xFF) << np.uint64(56)


def _dec(b) -> str:
    return b.rstrip(b"\0").decode("ascii")


def _from_flat(st, structure_id: str) -> Protein | None:
    """Vectorised version of :func:`_from_model` on ``gemmi.FlatStructure`` (same results); returns
    None when a guard fails (8-character names that FlatStructure may have truncated, residue
    grouping that does not match the model) so that the caller uses the per-atom loop.  Names are
    compared as 8-byte integers."""
    model = st[0]
    f = gemmi.FlatStructure(st)
    f.strings_as_numbers = False
    mn = f.model_num
    n = int(np.argmax(mn != model.num)) if (len(mn) and (mn != model.num).any()) else len(mn)
    if n == 0 or n != sum(len(r) for ch in model for r in ch):
        return None
    an64 = np.ascontiguousarray(f.atom_names[:n]).view(np.uint64)
    rn64 = np.ascontiguousarray(f.residue_names[:n]).view(np.uint64)
    ch64 = np.ascontiguousarray(f.chain_ids[:n]).view(np.uint64)
    if ((an64 & _TOP) != 0).any() or ((rn64 & _TOP) != 0).any() or ((ch64 & _TOP) != 0).any():
        return None
    pos = f.pos[:n]
    resnum = f.resnums[:n]
    icode = f.icodes[:n]
    het = f.het_flags[:n]
    alt = f.altlocs[:n]
    brk = np.ones(n, bool)
    brk[1:] = ((ch64[1:] != ch64[:-1]) | (resnum[1:] != resnum[:-1]) | (icode[1:] != icode[:-1])
               | (rn64[1:] != rn64[:-1]) | (het[1:] != het[:-1]))
    rstart = np.flatnonzero(brk)
    n_res = len(rstart)
    if n_res != sum(1 for ch in model for r in ch if len(r)):
        return None
    rix = np.cumsum(brk) - 1
    rc64, rn64r = ch64[rstart], rn64[rstart]
    uc, cinv = np.unique(rc64, return_inverse=True)
    cinv = cinv.reshape(-1)
    uc_s = [_dec(x.tobytes()) for x in uc]
    urn, rinv = np.unique(rn64r, return_inverse=True)
    rinv = rinv.reshape(-1)
    urn_s = [_dec(x.tobytes()) for x in urn]
    rnum = resnum[rstart]
    ic = icode[rstart]
    ric_c = np.where((ic == 0) | (ic == 32), 0, ic).astype(np.int64)
    rhet = het[rstart] == ord("H")
    # first N / first C / any CA per residue
    n_first = np.full(n_res, -1, np.int64)
    c_first = np.full(n_res, -1, np.int64)
    for target, first in ((_N64, n_first), (_C64, c_first)):
        w = np.flatnonzero(an64 == target)
        if len(w):
            u, fi = np.unique(rix[w], return_index=True)
            first[u] = w[fi]
    has_ca = np.zeros(n_res, bool)
    has_ca[rix[an64 == _CA64]] = True
    water = np.isin(rn64r, _WATER64)
    tab = np.array([_is_tabulated_aa(x) for x in urn_s], bool)[rinv]
    aa_like = ~water & (tab | ((n_first >= 0) & (c_first >= 0) & has_ca))
    # peptide-bonded fragments (same arithmetic as load_structure: ((0 + dx^2) + dy^2) + dz^2)
    link = np.zeros(n_res, bool)
    if n_res > 1:
        cand = (rc64[1:] == rc64[:-1]) & aa_like[1:] & aa_like[:-1] & (c_first[:-1] >= 0) & (n_first[1:] >= 0)
        j = np.flatnonzero(cand)
        if len(j):
            dd = pos[c_first[j]] - pos[n_first[j + 1]]
            d2 = (dd[:, 0] ** 2 + dd[:, 1] ** 2) + dd[:, 2] ** 2
            link[j[d2 <= PEPTIDE_BOND_MAX ** 2] + 1] = True
    frag = np.arange(n_res)
    for i in np.flatnonzero(link).tolist():
        frag[i] = frag[i - 1]
    atom_record = aa_like & ~rhet
    frag_has_atom = np.zeros(n_res, dtype=bool)
    np.logical_or.at(frag_has_atom, frag, atom_record)
    is_prot = aa_like & frag_has_atom[frag]
    # atoms kept: protein residues, no H/D, altloc "first" per residue position
    el16 = np.ascontiguousarray(f.element_names[:n]).view(np.uint16)
    uel, einv = np.unique(el16, return_inverse=True)
    einv = einv.reshape(-1)
    el_u = [_el2(_dec(x.tobytes())) for x in uel]
    is_h = np.array([e in ("H", "D") for e in el_u], bool)[einv]
    keep = is_prot[rix] & ~is_h
    ha = np.flatnonzero(alt != 0)
    if len(ha):
        key = (cinv.astype(np.int64) * (1 << 40) + (rnum.astype(np.int64) + (1 << 31)) * 256 + ric_c)[rix[ha]]
        u, fi, apk = np.unique(key, return_index=True, return_inverse=True)
        apk = apk.reshape(-1)
        chosen = alt[ha[fi]]
        keep[ha] &= alt[ha] == chosen[apk]
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        return None
    kept_res, res_index = np.unique(rix[idx], return_inverse=True)
    res_index = res_index.astype(np.int32).reshape(-1)
    Lr = max(len(x) for x in urn_s)
    Lc = max(len(x) for x in uc_s)
    # strings of the kept atoms via their distinct values
    nm32 = (an64[idx] & np.uint64(0xFFFFFFFF))
    unm, ninv = np.unique(nm32, return_inverse=True)
    ninv = ninv.reshape(-1)
    nm_u = [_nm4(_dec(np.uint64(x).tobytes())) for x in unm]
    atom_name = np.array(nm_u, "<U4")[ninv]
    element = np.array(el_u, "<U2")[einv[idx]]
    rn_up_u = [_up(x) for x in urn_s]
    rk = rinv[kept_res]
    res_name = np.array(rn_up_u, f"<U{Lr}")[rk][res_index]
    chain = np.array(uc_s, f"<U{Lc}")[cinv[kept_res]][res_index]
    ric_s = ["" if c == 0 else chr(c) for c in ric_c[kept_res].tolist()]
    residue_ids = np.array([residue_id(np.str_(uc_s[c]), np.int32(k), np.str_(i)) for c, k, i in
                            zip(cinv[kept_res].tolist(), rnum[kept_res].tolist(), ric_s)], dtype=object)
    coords = pos[idx].astype(np.float32)
    plddt = f.b_iso[:n][idx].astype(np.float32)
    # chemistry rows per distinct (element, atom name, residue name)
    tk = (einv[idx].astype(np.int64) * len(unm) + ninv) * len(urn) + rk[res_index]
    ut, tinv = np.unique(tk, return_inverse=True)
    rows_u = np.empty(len(ut), np.int64)
    for q, t in enumerate(ut.tolist()):
        a, rest = divmod(t, len(unm) * len(urn))
        b, c = divmod(rest, len(urn))
        rows_u[q] = _row((el_u[a], nm_u[b], rn_up_u[c]))
    rows = rows_u[tinv.reshape(-1)]
    return _finish(structure_id, coords, element, atom_name, res_name, res_index, residue_ids, chain, plddt, rows)
