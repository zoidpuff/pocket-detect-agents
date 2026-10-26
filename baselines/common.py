"""Shared helpers for the baseline wrappers (baselines/run_<tool>.py).

Common pipeline for every tool
  1. clean_structure(): read PDB / mmCIF (optionally .gz), keep model 1, drop waters, ligands/hetero
     non-polymer groups, hydrogens, alternate conformers (keep first altloc) and (by default)
     non-protein polymers; write <workdir>/<stem>.pdb (or .cif if PDB cannot represent it).
  2. run tool on the cleaned file(s).
  3. write_outputs(): <out>/<stem>.pockets.csv  (rank,score,center_x,center_y,center_z,residues)
                      <out>/<stem>.points.csv   (rank,x,y,z)           [if the tool provides points]
     plus a row in <out>/timing.csv.

Chain ids / residue numbers are the *author* ones (auth_asym_id / auth_seq_id) of the input.
Run with the shared project venv (needs gemmi, numpy): .venv/bin/python baselines/run_<tool>.py ...
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import shutil
import string
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Sequence

BASELINES = Path(__file__).resolve().parent
TOOLS = BASELINES / "tools"
PROJECT = BASELINES.parent

PDB_CHAIN_CHARS = string.ascii_uppercase + string.ascii_lowercase + string.digits + "".join(
    c for c in string.punctuation if c not in "_,;:\"'#")  # skip chars that clash with the residue/CSV syntax
POCKET_COLS = ["rank", "score", "center_x", "center_y", "center_z", "residues"]
POINT_COLS = ["rank", "x", "y", "z"]
STRUCT_EXT = (".pdb", ".ent", ".cif", ".mmcif", ".pdb.gz", ".ent.gz", ".cif.gz", ".mmcif.gz", ".bcif")


# --------------------------------------------------------------------------- inputs
def stem_of(path: str | Path) -> str:
    """File stem without structure extensions, e.g. pdb1abc.ent.gz -> 1abc ; 1abc-assembly1.cif.gz -> 1abc-assembly1"""
    n = Path(path).name
    for e in sorted(STRUCT_EXT, key=len, reverse=True):
        if n.lower().endswith(e):
            n = n[: -len(e)]
            break
    if n.startswith("pdb") and len(n) == 7 and Path(path).name.lower().endswith(".ent.gz"):
        n = n[3:]
    return n


def expand_inputs(items: Sequence[str]) -> list[Path]:
    """Items are structure files, directories (all structures inside), or @listfile (one path per line)."""
    out: list[Path] = []
    for it in items:
        if it.startswith("@"):
            out += [Path(l.strip()) for l in open(it[1:]) if l.strip() and not l.startswith("#")]
        elif os.path.isdir(it):
            out += sorted(p for p in Path(it).iterdir() if p.name.lower().endswith(STRUCT_EXT))
        else:
            out.append(Path(it))
    missing = [str(p) for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing[:5]}")
    seen = {}
    for p in out:
        s = stem_of(p)
        if s in seen:
            raise ValueError(f"duplicate stem {s}: {p} vs {seen[s]}")
        seen[s] = p
    return out


# --------------------------------------------------------------------------- cleaning
def clean_structure(path: str | Path, workdir: str | Path, *, keep_nucleic: bool = False,
                    prefer: str = "pdb", stem: str | None = None) -> tuple[Path, dict]:
    """Return (cleaned_file, info). prefer='pdb' writes PDB when representable, else mmCIF; 'cif' forces mmCIF."""
    import gemmi

    path = Path(path)
    stem = stem or stem_of(path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    if len(st) > 1:
        for i in range(len(st) - 1, 0, -1):
            del st[i]
    st.remove_ligands_and_waters()  # entity-based: waters + non-polymer (ligands, ions, ...)
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    st.remove_empty_chains()
    m = st[0]
    n_removed_nuc = 0
    if not keep_nucleic:
        nuc_types = {gemmi.PolymerType.Dna, gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid}
        for ch in m:
            nuc_sub = set()
            for sub in ch.subchains():
                ent = st.get_entity_of(sub)
                if (ent is not None and ent.polymer_type in nuc_types) or \
                        (ent is None and all(_is_nucleic(r) for r in sub)):
                    nuc_sub.add(sub.subchain_id())
            idx = [i for i, r in enumerate(ch) if r.subchain in nuc_sub or _is_nucleic(r)]
            for i in reversed(idx):
                del ch[i]
            n_removed_nuc += len(idx)
        st.remove_empty_chains()
    n_atoms = sum(len(r) for ch in m for r in ch)
    n_res = sum(len(ch) for ch in m)
    n_chains = len(m)
    if n_atoms == 0:
        raise ValueError("no protein atoms left after cleaning")
    chain_names = [ch.name for ch in m]
    chain_map = {}
    if prefer == "pdb" and any(len(c) > 1 for c in chain_names) and n_chains <= len(PDB_CHAIN_CHARS):
        # PDB format has 1-char chain ids: remap to unique single characters (undone in write_pockets)
        used = {c for c in chain_names if len(c) == 1}
        free = iter(c for c in PDB_CHAIN_CHARS if c not in used)
        for ch in m:
            if len(ch.name) > 1:
                new = next(free)
                chain_map[new] = ch.name
                ch.name = new
        chain_names = [ch.name for ch in m]
    pdb_ok = (all(len(c) == 1 for c in chain_names) and n_atoms < 99999
              and all(r.seqid.num < 10000 for ch in m for r in ch))
    if prefer == "pdb" and pdb_ok:
        out = workdir / f"{stem}.pdb"
        st.write_pdb(str(out))
    else:
        out = workdir / f"{stem}.cif"
        st.make_mmcif_document().write_file(str(out))
        if chain_map:  # cif keeps (possibly remapped) names as written; map is still valid
            pass
    return out, dict(n_atoms=n_atoms, n_res=n_res, n_chains=n_chains, fmt=out.suffix[1:], n_nuc_removed=n_removed_nuc,
                      chain_map=chain_map)


def _is_nucleic(res) -> bool:
    import gemmi
    info = gemmi.find_tabulated_residue(res.name)
    if info is not None:
        return info.is_nucleic_acid()
    # unknown residue name inside polymer: nucleic if it has P and C4'/C1' atoms and no CA
    names = {a.name for a in res}
    return ("CA" not in names) and ("P" in names or "C4'" in names or "C1'" in names)


# --------------------------------------------------------------------------- residues
def fmt_res(chain: str, num: int | str, ins: str = "") -> str:
    ins = (ins or "").strip()
    return f"{chain}:{num}{ins}"


_RES_RE = re.compile(r"^(?P<chain>.+?)_(?P<num>-?\d+)(?P<ins>[A-Za-z]?)$")


def parse_p2rank_res(tok: str) -> str:
    """'A_103' / 'A_103a' / 'AA_-5' -> 'A:103' / 'A:103A' / 'AA:-5'"""
    m = _RES_RE.match(tok)
    if not m:
        raise ValueError(f"bad residue token {tok}")
    return fmt_res(m["chain"], m["num"], m["ins"])


# --------------------------------------------------------------------------- output
def _res_key(r: str):
    c, _, rest = r.partition(":")
    m = re.match(r"(-?\d+)(.*)", rest)
    return (c, int(m[1]), m[2]) if m else (c, 0, rest)


def write_pockets(outdir: Path, stem: str, pockets: list[dict], points: list[tuple] | None = None,
                  chain_map: dict | None = None):
    """pockets: dicts with keys rank,score,center(3),residues(list[str]); points: (rank,x,y,z) tuples."""
    outdir.mkdir(parents=True, exist_ok=True)
    if chain_map:
        def unmap(r):
            c, _, rest = r.partition(":")
            return f"{chain_map.get(c, c)}:{rest}"
        pockets = [dict(p, residues=[unmap(r) for r in p["residues"]]) for p in pockets]
    pockets = [dict(p, residues=sorted(p["residues"], key=_res_key)) for p in pockets]
    with open(outdir / f"{stem}.pockets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(POCKET_COLS)
        for p in sorted(pockets, key=lambda p: p["rank"]):
            c = p["center"]
            w.writerow([p["rank"], f"{p['score']:.5g}", f"{c[0]:.3f}", f"{c[1]:.3f}", f"{c[2]:.3f}",
                        " ".join(p["residues"])])
    if points is not None:
        with open(outdir / f"{stem}.points.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(POINT_COLS)
            for r, x, y, z in sorted(points, key=lambda t: t[0]):
                w.writerow([r, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])


def append_timing(outdir: Path, rows: list[dict]):
    p = outdir / "timing.csv"
    new = not p.exists()
    cols = ["stem", "status", "n_atoms", "n_res", "n_chains", "fmt", "clean_s", "tool_s", "total_s", "note"]
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def read_pockets_csv(path) -> list[dict]:
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append(dict(rank=int(r["rank"]), score=float(r["score"]),
                            center=(float(r["center_x"]), float(r["center_y"]), float(r["center_z"])),
                            residues=r["residues"].split()))
    return out


# --------------------------------------------------------------------------- CLI helpers
def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("inputs", nargs="+", help="structure files (.pdb/.cif[.gz]), directories, or @listfile")
    ap.add_argument("-o", "--out", required=True, help="output directory")
    ap.add_argument("-j", "--procs", type=int, default=1, help="parallel worker processes / threads")
    ap.add_argument("--keep-raw", action="store_true", help="keep raw tool output under <out>/raw/<stem>/")
    ap.add_argument("--keep-nucleic", action="store_true", help="do not strip DNA/RNA polymer chains")
    ap.add_argument("--no-clean", action="store_true",
                    help="skip ligand/water stripping and feed the input file to the tool unchanged")
    ap.add_argument("--overwrite", action="store_true", help="recompute even if <stem>.pockets.csv exists")
    return ap


def todo(inputs: list[Path], outdir: Path, overwrite: bool) -> list[Path]:
    if overwrite:
        return inputs
    return [p for p in inputs if not (outdir / f"{stem_of(p)}.pockets.csv").exists()]


def prepare(path: Path, workdir: Path, args, prefer="pdb"):
    """clean (or copy/decompress) one input -> (file, info, seconds)"""
    t = time.time()
    if args.no_clean:
        if path.name.endswith(".gz"):
            dst = workdir / path.name[:-3]
            with gzip.open(path, "rb") as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        else:
            dst = workdir / path.name
            shutil.copy(path, dst)
        # normalise stem
        s = stem_of(path)
        final = workdir / (s + dst.suffix)
        if final != dst:
            dst.rename(final)
        return final, dict(fmt=final.suffix[1:]), time.time() - t
    f, info = clean_structure(path, workdir, keep_nucleic=args.keep_nucleic, prefer=prefer)
    return f, info, time.time() - t


def pmap(fn: Callable, items: Iterable, procs: int):
    items = list(items)
    if procs <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    with ProcessPoolExecutor(procs) as ex:
        return list(ex.map(fn, items))


def safe(fn):
    """Decorator turning exceptions into ('error', traceback) so one failure doesn't kill a batch."""
    def w(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa
            return dict(status="error", note=f"{type(e).__name__}: {e}"[:300], tb=traceback.format_exc())
    w.__name__ = fn.__name__
    return w


def finish(out: Path, rows: list[dict], extra: str = ""):
    append_timing(out, rows)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"done: {ok}/{len(rows)} ok {extra}")
    for r in rows:
        if r["status"] != "ok":
            print("  FAILED", r["stem"], r.get("note", ""), file=sys.stderr)


def run_cmd(cmd, timeout=None, **kw):
    """subprocess.run with captured output; raises RuntimeError with tail of output on failure/timeout."""
    import subprocess
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"timeout after {timeout}s")
    if r.returncode != 0:
        raise RuntimeError(f"rc={r.returncode}: " + (r.stdout + r.stderr)[-400:])
    return r


def reexec_in_venv(venv: Path):
    """If not already running inside `venv`, re-exec this script with the venv's python (so wrappers can be
    launched with any python, e.g. the project .venv). Call at the top of a wrapper's main()."""
    py = Path(venv) / "bin" / "python"
    if Path(sys.prefix).resolve() != Path(venv).resolve():
        os.execv(str(py), [str(py), "-I", *sys.argv] if False else [str(py), *sys.argv])
