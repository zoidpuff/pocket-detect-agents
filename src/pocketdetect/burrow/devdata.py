"""Development-data helpers: manifests, cached ground truth, point labels, sequence groups."""
from __future__ import annotations

import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..ligands import GroundTruth, build_ground_truth

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "results/dev/cache"

ION_MAX_ATOMS = 1  # single-atom ligands are ions (3 Å label radius)


def read_manifest(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ("ligand_spec", "chains"):
        if c not in df.columns:
            df[c] = ""
    return df


def _gt_job(args):
    sid, path, rules, spec, chains = args
    try:
        ch = [c for c in chains.split(",") if c] if chains and chains != "*" else None
        return build_ground_truth(path, structure_id=sid, rules=rules, spec=spec or None, chains=ch), None
    except Exception as e:  # reported, never silently dropped
        return None, f"{sid}: {type(e).__name__}: {e}"


def load_gts(manifest, rules: str = "p2rank", jobs: int = 16) -> dict[str, GroundTruth]:
    manifest = Path(manifest)
    h = hashlib.sha256(manifest.read_bytes() + rules.encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / ".gitignore").write_text("*\n")
    f = CACHE / f"gt_{manifest.stem}_{rules}_{h}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    df = read_manifest(manifest)
    tasks = list(zip(df.structure_id, df.structure_path, [rules] * len(df), df.ligand_spec, df.chains))
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(_gt_job, tasks, chunksize=4))
    errs = [e for g, e in res if e]
    if errs:
        raise RuntimeError("ground-truth errors:\n" + "\n".join(errs[:20]))
    gts = {g.structure_id: g for g, _ in res}
    f.write_bytes(pickle.dumps(gts))
    return gts


def ligand_distance(points, gt: GroundTruth):
    """Per point: min distance to any relevant-ligand heavy atom, and to ion atoms only."""
    pts = np.asarray(points, np.float64)
    d_all = np.full(len(pts), np.inf)
    d_ion = np.full(len(pts), np.inf)
    if len(pts) == 0:
        return d_all, d_ion
    for lig in gt.ligands:
        d, _ = cKDTree(lig.coords).query(pts, k=1)
        d_all = np.minimum(d_all, d)
        if lig.n_atoms <= ION_MAX_ATOMS:
            d_ion = np.minimum(d_ion, d)
    return d_all, d_ion


def point_labels(d_all, d_ion, r: float = 4.0, r_ion: float = 3.0):
    return ((d_all <= r) & ~(np.isfinite(d_ion) & (d_ion <= d_all) & (d_all > r_ion))).astype(np.int8)


def sequence_groups(gts: dict, manifest, min_id: float = 0.3, tmp: Path | None = None) -> dict[str, int]:
    """Structure-level groups = connected components of MMseqs2 30 % clusters of their chains."""
    import subprocess
    import tempfile

    from ..structure import load_structure
    df = read_manifest(manifest)
    tmp = Path(tempfile.mkdtemp(dir=tmp))
    fa = tmp / "seqs.fa"
    three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
             "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
             "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}
    with open(fa, "w") as fh:
        for sid, path in zip(df.structure_id, df.structure_path):
            s = load_structure(path, protein_only=True)
            for ch in s.chains:
                m = s.r_chain == ch
                seq = "".join(three.get(r, "X") for r in s.r_name[m])
                if len(seq) >= 10:
                    fh.write(f">{sid}|{ch}\n{seq}\n")
    mm = "/workspace/tools/bin/mmseqs"
    subprocess.run([mm, "easy-cluster", str(fa), str(tmp / "cl"), str(tmp / "w"), "--min-seq-id", str(min_id),
                    "-c", "0.5", "--cov-mode", "0", "-v", "1"], check=True)
    cl = pd.read_csv(tmp / "cl_cluster.tsv", sep="\t", header=None, names=["rep", "mem"])
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for rep, mem in zip(cl.rep, cl.mem):
        a, b = find(rep.split("|")[0]), find(mem.split("|")[0])
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = sorted({find(s) for s in df.structure_id})
    rid = {r: i for i, r in enumerate(roots)}
    return {s: rid[find(s)] for s in df.structure_id}
