"""Sequence-leakage rule (DESIGN.md §9, 'Leakage rule (complete)').

A training chain is removed if ANY MMseqs2 hit to a protected set (test union, or train for dev)
satisfies  E < 1e-3  AND  identity >= 30%  AND  (coverage >= 50% of either chain OR >= 80 aligned
residues).  Identity = MMseqs2 ``fident`` (identical / alignment columns), coverage = aligned span /
chain length (``qcov``/``tcov``), aligned residues = ``alnlen``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

__all__ = ["RULE", "M8_COLUMNS", "apply_rule", "mmseqs_search", "read_m8"]

RULE = dict(max_evalue=1e-3, min_identity=0.30, min_cov_either=0.50, min_aligned=80)
M8_COLUMNS = ["query", "target", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits"]
MMSEQS = Path(__file__).resolve().parents[3] / "tools" / "bin" / "mmseqs"


def apply_rule(m8: pd.DataFrame, rule: dict = RULE) -> pd.Series:
    """Boolean mask of hits that count as leakage."""
    return (
        (m8["evalue"] < rule["max_evalue"])
        & (m8["fident"] >= rule["min_identity"])
        & ((m8["qcov"] >= rule["min_cov_either"]) | (m8["tcov"] >= rule["min_cov_either"]) | (m8["alnlen"] >= rule["min_aligned"]))
    )


def read_m8(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=M8_COLUMNS,
                       dtype={"query": str, "target": str})


def mmseqs_search(query_fasta: str | Path, target_fasta: str | Path, out_m8: str | Path, tmp: str | Path,
                  threads: int = 48, sensitivity: float = 7.5, max_seqs: int = 5000, evalue: float = 1e-3,
                  mmseqs: str | Path = MMSEQS) -> pd.DataFrame:
    """Sensitive local MMseqs2 search (``easy-search``), E-value cut at ``evalue``; returns all hits."""
    tmp = Path(tmp)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    cmd = [str(mmseqs), "easy-search", str(query_fasta), str(target_fasta), str(out_m8), str(tmp),
           "-s", str(sensitivity), "--max-seqs", str(max_seqs), "-e", str(evalue), "--threads", str(threads),
           "--format-output", ",".join(M8_COLUMNS), "-v", "1"]
    subprocess.run(cmd, check=True, env={**os.environ, "OMP_NUM_THREADS": str(threads)})
    shutil.rmtree(tmp, ignore_errors=True)
    return read_m8(out_m8)
