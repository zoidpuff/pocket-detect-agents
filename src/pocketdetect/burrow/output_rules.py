"""Output rules applied to the score-ordered, NMS'd pocket list (v3b, track T2).

The frozen v2 floor (formation.min_score = 0.03, min_keep = 3) keeps the first max(3, #score >= 0.03)
pockets of the full list; every rule below is a post-filter of the same full list (`--min-score none`
output), so it can be applied identically after either backend:

  v1               keep everything
  v2               keep rank < 3 or score >= 0.03
  a<rho>           keep rank < 3 or score >= min(0.03, rho * best score of the protein)
  b<k>             keep rank < max(3, k * ceil(n_res / 150)) or score >= 0.03
  c<R>_<lo>        rule v2, then additionally keep (in rank order) any pocket with score >= lo whose
                   centre is > R Å from every pocket kept so far (best pocket of a new surface region)
  <a-rule>+<c-rule>  a-rule as the base, then the region rule
"""
from __future__ import annotations

import math
import re

import numpy as np

TAU, MIN_KEEP, RES_PER_UNIT = 0.03, 3, 150.0


def _base(rule: str, s: np.ndarray, n_res: float) -> np.ndarray:
    n = len(s)
    rank = np.arange(n)
    if rule == "v1":
        return np.ones(n, bool)
    if rule == "v2":
        return (rank < MIN_KEEP) | (s >= TAU)
    m = re.fullmatch(r"a([0-9.]+)", rule)
    if m:
        thr = min(TAU, float(m.group(1)) * (s.max() if n else 0.0))
        return (rank < MIN_KEEP) | (s >= thr)
    m = re.fullmatch(r"b([0-9.]+)", rule)
    if m:
        mk = max(MIN_KEEP, int(float(m.group(1)) * math.ceil(max(n_res, 1.0) / RES_PER_UNIT)))
        return (rank < mk) | (s >= TAU)
    raise ValueError(f"unknown output rule {rule!r}")


def apply_rule(rule: str, scores, centers, n_res: float = 300.0) -> np.ndarray:
    """Indices (rank order) of the pockets kept by ``rule``; ``scores`` must be in rank order."""
    s = np.asarray(scores, float)
    C = np.asarray(centers, float).reshape(-1, 3)
    parts = rule.split("+")
    region = None
    if re.fullmatch(r"c[0-9.]+_[0-9.]+", parts[-1]):
        region = parts.pop()
    base = parts[0] if parts else "v2"
    keep = _base(base, s, n_res)
    if region is not None:
        R, lo = map(float, region[1:].split("_"))
        kept = list(np.flatnonzero(keep))
        for i in range(len(s)):
            if keep[i] or s[i] < lo:
                continue
            if all(np.linalg.norm(C[i] - C[j]) > R for j in kept):
                keep[i] = True
                kept.append(i)
    return np.flatnonzero(keep)


def filter_prediction_files(out_dir, stem: str, rule: str, n_res: float) -> int:
    """Apply ``rule`` to ``<out>/<stem>.csv`` (+ ``.points.csv``) written with the full list; ranks are
    renumbered 1..k in the original order.  Returns the number of pockets kept."""
    from pathlib import Path
    import pandas as pd
    out = Path(out_dir)
    f = out / f"{stem}.csv"
    df = pd.read_csv(f, keep_default_na=False, dtype={"residues": str})
    if len(df) == 0:
        return 0
    df = df.sort_values("rank", kind="stable").reset_index(drop=True)
    keep = apply_rule(rule, df["score"].to_numpy(float), df[["center_x", "center_y", "center_z"]].to_numpy(float), n_res)
    new = df.iloc[keep].copy()
    old_ranks = new["rank"].tolist()
    new["rank"] = np.arange(1, len(new) + 1)
    lines = ["rank,score,center_x,center_y,center_z,residues"]
    # keep the writer's exact text for unchanged fields
    raw = f.read_text().splitlines()[1:]
    by_rank = {int(l.split(",", 1)[0]): l.split(",", 1)[1] for l in raw if l.strip()}
    for r_new, r_old in zip(new["rank"], old_ranks):
        lines.append(f"{r_new},{by_rank[int(r_old)]}")
    f.write_text("\n".join(lines) + "\n")
    p = out / f"{stem}.points.csv"
    if p.exists():
        remap = {int(o): int(n) for o, n in zip(old_ranks, new["rank"])}
        pl = p.read_text().splitlines()
        res = [pl[0]] + [f"{remap[int(l.split(',', 1)[0])]},{l.split(',', 1)[1]}" for l in pl[1:]
                         if l.strip() and int(l.split(",", 1)[0]) in remap]
        p.write_text("\n".join(res) + "\n")
    return len(new)
