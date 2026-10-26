#!/usr/bin/env python
"""P2Rank (-c alphafold) vs BURROW on a random 500-protein subset of the human AFDB proteome.

Speed (wall-clock for the whole subset, same CPU budget: P2Rank one JVM x 24 threads; BURROW fast
driver 3 GPUs x 8 workers) and the hallucination proxy (fraction of top-3 pockets whose mean residue
pLDDT is < 50 / < 70).  Writes data/processed/proteome_human/p2rank_subset/summary.json, consumed by
scripts/proteome_summary.py.

    python scripts/proteome_p2rank_subset.py --code-root <worktree>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/processed/proteome_human"
PY = str(ROOT / ".venv/bin/python")
SEED = 20261026


def residue_plddt_map(path):
    sys.path.insert(0, str(ROOT / "src"))
    from pocketdetect.structure import load_structure
    from pocketdetect.burrow.af import residue_plddt
    s = load_structure(path)
    s = s.select(s.protein_heavy_mask())
    return dict(zip(s.residue_ids.tolist(), residue_plddt(s).tolist()))


def top3_fracs(per_struct):
    v = [x for lst in per_struct for x in lst[:3] if x == x]
    v = np.array(v) if v else np.zeros(0)
    return round(float((v < 50).mean()), 4) if len(v) else None, round(float((v < 70).mean()), 4) if len(v) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--code-root", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    out = D / "p2rank_subset"
    out.mkdir(parents=True, exist_ok=True)
    paths = [l.strip() for l in (D / "inputs.list").read_text().splitlines() if l.strip()]
    stem = lambda p: Path(p).name.split(".")[0]
    frag = {re.match(r"AF-([^-]+)-F(\d+)", stem(p))[1] for p in paths if "-F1-" not in p}
    pool = [p for p in paths if "-F1-" in p and re.match(r"AF-([^-]+)-F", stem(p))[1] not in frag]
    random.Random(SEED).shuffle(pool)
    sub = sorted(pool[:a.n])
    (out / "subset.list").write_text("\n".join(sub) + "\n")
    rows = []
    env = dict(os.environ, PYTHONPATH=str(a.code_root / "src"))
    # ---- BURROW (fast driver), both modes
    for mode, extra in (("naive", []), ("afmode", ["--af-mode"])):
        od = out / f"burrow_{mode}"
        t = time.perf_counter()
        subprocess.run([PY, "-m", "pocketdetect.burrow.fast", "predict", f"@{out / 'subset.list'}", "--out", str(od),
                        "--device", "cuda:0,1,3", "--workers", "8", "--batch", "32", "--compact", "20",
                        "--ranker", "all", "--preset", "default"] + extra, check=True, env=env, cwd=a.code_root,
                       capture_output=True)
        wall = time.perf_counter() - t
        per = {}
        with open(od / "pockets.tsv") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                per.setdefault(r["stem"], []).append(float(r["mean_plddt"]))
        npk = [int(r["n_pockets"]) for r in csv.DictReader(open(od / "structures.tsv"), delimiter="\t")]
        f50, f70 = top3_fracs(list(per.values()))
        rows.append(dict(method=f"BURROW {mode} (3 GPUs x 8 workers)", wall_s=round(wall, 1),
                         per_min=round(60 * len(sub) / wall, 1), top3_lt50=f50, top3_lt70=f70,
                         median_pockets=float(np.median(npk))))
        print(rows[-1], flush=True)
    # ---- P2Rank 2.5.1 -c alphafold, one JVM
    od = out / "p2rank_af"
    t = time.perf_counter()
    subprocess.run([PY, str(ROOT / "baselines/run_p2rank.py"), f"@{out / 'subset.list'}", "-o", str(od),
                    "-j", str(a.threads), "--config", "alphafold"], check=True, capture_output=True)
    wall = time.perf_counter() - t
    per, npk = [], []
    for p in sub:
        f = od / f"{stem(p)}.pockets.csv"
        if not f.exists():
            continue
        pl = residue_plddt_map(p)
        lst = []
        with open(f) as fh:
            rr = list(csv.DictReader(fh))
        npk.append(len(rr))
        for r in rr[:3]:
            v = [pl[x] for x in r["residues"].split() if x in pl]
            lst.append(float(np.mean(v)) if v else float("nan"))
        per.append(lst)
    f50, f70 = top3_fracs(per)
    rows.append(dict(method=f"P2Rank 2.5.1 -c alphafold (one JVM, {a.threads} threads)", wall_s=round(wall, 1),
                     per_min=round(60 * len(sub) / wall, 1), top3_lt50=f50, top3_lt70=f70,
                     median_pockets=float(np.median(npk)) if npk else None))
    print(rows[-1], flush=True)
    note = (f"{len(sub)} random non-fragmented human AFDB proteins (seed {SEED}). Wall-clock includes parsing, JVM / "
            "process start-up and writing; same node, run after the proteome jobs. Pocket pLDDT = mean residue pLDDT "
            "of the pocket's residues (P2Rank: its residue list; BURROW: residues <= 4 Å of pocket points).")
    (out / "summary.json").write_text(json.dumps(dict(n=len(sub), rows=rows, note=note,
                                                      load=open("/proc/loadavg").read().split()[:3]), indent=1))


if __name__ == "__main__":
    main()
