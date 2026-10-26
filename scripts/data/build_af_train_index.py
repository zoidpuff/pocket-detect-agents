#!/usr/bin/env python
"""Track C2 step 1: accession -> tar index over the whole local AFDB v6 copy, and extraction of the
F1 models of all train/dev UniProts (Track A manifests), in ONE streaming pass per tar.

Tars: /cluster/work/beltrao/shared/alphafold/v6/*.tar (48 proteome tars + swissprot_cif_v6.tar;
swissprot_pdb_v6.tar is skipped: same models as PDB format). Tars are processed in parallel
(<= 16 processes). Each pass records every member (offset + size, so later random access needs
no rescan) and stages the F1 cif.gz bytes of needed accessions.

Merge: an accession is *fragmented* if any tar has a member with F > 1 (AFDB splits proteins
> 2,700 aa); such accessions are skipped. Otherwise the F1 cif.gz is taken from a proteome tar if
present, else from the SwissProt tar (identical v6 models; sha256 compared when both exist).

Outputs (data/processed/af_train/):
  afdb_index.parquet        acc, frag, version, ext, tar, member, offset_data, size  (all members)
  models/AF-<ACC>-F1-model_v6.cif.gz
  model_index.csv           per needed accession: status (extracted|fragmented|absent), source tar,
                            sha256, n_fragments, n_tars, sha_consistent
  index_info.json           per-tar member counts / timings, counts

Usage: nohup .venv/bin/python scripts/data/build_af_train_index.py --jobs 16 > log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AFDB = Path("/cluster/work/beltrao/shared/alphafold/v6")
OUT = ROOT / "data/processed/af_train"
MEMBER_RE = re.compile(r"^AF-(?P<acc>[A-Z0-9]+(?:-\d+)?)-F(?P<frag>\d+)-model_v(?P<ver>\d+)\.(?P<ext>cif\.gz|pdb\.gz)$")


def needed_accessions() -> set[str]:
    accs = set()
    for split in ("train", "dev"):
        m = pd.read_csv(ROOT / f"data/processed/{split}/manifest.csv", dtype=str, keep_default_na=False)
        for u in m["uniprot"]:
            accs |= {a for a in u.split(",") if a}
    return accs


def scan(tar_path: str, needed: frozenset, stage_dir: str):
    t0 = time.time()
    rows, staged = [], {}
    stage = Path(stage_dir) / Path(tar_path).stem
    stage.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            mm = MEMBER_RE.match(name)
            if not mm:
                continue
            acc, frag = mm["acc"], int(mm["frag"])
            rows.append((acc, frag, int(mm["ver"]), mm["ext"], Path(tar_path).name, m.name, m.offset_data, m.size))
            if frag == 1 and mm["ext"] == "cif.gz" and acc in needed:
                data = tf.extractfile(m).read()
                (stage / name).write_bytes(data)
                staged[acc] = (str(stage / name), hashlib.sha256(data).hexdigest())
    return tar_path, rows, staged, round(time.time() - t0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.jobs > 16:
        sys.exit("at most 16 processes (PI constraint)")
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    stage_dir = out / "models" / "_staging"
    needed = frozenset(needed_accessions())
    tars = sorted(p for p in AFDB.glob("*.tar") if p.name != "swissprot_pdb_v6.tar")
    # largest first for better load balance
    tars.sort(key=lambda p: -p.stat().st_size)
    print(f"{len(needed)} needed accessions; {len(tars)} tars", flush=True)
    t0 = time.time()
    all_rows, staged_by_tar, per_tar = [], {}, {}
    with ProcessPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(scan, str(t), needed, str(stage_dir)) for t in tars]
        for f in as_completed(futs):
            tp, rows, staged, dt = f.result()
            all_rows += rows
            staged_by_tar[Path(tp).name] = staged
            per_tar[Path(tp).name] = dict(n_members=len(rows), n_staged=len(staged), seconds=dt)
            print(f"  {Path(tp).name}: {len(rows)} members, {len(staged)} staged, {dt}s (t={time.time()-t0:.0f}s)", flush=True)
    idx = pd.DataFrame(all_rows, columns=["acc", "frag", "version", "ext", "tar", "member", "offset_data", "size"])
    idx.to_parquet(out / "afdb_index.parquet", index=False)

    frag_max = idx.groupby("acc")["frag"].max()
    n_frag = idx[idx.ext == "cif.gz"].groupby("acc")["frag"].nunique()
    recs = []
    for acc in sorted(needed):
        cands = [(t, *staged_by_tar[t][acc]) for t in staged_by_tar if acc in staged_by_tar[t]]
        if acc not in frag_max.index:
            recs.append(dict(acc=acc, status="absent", n_fragments=0, n_tars=0, source="", sha256="", sha_consistent=""))
            continue
        if frag_max[acc] > 1:
            recs.append(dict(acc=acc, status="fragmented", n_fragments=int(n_frag.get(acc, 0)), n_tars=len(cands),
                             source="", sha256="", sha_consistent=""))
            continue
        if not cands:
            recs.append(dict(acc=acc, status="no_cif", n_fragments=1, n_tars=0, source="", sha256="", sha_consistent=""))
            continue
        cands.sort(key=lambda c: (c[0].startswith("swissprot"), c[0]))  # proteome tar first
        src, path, sha = cands[0]
        dest = out / "models" / Path(path).name
        shutil.copyfile(path, dest)
        recs.append(dict(acc=acc, status="extracted", n_fragments=1, n_tars=len(cands), source=src, sha256=sha,
                         sha_consistent=len({c[2] for c in cands}) == 1))
    shutil.rmtree(stage_dir)
    mi = pd.DataFrame(recs)
    mi.to_csv(out / "model_index.csv", index=False)
    info = dict(n_needed=len(needed), n_tars=len(tars), n_index_rows=len(idx), n_index_accessions=int(idx.acc.nunique()),
                counts=mi.status.value_counts().to_dict(),
                sha_inconsistent=int((mi.sha_consistent.astype(str) == "False").sum()),
                per_tar=per_tar, wall_s=round(time.time() - t0, 1))
    (out / "index_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "per_tar"}, indent=2))


if __name__ == "__main__":
    main()
