#!/usr/bin/env python
"""Track C step 1: single streaming pass over the AFDB v6 human proteome tar.

Extracts ``AF-<ACC>-F1-model_v6.cif.gz`` for every UniProt accession in the LIGYSIS human site
table (LIGYSIS_human_sites_APRIL_2024.pkl, 3,518 accessions) plus optional extra accessions, into
``data/processed/af/models/``. Proteins modelled as several fragments (AFDB splits proteins
> 2,700 aa into F1..Fn) are NOT extracted and are recorded as excluded.

Outputs
  data/processed/af/models/AF-<ACC>-F1-model_v6.cif.gz
  data/processed/af/extract_index.csv   acc, status (extracted|fragmented|absent), n_fragments,
                                        member, bytes, sha256
  data/processed/af/extract_info.json   tar path/size/mtime, counts, wall time

Accessions absent from the proteome tar are searched once in the SwissProt tar (--fallback-tar).

Usage: .venv/bin/python scripts/data/build_af_extract.py [--extra-accs FILE] [--tar PATH] [--fallback-tar PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_pickle import safe_load  # noqa: E402

AFDB = Path("/cluster/work/beltrao/shared/alphafold/v6")
HUMAN_TAR = AFDB / "UP000005640_9606_HUMAN_v6.tar"
LIG_RES = ROOT / "data/raw/ligysis/lbs_comparison_results"
OUT = ROOT / "data/processed/af"
MEMBER_RE = re.compile(r"^AF-(?P<acc>[A-Z0-9]+(?:-\d+)?)-F(?P<frag>\d+)-model_v(?P<ver>\d+)\.(?P<ext>cif\.gz|pdb\.gz)$")


def ligysis_accessions() -> list[str]:
    hs = safe_load(LIG_RES / "LIGYSIS_human_sites_APRIL_2024.pkl")
    return sorted(set(hs["ACC"].astype(str)))


def scan_tar(tar_path: Path, accs: set, t0: float):
    """Stream ``tar_path`` once ("r|": no seeking). Returns fragment numbers and AFDB versions per
    target accession and the bytes of every F1 cif.gz. Fragmentation is decided by the caller after
    the whole tar has been seen (fragments may come in any order)."""
    frags: dict[str, set[int]] = defaultdict(set)
    versions: dict[str, set[str]] = defaultdict(set)
    f1_bytes: dict[str, bytes] = {}
    f1_member: dict[str, str] = {}
    n_members = 0
    with tarfile.open(tar_path, mode="r|") as tf:
        for m in tf:
            n_members += 1
            if not m.isfile():
                continue
            mm = MEMBER_RE.match(os.path.basename(m.name))
            if not mm or mm["acc"] not in accs:
                continue
            acc, frag = mm["acc"], int(mm["frag"])
            frags[acc].add(frag)
            versions[acc].add(mm["ver"])
            if frag == 1 and mm["ext"] == "cif.gz":
                f1_bytes[acc] = tf.extractfile(m).read()
                f1_member[acc] = m.name
            if n_members % 200000 == 0:
                print(f"  {tar_path.name}: {n_members} members, {len(frags)} target accs seen, {time.time()-t0:.0f}s", flush=True)
    return frags, versions, f1_bytes, f1_member, n_members


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", default=str(HUMAN_TAR))
    ap.add_argument("--extra-accs", default=None, help="file with one extra UniProt accession per line")
    ap.add_argument("--fallback-tar", default=str(AFDB / "swissprot_cif_v6.tar"),
                    help="second tar searched only for accessions absent from --tar ('' = none)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    mdir = out / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    accs = set(ligysis_accessions())
    if args.extra_accs:
        accs |= {l.strip() for l in open(args.extra_accs) if l.strip()}
    print(f"{len(accs)} target accessions", flush=True)

    t0 = time.time()
    tar_path = Path(args.tar)
    frags, versions, f1_bytes, f1_member, n_members = scan_tar(tar_path, accs, t0)
    source = {a: tar_path.name for a in frags}
    fb_info = None
    if args.fallback_tar:
        # second pass only for accessions entirely absent from the proteome tar (e.g. entries not in
        # the reference proteome); recorded per accession in column ``source``
        missing = {a for a in accs if a not in frags}
        fb = Path(args.fallback_tar)
        fr2, ve2, b2, m2, n2 = scan_tar(fb, missing, t0)
        for a in fr2:
            frags[a], versions[a], source[a] = fr2[a], ve2[a], fb.name
            if a in b2:
                f1_bytes[a], f1_member[a] = b2[a], m2[a]
        fb_info = dict(tar=str(fb), n_members=n2, n_searched=len(missing), n_found=len(fr2))

    rows = []
    for acc in sorted(accs):
        nf = len(frags.get(acc, ()))
        if nf == 0:
            rows.append(dict(acc=acc, status="absent", n_fragments=0, member="", bytes=0, sha256="", source=""))
            continue
        if nf > 1 or max(frags[acc]) > 1:
            rows.append(dict(acc=acc, status="fragmented", n_fragments=nf, member="", bytes=0, sha256="", source=source[acc]))
            continue
        if acc not in f1_bytes:
            rows.append(dict(acc=acc, status="no_cif", n_fragments=nf, member="", bytes=0, sha256="", source=source[acc]))
            continue
        data = f1_bytes[acc]
        dest = mdir / os.path.basename(f1_member[acc])
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        rows.append(dict(acc=acc, status="extracted", n_fragments=1, member=f1_member[acc], bytes=len(data),
                         sha256=hashlib.sha256(data).hexdigest(), source=source[acc]))
    idx = pd.DataFrame(rows)
    idx.to_csv(out / "extract_index.csv", index=False)
    st = tar_path.stat()
    info = dict(tar=str(tar_path), tar_bytes=st.st_size, tar_mtime=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(st.st_mtime)),
                n_tar_members=n_members, n_target_accs=len(accs), counts=idx["status"].value_counts().to_dict(),
                afdb_versions=sorted({v for s in versions.values() for v in s}), fallback=fb_info, wall_s=round(time.time() - t0, 1))
    (out / "extract_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
