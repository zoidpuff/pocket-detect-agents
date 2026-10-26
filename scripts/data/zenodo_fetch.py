#!/usr/bin/env python3
"""Download all (or selected) files of a Zenodo record, verify Zenodo's md5, record sha256.

Usage: python -I zenodo_fetch.py RECORD_ID OUTDIR [--include REGEX] [--exclude REGEX]

Writes OUTDIR/MANIFEST.tsv (file, sha256, bytes, retrieved_utc, url) and
OUTDIR/zenodo_record_<id>.json (record metadata as retrieved). Stdlib only.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request

API = "https://zenodo.org/api/records/{}"


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def hash_file(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("outdir")
    ap.add_argument("--include", default=None)
    ap.add_argument("--exclude", default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rec = get_json(API.format(a.record))
    with open(os.path.join(a.outdir, f"zenodo_record_{a.record}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    files = rec["files"]
    if a.include:
        files = [x for x in files if re.search(a.include, x["key"])]
    if a.exclude:
        files = [x for x in files if not re.search(a.exclude, x["key"])]
    man_path = os.path.join(a.outdir, "MANIFEST.tsv")
    done = set()
    if os.path.exists(man_path):
        done = {l.split("\t")[0] for l in open(man_path)}
    print(f"record {a.record}: {rec['metadata']['title']} | {len(files)} files, "
          f"{sum(x['size'] for x in files)/1e9:.2f} GB", flush=True)
    bad = 0
    for x in files:
        key = x["key"]
        out = os.path.join(a.outdir, os.path.basename(key))
        url = x["links"]["self"]
        algo, want = x["checksum"].split(":", 1)
        if not (os.path.exists(out) and os.path.getsize(out) == x["size"]):
            tmp = out + ".part"
            with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            os.replace(tmp, out)
        got = hash_file(out, algo)
        if got != want:
            print(f"CHECKSUM MISMATCH {key}: {algo} {got} != {want}", file=sys.stderr)
            bad += 1
            continue
        if key not in done:
            sha = hash_file(out, "sha256")
            with open(man_path, "a") as m:
                m.write(f"{os.path.basename(key)}\t{sha}\t{x['size']}\t"
                        f"{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\t{url}\n")
        print(f"ok {key} ({x['size']} B, {algo} verified)", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
