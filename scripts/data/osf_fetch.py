#!/usr/bin/env python3
"""Recursively download an OSF project's osfstorage files, verifying OSF's sha256.

Usage: python -I osf_fetch.py NODE_ID OUTDIR [--include REGEX] [--exclude REGEX]
REGEX is matched against the OSF materialized path (e.g. /cryptobench/cryptobench-dataset/...).
Writes OUTDIR/MANIFEST.tsv (path, sha256, bytes, retrieved_utc, url). Stdlib only.
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


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def walk(url):
    while url:
        d = get_json(url)
        for x in d["data"]:
            a = x["attributes"]
            if a["kind"] == "folder":
                yield from walk(x["relationships"]["files"]["links"]["related"]["href"])
            else:
                yield a["materialized_path"], a["size"], (a.get("extra") or {}).get("hashes", {}).get("sha256"), x["links"]["download"]
        url = d["links"].get("next")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("node")
    ap.add_argument("outdir")
    ap.add_argument("--include")
    ap.add_argument("--exclude")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    man = os.path.join(a.outdir, "MANIFEST.tsv")
    done = {l.split("\t")[0] for l in open(man)} if os.path.exists(man) else set()
    bad = 0
    for path, size, want, url in walk(f"https://api.osf.io/v2/nodes/{a.node}/files/osfstorage/"):
        if a.include and not re.search(a.include, path):
            continue
        if a.exclude and re.search(a.exclude, path):
            continue
        rel = path.lstrip("/")
        if ".." in rel.split("/"):
            raise SystemExit(f"refusing suspicious path {path}")
        out = os.path.join(a.outdir, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if not (os.path.exists(out) and os.path.getsize(out) == size):
            with urllib.request.urlopen(url, timeout=600) as r, open(out + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            os.replace(out + ".part", out)
        got = sha256(out)
        if want and got != want:
            print(f"CHECKSUM MISMATCH {rel}: {got} != {want}", file=sys.stderr)
            bad += 1
            continue
        if rel not in done:
            with open(man, "a") as m:
                m.write(f"{rel}\t{got}\t{size}\t{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\t{url}\n")
        print(f"ok {rel} ({size} B, sha256 {'verified' if want else 'computed'})", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
