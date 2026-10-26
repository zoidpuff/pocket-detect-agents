#!/usr/bin/env python3
"""Download the files under a path prefix of a GitHub repo at a pinned commit.

Usage: python -I github_subtree_fetch.py OWNER/REPO COMMIT PREFIX OUTDIR
Each file is verified against its git blob SHA-1 (from the tree API) and its sha256 is
recorded in OUTDIR/MANIFEST.tsv. Git-LFS pointer files are reported. Stdlib only.
"""
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request


def main():
    repo, commit, prefix, outdir = sys.argv[1:5]
    prefix = prefix.strip("/")
    tree = json.load(urllib.request.urlopen(
        f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1", timeout=120))
    if tree.get("truncated"):
        print("WARNING: tree listing truncated", file=sys.stderr)
    os.makedirs(outdir, exist_ok=True)
    man = os.path.join(outdir, "MANIFEST.tsv")
    done = {l.split("\t")[0] for l in open(man)} if os.path.exists(man) else set()
    bad = 0
    n = 0
    for t in tree["tree"]:
        if t["type"] != "blob" or not (t["path"] == prefix or t["path"].startswith(prefix + "/")) and prefix:
            continue
        rel = t["path"]
        if ".." in rel.split("/"):
            raise SystemExit(f"suspicious path {rel}")
        out = os.path.join(outdir, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(rel)}"
        if not (os.path.exists(out) and os.path.getsize(out) == t["size"]):
            with urllib.request.urlopen(url, timeout=600) as r, open(out + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            os.replace(out + ".part", out)
        data = open(out, "rb").read()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if blob != t["sha"]:
            # raw.githubusercontent serves LFS objects resolved; the tree holds the pointer
            print(f"BLOB SHA MISMATCH {rel} (LFS object?)", file=sys.stderr)
            bad += 1
        if rel not in done:
            with open(man, "a") as m:
                m.write(f"{rel}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\t"
                        f"{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\t{url}\n")
        n += 1
    print(f"{repo}@{commit} {prefix}: {n} files, {bad} blob mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
