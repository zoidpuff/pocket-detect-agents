#!/usr/bin/env python
"""THE ONLY entry point for test-set evaluation (PREREG "Test gating").

  python scripts/run_test_eval.py --benchmark ligysis --pred-dir <dir> --method BURROW-all \
      --version v1 --model models/burrow_all_s0.lgb [--model ...] [--final] [--n-boot 1000]

* Refuses to run if tracked files in the git tree are modified/staged (untracked files are ignored).
* Records commit hash, sha256 of every --model file, of src/pocketdetect/{evaluate,ligands,site_eval,
  sites,benchmarks,clusters}.py, of docs/PREREG.md, of the ground-truth table and of the prediction
  directory (sha256 over sorted per-file hashes).
* At most 3 distinct frozen versions may be evaluated on test before the final run (``--final``,
  allowed once per benchmark x method); a 4th non-final version is refused.
* Writes results/test/<version>/<benchmark>/<method>/ and appends a row to results/TEST_LEDGER.md.
Baseline methods (no model file) pass ``--model none``; the ledger records that explicitly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketdetect.benchmarks import REGISTRY, run_benchmark  # noqa: E402
from pocketdetect.seeds import BOOTSTRAP_SEED  # noqa: E402

RESULTS = ROOT / "results"
LEDGER = RESULTS / "TEST_LEDGER.md"
CODE_FILES = ["src/pocketdetect/evaluate.py", "src/pocketdetect/ligands.py", "src/pocketdetect/site_eval.py",
              "src/pocketdetect/sites.py", "src/pocketdetect/benchmarks.py", "src/pocketdetect/clusters.py"]
# every file that influences scoring must be tracked and unmodified (the rest of the tree may be dirty:
# method development continues in parallel; the prediction code is pinned by --code-commit instead)
SCORING_FILES = CODE_FILES + ["src/pocketdetect/structure.py", "src/pocketdetect/ligysis.py", "src/pocketdetect/af_eval.py",
                              "src/pocketdetect/prereg.py", "src/pocketdetect/controls.py", "src/pocketdetect/seeds.py",
                              "scripts/eval_dimer.py", "scripts/run_test_eval.py", "scripts/run_prereg_analysis.py", "docs/PREREG.md"]
LEDGER_HEADER = (
    "# TEST LEDGER — every test-set evaluation (appended automatically by scripts/run_test_eval.py; never edit rows)\n\n"
    "| utc time | version | final | benchmark | method | scoring commit | prediction code commit | model sha256 (12) | scoring code sha256 (12) | PREREG sha256 (12) | "
    "GT sha256 (12) | preds sha256 (12) | missing preds | headline (value [95% CI]) | results dir |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_sha256(d: Path) -> tuple[str, int]:
    files = sorted(p for p in d.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(d)).encode())
        h.update(sha256(p).encode())
    return h.hexdigest(), len(files)


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def gate_check(allow_dirty: bool = False) -> str:
    """Refuse unless every scoring file is tracked and unmodified. Returns the dirty listing."""
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", *SCORING_FILES)
    if dirty and not allow_dirty:
        raise SystemExit("refused: scoring files modified — commit first:\n" + dirty)
    for f in SCORING_FILES:
        if subprocess.run(["git", "ls-files", "--error-unmatch", f], cwd=ROOT, capture_output=True).returncode != 0 and not allow_dirty:
            raise SystemExit(f"refused: {f} is not tracked by git")
    return dirty


def ledger_rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text().splitlines():
        if line.startswith("| 20"):
            c = [x.strip() for x in line.strip("|").split("|")]
            rows.append({"time": c[0], "version": c[1], "final": c[2], "benchmark": c[3], "method": c[4]})
    return rows


def check_versions(version: str, final: bool, benchmark: str, method: str) -> None:
    rows = [r for r in ledger_rows() if r["benchmark"] == benchmark and r["method"] == method]
    if final:
        if any(r["final"] == "yes" for r in rows):
            raise SystemExit(f"a final run for {benchmark}/{method} is already in the ledger")
        return
    versions = {r["version"] for r in rows if r["final"] != "yes"}
    if version not in versions and len(versions) >= 3:
        raise SystemExit(f"refused: {benchmark}/{method} already has 3 non-final test versions {sorted(versions)} (PREREG cap)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=[k for k, b in REGISTRY.items() if b.split == "test"])
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--method", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--model", action="append", required=True, help="model file(s); 'none' for baselines without a model")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--code-commit", default=None, help="commit of the code that produced the predictions (e.g. frozen worktree)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)  # harness self-test only
    ap.add_argument("--results-root", type=Path, default=None, help=argparse.SUPPRESS)  # harness self-test only
    a = ap.parse_args(argv)
    global RESULTS, LEDGER
    if a.results_root is not None:
        if not a.allow_dirty:
            raise SystemExit("--results-root is for the self-test only")
        RESULTS, LEDGER = a.results_root, a.results_root / "TEST_LEDGER.md"
    if not re.fullmatch(r"[A-Za-z0-9._+-]+", a.version) or not re.fullmatch(r"[A-Za-z0-9._+-]+", a.method):
        raise SystemExit("version/method must match [A-Za-z0-9._+-]+")

    dirty = gate_check(a.allow_dirty)
    check_versions(a.version, a.final, a.benchmark, a.method)

    models = {}
    for m in a.model:
        if m == "none":
            models["none"] = "none"
            continue
        p = Path(m)
        if not p.is_file():
            raise SystemExit(f"model file {m} not found")
        models[str(p)] = sha256(p)
    if not a.pred_dir.is_dir():
        raise SystemExit(f"prediction dir {a.pred_dir} not found")
    bench = REGISTRY[a.benchmark]
    code_sha = hashlib.sha256("".join(sha256(ROOT / f) for f in CODE_FILES).encode()).hexdigest()
    rec = {
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git("rev-parse", "HEAD"), "dirty": bool(dirty), "host": socket.gethostname(),
        "version": a.version, "final": a.final, "benchmark": a.benchmark, "method": a.method, "prediction_code_commit": a.code_commit,
        "scoring_files_sha256": {f: sha256(ROOT / f) for f in SCORING_FILES},
        "models": models, "code_sha256": {f: sha256(ROOT / f) for f in CODE_FILES}, "code_sha256_combined": code_sha,
        "prereg_sha256": sha256(ROOT / "docs/PREREG.md"),
        "gt_sha256": (dir_sha256(bench.abspath)[0] if bench.abspath.is_dir() else sha256(bench.abspath)) if bench.abspath.exists() else None,
        "seed": BOOTSTRAP_SEED, "n_boot": a.n_boot, "argv": sys.argv,
    }
    rec["preds_sha256"], rec["n_pred_files"] = dir_sha256(a.pred_dir)
    out = RESULTS / "test" / a.version / a.benchmark / a.method
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refused: {out} already exists (results are never overwritten)")
    info = run_benchmark(bench, a.pred_dir, out, n_boot=a.n_boot, jobs=a.jobs)
    rec["evaluation"] = info
    (out / "run_record.json").write_text(json.dumps(rec, indent=2, default=str))

    def fmt(d):
        return f"{100 * d['value']:.2f} [{100 * d['ci_low']:.2f}, {100 * d['ci_high']:.2f}]" if d["value"] <= 1 else f"{d['value']:.0f}"

    head = "; ".join(f"{k} {fmt(v)}" for k, v in info["headline"].items())
    model_sha = ",".join(v[:12] if v != "none" else "none" for v in models.values())
    if not LEDGER.exists():
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(LEDGER_HEADER)
    row = (f"| {rec['utc']} | {a.version} | {'yes' if a.final else 'no'} | {a.benchmark} | {a.method} | {rec['commit'][:10]}{' (dirty)' if dirty else ''} | "
           f"{(a.code_commit or '')[:10]} | "
           f"{model_sha} | {code_sha[:12]} | {rec['prereg_sha256'][:12]} | {(rec['gt_sha256'] or '')[:12]} | {rec['preds_sha256'][:12]} | "
           f"{info['n_pred_missing']} | {head} | {os.path.relpath(out, ROOT)} |\n")
    with open(LEDGER, "a") as f:
        f.write(row)
    print(row)


if __name__ == "__main__":
    main()
