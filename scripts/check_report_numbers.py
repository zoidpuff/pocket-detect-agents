#!/usr/bin/env python3
"""Check every source-tagged number in REPORT.md against the committed results files.

A tagged number is a number immediately followed by an HTML comment (invisible when rendered):

    72.47<!--src:v2:family1{endpoint=C1;seed_index=0;comparator=published:fpocket+PRANK}/a-->

Selector forms (after ``src:``):

* ``json:PATH#QUERY`` – JSON file; aliases ``v1:QUERY`` / ``v2:QUERY`` for
  ``results/test/v{1,2}/summary.json``. QUERY is a ``/``-separated path; a list step may carry a
  filter ``name{key=value;key=value}`` that must select exactly one element (values compared as
  strings, ``seed_index=0`` etc.). Example: ``descriptive/secondary{benchmark=ligysis;
  method=BURROW[burrow_all_default_s0];metric=recall_max}/value``.
* ``md:PATH::SECTION::ROW::EXTRACT`` – text/markdown file. SECTION (regex, may be empty) sets the
  search start (first matching line); ROW (regex) selects the first matching line after it.
  EXTRACT is ``cN.k`` (k-th number, 0-based, in table cell N, 0-based, leading pipe ignored),
  ``nK`` (K-th number on the line) or ``gK`` (regex group K of ROW).
* ``calc:EXPR`` – arithmetic over nested selectors written ``{{selector}}``.

Any selector may end with ``*F`` (multiply by F, e.g. ``*100`` for fractions shown as %).

Numbers are compared with tolerance = half a unit of the displayed last digit (0.05 for one decimal,
0.5 for integers, 0.005 for two decimals).
By default source files are read from git ``HEAD`` (i.e. the *committed* results); use
``--worktree`` to read the working tree instead. Exit code 0 iff every tag resolves and matches.
TODO markers in the report are counted and listed (they do not fail the check unless --no-todo).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "v1": "results/test/v1/summary.json",
    "v2": "results/test/v2/summary.json",
}
NUM_RE = re.compile(r"[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TAG_RE = re.compile(
    r"(?P<num>[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(?P<pct>%?)\s*<!--src:(?P<sel>.*?)-->",
    re.S,
)


class SelectorError(Exception):
    pass


def to_float(tok: str) -> float:
    return float(tok.replace("−", "-").replace(",", "").replace("+", ""))


def decimals(tok: str) -> int:
    return len(tok.split(".", 1)[1]) if "." in tok else 0


class Source:
    def __init__(self, ref: str | None):
        self.ref = ref

    @lru_cache(maxsize=None)
    def text(self, path: str) -> str:
        if self.ref is None:
            p = ROOT / path
            if not p.exists():
                raise SelectorError(f"file not found: {path}")
            return p.read_text()
        try:
            out = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"{self.ref}:{path}"],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError:
            raise SelectorError(f"not committed at {self.ref}: {path}")
        return out.stdout

    @lru_cache(maxsize=None)
    def json(self, path: str):
        return json.loads(self.text(path))


def json_query(obj, query: str):
    step_re = re.compile(r"^(?P<name>[^{}]*)(?:\{(?P<filt>[^{}]*)\})?$")
    for step in [s for s in query.split("/") if s != ""]:
        m = step_re.match(step)
        if not m:
            raise SelectorError(f"bad query step {step!r}")
        name, filt = m.group("name"), m.group("filt")
        if name:
            if isinstance(obj, list):
                obj = obj[int(name)]
            elif isinstance(obj, dict):
                if name not in obj:
                    raise SelectorError(f"key {name!r} not found")
                obj = obj[name]
            else:
                raise SelectorError(f"cannot index {type(obj).__name__} with {name!r}")
        if filt is not None:
            conds = [c.split("=", 1) for c in filt.split(";") if c.strip()]
            if not isinstance(obj, list):
                raise SelectorError("filter applied to non-list")
            hits = [e for e in obj if all(str(e.get(k.strip())) == v.strip() for k, v in conds)]
            if len(hits) != 1:
                raise SelectorError(f"filter {filt!r} matched {len(hits)} elements")
            obj = hits[0]
    if isinstance(obj, bool) or not isinstance(obj, (int, float)):
        raise SelectorError(f"query did not end on a number (got {obj!r})")
    return float(obj)


def md_query(text: str, section: str, row: str, extract: str) -> float:
    lines = text.splitlines()
    start = 0
    if section:
        sre = re.compile(section)
        for i, ln in enumerate(lines):
            if sre.search(ln):
                start = i
                break
        else:
            raise SelectorError(f"section {section!r} not found")
    rre = re.compile(row)
    for ln in lines[start:]:
        m = rre.search(ln)
        if not m:
            continue
        if extract.startswith("g"):
            return to_float(m.group(int(extract[1:])))
        if extract.startswith("n"):
            nums = NUM_RE.findall(ln)
            return to_float(nums[int(extract[1:])])
        if extract.startswith("c"):
            cell, _, k = extract[1:].partition(".")
            cells = ln.strip().strip("|").split("|")
            nums = NUM_RE.findall(cells[int(cell)])
            return to_float(nums[int(k or 0)])
        raise SelectorError(f"bad extract {extract!r}")
    raise SelectorError(f"row {row!r} not found")


def resolve(sel: str, src: Source) -> float:
    sel = sel.strip()
    factor = 1.0
    m = re.search(r"\s*\*\s*([0-9.]+)$", sel)
    if m and not sel.startswith("calc:"):
        factor = float(m.group(1))
        sel = sel[: m.start()]
    if sel.startswith("calc:"):
        expr = sel[5:]
        vals = []

        def sub(mm):
            vals.append(resolve(mm.group(1), src))
            return f"({vals[-1]!r})"

        expr_num = re.sub(r"\{\{(.+?)\}\}", sub, expr)
        if not re.fullmatch(r"[0-9eE.+\-*/() ]+", expr_num):
            raise SelectorError(f"unsafe calc expression {expr_num!r}")
        return float(eval(expr_num, {"__builtins__": {}}, {}))  # noqa: S307 (numbers/operators only)
    head, _, rest = sel.partition(":")
    if head in ALIASES:
        return factor * json_query(src.json(ALIASES[head]), rest)
    if head == "json":
        path, _, q = rest.partition("#")
        return factor * json_query(src.json(path), q)
    if head == "md":
        parts = rest.split("::")
        if len(parts) != 4:
            raise SelectorError("md selector needs PATH::SECTION::ROW::EXTRACT")
        path, section, row, extract = parts
        return factor * md_query(src.text(path), section, row, extract)
    raise SelectorError(f"unknown selector type {head!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="REPORT.md")
    ap.add_argument("--ref", default="HEAD", help="git ref to read result files from (default HEAD)")
    ap.add_argument("--worktree", action="store_true", help="read result files from the working tree")
    ap.add_argument("--no-todo", action="store_true", help="fail if the report still contains TODO markers")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    report = (ROOT / args.report).read_text()
    src = Source(None if args.worktree else args.ref)
    n_ok, failures = 0, []
    files = set()
    for m in TAG_RE.finditer(report):
        tok, sel = m.group("num"), m.group("sel")
        line = report.count("\n", 0, m.start()) + 1
        shown = to_float(tok)
        tol = 0.5 * 10 ** (-decimals(tok)) + 1e-6
        try:
            val = resolve(sel, src)
        except Exception as e:  # noqa: BLE001
            failures.append((line, tok, sel, f"unresolved: {e}"))
            continue
        for part in re.findall(r"(?:json:|md:)([^#:]+)", sel):
            files.add(part)
        for a in ALIASES:
            if re.search(rf"(^|\{{\{{){a}:", sel):
                files.add(ALIASES[a])
        if abs(val - shown) <= tol:
            n_ok += 1
            if args.verbose:
                print(f"ok   L{line}: {tok} == {val:.4g}   [{sel}]")
        else:
            failures.append((line, tok, sel, f"source value {val:.4f} (|Δ| {abs(val - shown):.3f} > {tol:.3f})"))

    todos = [(report.count("\n", 0, t.start()) + 1, report[t.start(): t.start() + 90].splitlines()[0])
             for t in re.finditer(r"TODO", report)]
    where = "working tree" if args.worktree else f"git {args.ref}"
    print(f"check_report_numbers: {n_ok + len(failures)} tagged numbers, {n_ok} match, "
          f"{len(failures)} fail; {len(files)} source files read from {where}")
    for line, tok, sel, msg in failures:
        print(f"FAIL L{line}: {tok}  [{sel}]  -> {msg}")
    if todos:
        print(f"{len(todos)} TODO marker(s):")
        for line, txt in todos:
            print(f"  L{line}: {txt}")
    bad = bool(failures) or (args.no_todo and todos) or (n_ok == 0)
    print("RESULT:", "FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
