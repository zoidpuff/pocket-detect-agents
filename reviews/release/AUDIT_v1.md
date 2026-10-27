# Independent pre-release audit: release/burrow

Auditor: independent agent. Date: 2026-10-27. Repository: `/cluster/project/beltrao/gmagnus/pocket_detect/release/burrow`
(HEAD `f805331`, one commit, 854 blobs + 207 trees + 1 commit = 1,062 objects, all reachable; no packs, no dangling
or unreachable objects; working tree == HEAD, no untracked or ignored files present).

I did not use `scripts/check_release_clean.py`. My own tools are in `/tmp/claude-548918/audit_release/`:
`scan_blobs.py` (scans every git object via `git cat-file`, gunzips `.gz` blobs, writes `hits_<category>.tsv`),
`links.py` (broken relative markdown links), plus the commands described below. The repo has no `.parquet`,
`.npz`, notebooks, PDFs or images, so there was no binary metadata to inspect. Every gzip header has mtime 0 and no
embedded file name, except the unmodified AFDB example file (upstream header).

## Verdict per category

| # | Category | Verdict |
|---|---|---|
| 1 | Personal data | **FAIL**: one real leak (F1). Everything else is clean or for the PI to decide. |
| 2 | Confidential / restricted data | **PASS**, with documentation fixes (F5, F6) |
| 3 | Hygiene | **PASS**, with minor issues (F7–F10) |
| 4 | Functional sanity | **PASS with a major caveat**: a non-editable install loses its models and the fast backend (F2, F3) |

## Findings

### F1 (MUST FIX): personal identifiers in shipped code
`scripts/check_release_clean.py:27`: `"user": r"gmagnus|magnusson|beltrao",`. This contains the user's cluster
username stem, the user's surname, and the PI/lab name. Lines 28, 39 and 42 also hard-code `ethz`, `euler`,
`eu-login`, `rsyncPDB` and the `jurgjn/euler-vibe` allow-list. The file is in git history (blob in commit `f805331`).
**Fix:** remove the script from the release, or load the identifying patterns from an untracked local file or an
environment variable. Because the history has only one commit, amend or re-create it so the blob is gone, and check
afterwards with `git rev-list --objects --all`.

### F2 (SHOULD FIX): non-editable install cannot find the models
`src/pocketdetect/paths.py:15-17`: `REPO_ROOT = Path(__file__).resolve().parents[2]`, so after `pip install .` the
code looks for models in `<venv>/lib/python3.12/models/`. Without `BURROW_MODELS`, `burrow predict examples/1a28_A.pdb`
fails with `FileNotFoundError: .../venv/lib/python3.12/models/burrow_all_default_s0.json`. The README (lines 72–79)
documents that `-e` is required, so this is documented, but it is fragile. **Fix:** raise a clear error that names
`BURROW_MODELS`, or ship the models as package data or a download.

### F3 (SHOULD FIX): non-editable install silently disables the fast backend
`src/pocketdetect/burrow/fast/validated.json` is not packaged: `pyproject.toml` has no `package-data`, and it is the
only non-`.py` file under `src/`. In the installed wheel, `guard.py:35` cannot find the file, so every stage warns
"reference code changed since validation ... falling back to the reference implementation". Outputs are still
correct (identical to `examples/expected/`), but the speed claims do not hold for such installs:
- 1A28 took 12.2 s;
- `scripts/smoke.py` still prints PASSED, so its "reference == fast" check is vacuous here;
- `pytest` (with `BURROW_MODELS` set) gives **36 failed** / 134 passed: 11 from the guard, plus 25 cascading
  `RuntimeError: Cannot set NUMBA_NUM_THREADS ... (currently have 1, trying to set 64)`.

**Fix:** add `[tool.setuptools.package-data] pocketdetect = ["burrow/fast/validated.json"]`. Also make `smoke.py`
fail when `guard._status()` is not all True.

### F4 (PI decision): every mention of the cluster, platform or third-party users
- `PROMPT.md:23`: "Briefing: you are running inside a Singularity container on the ETH Euler cluster"
- `PROMPT.md:31`: "[redacted: shared data path] on euler has:". This is the only difference from the original
  `/cluster/project/beltrao/gmagnus/pocket_detect/PROMPT.md` (original: `/cluster/work/beltrao/shared on euler has:`).
  The redaction is documented in `README.md:172-173` and `prompts/README.md:9`.
- `PROMPT.md:33`: `rsyncPDB/  - copy of the PDB` (local directory name)
- `AGENT_SETUP.md:6`: "...inside a Singularity container on the ETH Euler cluster" (also: "shared with other users")
- `prompts/04_manuscript_and_repo.md:3`: URL `github.com/jurgjn/euler-vibe/blob/main/home/euler-claude/.claude/skills/...`.
  This is the GitHub username of a third person and a path that implies an Euler home directory. It is a verbatim
  human prompt and a public URL.
- `docs/research_record/RESEARCH_TREE.md:36`: "(jurgjn/euler-vibe)"
- `docs/research_record/RESEARCH_TREE.md:8`: hardware details, "1.7 TB local /tmp NVMe", "internet via proxy works".
  No proxy address is given.
- `docs/PERF.md:3,35,116`: host redacted to `<compute node>` (fine); the node load averages remain.
- Commit timezone `+0100` (CET) in the commit object; harmless.

No other occurrences of ethz, euler, beltrao, gmagnus or magnusson outside F1. The only emails are
`burrow-authors@users.noreply.github.com` and `noreply@anthropic.com`. Nothing else was found:

| Checked for | Result |
|---|---|
| IP addresses | none |
| UUIDs / Claude session IDs / `/tmp/claude-*` | none |
| home directories / absolute paths | none (only `<repo>/data`-style placeholders and `~/.cache/pocketdetect` in `docs/PERF.md:73`) |
| API keys / tokens | none; all `AKIA…` and `PASSW` hits are protein sequences |
| proxy settings | none; the `:3128` hits are residue numbers |
| PIDs | none |
| personal names beyond cited authors | none (the AFDB example lists AlphaFold authors, upstream) |

### F5 (SHOULD FIX): DATA.md and benchmarks/README.md overstate "no third-party structure files"
`DATA.md:3` and `benchmarks/README.md:3-4` say no third-party structures are redistributed. However:
- `examples/AF-P00918-F1-model_v6.cif.gz` is a verbatim AFDB file (CC-BY-4.0; redistribution allowed with attribution).
- `examples/1a28_A.pdb` is derived from the PDB (CC0).

`README.md:147` states both licenses correctly. **Fix:** add "except the two example structures in `examples/`" and
the AFDB attribution to DATA.md.

### F6 (note): third-party data licenses verified
- **LIGYSIS** Zenodo 14645504 and 14033286: `cc-by-4.0`, confirmed from the local Zenodo record JSON. The adapted
  `benchmarks/ligysis/sites.tsv.gz` carries attribution and a change note in `benchmarks/THIRD_PARTY_LICENSES.md`. OK.
- **CryptoBench**: the GitHub LICENSE is MIT, and the OSF node pz4a9 license is MIT (checked via the OSF API). The MIT
  notice is included. OK.
- **PocketWatch**: MIT, confirmed from the local HF dataset card (`license: mit`). OK.
- **P2Rank datasets** (no license): only IDs, chain, upstream path, sha256 and the `(mlig)` ligand codes (derived
  labels from Binding MOAD) are shipped, plus BURROW predictions and summary scores. No structures. OK.
- **BioLiP2**: only entry IDs, plus derived relevant-ligand annotations in `benchmarks/dimer/*`. This is disclosed
  in `DATA.md:65-67`. OK.
- **AFDB / PDB / UniProt / SIFTS**: IDs, sequences, derived tables. OK.
- **scPDB**: not present. Its only mentions are literature text and file names in the LIGYSIS Zenodo checksum list.
  For transparency: `docs/research_record/RESEARCH_TREE.md:67` records that scPDB was downloaded during the research,
  while `scripts/data/README.md:72` says "not downloaded". Consider rewording to "not downloaded by these scripts".
- **HOTPocket / AF2BIND / PocketMiner**: no data shipped.
- No third-party baseline prediction files are shipped (only summary CSV/JSON).

### F7 (minor): prompt provenance inconsistencies
- `prompts/` orders the follow-ups extension → questions → finalize. The main repo's
  `manuscript/FOLLOWUP_PROMPTS.md` orders them questions → extension → finalize.
- `prompts/02_questions.md` joins a multi-line message into one line, yet is labelled "verbatim".
- `AGENT_SETUP.md:6` cites `manuscript/FOLLOWUP_PROMPTS.md`, which is not in the release.

The PI should confirm the true order.

### F8 (minor): references to files that are not shipped
`AGENT_SETUP.md:19-36` cites `reviews/*`, `results/review_dev/REVIEW.md`, `scripts/check_report_numbers.py` and
`manuscript/*`. `docs/README.md:17` explains this for docs, but AGENT_SETUP does not. No broken markdown links were found.

### F9 (minor): .gitignore
The `.gitignore` is sensible: caches, venvs, `data/`, `*.npz`/`*.pkl`/`*.log`, archives. Suggest adding `.env`,
`.cache/` and `*.nbi`/`*.nbc`. The `!models/` and similar negations have no effect, which is harmless.

### F10 (info): hygiene checks that passed
- Largest file is 8.8 MB (`benchmarks/ligysis/sites.tsv.gz`); nothing is over 50 MB.
- No `__pycache__`, `.venv`, egg-info, build output or logs are committed.
- No absolute paths in code, configs, model JSON `meta`, or `run_record.json`/`eval_info.json`; all are repo-relative.
- `LICENSE` (MIT, "BURROW authors"), `pyproject.toml` (`license = MIT`) and `README.md:187-191` are consistent.
- `benchmarks/SHA256SUMS`, `models/SHA256SUMS` and `predictions/MANIFEST.tsv` (11/11) all verify.
- The README states the AI-agent nature at the top (`README.md:3-12`): "**This method and its benchmarking were
  produced by AI agents, as an experiment in testing how well agents can run a bioinformatics project
  autonomously.** ... The human did not write code, choose methods or edit results. ... Treat the results
  accordingly: they were checked by independent agent reviewers and verifiers (described below), but not by human
  experts."

## Functional sanity (fresh venv, Python 3.12.3, `pip install './clone[test]'` non-editable, from a git clone in /tmp)
- `burrow predict` with `BURROW_MODELS` set produced output identical (`cmp`) to `examples/expected/` for all three
  cases: 1a28_A (default), 1a28_A (`--preset fast`) and AF-P00918 (`--af-mode`). Running on the whole `examples/`
  directory also worked. All of these used the reference fallback (F3).
- Without `BURROW_MODELS`: FileNotFoundError (F2).
- `python scripts/smoke.py`: PASSED in 15 s, but with the fast backend disabled (F3).
- `pytest` from source (`PYTHONPATH=src`, equivalent to editable): 170 passed, 62 skipped (GPU/data), 0 failed in 4
  repeated runs, including one from a cold numba cache. One early run gave 25 numba-thread failures that I could
  not reproduce. The same suite run under the research `.venv` also showed 28 such failures once, so the numba
  thread-count handling may be order- or environment-sensitive (`cli.py:187` sets `NUMBA_NUM_THREADS` in-process).
- `pytest` against the installed wheel: 36 failed (F3).
