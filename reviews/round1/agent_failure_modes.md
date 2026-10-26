# Review round 1 — AI-agent failure modes (DESIGN v1)

Reviewer: agent-failure-modes specialist. Scope: process safeguards only; no changes to goals.
Observed state (15:29): `docs/EVALUATION.md` does not exist yet; `results/` empty; uncommitted edits by a concurrent agent to `src/pocketdetect/ligands.py` (label extraction) and an untracked download script; `seeds.py` has `SEEDS=(20261026,1,…,9)`, `BOOTSTRAP_SEED=20261026`, but DESIGN §3 says `SEEDS=[0,1,2]`, bootstrap seed 0.

## A. Test-set leakage through iteration
1. **MUST — LIGYSIS is the primary test set *and* the obvious tuning target.** The BioLiP dev split is distribution-shifted from LIGYSIS, so agents will be tempted to "just check" LIGYSIS. Carve a fixed **LIGYSIS-dev** slice (e.g. 15% of LIGYSIS UniProts, cluster-disjoint at 30% from the rest, chosen with BOOTSTRAP_SEED, IDs committed) for development; report O1 only on the remainder. Same for COACH420/HOLO4K: no dev use at all.
2. **MUST — test access gate.** A single entry point `scripts/run_test_eval.py` that refuses to run unless the model artifact's sha256 + git commit are clean and appends to `results/TEST_LEDGER.md` automatically (not by hand). Budget: ≤ 2 test evaluations per objective (one interim at E3, one final at E9). Any change made after seeing a test number is labelled "post-hoc" in REPORT.md.
3. **MUST — ligand stripping test.** Inputs to our method and to baselines must contain no ligand/water/HETATM-derived atoms (buriedness would trivially "see" the ligand). Unit test: predictions on a holo file are identical with and without ligand records. Also assert B-factor is ignored outside `--af-mode`.
4. **MUST — independent leakage audit.** After the training set is final, a different agent recomputes max seq-id (MMseqs2 `easy-search`, not the clustering output) of every test chain vs training+dev, plus PDB-ID and UniProt overlap (incl. AF-mode training UniProts vs O5 test UniProts). Script asserts zero violations; output committed.

## B. Silent metric / label drift
5. **MUST — freeze evaluation and label code before E1.** Write `docs/EVALUATION.md`, then record the sha256 of `evaluate.py`, `ligands.py` and the active `LigandRules.as_dict()` in every results JSON. The test-gate script refuses mismatches. Changes after freeze require: PI approval, decision note in RESEARCH_TREE, re-run of all previously reported numbers (method *and* baselines).
6. **MUST — reconcile seeds now.** One source of truth (`seeds.py`); DESIGN must quote it, not a different list.
7. **SHOULD — co-report DCA 4 Å and pocket size** (residues, volume) next to DCC 12 Å top-N+2 for every method, so a ranker that inflates pockets is visible (proxy optimisation).

## C. Broken pipelines that look fine
8. **MUST — baseline reproduction gate.** Our re-run P2Rank on COACH420/HOLO4K must land within ±2 pp of published 72.0/68.6 (top-n DCA 4 Å), and our harness must reproduce LIGYSIS authors' code on their published predictions to ≤0.5 pp. If not, stop and debug before any method comparison.
9. **MUST — no silent drops.** Every eval prints and stores `n_expected / n_parsed / n_predicted / n_failed`; a missing or crashed prediction counts as a miss, never excluded. Assert identical structure sets across all methods in a comparison table.
10. **MUST — controls on every eval run** (cheap): random-surface-point (negative) and ligand-centroid oracle (positive) alongside each table; shuffled-label model once at E2. Flag if method < random+5 pp or oracle < 95%.
11. **SHOULD — CPU/GPU equivalence test** on ≥ 50 real structures (pocket centres within 0.5 Å, rank correlation ≥ 0.99) in CI, since GPU speed claims are only valid if outputs match.
12. **SHOULD — training sanity**: log positive-point fraction, #chains, #sites per split; feature importance checked for leak-prone features (chain count, residue index, any HETATM-derived field).

## D. Fabrication, copy-paste, overclaiming
13. **MUST — provenance column.** Every number in tables is tagged `re-run (results/…json@commit)` or `published (citation)`. Published LIGYSIS predictions downloaded from Zenodo are "published", never presented as our runs.
14. **MUST — independent recomputation of headline numbers.** Before REPORT.md, a fresh agent that wrote no method code recomputes O1/O2/O5/O7 headline numbers from raw prediction CSVs with the frozen harness, and `scripts/check_report_numbers.py` asserts every number in REPORT.md matches a committed results file.
15. **MUST — pre-registered verdict.** The "clearly beat" rule (§1) is applied mechanically by script; a failed objective is reported as failed. Speed comparisons state whether JVM startup / I/O is included for P2Rank (batch vs per-file).

## E. Coordination, context, time
16. **MUST — file ownership.** One owner agent per module (`evaluate.py`/`ligands.py`: harness owner only; `DESIGN.md`, `RESEARCH_TREE.md`, `LESSONS.md`: PI only). Others propose diffs. Commits only by the PI or owners with explicit paths — never `git add -A` (current tree has another agent's uncommitted label-code edits that would be swept in).
17. **SHOULD — shared `.venv` lock.** Commit `uv.lock`; package installs only via the PI, to avoid one agent breaking another's environment.
18. **SHOULD — sub-agent report contract.** Every report states: commit hash, commands run, result file paths, what was *not* verified. PI checks one number per report against the file.
19. **MUST — timeboxed checkpoints** (CET, 27 Oct): 00:00 E3 end-to-end on dev or fall back to E2 + geometric pockets; 06:00 E5/E6 started else drop DL baselines and fold-disjoint ablation to COULD; 10:00 code freeze; 11:00 final test evals; 12:30 independent verification; 14:00 REPORT.md. DL baselines (GrASP, DeepPocket, PUResNet, VN-EGNN) hard-capped at 1 h each install+run.
20. **SHOULD — stop rule for iteration**: abandon a branch after two consecutive dev changes whose improvement lies inside the dev bootstrap CI; record as [-] in the tree.
21. **COULD — 30-min heartbeat** line in RESEARCH_TREE (time, running agents, next checkpoint) to survive context resets.
