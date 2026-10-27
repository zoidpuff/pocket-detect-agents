# LESSONS — process rules for running this project efficiently

One-line rules with evidence. Final prune 2026-10-27 ~07:20 after a fresh-agent retrospective (reviews/process/FINAL_retro.md; earlier: M2_retro.md).

1. Inspect mounts, tools and *real* resource caps before planning (evidence: no Java/nvcc/conda; PAE endpoint 403; effective container memory ~79 GB, not 128 — P2Rank training OOM).
2. Only the PI commits, explicit paths only; sub-agents never git add/commit/reset (evidence: 8.9k unreachable objects ≈169 MB from an agent's staging).
3. Commit at each hand-off and at least hourly, one commit per result including its tree note; take timestamps from git/`date`, not memory (evidence: ~40 uncommitted paths at M2; several decision times written 20–50 min off; ~30% of commits were bookkeeping-only).
4. One source of truth for constants (evidence: DESIGN v1 seeds vs src/pocketdetect/seeds.py mismatch).
5. Run every job > ~5 min detached with a log and a completion marker; cap workers on a shared node (evidence: two restarts + a 4.5 h API pause cost only wall-clock; thread-limit hangs).
6. After a restart or pause, check git status + disk and resume agents with their file lists rather than relaunching (evidence: 5 agents resumed twice, zero lost work).
7. Kill only PIDs you stored at launch; never `pkill -f` or `ps | grep | xargs kill` (evidence: a broad pkill killed the test-v1 cryptobench job; an agent killed its own shell 4×).
8. Reproduce published numbers before building on a harness (evidence: LIGYSIS 33/33 variants + P2Rank evaluator to 0.1% made every comparison credible).
9. "Done" for data/leakage requires an independent re-implementation (evidence: builder verify PASS vs independent audit FAIL → 186 entries dropped, D6).
10. Pre-register gates, endpoints *and output-list-length policy*; log every deviation as a dated decision (evidence: G1 deviation D7; the v2 output floor added ~8 pts on C1 while C1-strict was unchanged — decide list length before the first test run).
11. Every mechanism or negative claim needs a control; never tune a metric's leniency (evidence: random-pruning control; byte-identical no-op; NMS 14 Å rejected; shuffled-label −28/−36).
12. Decompose headline gains by composition (ligand class, site type) before claiming them (evidence: ions carried ~2/3 of the `all` gain; drug-like "other" sites +2.2 only).
13. Select on train cluster-CV, touch dev rarely, and disclose any test-informed steering (evidence: phase 3 selected on CV, dev once; D14 v3 targeted O6 after test v1 — disclosed).
14. Time-box exploratory branches with a kill criterion; optimise feature-dependent code only after the freeze (evidence: AF side-chain tolerance −16 pts; v3 +0.3…+1.9 vs +3 gate; backend feature stage re-ported after freeze).
15. Run frozen-version test predictions from an isolated git worktree, and test exactness at every branch boundary value (evidence: v2 edits began minutes after FREEZE v1, D10; Triton fp32 bug only at pLDDT = 70, D11).
