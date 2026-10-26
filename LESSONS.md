# LESSONS — process rules for running this project efficiently

One-line rules with evidence. Pruned at M2 (2026-10-26 ~20:20) after a fresh-agent retrospective (reviews/process/M2_retro.md).

1. Inspect mounts/tools before planning (evidence: no Java/nvcc/conda; PAE endpoint 403 → plan around what exists).
2. Only the PI commits, explicit paths only; sub-agents never git add/commit/reset (evidence: 8.9k unreachable objects ≈169 MB from an agent's staging; in-progress files swept into early PI commits).
3. One source of truth for constants (seeds in src/pocketdetect/seeds.py; docs reference, never restate) (evidence: DESIGN v1 vs seeds.py mismatch).
4. After a session restart, check git status + disk and resume agents via SendMessage with their file list (evidence: two restarts, 5 agents resumed, no work lost).
5. Launch any job > ~5 min with nohup + log; cap workers (shared 4096-thread user limit) (evidence: detached jobs survived restarts; thread-limit hangs in bench runs).
6. Reproduce published numbers before building on a harness (evidence: LIGYSIS 33/33 variants + P2Rank evaluator to 0.1% made all later comparisons trustworthy).
7. Data/leakage "done" requires an independent re-implementation, not the builder's own verifier (evidence: Track A verify PASS; audit found 7 missed assembly partners + 85 borderline hits → D6).
8. Pre-register gates and endpoints; log every deviation as a dated decision with reason (evidence: G1 failed on BioLiP dev but fallback was worse → documented D7, not silently dropped).
9. Every mechanism/negative claim needs a control (evidence: random-pruning control showed pLDDT pruning ≠ removing atoms; byte-identical no-op check).
10. Select on train cluster-CV, touch dev rarely, never tune a metric's leniency (evidence: NMS 14 Å would inflate LIGYSIS top-N+2 to 72.6 by spreading pockets — rejected via one-to-one P1K).
11. Commit at each agent hand-off and at least hourly; use git/`date` timestamps, not memory (evidence: ~40 uncommitted paths at 20:11; D1/D2 times written ~20 min off).
12. Cap exploratory branches with a kill criterion; optimise feature-*dependent* code only after the freeze (evidence: AF side-chain tolerance −16 pts; backend's feature stage must be re-ported after freeze, while feature-independent parts — trees, NMS — were safe to port early).
13. Run frozen-version test predictions from an isolated git worktree at the freeze commit; development continues on main (evidence: v2 edits to burrow/pipeline.py started minutes after FREEZE v1).
