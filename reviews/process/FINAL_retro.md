# Final retro (T+16.5 h), fresh-eyes review

## Stop doing
- Committing "tree:" bookkeeping separately: ~30 of 97 commits (e.g. "tree: time fix", "tree: fix time", "tree: nonion analysis"); fold into the result commit.
- Letting results sit uncommitted again: git status at wrap-up shows dozens of untracked results/dev/*, ablations/*, baselines/p2rank_ours; the M2 retro flagged this and it recurred.
- Spending effort on low-yield branches: 8 optional DL/geometric baselines wrapped (1437c6e), v3 cycle (+0.3..+1.9 vs +3 gate, D15), AF geometry mechanisms equal to a pLDDT<70 filter; give each a written time box.
- Choosing a headline metric whose leniency the method can exploit: v2 output floor (~42 to 6-8 pockets) gave C1 +8 pts while C1-strict was unchanged (verification 8.3); decide list-length policy before the first test run.
- Pre-registering a headline that ions dominate: review 5.3 found ion sites carry ~2/3 of the `all` gain and non-ion is about +5; do composition breakdowns before fixing claims.
- Making decisions from test results mid-stream: D14 v3 targeted O6 after seeing test v1; v1 predictions existed 70 min before FREEZE v2.

## What worked well
- Reproduction gate: 33/33 LIGYSIS variants and P2Rank evaluator to 0.1% made every comparison credible.
- Independent agents: leakage audit (D6, 186 entries dropped), dev review CONFIRMED with caveats, final recomputation matches within 0.2 pt.
- Pre-registration plus honest negatives: G1 deviation (D7), v3 not frozen (D15), apo-dev negative reported, ledger of test versions.
- Controls: shuffled labels -28/-36, fold-disjoint, random pruning, byte-identical no-op, P2Rank retrained on our data (7259810).
- Resilience: 2 restarts and a 4.5 h API pause (D17) lost no work because jobs were detached with ALL_DONE markers.
- Worktree-isolated frozen test runs (D10) plus exactness tests (60/60 CPU/GPU) caught the pLDDT=70 fp32 bug (D11).

## Proposed final LESSONS.md
1. Inspect mounts, tools and real resource caps (container ~79 GB, not 128) before planning (no Java/nvcc; P2Rank OOM at 79 GB).
2. Only the PI commits, explicit paths only; sub-agents never git add/commit/reset (8.9k orphan objects, 169 MB).
3. Commit at each hand-off and at least hourly, one commit per result including its tree note; take times from git/`date` (~40 uncommitted paths at M2 and again at wrap-up; D3 times 20 min off).
4. One source of truth for constants; docs reference, never restate (DESIGN v1 vs seeds.py).
5. Jobs over 5 min run detached (nohup) with logs and completion markers; cap workers (restarts and 4.5 h API pause cost only wall-clock).
6. After a restart, check git status and disk, resume agents via SendMessage with their file lists (5 agents, 0 lost).
7. Kill only PIDs you stored at launch; never pkill -f or ps|grep|xargs kill on a shared node (killed test-v1 job; agent killed own shell 4x).
8. Reproduce published numbers before building on a harness (LIGYSIS 33/33).
9. "Done" for data/leakage needs an independent re-implementation, run before building on it (verify PASS vs audit FAIL, D6).
10. Pre-register gates, endpoints and output-length policy; log every deviation as a dated decision (G1/D7, D13).
11. Every mechanism or negative claim needs a control, and never tune a metric's leniency (shuffled labels, random pruning; NMS 14 Å rejected; v2 floor C1 gain vs unchanged C1-strict).
12. Break headline gains down by composition (ion vs non-ion, apo vs holo) before claiming them (ions ~2/3 of gain).
13. Select on train cluster-CV, touch dev rarely, and do not let test results steer later versions without disclosing it (D7, D14).
14. Time-box exploratory branches with a written kill criterion; optimise feature-dependent code only after the freeze (AF tolerance -16 pts; 586 to 166 ms rework).
15. Run frozen-version test predictions from an isolated worktree, and test exactness at every branch boundary value (D10; Triton bug only at pLDDT exactly 70).
