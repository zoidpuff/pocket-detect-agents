# M2 retro (T+5.3 h), fresh-eyes review

## Stop doing
- Leaving results uncommitted: git status shows ~40 untracked/modified paths (results/dev*, scripts/af_dev_*, baselines/*) while the last commit is 20:11; PROMPT requires committed results.
- Letting LESSONS.md go stale: last edit 16:31, yet audit FAIL (18:22), G1 failure (D7), and a 20-min timestamp error (D3) produced no rules.
- Building the audit after the train set is declared done: Track A "verify PASS" 18:10, independent audit found a marginal FAIL at 18:22 and forced D6 (186 entries dropped); gate before declaring done.
- Chasing AF geometric mechanisms that tree shows do not beat a pLDDT<70 filter (prune/soft-occlusion +0.0, side-chain tolerance -16 pts); cap such branches at one experiment.
- Porting a fast backend before features are frozen (backend 18:05, freeze 23:00, CPU still 166 ms vs D5 <=100 ms): expect rework.
- Tracking commits ("tree:" ~12 of 40 commits) and stating times from memory (D3 note); batch tree updates per milestone and use git time.

## What worked
- Reproduction gate: harness matches published LIGYSIS for 33/33 variants and P2Rank's evaluator to 0.1%, which makes later comparisons credible.
- Independent critique: 6 design reviewers gave DESIGN v2/v3 + frozen PREREG; the audit agent caught what the builder's verifier missed (7 partners, 85 borderline hits).
- Restart resilience: 16:18 restart, 5 agents resumed from disk, no lost work; nohup jobs survived.
- Honest negatives with controls: G1 deviation documented, random-pruning control, byte-identical no-op, NMS chosen by one-to-one P1K rather than the inflated top-N+2 (72.6).
- Early de-risking: v0 pipeline on P2Rank training sets before the train set existed passed G1 at 91% (16:35).
- Exactness tests for the backend: 60/60 CPU/GPU identical, rotation invariance 1.5e-6.

## Proposed LESSONS.md (12)
1. Inspect mounts/tools before planning (no Java/nvcc/conda; baselines bring own JDK).
2. Only the PI commits, explicit paths only, never `git add -A` or reset by agents (R1 review: in-progress edits swept in; 8.9k orphan objects, 169 MB).
3. Constants have one source (seeds.py); docs reference it (DESIGN said [0,1,2], code differs).
4. After restart: check git status and disk, resume agents via SendMessage with file lists (16:18: 5 agents, 0 lost).
5. Jobs >5 min run under nohup with logs (16:29 restart: Track A and geometry bench survived).
6. Reproduce published numbers before building on the harness (LIGYSIS 33/33).
7. Any leakage/data check needs an independent re-implementation before "done" (verify PASS vs audit FAIL).
8. Pre-register gates and thresholds; log deviations as dated decisions (G1/D7).
9. Every negative or mechanism claim needs a control (random pruning, no-op identity, shuffled labels).
10. Select on train-CV, touch dev/test rarely (D7); never tune metric definitions to inflate (NMS 14 Å would give 72.6).
11. Commit at each agent hand-off and at least hourly; record git time, not estimated time. (evidence: ~40 uncommitted paths at T+5 h; D3 timestamps 20 min off)
12. Cap exploratory branches (one experiment, written kill criterion) and port/optimise only after the feature freeze (AF side-chain tolerance, 586 ms to 166 ms rework).
