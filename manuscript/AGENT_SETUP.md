# Agent setup used to run the project (source for the Methods section)

Written by the PI agent from its own session record. Times are CET. Git history (main repository) is the authoritative timeline.

## Framing
The whole project — literature review, design, data curation, method development, benchmarking, and the written report — was carried out by AI agents (Claude Code; models Claude Opus 5.5 and Claude Sonnet 5.5) running autonomously on one compute node (64-core AMD EPYC 9554, 4× RTX 4090, shared with other users; effective memory cap ≈79 GB) inside a Singularity container on the ETH Euler cluster. A human (the user) wrote the initial prompt (PROMPT.md), later granted a 3–4 h extension, asked three follow-up questions (multimers, GPU use, remaining ideas), asked whether the work was finalized, and requested this manuscript (verbatim follow-ups: manuscript/FOLLOWUP_PROMPTS.md). Agent briefs were written by the PI agent and are not archived as files. The human did not write code, choose methods, or edit results.

## Roles
- **PI agent** (Claude Opus 5.5, one long session): planned, delegated, reviewed and integrated; owned DESIGN.md, PREREG.md amendments, RESEARCH_TREE.md (research tree + decision log D0–D19), LESSONS.md, all git commits. Did not write method code.
- **Sub-agents** (each a fresh Claude Code agent with its own context, launched with a written brief; could be resumed with follow-up messages). Model per role in brackets.

| # | Role | Model | Main outputs |
|---|---|---|---|
| 1 | Literature reviewer (used 2 research sub-agents) | Opus | LITERATURE.md |
| 2 | Data acquisition | Opus | DATA.md, download/verify scripts, MMseqs2/Foldseek |
| 3 | Baselines engineer | Sonnet | P2Rank 2.4.2/2.5.1, fpocket 4.0.3/4.2.3, fpocket+PRANK + 8 optional tools; baseline runs on all test sets; P2Rank retrained on our data; apo-dev; PocketMiner/CryptoBench-holo analyses |
| 4 | Evaluation engineer | Opus | I/O + metrics harness, LIGYSIS reproduction, cluster bootstrap, gated test runner + ledger, pre-registered analysis, descriptive/post-hoc rows |
| 5 | Geometry/HPC engineer | Opus | geometric primitives (CPU+GPU), exact fast backend, proteome run, ablations, speed work |
| 6–11 | Design reviewers, round 1: external PI; prior work & data; statistics; algorithms & compute; AI-agent failure modes; biophysics & AlphaFold | Opus ×6 | reviews/round1/*.md |
| 12–13 | Design reviewers, round 2: external PI + statistics; data + agent risks | Opus ×2 | reviews/round2/*.md |
| 14 | Training/dev set builder (Track A, A2) | Opus | leakage-controlled BioLiP2 train/dev; apo counterparts |
| 15 | AF test set + AF training labels (Tracks C, C2) | Opus | AF benchmark; AF train/dev labels |
| 16 | Dimer benchmark builder (Track D) | Opus | dimer benchmark + evaluator |
| 17 | Method developer | Opus | BURROW reference implementation, training, presets, v2/v3 cycles, packaging |
| 18 | AF-mode developer | Opus | AF mode mechanisms, ablations, optional AF ranker |
| 19 | Independent leakage auditor | Sonnet | results/audit/LEAKAGE_AUDIT.md (found residual leakage → 186 entries removed) |
| 20 | Independent dev-result reviewer | Opus | results/review_dev/REVIEW.md |
| 21 | Independent test verifier (no project imports; used LIGYSIS authors' code) | Opus | results/verify_test/VERIFY.md |
| 22–23 | Process retrospectives ("what should we stop doing / what worked") | Sonnet ×2 | reviews/process/*.md |
| 24 | Report writer + number checker | Opus | REPORT.md, scripts/check_report_numbers.py |
| 25 | External report reviewer | Opus | reviews/report/REPORT_REVIEW.md |
| 26 | Release-repository builder | Opus | release/burrow (scrubbed MVP repo) |
| 27 | Figure maker (used the `dataviz` skill) | Opus | manuscript/figures |
| 28 | Manuscript writer (used the edit-for-clarity skill) | Opus | manuscript/main.*, S1–S2 Text |
| 29 | Reference verifier | Opus | verified bibliography |
| 30 | Manuscript reviewer | Opus | manuscript/REVIEW.md |

## Process rules (enforced by the PI)
- Design by critique: DESIGN v1 → 6 independent reviewers → v2 → 2 reviewers → v3; claims, endpoints, comparators, margins and multiplicity frozen in docs/PREREG.md (sha256 recorded) before any test-set run; amendments A1–A3 dated and made before the relevant predictions existed.
- Reproduction gates before comparisons: harness vs P2Rank's own evaluator (COACH420/HOLO4K) and vs the published LIGYSIS table (33/33 method variants); our P2Rank/fpocket re-runs within ±2 points of published.
- Independent re-implementation for anything a result depends on: leakage audit, dev-result review, test-number recomputation, report review.
- Test discipline: test sets only via a gated script (clean scoring code, model sha256, append-only ledger); ≤3 frozen versions before the final run; dev/train-CV for all selection; predictions for frozen versions generated from isolated git worktrees.
- File ownership per agent; only the PI commits (explicit paths); detached jobs with logs for anything > 5 min.
- Research tree with status legend and dated decisions as the PI's memory; LESSONS.md pruned at milestones after fresh-agent retrospectives.

## Timeline (CET; from git commit times)
- 2026-10-26 14:54 start; 15:26 LITERATURE.md + DESIGN v1; 15:35 DESIGN v2 (6 reviews); 15:39 DESIGN v3 + PREREG frozen (2 reviews); 15:48 / 16:46 / 20:39 PREREG amendments A1–A3 (each before the relevant predictions existed).
- ~16:18 and ~16:29 two session restarts (all agents resumed from disk; no work lost).
- 16:34 LIGYSIS reproduction gate passed (33/33 variants); 18:10 train/dev built; 18:22 independent leakage audit FAIL (marginal) → 186 entries removed (D6).
- 20:10 first BioLiP2 dev results; 21:29 FREEZE v1; 23:09 FREEZE v2 (adopted before any v1 test score was computed; v1 prediction files already existed).
- 2026-10-27 00:27 test v1 scored; 01:14 test v2 scored; 01:41 v3 (interface) cycle negative.
- ~02:40–07:10 API usage limit paused all agents (detached jobs completed).
- 07:10–10:00 descriptive rows, report, external review, fast preset.
- 11:31–12:35 user-granted extension: speed engineering, isolated-chain dimers, per-region output rule (negative vs gate).
- Afterwards: manuscript and release repository.

## Notable agent failure modes observed (and caught)
- Builder's own leakage check passed while an independent audit found residual leakage (7 assembly partners missed; 85 borderline hits).
- A GPU kernel received a float threshold as fp32: outputs differed only for atoms at pLDDT exactly 70; caught by an exactness test, fixed, AF-mode test predictions regenerated on CPU.
- A broad `pkill -f` killed another agent's test job; a `ps|grep|kill` pattern killed an agent's own shell 4×.
- An agent's staging left 169 MB of unreachable git objects.
- A stratum filter bug made a descriptive O6 table use all sites instead of interface sites (found by the external report reviewer).
- Timestamps written from memory were 20–50 min off (git history used instead).
- The output-score floor (v2) improved the headline metric mainly by shortening output lists, a property of the LIGYSIS closest-pocket rule; flagged by reviewers and reported.
