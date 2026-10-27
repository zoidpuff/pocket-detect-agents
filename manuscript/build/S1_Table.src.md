# S1 Table. Agent roster

@@@TABLE roster
title: Agent roster.
caption: All agent roles in the BURROW project (Table 4 of the main text lists roles 1–25; roles 26–27 produced this manuscript), with the model used, the main files each role produced (paths in the project repository), and how the role's output was checked by an agent that did not produce it. Opus, Claude Opus 5.5; Sonnet, Claude Sonnet 5.5.
colspec: p{0.6cm}p{3.0cm}p{1.3cm}p{5.0cm}p{4.0cm}
| # | Role | Model | Main outputs | Independent check |
|---|---|---|---|---|
| – | PI agent | Opus | DESIGN.md, docs/PREREG.md amendments, RESEARCH_TREE.md, LESSONS.md, all commits | Design reviews (6–13); process retrospectives (22–23) |
| 1 | Literature reviewer (2 research sub-agents) | Opus | LITERATURE.md | Design reviewers for prior work (7, 12) |
| 2 | Data acquisition | Opus | DATA.md; download and verification scripts; MMseqs2 and Foldseek set-up | Checksums; leakage auditor (19) |
| 3 | Baselines engineer | Sonnet | P2Rank 2.4.2/2.5.1, fpocket 4.0.3/4.2.3, fpocket+PRANK and 8 optional tools; baseline runs; P2Rank retrained on our data; apo-dev; PocketMiner and CryptoBench-holo analyses | Reproduction gate against published LIGYSIS rows; test verifier (21) |
| 4 | Evaluation engineer | Opus | Evaluation harness and metrics; LIGYSIS reproduction; cluster bootstrap; gated test runner and ledger; pre-registered analysis | Reproduction of 33/33 published variants; test verifier (21) with no project imports |
| 5 | Geometry and HPC engineer | Opus | Geometric primitives (CPU and GPU); exact fast backend; proteome run; ablations; speed work | Exactness tests against the reference; dev reviewer (20) |
| 6–11 | Design reviewers, round 1 | Opus ×6 | reviews/round1/*.md (external PI; prior work and data; statistics; algorithms and compute; AI-agent failure modes; biophysics and AlphaFold) | Changes mapped in DESIGN.md §8 |
| 12–13 | Design reviewers, round 2 | Opus ×2 | reviews/round2/*.md (external PI and statistics; data and agent risks) | Changes in DESIGN.md §9 |
| 14 | Training and dev set builder | Opus | Leakage-controlled BioLiP2 training and dev sets; apo counterparts (docs/TRAINSET.md) | Leakage auditor (19): marginal fail, 186 entries removed |
| 15 | AlphaFold test set and AlphaFold training labels | Opus | AlphaFold benchmark (docs/AF_BENCHMARK.md); AlphaFold training and dev labels | Test verifier (21) for O5 |
| 16 | Dimer benchmark builder | Opus | Dimer benchmark and evaluator (docs/DIMER_BENCHMARK.md) | Verification script; external report reviewer (25) found a stratum bug |
| 17 | Method developer | Opus | BURROW reference implementation, training, presets, v2 and v3 cycles, packaging (docs/METHOD.md) | Dev reviewer (20); test verifier (21) |
| 18 | AlphaFold-mode developer | Opus | AlphaFold-mode mechanisms, ablations, optional AlphaFold ranker (docs/AF_MODE.md) | Pre-registered pLDDT-filter control; dev reviewer (20) |
| 19 | Independent leakage auditor | Sonnet | results/audit/LEAKAGE_AUDIT.md | – |
| 20 | Independent dev-result reviewer | Opus | results/review_dev/REVIEW.md | – |
| 21 | Independent test verifier | Opus | results/verify_test/VERIFY.md (all headline values within 0.2 points) | – |
| 22–23 | Process retrospectives | Sonnet ×2 | reviews/process/M2_retro.md, FINAL_retro.md | – |
| 24 | Report writer and number checker | Opus | REPORT.md; scripts/check_report_numbers.py | External report reviewer (25) |
| 25 | External report reviewer | Opus | reviews/report/REPORT_REVIEW.md | – |
| 26 | Manuscript writer | Opus | This manuscript (manuscript/) | Submitting authors |
| 27 | Figure maker | [to be confirmed] | manuscript/figures/ (every plotted value read from committed result files) | Submitting authors |
@@@END

@@TABLE roster@@
