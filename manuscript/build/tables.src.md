@@@TABLE origins
title: Origins of BURROW's design choices.
caption: Ideas taken from published work, and what BURROW changed. "Agents" marks choices that came from the agents' design reviews rather than from a single published source.
colspec: p{3.1cm}p{5.3cm}p{6.3cm}
| Design element | Source | What BURROW does |
|---|---|---|
| Score points, cluster them, rank the clusters | P2Rank and PRANK [@krivak2018; @krivak2015] | Same three-stage architecture. Points are volume elements on a grid rather than surface points. Gradient-boosted trees replace random forests. A separate pocket ranker is trained on cross-fitted point scores. |
| Candidate recall as the target | LIGYSIS analysis: fpocket's list contains about 91% of sites, P2Rank's 57% [@utges2024] | Own grid candidates reach 99.3% maximum recall on LIGYSIS before the output floor (at 32.9 pockets per chain). |
| Protein–solvent–protein buriedness | LIGSITE [@hendlich1997; @huang2006] | Used as a cheap first filter that decides which points are kept at full resolution. |
| Buriedness from ray casting, on several scales | PocketPicker [@weisel2007]; multi-scale buriedness in Ghecom [@kawabata2010] | Fibonacci rays at 6–24 Å through a full-atom grid and a backbone+Cβ grid (the latter tolerates side-chain changes). |
| Protrusion | P2Rank's most important feature [@krivak2018] | Kept as one feature among many. |
| Enclosure and polarity as druggability descriptors | SiteMap [@halgren2009]; fpocket druggability score [@schmidtke2010b] | First-hit chemistry: the chemical class of the first atom that each ray hits. |
| Weighted contact number as a flexibility measure | Lin et al. [@lin2008] | Mean weighted contact number around each point. |
| Cross-fitting of first-stage predictions | Double/debiased machine learning [@chernozhukov2018] | Ranker trained on out-of-fold point scores. |
| Site definition, redundancy rule, closest-pocket protocol, top-N+2 metric | LIGYSIS [@utges2024] | Training sites built the same way (fingerprint clustering, relative intersection, cut 0.5); metric and protocol ported and verified against the authors' code. |
| Relevant-ligand labels | BioLiP2 [@zhang2024] | Training labels, including metal ions. |
| Removal of low-confidence segments | AF2BIND (termini and internal runs of ≥ 7 residues with pLDDT < 50) [@gazizov2026] | Used unchanged as a fixed prior in `--af-mode`. |
| Pocket filter at mean pLDDT 70 | HOTPocket; Zillmer and Walther [@carpenter2025; @zillmer2025] | Used as the control baseline that AlphaFold mode had to beat (it did not). |
| Leakage-aware and temporal evaluation | CryptoBench, PocketWatch, PLINDER [@skrhak2025; @portnoi2026; @durairaj2024] | Test union frozen before training; sequence filter with domain-level coverage; fold-disjoint ablation; independent audit. |
| Metal-site candidates; ions in the headline model | Agents (design review); metal-site location prediction, e.g. Metal3D [@durr2023] | Extra candidate points at metal-coordinating side chains; separate `all` and `small` rankers. |
| Pre-registration, Holm correction, cluster bootstrap | Agents (statistics review) [@holm1979] | Frozen claims, endpoints, margins and seeds before any test-set run. |
@@@END

@@@TABLE benchmarks
title: Test benchmarks.
caption: All benchmarks were frozen before any BURROW test prediction. Endpoint codes refer to the pre-registered claims (C, family 1; O, family 2). Sites and units are scored sites and cluster-bootstrap units.
colspec: p{3.0cm}p{4.2cm}p{3.0cm}p{4.0cm}
| Benchmark | Endpoint | Size | Input |
|---|---|---|---|
| LIGYSIS [@utges2024] (C1, C1-strict, C2) | Top-N+2 recall at DCC ≤ 12 Å (closest pocket); DCA ≤ 4 Å; P1K | 2,775 chains, 6,721 non-redundant sites of 6,882 (6,387 with ligand atoms) | Authors' ligand-free representative chains |
| LIGYSIS without ions (descriptive) | As above | 2,003 chains, 4,141 sites | As above |
| COACH420, HOLO4K (mlig) [@krivak2018] (O2a, O2b) | DCA ≤ 4 Å, top-n, P2Rank rules | 379 and 6,874 ligands | P2Rank dataset files |
| PocketWatch test [@portnoi2026] (O3) | Top-N+2 at DCC ≤ 12 Å, authors' mask | 1,566 structures, 1,573 sites | Authors' predicted structures, `--af-mode` (no effect: no pLDDT) |
| CryptoBench test [@skrhak2025] (O4) | Top-N+2 at DCC ≤ 12 Å to the site centre on the apo structure | 222 apo structures, 270 sites | Apo structures |
| AlphaFold set, built here (O5) | Top-N+2 at DCC ≤ 12 Å | 2,526 AFDB v6 human models, 6,502 sites | Full-length models, `--af-mode` |
| Dimer set, built here (O6) | Interface top-N+2 at DCC ≤ 12 Å | 400 assemblies, 807 unique sites (243 at interfaces) | PDB assembly 1 |
@@@END

@@@TABLE prereg
title: Pre-registered test results (seed 0).
caption: Comparator values are percentages; Δ is BURROW minus the comparator in percentage points, with 95% paired cluster-bootstrap CIs. Family 1 comparator: published fpocket+PRANK predictions. Family 2 comparator: the better of P2Rank 2.5.1 and fpocket+PRANK on identical inputs (P2Rank's AlphaFold configuration for O3 and O5). Seeds 1 and 2 give the same decisions as seed 0 in both versions. The O4 point estimate varies most across seeds (v2: −2.6 to +1.1).
colspec: p{1.4cm}p{2.8cm}p{1.6cm}p{2.3cm}p{1.8cm}p{2.3cm}p{1.8cm}
| Endpoint | Benchmark and metric | Comparator | v1 Δ [CI] | v1 decision | v2 Δ [CI] | v2 decision |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | LIGYSIS top-N+2 (DCC 12 Å) | 60.4 | +3.9 [2.0, 6.0] | pass | +12.1 [9.9, 14.2] | pass |
| C1-strict | LIGYSIS top-N+2 (DCA 4 Å) | 31.6 | +22.9 [20.2, 25.9] | pass | +22.6 [20.0, 25.5] | pass |
| C2 | LIGYSIS P1K | 81.7 | +14.0 [10.1, 18.6] | pass (CI > 0) | +14.4 [10.2, 19.0] | pass (CI > 0) |
| O2a | COACH420 DCA4 top-n | 69.4 | +2.4 [−2.7, 7.1] | inconclusive | +1.6 [−3.0, 5.8] | inconclusive |
| O2b | HOLO4K DCA4 top-n | 76.6 | +1.8 [−1.0, 4.4] | non-inferior | +1.6 [−1.2, 4.3] | non-inferior |
| O3 | PocketWatch top-N+2 | 75.8 | −2.7 [−4.9, −0.5] | inconclusive | +0.8 [−1.4, 2.9] | non-inferior |
| O4 | CryptoBench apo top-N+2 | 71.9 | −5.2 [−11.1, 1.0] | inconclusive | −2.6 [−8.6, 3.0] | inconclusive |
| O5 | AlphaFold set top-N+2 | 51.4 | +7.8 [5.3, 10.3] | superior | +13.1 [11.0, 15.7] | superior |
| O6 | Dimer interface top-N+2 | 87.7 | −2.1 [−7.1, 2.8] | inconclusive | −3.7 [−8.7, 0.9] | inconclusive |
@@@END

@@@TABLE agents
title: Agent roles.
caption: Each row is one sub-agent role launched by the PI agent (Claude Code); roles 26–30 belong to the release and manuscript phase. Several roles were served by one long-lived agent that was resumed many times. Opus, Claude Opus 5.5; Sonnet, Claude Sonnet 5.5. S1 Table gives the full roster with outputs and checks.
colspec: p{0.6cm}p{5.6cm}p{1.6cm}p{6.3cm}
| # | Role | Model | Main outputs |
|---|---|---|---|
| – | PI agent (one long session) | Opus | Design, pre-registration amendments, research tree and decision log, lessons, all commits |
| 1 | Literature reviewer (with 2 research sub-agents) | Opus | Literature review |
| 2 | Data acquisition | Opus | Dataset list with checksums and licenses; download and verification scripts |
| 3 | Baselines engineer | Sonnet | P2Rank, fpocket+PRANK and 8 optional tools; all baseline runs; P2Rank retrained on our data |
| 4 | Evaluation engineer | Opus | Metrics harness, LIGYSIS reproduction, cluster bootstrap, gated test runner and ledger |
| 5 | Geometry and HPC engineer | Opus | Geometric primitives (CPU and GPU), exact fast backend, proteome run, ablations, speed |
| 6–11 | Design reviewers, round 1 (external PI; prior work and data; statistics; algorithms and compute; AI-agent failure modes; biophysics and AlphaFold) | Opus ×6 | Six reviews |
| 12–13 | Design reviewers, round 2 (external PI and statistics; data and agent risks) | Opus ×2 | Two reviews |
| 14 | Training and dev set builder | Opus | Leakage-controlled BioLiP2 sets; apo counterparts |
| 15 | AlphaFold test set and AlphaFold training labels | Opus | AlphaFold benchmark and labels |
| 16 | Dimer benchmark builder | Opus | Dimer benchmark and evaluator |
| 17 | Method developer | Opus | BURROW reference implementation, training, presets, v2 and v3 cycles |
| 18 | AlphaFold-mode developer | Opus | AlphaFold-mode mechanisms, ablations, optional ranker |
| 19 | Independent leakage auditor | Sonnet | Audit (found residual leakage; 186 entries removed) |
| 20 | Independent dev-result reviewer | Opus | Review of dev results |
| 21 | Independent test verifier (no project imports) | Opus | Recomputation of headline test numbers |
| 22–23 | Process retrospectives | Sonnet ×2 | "What should we stop doing?" and "what worked well?" |
| 24 | Report writer and number checker | Opus | Report; script that checks every reported number |
| 25 | External report reviewer | Opus | Report review |
| 26 | Release-repository builder | Opus | Release repository (scrubbed) |
| 27 | Figure maker | Opus | Figures |
| 28 | Manuscript writer | Opus | Manuscript and supporting texts |
| 29 | Reference verifier | Opus | Verified bibliography |
| 30 | Manuscript reviewer | Opus | Referee-style review of the draft |
@@@END

@@@TABLE decisions
title: Decisions that affect how the results should be read.
caption: D numbers refer to the decision log in the project's research tree. Each entry gives what was decided, why, and the consequence for interpretation.
colspec: p{3.2cm}p{5.7cm}p{5.8cm}
| Decision | Reason | Consequence |
|---|---|---|
| Ions included in the headline model (D1) | LIGYSIS and the published bar include ion sites; metal sites are common and poorly covered by surface-point methods | Ion sites carry much of the C1 margin; a pre-registered `small` ranker and post-hoc ion-free analyses separate the effect |
| Two rankers, mapped to benchmarks by ligand definition (D1) | Benchmarks differ in whether ions count; mapping fixed before results | Fair comparison on ion-free benchmarks; two models to maintain |
| LIGYSIS-style training labels from BioLiP2 (D1) | Train on the same site definition as the main benchmark; keep all 2,775 LIGYSIS chains for comparability | May favour LIGYSIS over benchmarks with other site definitions |
| Conservative leakage fix (D6) | Independent audit found residual leakage; all 186 flagged entries removed, not only the 8 hard failures | Smaller training set; no full re-audit after the fix |
| Accept failure of the candidate-recall gate G1 (D7) | Gate failed on dev (58.6% vs 88% at ≤ 40 geometric candidates); the pre-specified fallback (fpocket candidates) was worse | Deviation from the design, disclosed; own candidates kept |
| Speed budget (D5): default ≤ about 100 ms per core, at least 5 times faster than P2Rank | Speed was a goal of the prompt | Default: 96 ms median per core, about 1.5 times faster than P2Rank per process; the 5-fold target was missed; the fast preset was added later |
| Suppression radius 10 Å (D7) | Chosen by one-to-one P1K on cross-validation; 12–14 Å would raise top-N+2 by spreading pockets | Avoids tuning to the 12 Å tolerance of the metric |
| Output floor adopted as v2 before scoring v1 (D12, D13) | Dev evidence that the closest-pocket rule gives less credit to long lists; τ chosen on training-set cross-validation | Adopted 70 min after v1 test predictions existed; v1 also reported and passes alone; C1 gain is an output-length effect |
| AlphaFold mode = pruning and soft occlusion with the standard ranker; AlphaFold ranker optional (D8) | The AlphaFold ranker lost recall near disorder on dev | Default unchanged; `--af-ranker` recommended only for AFDB screening (test-informed) |
| v3 interface cycle with a fixed gate (D14, D15) | Motivated by the O6 test result (test-informed, disclosed) | Gate failed; nothing changed |
| No further test version (D16) | Three dev cycles showed saturation on non-ion ranking | Remaining time used for checks, descriptive rows and reviews |
| No default switched on test evidence (D18, D19) | Fast preset and per-region floor evaluated as options only | Fast preset looked better on some test rows but the default was kept |
@@@END
