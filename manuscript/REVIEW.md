# Referee report: "BURROW: a fast protein pocket finder and ranker, designed, built and benchmarked end to end by AI agents"

Reviewer role: PLOS Computational Biology referee (structural bioinformatics and ML benchmarking), also checking how the AI-agent research is reported.
Files reviewed: manuscript/main.md, S1_Text.md, S2_Text.md, S1_Table.md and all 8 figure PNGs.
Ground truth used: REPORT.md, results/test/v{1,2}/SUMMARY.md, results/test/v2/DESCRIPTIVE.md, results/test/v2/POSTHOC.md, docs/PREREG.md, docs/PERF.md, RESEARCH_TREE.md, manuscript/AGENT_SETUP.md, manuscript/FOLLOWUP_PROMPTS.md, PROMPT.md, plus spot checks in docs/METHOD.md, docs/TRAINSET.md, docs/DIMER_BENCHMARK.md, docs/AF_BENCHMARK.md, DESIGN.md, results/dev/biolip/{PHASE2,V2_CYCLE}.md and results/validation/ligysis_reproduction.md.

## Overall assessment

This is an unusually careful and honest manuscript. Almost every headline number traces to REPORT.md (which a script checks against committed results), and the main caveats (ion share, output length, the timing of v2, the pLDDT-filter equivalence, inconclusive apo and interface results) are disclosed. I spot-checked about 150 numbers in main.md against REPORT.md and the result files. All but the items listed below matched.

The remaining problems are of four kinds:
1. A few factual errors: the pre-registered rule for C1-strict in the Fig 3 caption, the shuffled-label control, the speed target in S1 Text, and the agent roster and counts.
2. One substantive overclaim. C1-strict (DCA4) is presented as the "robust evidence", but on non-ion sites the DCA4 margin is only +3.1 points. Most of the +22.6 is probably ion sites, like the C1 margin.
3. Stale or placeholder content: S2 Text §9, S1 Table role 27, references 15/19/20/21, the repository URL, and missing funding and competing-interests statements.
4. Missing details on the agent setup that the user's request (FOLLOWUP_PROMPTS item 4) and good reporting practice for agent research require.

Recommendation: minor-to-moderate revision. None of the issues changes a pre-registered decision.

---

## MUST fix (before submission)

**M1. Fig 3 caption misstates the pre-registered rule for C1-strict (factual error).**
- Text (Fig 3 caption): "C1 and C1-strict required Δ ≥ +2 with a CI above 0 for every seed and both comparators."
- Source: docs/PREREG.md family 1, item 2: C1-strict only requires that the CI of Δ excludes 0. The +2-point bar applies to C1 only. The manuscript's own Statistics section states this correctly ("for C1-strict, CI > 0").
- The figure also extends the dashed "+2 bar" line through the two C1-strict rows. Shorten it so it covers only the C1 rows.
- Replacement: "C1 required Δ ≥ +2 points with a CI above 0, and C1-strict required a CI above 0, for every seed and both comparators. C2 was a non-inferiority test at −3."

**M2. The shuffled-label control is described wrongly (factual error).**
- Text (Controls and ablations): "A model trained on shuffled labels fell to the level of evenly spaced surface pockets (LIGYSIS top-N+2 −28.4 points against v1)."
- Source: v1 64.3 − 28.4 = 35.9%. The LIGYSIS test controls (REPORT §4.7) are surface tiling 18.3%, random surface 28.9% and a single pocket at the protein centroid 37.0%. So the shuffled model is at the centroid level, about twice the tiling level. REPORT says "about the surface-tiling or centroid level".
- Replacement: "A model trained on shuffled labels fell to 35.9% LIGYSIS top-N+2 (−28.4 points against v1). This is about the level of a single pocket placed at the protein centroid (37.0%) and well above evenly spaced surface pockets (18.3%)."

**M3. The "robust evidence" (DCA4) is also dominated by ion sites (overclaim / missing caveat).**
- Text (Output-length section): "We regard C1-strict, which the floor does not change, as the robust evidence."
- Text (Discussion): "The robust evidence is localisation: the DCA4 recall is 22.6 points higher than fpocket+PRANK's, and the floor does not change it."
- Source: REPORT §4.2, post-hoc LIGYSIS_NI, `small` ranker. On non-ion sites the DCA4 top-N+2 margin is +3.1 [1.5, 4.8] against fpocket+PRANK (v2 s0; +3.2 to +3.5 across seeds) and +6.7 [5.2, 8.4] against P2Rank. fpocket+PRANK's DCA4 is 48.2% on non-ion sites and 31.8% overall, so its DCA4 on ion sites must be very low. The AF set shows the same pattern (ion DCA4: BURROW 50.2 vs P2Rank-AF 5.3).
- So the floor cannot explain the DCA4 margin, but the ions can. The text presents this metric as the margin that survives every caveat, which overstates it.
- Replacement (Discussion): "The localisation margin is robust to the output floor (DCA4 +22.6 points against fpocket+PRANK, unchanged from v1), but it is not robust to ligand class. On LIGYSIS without ion sites (post hoc, `small` ranker) the DCA4 margin is +3.1 [1.5, 4.8] against fpocket+PRANK and +6.7 [5.2, 8.4] against P2Rank."
- Make the matching change in the Output-length section, and add one clause to the Abstract (see M4).

**M4. The Abstract quotes the v2 number without the v1 number or list lengths.**
- The headline 72.5% comes from v2. v2 was adopted 70 min after the v1 test prediction files existed, and its +8.2-point gain over v1 is an output-length effect. The Abstract says only "Part of the gain came from shorter output lists". A reader of the Abstract alone cannot judge the size of that part.
- Replacement for the two sentences that follow "On LIGYSIS (2,775 chains, 6,721 sites)":
  "BURROW's top-N+2 recall was 72.5% (version 2, 9.1 pockets per chain), against 60.4% for the best published method (fpocket+PRANK, 20.9 pockets per chain) and 52.6% for P2Rank. Version 1, which was frozen before any test predictions existed, reached 64.3%. The difference between the versions comes from shorter output lists, which score better under the benchmark's closest-pocket rule. Success within 4 Å of ligand atoms was 54.2% against 31.6% in both versions. Most of the margin came from ion sites (+29.5 points); on non-ion sites the gain was +3.0 to +3.5 points (+3.1 at 4 Å)."
- Check that the Abstract stays at or under 300 words. It is now about 294 words, so trim elsewhere, e.g. "We describe the agent setup, its decisions and the errors it caught".

**M5. The agent counts and the roster contradict the source and each other (agent-reporting accuracy).**
- Box: "about 25 sub-agents". Methods: "Table 4 lists the 25 roles". S1 Table: 27 roles, with role 26 = manuscript writer and role 27 = figure maker, model "[to be confirmed]".
- Source: manuscript/AGENT_SETUP.md lists 30 roles plus 2 research sub-agents of the literature reviewer: 26 release-repository builder, 27 figure maker (Opus, dataviz skill), 28 manuscript writer (Opus), 29 reference verifier (Opus), 30 manuscript reviewer (Opus).
- Fix:
  - (a) Box: "a principal-investigator (PI) agent and about 30 sub-agents (25 roles during the project and 5 for the release repository and manuscript)".
  - (b) Table 4: add rows 26–30, or add the caption sentence "Roles 26–30 (release repository, figures, manuscript, reference check, manuscript review) are listed in S1 Table."
  - (c) S1 Table: renumber to match AGENT_SETUP.md. Replace "[to be confirmed]" with "Opus". Add the release builder, reference verifier and manuscript reviewer, and change the caption from "roles 26–27 produced this manuscript" to "roles 26–30".
  - (d) Methods "Manuscript": "This manuscript was written by a sub-agent from the verified report and result files. Other sub-agents made the figures, checked the references and reviewed the draft. A separate sub-agent built the release repository."

**M6. S2 Text §2 and §9 are out of date and contain a placeholder.**
- §2: "The exact wording of the follow-up messages was not saved in the repository (see Section 9)."
- §9: "[The verbatim text of these messages should be added here by the submitting authors from the session transcript.]"
- Source: manuscript/FOLLOWUP_PROMPTS.md now holds all four human follow-up messages verbatim, plus the two content-free "Continue" restarts. The main text already says that S2 Text gives "a record of the follow-up requests".
- Fix: paste FOLLOWUP_PROMPTS.md verbatim into §9 and delete the placeholder.
  - §2 replacement: "The verbatim follow-up messages are reproduced in Section 9."
  - Add to the list of human actions in §2: "asked whether the work was finalized".
  - Note that agent-to-agent briefs and session transcripts were not archived (see S3).

**M7. Placeholders and missing PLOS CB statements.**
- Author list, affiliations and corresponding author are missing. Author contributions use "[to be completed]"; PLOS requires CRediT roles.
- PLOS CB requires **Funding** and **Competing interests** statements. Both are absent. A **Data Availability** statement must give a final URL and should give an archived DOI (e.g. Zenodo) for the code, models and predictions. "https://github.com/<to-be-published>/burrow" is a placeholder.
- An ethics statement is not needed, but add "No human or animal data were used."
- References with placeholder titles or authors: [15] (PocketWatch/zPocket, authors without initials, title missing), [19] (HOTPocket), [20] (Zillmer & Walther), [21] (AF2BIND). [32] (Clark, Martin & Mahowald 2026, arXiv:2610.29162) should be checked against the source in manuscript/skill/SKILL.md, which matches. [4] should use its DOI (10.1186/s13321-015-0059-5) instead of a PMC ID.
- Add a short running title (≤ 70 characters), e.g. "BURROW: agent-built pocket detection".

**M8. S1 Text §16 understates how far the speed target was missed (factual).**
- Text: "the default preset took 96.0 ms (median, excluding parsing; 90th percentile 264.9 ms), so the target was missed on larger chains."
- Source: DESIGN.md N1 target "≤ 70 ms/structure/core". The median is 96.0 ms, which is above 70 ms, so the target was missed for the typical chain, not only for larger ones. Only `--preset fast` (28.4 ms median) meets it. D5 also set "≥ 5× faster than P2Rank", but per process the default is 1.5× faster.
- Replacement: "the default preset took 96.0 ms (median, excluding parsing; 90th percentile 264.9 ms), so it missed the target at the median. Only the fast preset, added after the test runs, met it (28.4 ms)."
- In the main text, add this to the list of unmet goals in the Discussion (see S1).

---

## SHOULD fix

### (1) Factual or traceability issues

| # | Exact text | Problem | Correct value / source | Fix |
|---|---|---|---|---|
| F1 | Fig 2 caption: "For these, no CIs or pocket counts were computed" (published predictions) | The figure shows a CI and a pocket count (20.9) for *published* fpocket+PRANK. | REPORT §4.2: published fpocket+PRANK pockets/chain 20.9; family-1 CIs exist against it | "For these, except fpocket+PRANK (the C1 comparator), no CIs or pocket counts were computed." |
| F2 | Benchmark design: "reproduced all 33 published LIGYSIS method variants to within 0.5 points"; Fig 2 caption: "to within 0.4 points" | Both are true but read as a contradiction: 0.5 is the gate and 0.38 is the largest observed deviation (DeepPocket-Segmented). | results/validation/ligysis_reproduction.md | Use "to within 0.4 points (gate: 0.5)" in both places. |
| F3 | Controls: "Results changed by amounts within noise on 8 of 9 test rows" | S2 Fig C plots 10 test rows (it adds AF-set DCA4). | REPORT §1 defines the 9 rows (8 benchmarks, LIGYSIS twice) | Either drop the AF-set DCA4 row from S2 Fig C, or say "9 of 10 rows". |
| F4 | AF section: "only 95 sites have pLDDT below 70" | The scored pLDDT strata hold 13 + 76 = 89 sites (REPORT §4.3); 95 is the count at benchmark build (docs/AF_BENCHMARK.md). | | "only 95 benchmark sites (89 scored) have mean pLDDT below 70". |
| F5 | Training data: "The independent audit found 7 …, 92 …, and one UniProt overlap. All 186 flagged entries were removed" | 7 + 92 + 1 = 100, not 186. The other entries were flagged under lower tiers (D6). Table 5 adds "8 hard failures" without explanation. | RESEARCH_TREE D6; results/audit/LEAKAGE_AUDIT.md | "… one UniProt overlap (8 hard failures in total). The agents removed every entry that any tier of the audit flagged (186 entries, 2.6%), which left …" |
| F6 | "removed 41% of training inputs"; elsewhere "6,475 training … chains" | Readers cannot see that the 41% refers to 8,184 *inputs* (representative chains plus assembly inputs), not 6,475 chains. | REPORT §4.7 (8,184 → 4,839) | "which removed 41% of the 8,184 training inputs (representative chains and assemblies)". |
| F7 | AF section: "A model trained without ion labels reached 78.2%" | This is the CHEN11-only ablation: 251 structures from a different dataset. "Without ion labels" hides a 30-fold smaller and different training set. | REPORT §4.4, §4.7 | "The CHEN11-only ablation model (251 P2Rank training structures, no ion labels) reached 78.2%, which suggests, but does not show, that …" |
| F8 | Table 3 caption: "Seeds 1 and 2 give the same decisions, except that seed-to-seed variation on O4 spans −2.6 to +1.1." | Not an exception: O4 is inconclusive for every seed. The table also has no v1 decision column, although v1 and v2 differ on O3 (inconclusive vs non-inferior). | REPORT §4.1 | Add a "v1 decision" column. Caption: "Seeds 1 and 2 give the same decisions as seed 0 in both versions. The O4 point estimate varies most across seeds (v2: −2.6 to +1.1)." |
| F9 | Runtime: "repeated sessions differed by up to about 1.5-fold" | The example in REPORT is P2Rank 270 vs 209 ms (1.29×). I could not trace 1.5 to a pair of measurements in docs/PERF.md. | REPORT §6.7 | Name the pair that gives 1.5×, or write "by up to about 1.3-fold (P2Rank 270 vs 209 ms)". |
| F10 | Runtime: "The human AlphaFold proteome (20,550 proteins) ran in 129 s on three GPUs in AlphaFold mode." | This was the frozen **v1** default model, measured before the T1 speed-up (REPORT §4.4). | | Append "(frozen v1 model, before the final speed work)". |
| F11 | Box and AI-contribution statement: "asked three follow-up questions" | The human also asked "Are we done then? Is this finalized?" (FOLLOWUP_PROMPTS item 3). | | "asked four follow-up questions (about multimers, GPU use, remaining ideas and whether the work was finished)". |

### (2) Overclaiming or missing caveats

- **O1. Discussion, first paragraph.** "It clearly beats the published methods on LIGYSIS, with recall, atom-level localisation and precision all better under pre-registered tests."
  - P1K was pre-registered as non-inferiority only. Comparisons with the deep-learning methods are descriptive (no paired CI).
  - Replacement: "On LIGYSIS it passed all three pre-registered tests against fpocket+PRANK, the best published method: superiority in recall at 12 Å and at 4 Å, and non-inferiority in precision. The precision CI also excluded zero. Against other published methods the comparison is descriptive."
- **O2. "This precision was pre-registered only as non-inferiority, but it was superior."**
  - Replacement: "P1K was pre-registered as a non-inferiority test; its CI [10.2, 19.0] also excludes zero, but superiority was not a pre-registered test."
- **O3. "calibrated learned ranking" / "comparable across proteins".**
  - The section title, Abstract and Table 1 claim calibration, but no calibration analysis (reliability curve, ECE) is shown. The only evidence is pooled P1K.
  - Either show a reliability plot on dev, or write "a learned ranking trained so that its scores can be pooled across proteins (evidence: pooled P1K)". Change the heading to "high-recall candidates and a learned ranking".
- **O4. PocketWatch and AlphaFold mode.**
  - Table 2 and the Abstract present O3 as a test of AlphaFold-mode input. But PocketWatch models carry B-factor 100 everywhere, so AF mode was inert (REPORT §1, §4.4).
  - Add to Table 2 (O3 input) and to the Secondary-endpoints paragraph: "PocketWatch models carry no pLDDT (B-factor 100), so `--af-mode` had no effect there."
- **O5. The "optional ranker that uses pLDDT descriptors raised P1K to 73.9%."**
  - Add the cost: "but lowered PocketWatch top-N+2 by 3.4 points and P1K by 14.6 points, where no pLDDT is available."
- **O6. "The geometric mechanisms therefore add hallucination control, which the filter also provides. They do not add recall."**
  - This contradicts the +1.0 [0.1, 2.0] gain over naive input two sentences earlier.
  - Replacement: "The geometric mechanisms remove hallucinated pockets, which the filter also does, and they add no recall beyond the filter."
- **O7. Max-recall comparison in the Discussion.** "the candidate set has 99.3% maximum recall before the floor, against 57.7% for P2Rank".
  - Max recall grows with list length: v1 output 32.9 pockets per chain, P2Rank about 5.
  - Add "(at 32.9 pockets per chain, against about 5 for P2Rank; fpocket reaches 91% at 20.9)".
- **O8. Dimer section.** Readers will compare "84.0%" (assembly, site level) with "69.6%" (chain mode, chain-copy level) and then with "lost little (−1.2)".
  - Say in the main text that 69.6% has a different denominator (634 chain-copy rows), and that the −1.2 is a site-level comparison ("found by any isolated chain").
  - Also note that only 9 of the 243 interface sites are ions (docs/DIMER_BENCHMARK.md). O6 is therefore effectively a non-ion test, which fits the small non-ion margins elsewhere.
- **O9. Unmet goals in the Discussion.**
  - The list of unmet goals omits two speed targets: the ≤ 70 ms/core target was missed by the default at the median, and the self-set target of "≥ 5× faster than P2Rank" (D5) was missed (1.5× per process).
  - Add: "The default preset also missed the speed target set in the design (≤ 70 ms per structure per core; median 96 ms). Only the fast preset met it."
- **O10. Generalisation from one run.**
  - Author summary: "The study shows what agent-run research can deliver and where human oversight is still needed." Replacement: "This single run shows what one agent-run project delivered and where human oversight would have helped."
  - Discussion: "The weaknesses were also typical of fast research." Unsupported; delete it or write "These weaknesses also occur in human projects that run under time pressure."
- **O11. Methods.** "Anything that a result depended on was re-implemented by another agent."
  - Too broad: the method, the dimer evaluator and the benchmarks themselves were not re-implemented.
  - Replacement: "Four checks were done by agents that had not produced the checked work: the leakage audit (re-implemented), the review of dev results, the recomputation of headline test numbers (without project code) and the review of the report."
- **O12. Human verification of the manuscript.**
  - PLOS policy on AI tools expects the authors to state how AI-generated content was checked. The box says the human "did not … edit text".
  - The manuscript's own numbers have not been run through scripts/check_report_numbers.py; only REPORT.md has. Run the checker (or an equivalent) on main.md and the SI files.
  - Add one sentence to the AI-contribution statement on what the human authors verified before submission.

### (3) Internal inconsistencies

- **I1. Terminology for the chemistry feature.** "enclosure by chemistry" (Results, Table 1) and "first-hit chemistry" (Fig 1, Methods, S2 Fig) name the same feature. Use one term, preferably "first-hit chemistry".
- **I2. Per-process and per-core timings.** These are kept apart correctly in the main text and Fig 5. Do not import RESEARCH_TREE's "≈ 6.6×" or the "≈ 1.9× per core" claim, which compare BURROW per core against P2Rank per process. No change is needed; flagged as a check passed.
- **I3. Order of SI citations.** S2 Text is cited (box, Introduction) before S1 Text, and S2 Fig (Controls) before S1 Fig. PLOS asks for SI files in order of first citation. Swap the numbers, or cite S1 Fig before S2 Fig.
- **I4. "we" and "the agents".** The text switches between "we pre-registered", "the agents found" and "our evaluation code". Since all the work was done by agents and the authors are humans, define once (e.g. at the end of the Introduction): "'We' refers to the agent team."
- **I5. Table 4 placement.** It is first cited in Results ("The agent-run process") but placed in Methods. Move it, or cite it first in Methods.
- **I6. Decision IDs.** Methods says "20 dated decisions (D0–D19)" and S2 says "D0–D19 for the project, D20 for the manuscript". These are consistent. REPORT.md's header still says D0–D17, but the manuscript does not cite it. OK.

### (4) PLOS CB format

- Abstract: about 294 words (limit 300). OK, but tight; see M4.
- Author summary: about 195 words (range 150–200). OK.
- Section order (Abstract, Author summary, Introduction, Results, Discussion, Materials and methods, Supporting information, Acknowledgments, References) matches the PLOS LaTeX template.
- The "How this work was produced" box: PLOS CB research articles have no boxes. Keep the AI-contribution statement after the author list, and move the box text into the first paragraph of the Introduction or into the Methods. Keep the flag at the very top; the user explicitly asked for this.
- Figures: captions follow the "Fig N. Title. Legend." pattern. Submit TIFF/EPS files at ≥ 300 dpi. Fig 1 and Fig 2 captions repeat Methods text; they can be cut by about one third.
- In S2 Fig panels A and B, the printed labels for the A3 arrows overlap and are hard to read. Move them or stack them.
- SI captions have the correct form ("S1 Fig. Title. Legend"). S1 Table caption: fix as in M5.
- Data availability (M7): PLOS requires that the data underlying every figure be available. Deposit figure_data.csv with the release.

### (5) Writing rules (edit-for-clarity SKILL.md): sample of 47 sentences

I checked 47 sentences: Abstract 6, Author summary 4, Introduction 6, Results 20, Discussion 6, Methods 6, SI 3 (excerpts below). **28 had no violation.** The 19 with a violation or a precision problem are listed with rewrites. The text generally follows the guide well: it has few denominal verbs or inanimate subjects with human verbs, and keeps "that" in relative clauses. The main residual problems are stacked nouns (pattern 5), coined phrases (pattern 1) and long subject–verb gaps (pattern 6).

Sentences with no violation (abbreviated): Abstract "Finding and ranking ligand-binding pockets…", "Claims and statistical tests were fixed…"; Author summary "Many computer programs predict pockets…", "Test proteins often resemble…"; Introduction "We report the agents' method…", "Geometric methods find concave regions…", "Examples are LIGSITE…", "Their analysis also showed that fpocket's candidate list…", "Low-confidence disordered regions can pack…"; Results "BURROW removes everything except protein atoms…", "Each grid point stands for…", "Of the remaining points…", "Extra candidate points are placed…", "The mapping from benchmark to ranker was fixed…", "The smallest difference was +11.6 points…", "Two properties explain much…", "Each site is credited only to…", "Long output lists therefore lose credit.", "The standard ranker is used unchanged.", "BURROW takes any number of chains…", "A missing partner chain therefore mostly reduces…"; Discussion "Because the baselines were not given a tuned floor…", "Where does the gain come from?", "Human oversight would be most useful at three points…", "Fold-disjoint retraining is evidence…"; Methods "Each sub-agent was a fresh Claude Code agent…", "The ranker is trained with cross-entropy on the target…", "τ was chosen by cross-validation…".

| # | Section | Original | Pattern / problem | Rewrite |
|---|---|---|---|---|
| W1 | Abstract | "BURROW places candidate volume elements on a 1 Å grid and describes each point by …" | 5 (stacked nouns: "candidate volume elements") | "BURROW places candidate points on a 1 Å grid, each standing for a small volume of space, and describes each point by …" |
| W2 | Abstract | "Per process, BURROW took 142 ms per structure against 209 ms for P2Rank" | Ambiguous "per process" | "Run as a single process over 200 structures, including start-up and file input and output, BURROW took 142 ms per structure and P2Rank 209 ms" |
| W3 | Author summary | "The study shows what agent-run research can deliver …" | Overgeneralisation | See O10 |
| W4 | Introduction | "BURROW uses volume-element candidate points, buriedness from ray casting on full-atom and backbone-only grids, …" | 5 ("volume-element candidate points") | "BURROW uses candidate points that each stand for a small volume, buriedness from rays cast through …" |
| W5 | Results | "We call this enclosure by chemistry: it describes what a ligand at that point would touch, and it is a cheap approximation of desolvation." | 1 (coined phrase); unsupported claim | "We call this feature first-hit chemistry. It records which kinds of atoms a ligand at that point would contact, as a rough proxy for the cost of desolvation." |
| W6 | Results | "Together, this makes its scores comparable across proteins." | Unclear referent "this"; unsupported (see O3) | "These two choices are meant to make scores comparable across proteins; pooled P1K (below) tests this." |
| W7 | Results | "Confidence intervals (CIs) are 95% paired cluster-bootstrap intervals over 30% sequence-identity clusters." | 5 | "Confidence intervals (CIs) are 95% intervals from a paired bootstrap that resamples clusters of chains with ≥ 30% sequence identity." |
| W8 | Results | "… but 70 minutes after the v1 test prediction files had been written." | 5 ("v1 test prediction files") | "… but 70 minutes after the files with the v1 predictions on the test sets had been written." |
| W9 | Results | "This precision was pre-registered only as non-inferiority, but it was superior." | 2 (precision "was pre-registered"); overclaim | See O2 |
| W10 | Results | "The PocketWatch decision was narrow: its non-inferiority p-value (0.003) was below the Holm threshold (0.005) by a margin close to the 0.001 resolution of 1,000 resamples." | 5, 6 | "The PocketWatch decision was narrow. The p-value for non-inferiority was 0.003 and the Holm threshold 0.005. With 1,000 resamples, p-values are resolved only to 0.001." |
| W11 | Results | "In 403 of its misses, a pocket ranked beyond N+2 was closer to the site." | Precision | "In 403 of its 417 misses, the pocket closest to the site lay within 12 Å but was ranked beyond N+2." (V2_CYCLE.md) |
| W12 | Results | "on apo cryptic sites, v1 found …" | "apo cryptic sites" undefined at this point | "on cryptic sites in the apo development set (CryptoBench training folds), v1 found …" |
| W13 | Results | "For one process that includes start-up and file input and output, BURROW's default preset took 142 ms per structure, P2Rank 2.5.1 in batch mode 209 ms, and fpocket 4.2.3 605 ms." | 6 (long opening phrase); ellipsis | "We ran each tool as one process over the 200 chains and divided the total time, including start-up and file input and output, by 200. BURROW's default preset took 142 ms per structure, P2Rank 2.5.1 (batch mode) took 209 ms and fpocket 4.2.3 took 605 ms." |
| W14 | Results | "It then lowers the opacity of the remaining atoms in proportion to their pLDDT …" | Imprecise (the ramp is clipped) | "It then makes atoms with pLDDT below 80 partly transparent, linearly down to full transparency at pLDDT 40, so that …" |
| W15 | Results | "Side-chain tolerance, which makes side-chain atoms partly transparent, reduced recall on dev (for example −10.9 points at λ = 0.25 on all side chains) and is off by default." | 6; λ undefined in the main text | "Side-chain tolerance reduced recall on dev and is off by default. It makes side-chain atoms partly transparent; at an opacity factor λ = 0.25 on all side chains, recall fell by 10.9 points." |
| W16 | Results | "… predictions of every method are collapsed by the same label-free rule." | 1 ("label-free rule") | "… for every method, predicted pockets whose residue sets overlap by a Jaccard index of 0.5 or more are merged, by a rule that does not use the site labels." |
| W17 | Results | "The 4.5 h pause, caused by a usage limit of the API, cost only wall-clock time, because long jobs ran detached and wrote their state to disk." | 6; 1 ("ran detached") | "An API usage limit paused all agents for 4.5 h. The pause cost only wall-clock time, because long jobs ran as background processes that wrote their state to disk." |
| W18 | Discussion | "The weaknesses were also typical of fast research." | 1 ("fast research"); unsupported | See O10 |
| W19 | Methods | "Metal candidates are placed 1.95–2.35 Å from a side-chain coordinator that has at least two coordinating atoms (His ND1/NE2, Cys SG, Asp/Glu O, Met SD, backbone O) within 2.6 Å, with no other atom closer than 2.4 Å." | 6; ambiguous (a "side-chain coordinator" includes backbone O; it is unclear whether "within 2.6 Å" refers to the point or the atom) | Check against docs/METHOD.md §4, then e.g.: "A metal candidate is a point 1.95–2.35 Å from a coordinating atom (His ND1/NE2, Cys SG, Asp or Glu carboxylate O, Met SD or backbone O). The point must have at least two coordinating atoms within 2.6 Å and no other atom closer than 2.4 Å." |

Other precision issues found while sampling:
- Methods: "The geometric value of a point, b, is the mean blocked fraction at 10 Å, 16 Å and the longest length." In the default preset the longest length is 16 Å, so 16 Å is counted twice. State this ("the mean over 10, 16 and 16 Å in the default preset; 10, 16 and 24 Å in the accurate preset") or correct it.
- SI: S2 Text §2 "The exact wording … was not saved" is outdated (M6). S1 Text §16 "missed on larger chains" is wrong (M8). S1 Text §4 "the level of evenly spaced surface pockets" (14.6 on dev): give the dev tiling value it is compared with, or delete the phrase.

### (6) Missing important decisions or methods

- **D-a. How the hyperparameters were chosen.** No values are justified in the main text. Examples: the 1 Å grid, 30 rays, ray lengths 6/10/16 Å, buriedness thresholds 0.40–0.85, score thresholds 0.15–0.50, b ≥ 0.7 for scoring, the 6 Å and 12 Å split lengths, and 250×63 and 300×15 trees. Add one sentence: "Hyperparameters were set on cross-validation over the P2Rank training set (CHEN11) and train-CV, under the speed budget of decision D5; dev was used only for confirmation (PHASE2.md, PHASE3.md)."
- **D-b. Why gradient-boosted trees** rather than random forests (P2Rank) or a neural network. One sentence is enough: speed, tabular features, and the prompt's warning about deep-learning overfitting.
- **D-c. Pre-registered analyses not run.** These are not disclosed in the manuscript:
  - the UniProt-level bootstrap sensitivity analysis (PREREG "Fixed rules"; REPORT §4.3 says it was not run);
  - the sensitivity analysis on the common chain subset for published predictions (PREREG family 1).
  - Add both to the Statistics or Amendments paragraph.
- **D-d. The AF-mode default (D8)** is not in Table 5. Add: "AlphaFold mode = pruning + soft occlusion with the standard ranker; AlphaFold ranker optional (D8) | the AF ranker lost recall near disorder | default unchanged; `--af-ranker` recommended only for AFDB screening (test-informed)".
- **D-e. The speed budget (D5)** and its outcome are not in Table 5 (see O9).
- **D-f. Literature credit (requested by the user).** Table 1 credits the main sources. Credit is missing for:
  - weighted contact number (cite e.g. Lin et al. 2008, Proteins);
  - Fibonacci sphere sampling of ray directions;
  - cross-fitting (Chernozhukov et al. 2018) for the cross-fitted ranker;
  - prior metal-site prediction from coordination geometry (e.g. MIB, Metal3D or BioMetAll). Table 1 currently attributes "metal-site candidates" to the agents alone.
- **D-g. Reporting of the agent setup.** Add to Methods or S2 Text:
  - the Claude Code version and the dates the models were used;
  - whether any skills, tools or MCP servers were enabled (S1 Table says the figure maker used a `dataviz` skill);
  - compute and token or cost usage, if available;
  - that agent briefs and session transcripts were not archived, so individual delegation steps cannot be audited (stated only in FOLLOWUP_PROMPTS.md);
  - the number of human messages (4, plus 2 automatic restarts).
  - These are the minimum needed for readers to judge or reproduce an agent-run study.
- **D-h. Training-set composition.** Give the number of sites and the ion share (14,073 sites, 37.7% ion-only, before the audit fix; docs/TRAINSET.md). It is central to the ion argument.

### (7) Do the figures match the text?

| Figure | Matches text? | Issues |
|---|---|---|
| Fig 1 | Yes | Box labels "first-hit chemistry" vs text "enclosure by chemistry" (I1). |
| Fig 2 | Yes; values match REPORT (72.5, 54.2, 96.1, 9.1, 20.9, 32.9) | Caption contradicts the plotted CI and count for published fpocket+PRANK (F1). "0.4" vs "0.5" (F2). |
| Fig 3 | Values match Table 3 and REPORT for all seeds | The dashed +2 bar covers the C1-strict rows and the caption states the wrong rule (M1). |
| Fig 4 | Values match (ion +29.5, nucleotide +4.7, other +2.2, non-ion v1 −4.6 / v2 +3.0; AF ion +32.1, non-ion +1.9, nucleotide +0.1, other +2.3) | Legend says "P2Rank (published)" while Fig 2 and the text use the re-run P2Rank 2.5.1 (52.6). Both are fine, but say which in the caption. |
| Fig 5 | Values match docs/PERF.md (48/142/271/209/605 ms; 545/646, 171/411, 88/338 structures/s) | The x-axis is per process, consistent with the text. Panel B's fpocket point is fpocket alone (not +PRANK); the caption says so implicitly; make it explicit. |
| Fig 6 | Values match (63.5/64.1/64.5/51.4; 2.80/0.00/0.12/0.26%) | Panel B, right ("All sites") is not discussed in the text; add one clause or drop the panel. |
| S1 Fig | Values match REPORT §4.3 and POSTHOC | – |
| S2 Fig | Dev values match REPORT §4.7 | Panel C has 10 rows vs "8 of 9" (F3). The A3 labels overlap in A and B. |

---

## COULD fix

- C1. The main text is long (about 11,700 words including tables and references), and the user asked for "short and to the point". Candidates for cutting:
  - repeated definitions of top-N+2 and DCA4 (Fig 2 caption, Table 2, Statistics);
  - the two paragraphs on the dimer and apo sections that repeat Fig 6B;
  - Table 2's "Input" column (move it to Methods).
- C2. Discussion: "largely because the baselines were not trained to find ions" is a causal claim. Write "probably because …". A P2Rank model trained on ions would test it; P2Rank-ours trained on our labels including ions did not reach BURROW on LIGYSIS, which weakly supports the claim.
- C3. Table 1, "Candidate recall as the target": add the list length to "99.3% maximum recall" (see O7).
- C4. Add the seed-to-seed SD on dev (≈ 0.7 points) where single-seed ablations are discussed.
- C5. Discussion: the follow-up list could note that a pre-registered list-length policy for *baselines* (a tuned floor for P2Rank and fpocket+PRANK) is the most direct test of the output-length caveat.
- C6. The independent verifier "could not rule out that [v1 prediction files] had been inspected". If tool-call logs exist, say whether any agent read those files between 20:58 and 22:09 UTC. Even a negative ("logs not retained") is informative.

---

## Compliance with the user's request (FOLLOWUP_PROMPTS item 4)

| Element | Status | Note |
|---|---|---|
| Flag at the beginning that this was a test of agents | Met | Title, AI-contribution statement and box. Keep the flag if the box is converted (PLOS format). |
| Methods explain the agent and sub-agent setup | Partly met | Roles, models, rules, gating and interruptions are covered. The roster and counts are inaccurate (M5). Tool versions, usage and transcript status are missing (D-g). |
| Credit for ideas taken from the literature | Mostly met | Table 1 is good. Credit for WCN, cross-fitting and metal-site precedents is missing (D-f). |
| Method explained conceptually and technically | Met | Results section 1 is conceptual and Methods technical. The calibration claim is unsupported (O3). |
| All important decisions explained | Mostly met | Table 5 is good. Missing: hyperparameter selection, the D5 speed budget, D8, and the pre-registered sensitivity analyses that were not run (D-a to D-e). |
| Critical, not overstated | Mostly met | Main exceptions: M3 (DCA4 "robust evidence"), M4 (Abstract), O1, O2, O10. |
| Precise, short | Partly met | Precise overall; long (C1). |
| Supplementary file on methods that failed | Met | S1 Text is thorough and well organised; fix §16 (M8). |
| Writing guide (edit-for-clarity) applied | Mostly met | 19 of 47 sampled sentences need edits, mostly stacked nouns and long gaps. |
| Original prompt included | Met | S2 Text §8 is byte-identical to PROMPT.md (checked with diff). |
