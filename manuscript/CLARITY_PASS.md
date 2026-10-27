# Clarity pass (edit-for-clarity skill, manuscript/skill/SKILL.md)

Procedure: every section was drafted with the six patterns in mind, then given a full sentence-by-sentence pass with the skill's checklist. Edits were made in the sources (`build/*.src.md`); `build/build.py` regenerates main.md/.tex, S1_Text, S2_Text and S1_Table. Meaning, numbers, code spans, file paths and the verbatim prompt (S2 Text §8) were not changed. The change list below covers the full pass (draft → final).

## Main text (main.src.md, tables.src.md)

### Pattern 1 — rare word combinations
- "cheap stand-in for desolvation" → "cheap approximation of desolvation"
- "the gain thus needs both parts, as the literature analysis predicted" → "Both parts are needed, as the literature analysis suggested"
- Abstract: "their rankings are often weaker on AlphaFold models" removed (also an over-generalisation; P2Rank is stable on AF models per LITERATURE.md); abstract rewritten to 295 words.

### Pattern 2 — inanimate subjects with verbs for living things
- "Proteome-wide pocket atlases deal with this by removing pockets" → "The authors of proteome-wide pocket atlases therefore remove pockets"
- "so that the ranker sees the same score distribution" / "it learns a graded target" → "so that the point scores in training follow the same distribution as at prediction time. Its training target is graded…"
- "The stricter test (C1-strict) asks whether…" → "In the stricter test (C1-strict), a site counts as found only if…"
- "which the closest-pocket rule of the benchmark favours" → "which score better under the closest-pocket rule of the benchmark" (abstract)
- "The size of the LIGYSIS margin needs careful reading" → "…should be read with care"
- "The LIGYSIS protocol also judges interface sites on a single chain" → "Under the LIGYSIS protocol, interface sites are also judged on a single chain"
- "Fold-disjoint retraining and the shuffled-label control argue against memorisation" → "Fold-disjoint retraining is evidence against memorisation of folds" (also removes a control that does not bear on folds)
- "which may favour LIGYSIS" → "which may give BURROW an advantage on LIGYSIS"
- "a human-led study would ideally have fixed the policy" → "Ideally, however, the policy on list length would have been fixed…"
- "Several of these rules caught real errors" → "Applying them caught real errors"
- "under guards that forbade losses in…" → "under the condition that … must not decrease"
- "Top-N+2 recall counts a site as found if…" → "Under top-N+2 recall, a site counts as found if…"
- Table 5: "the closest-pocket rule penalises long lists" → "gives less credit to long lists"
- Kept as standard technical usage: "a script checks", "the gated script refuses modified files", "the function returns".

### Pattern 3 — nouns used as verbs
- None found in the draft (checked "time-box", "benchmark", "gate", "floor"): "time-boxed" in S1 Text was replaced (see below); "benchmarked" is standard.

### Pattern 4 — missing "that"
- "the chemical class of the first atom hit" → "the first atom that the ray hits"
- "with all chains similar to any test chain removed" → "with every chain removed that is similar to any test chain"

### Pattern 5 — stacked nouns (three or more)
- "output score floor" → "output floor" (defined once: "pockets with a ranker score below 0.03 are dropped … (the output floor)")
- "protein language-model embeddings" → "embeddings from protein language models"
- "dimer pocket benchmark" → "public benchmark of pockets in dimers"
- "list-length policy" → "the policy on list length"
- "API usage limit" → "a usage limit of the API"
- "30% MMseqs2 sequence clusters" → "sequence clusters (30% identity, MMseqs2)"
- "training-set cluster cross-validation" → "cross-validation over clusters of the training set"
- "92 primary-tier inputs" → "92 evaluated inputs that violated it when sequences of observed residues were used"
- Figure captions: "stratum-filter bug" → "a filter bug"; "Holm-adjusted non-inferiority" → "non-inferiority with Holm correction"; "chaining control" → "control of chaining"; "atom- and residue-class densities" → "densities of atom and residue classes".

### Pattern 6 — long gaps between subject and verb
- "An optional fast preset, which was added after the test runs and was not pre-registered, took 48 ms" → split into two sentences.
- "However, a simple filter that drops pockets with mean pLDDT below 70, as used by HOTPocket and by Zillmer and Walther, was equally good (…)" → three sentences.
- "Retraining without any training protein that has a Foldseek TM-score of 0.5 or more to any test chain (41% …) left 8 of 9 test rows within noise" → "We also retrained BURROW without … which removed 41% of training inputs. Results changed by amounts within noise on 8 of 9 test rows…"
- "The union of all test sets (LIGYSIS, … every dimer chain) was frozen and checksummed before the training build" → "Before the training set was built, the agents froze and checksummed the union of all test sets. This union covered …"
- "Family 1 required, for every seed and against both …, a difference of …" → "Family 1 had three requirements, each for every seed and against both …: for C1, …; for C1-strict, …; and for C2, …"
- "For each point, LIGSITE protein–solvent–protein counts along 7 directions up to 8 Å are computed" → "For each point, BURROW counts LIGSITE protein–solvent–protein events along 7 directions up to 8 Å"
- "The target is clip(…), where q is …, trained with cross-entropy" → "The ranker is trained with cross-entropy on the target clip(…), where q is …"
- "A cheap test based on LIGSITE keeps the buried points at full resolution and keeps a coarser 3 Å sub-lattice of the remaining points, so that …" → two sentences.
- "The PocketWatch decision was narrow. Its Holm-adjusted p-value of 0.003 …" → corrected and restructured: "its non-inferiority p-value (0.003) was below the Holm threshold (0.005) by a margin close to the 0.001 resolution of 1,000 resamples" (the draft had mislabelled the p-value).

### Other corrections made during the pass (meaning checked against sources)
- Pre-registration freeze time: "about 16:10 CET" (header estimate) → "15:39 CET (git time)" (git history is authoritative; RESEARCH_TREE D3).
- "produced 136 git commits … wrote 20 dated decisions (D0–D19 … D20)" → "more than 130 git commits … logged 20 dated decisions (D0–D19)".
- Figure panel references added to match the final figures (Fig 3A/3B, Fig 4A–4D, Fig 6A/6B; S1 Fig is fold novelty only).
- Discussion: added one sentence on the limits of a single, unreplicated agent run.

## S1 Text
- Pattern 2: "Cheap augmentations do not by themselves teach a static scorer about conformational change" → "Cheap augmentations alone did not make a static scorer robust to conformational change"; "The ranker carries most of the gain" → "Most of the gain … comes from the ranker"; "moves points out of the distribution that the model has seen" → "outside the distribution of the training data"; "and so also demoted the few real sites" → "and so it also ranked the few real sites … lower".
- Pattern 3: "should have been time-boxed" → "should have been given a fixed time limit".
- Pattern 5: "Selection on train-CV with dev touched once kept dev optimism low" → "Selecting on train-CV and using dev only once kept the optimism of dev results low".

## S2 Text and S1 Table
- Drafted with the checklist; no further pattern instances found in the final pass. The verbatim prompt (S2 Text §8) is a quotation and was not edited. Cross-references to sections corrected (§8, §9).

## Figure captions
- Taken from manuscript/figures/CAPTIONS.md (figure agent) and edited for the patterns above; repository paths and "Sources:" lists were removed from the captions (they remain in CAPTIONS.md and figure_data.csv).

# Revision 1 (after manuscript/REVIEW.md and REFERENCES_CHECK.md)

All 19 rewrites W1–W19 from the referee's sample were applied:
- Pattern 5: W1 "candidate volume elements" → "candidate points … each standing for a small volume of space"; W4 "volume-element candidate points" → "candidate points that each stand for a small volume of space"; W7 "cluster-bootstrap intervals over 30% sequence-identity clusters" → "intervals from a paired bootstrap that resamples clusters of chains with at least 30% sequence identity"; W8 "v1 test prediction files" → "the files with the v1 predictions on the test sets"; W10 split into three statements.
- Pattern 1: W5 "enclosure by chemistry" → "first-hit chemistry" everywhere (also Table 1), with "a rough proxy for the cost of desolvation"; W16 "label-free rule" → explicit rule (Jaccard ≥ 0.5, does not use site labels); W17 "ran detached" → "ran as background processes"; W18 "typical of fast research" → "also occur in human projects that run under time pressure".
- Pattern 6: W13 long opening phrase on per-process timing split into two sentences; W15 side-chain tolerance sentence split, λ defined; W19 metal-candidate rule rewritten as three sentences (checked against src/pocketdetect/burrow/features.py); new sentence "The design target … was therefore met only by the fast preset" → "Only the fast preset therefore met the design target …".
- Precision/overclaim: W2 "Per process" made explicit; W3 author summary generalisation narrowed; W6 "Together, this makes its scores comparable" → "These two choices are meant to make scores comparable … we did not analyse calibration directly"; W9 P1K superiority wording; W11 miss counts made precise (403 misses with the closest pocket within 12 Å but ranked beyond N+2; 5 with no pocket within 12 Å); W12 "apo cryptic sites" defined; W14 opacity ramp stated exactly (transparent below pLDDT 40, opaque from 80).
- New text added in the revision (Methods agent setup, Data availability, dimer and AF caveats, Discussion speed targets) was drafted and re-scanned with the same checklist.

# Revision 2 (length)
Introduction–Discussion cut from ≈4,720 to ≈3,560 words and Methods from ≈3,120 to ≈1,910. The full technical methods and the longer Results paragraphs (runtime, AlphaFold mode, dimers, output length, agent process) moved verbatim to the new S3 Text. A script checked that no number from the previous version was lost across main + S1–S3 + S1 Table. The condensed sentences were re-scanned with the checklist. Pattern 6 fixes: the fold-disjoint sentence was split; the family-1 requirements were restored as a list. Pattern 2 fix: "The LIGYSIS margin needs two qualifications" → "Two qualifications apply to the LIGYSIS margin". Referee F3: "9 of 10 test rows", checked against results/test/v2/DESCRIPTIVE.md (only COACH420 has a CI that excludes 0). The Fig 3 caption now includes the figure agent's updated note on the +2 and −3 lines.
