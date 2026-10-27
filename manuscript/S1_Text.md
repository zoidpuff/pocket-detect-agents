# S1 Text. Methods that did not work

This text lists every major approach that the agents tried and then abandoned, or that gave a negative result. For each one it gives what was tried, the evidence (with the file in the project repository that holds it), and the lesson. Unless stated otherwise, numbers are percentage points of top-N+2 recall at DCC ≤ 12 Å under the LIGYSIS closest-pocket protocol, and CIs are 95% paired cluster-bootstrap intervals. "Dev" means the leakage-controlled BioLiP2 development set (579 chains). "Train-CV" means cross-validation over 30% sequence clusters of the training set. Gates were fixed by the PI agent before the corresponding results existed.

## 1. Candidate-recall gate G1 as specified

**What was tried.** The design required a gate on candidate recall before the main model was built: at least 88% of dev sites had to be reachable (DCC ≤ 12 Å and DCA ≤ 4 Å) with at most 40 geometric candidate pockets per protein, before any learned ranking. If the gate failed, the design specified a fallback: fpocket candidates rescored by BURROW's ranker.

**Evidence.** The gate passed on interim data from the P2Rank training sets (91.0%), then failed on BioLiP2 dev (58.6%). The fallback was worse: fpocket's candidates reached 47.3% under the same rule. The full pool of BURROW candidates before ranking reached 93.6%. (results/dev/G1.md, addendum; decision D7.)

**Lesson.** The gate measured the wrong quantity. A cap of 40 candidates ordered by geometry alone tests the geometric ordering, not the candidate pool. The agents accepted the deviation and kept their own candidates, because the fallback was worse on every measure. A gate should test the property that the design depends on (here, recall of the whole pool), and it should be checked on data that resemble the final dev set.

## 2. fpocket candidates with BURROW's ranker

**What was tried.** A mandatory ablation (and the G1 fallback): replace BURROW's grid candidates with fpocket 4.2.3 pockets and rank them with BURROW's ranker (not retrained on fpocket candidates, which is a conservative variant).

**Evidence.** On the pre-audit dev set (seed 0, `all` ranker), top-N+2 recall was similar (64.8 against 65.3 for BURROW's own candidates), but DCA4 recall fell from 56.2 to 39.7 and P1K from 72.1 to 59.4. fpocket's own ordering gave 47.7 top-N+2. (results/dev/biolip/PHASE2.md, "Ablations".)

**Lesson.** Most of the gain at the lenient 12 Å criterion comes from the ranker, but BURROW's volume-element candidates are what give precise centres (DCA4) and precision. A lenient metric can hide the value of a component; always report a stricter metric alongside.

## 3. Wider suppression radius

**What was tried.** A sweep of the suppression (non-maximum suppression) radius from 4 to 14 Å.

**Evidence.** Top-N+2 recall rose with the radius: 65.3 at 10 Å, 69.7 at 12 Å and 72.6 at 14 Å. Raw P1K fell from 72.1 to 65.3 and 60.1. One-to-one P1K peaked at 10 Å. A 12 Å radius would have added +4.4 top-N+2 on dev for −0.9 one-to-one P1K and −6.8 P1K. The agents kept 10 Å, which had been pre-selected by one-to-one P1K on cross-validation of the P2Rank training set (CHEN11) before BioLiP2 dev existed. (results/dev/biolip/PHASE2.md, "NMS sensitivity"; results/review_dev/REVIEW.md.)

**Lesson.** Spreading pockets to match the 12 Å tolerance of the metric raises the headline number without improving localisation. The agents rejected it as tuning to the metric. The same sweep contained a sign of the later output-length effect: with a minimum score of 0.05 at 10 Å, top-N+2 rose from 65.3 to 70.6 while maximum recall fell from 98.5 to 74.5. This signal was not interpreted until the v2 cycle. The lesson is to decide the policy on output length before the first test run.

## 4. Ranking without learning

**What was tried.** A learning-free baseline (`--ranker geometry`) that ranks BURROW's geometric candidates by buried volume, Σ vol · clip((b − 0.7)/0.3, 0, 1)², with the same suppression.

**Evidence.** 53.3 top-N+2 on dev against 54.1 for P2Rank 2.5.1. BURROW's own pockets with randomly shuffled ranks scored 14.6 on dev, the level of evenly spaced surface pockets. (results/dev/biolip/final_table.md; results/review_dev/REVIEW.md.)

**Lesson.** The candidates and features are analytical, but the accuracy comes from the learned scorer and ranker. The manuscript says so explicitly.

## 5. Phase-3 ranker variants

**What was tried.** Before the feature freeze, five ranker changes were fixed in advance and compared by train-CV (folds 0–4 of the training set; dev used once, only to confirm the selected version): V1, within-protein relative features; V2, V1 plus context from pairs of neighbouring pockets; V3, V2 trained with LambdaRank (one query per structure); V4, V2 with an ensemble of three ranker seeds; V5, an atom-type propensity potential (out-of-fold log-odds tables) used as ranker features. The selection criterion was the mean of top-N+2, DCA4 top-N+2 and P1K.

**Evidence.** V0 (no change) had the highest five-fold criterion (70.44). LambdaRank (V3) improved within-protein top-N (56.62 against 55.61) but lost P1K (82.14 against 83.46), giving a criterion of 69.92. On folds 0 and 1, V1 scored 69.68, V2 70.31 and V5 69.98, against 69.92 for V0 on the same folds. Under the protocol, only V0, V3 and V4 were run on all five folds. V4 was started but stopped because of node load before it produced a result, so the seed ensemble was not evaluated. V3 needed a fix for LightGBM's limit of 10,000 rows per query. (results/dev/biolip/PHASE3.md.)

**Lesson.** Within-protein ranking objectives trade off against precision across proteins, which the pooled P1K metric measures. Selecting on train-CV and using dev only once kept the optimism of dev results low. Variants that need new point features should be tested before the feature freeze, or not at all.

## 6. Training changes for apo structures (v2 cycle, H1–H4)

**What was tried.** After test v1 was frozen, the agents tried to make BURROW more robust to apo structures. Gate: apo-dev cryptic top-N+2 (CryptoBench training folds, never used for training) at least +3.0 with CI > 0 over v1, with no loss beyond −0.5 on BioLiP2 dev. Hypotheses: H1, AlphaFold models of training proteins as extra training inputs (2,844 proteins with transferred sites); H2, copies of every training chain with side chains removed at random (p = 0.5 within 8 Å of a ligand, 0.2 elsewhere); H1+H2 combined; H3, apo-oriented pocket descriptors for the ranker (backbone-only buriedness, side-chain enclosure deficit, mouth openness); H4, apo PDB chains of training proteins.

**Evidence.** Apo-dev cryptic differences against v1: H1 +1.9 [−0.0, 3.8]; H2 +0.0 [−2.4, 2.2]; H1+H2 −1.9 [−4.2, 0.3]; H3 +0.3 [−1.4, 1.8]; H4 +0.5. All failed the gate. A fifth variant, H5, was the output score floor: +9.7 [8.0, 11.5] on the same target. Diagnosis showed that v1's apo deficit came from long output lists under the closest-pocket rule, not from apo-specific ranking. H5 became v2. (results/dev/biolip/V2_CYCLE.md; results/dev_apo/APO_DEV.md.)

**Lesson.** Before trying to fix a weakness, find out what causes it. Here an error analysis would have pointed to the evaluation protocol immediately. Cheap augmentations alone did not make a static scorer robust to conformational change.

## 7. AlphaFold-mode geometry against a simple pLDDT filter

**What was tried.** The design required AlphaFold mode to beat a control baseline: BURROW on the unmodified model, followed by removal of pockets with mean pLDDT < 70. Mechanisms tested: removal of low-confidence segments, soft occlusion weighted by pLDDT (three weight curves), an oracle crop at pLDDT ≥ 70, and matched random pruning as a control.

**Evidence.** AlphaFold mode was equivalent to the filter, not better: +0.7 [−1.4, 2.8] on AlphaFold dev; −0.5 [−1.3, 0.2] and −0.7 [−1.4, 0.1] on the 2,844 out-of-fold training models with two base models. On the AlphaFold test set (descriptive): +0.4 [−0.5, 1.3], and the filter had higher P1K (+2.4). Near disordered segments, every geometric variant was at or below the filter (prune and soft occlusion −1.6 [−4.3, 1.2]; a stronger weight curve −3.6; oracle crop −4.8). Matched random pruning cost 2.5–3.1 points, so the removal of low-pLDDT segments is specifically harmless. Hallucinated pockets fell from 5.0% to 0.2% on dev. (results/dev_af/AF_DEV_RESULTS.md; docs/AF_MODE.md; results/test/v2/DESCRIPTIVE.md.)

**Lesson.** A mechanism needs a simple control. Without the pLDDT-filter baseline, the +1.0 gain of AlphaFold mode over the unmodified model would have looked like a success of the geometric mechanisms.

## 8. Side-chain tolerance

**What was tried.** Side-chain atoms beyond Cβ were made partly transparent (opacity × λ), either on all residues or only on residues with pLDDT < 90, to mimic side chains in apo-like conformations that block a cleft.

**Evidence.** λ on all side chains reduced recall on AlphaFold dev: −2.1 at λ = 0.5 and −15.7 at λ = 0.25 with the exploration base model, and −10.9 [−14.9, −7.2] at λ = 0.25 with the frozen model. Restricting λ to residues with pLDDT < 90 was neutral overall but hurt sites near low confidence (−3.1 to −4.2). Side-chain tolerance is implemented but off. (results/dev_af/AF_DEV_RESULTS.md.)

**Lesson.** The point model was trained on full occlusion, so changing the geometry at prediction time moves points outside the distribution of the training data. A geometric change of this kind needs retraining, which the side-chain dropout augmentation (H2 above) tried without gain.

## 9. PAE and an inter-domain stand-in

**What was tried.** The prompt suggested using PAE matrices to suppress fake inter-domain cavities. The AFDB file endpoint for PAE returned HTTP 403, so PAE could not be obtained for training or test models. As a substitute, the agents built a structure-only stand-in: spectral bisection of the Cα contact graph (8 Å), with descriptors of how a pocket spans the two parts and how densely the parts pack.

**Evidence.** The stand-in (`pldom`) added nothing to the AlphaFold ranker on train-CV. (docs/AF_BENCHMARK.md §6; results/dev_af/AF_DEV_RESULTS.md.)

**Lesson.** Check access to every data source before planning around it. PAE through a bulk download remains a follow-up.

## 10. AlphaFold-aware ranker as the default

**What was tried.** A pocket ranker that adds pocket pLDDT descriptors, trained on AlphaFold models of training proteins together with PDB rows.

**Evidence.** Pooled P1K improved by +6.7 to +6.9 (train-CV), but recall at sites near low confidence fell by −3.0 [−5.2, −0.6] with one base model (−0.8 [−3.2, 1.5] with the frozen model). Rankers trained on AlphaFold rows only failed non-inferiority on PDB dev (−2.2 [−3.9, −0.5]). On the test sets (descriptive), the ranker raised AlphaFold-set P1K from 58.9 to 73.9 but lowered PocketWatch top-N+2 by 3.4, because PocketWatch models carry no pLDDT. It is optional (`--af-ranker`), not the default (decision D8). (docs/AF_MODE.md; results/test/v2/DESCRIPTIVE.md.)

**Lesson.** The ranker learned a true prior (pockets in low-confidence regions are rarely sites), and so it also ranked the few real sites next to disorder lower. A trade-off of this kind should be offered as an option.

## 11. Interface hypotheses (v3 cycle)

**What was tried.** After the dimer-interface endpoint (O6) was the weakest test result, the agents ran a dev cycle targeted at interface sites. This step was informed by test data and is disclosed as such. Gate: interface top-N+2 on out-of-fold training assemblies (1,709 inputs, 319 interface sites) +3 with CI > 0 against v2. First they checked that the dev signal reproduced the test weakness: on the training assemblies, v2 was 2.2 points behind P2Rank on interface sites at 12 Å, but ahead at 4 Å (74.9 against 70.2) and on all assembly sites (+12.8). Hypotheses: a minimum number of kept pockets scaled by the number of chains; interface-specific pocket descriptors; both; and up-weighting assembly inputs in training.

**Evidence.** Minimum kept pockets scaled by chains +1.3; interface descriptors +0.6; both +1.9 [−0.6, 5.4]; up-weighting +0.3. None met the gate, and no v3 was frozen (decision D15). (results/dev/biolip/V3_CYCLE.md.)

**Lesson.** The interface stratum is small (17 interface sites in dev assemblies, 243 in the test set), and so are the effects. A larger interface benchmark and more multimer training data are needed before targeted changes can be evaluated.

## 12. Relative and per-region output floors (v3b cycle)

**What was tried.** On dev, 65% of v2's non-ion misses (151 of 232) were pockets that existed but fell below the global score floor (results/dev/biolip/NONION_ERRORS.md). The agents pre-declared four families of rules that keep more pockets: (a) a floor relative to the best score in the protein (ρ = 0.1, 0.2 or 0.3); (b) a minimum number of pockets scaled by protein size; (c) per region, adding every pocket with score ≥ 0.005 or 0.01 whose centre is more than 12, 16 or 20 Å from all kept pockets; (d) combinations of (a) and (c). Selection was on train-CV only, with one dev confirmation. Gate: non-ion closest-pocket top-N+2 at least +2.0 with CI > 0 against v2, with guards on maximum recall (+3), DCA4 (−0.5), P1K (−1), apo-dev and AlphaFold dev (−1).

**Evidence.** The best rule (per region, 16 Å, score ≥ 0.005) gained +1.2 [0.1, 2.4] with the `small` ranker and +0.5 [−0.1, 1.2] with `all` on dev, below the required +2.0. All guards passed: maximum recall rose by 7 to 10 points, and DCA4 and P1K were unchanged. The rule ships as the option `--output-rule region` for exhaustive discovery, and v2 stays the default (decision D19). (results/dev/biolip/V3b_CYCLE.md.)

**Lesson.** The remaining weakness on non-ion sites is the ranking of shallow, low-scoring sites, not the floor. A different floor recovers coverage but cannot move these sites into the top N+2.

## 13. P2Rank retrained on our training data

**What was tried.** A fairness control: P2Rank 2.5.1 with its default configuration, retrained on our BioLiP2 training set, to test whether our data alone explain BURROW's gain.

**Evidence.** Full and 50% training runs were killed by the memory limit of the container (about 79 GB). A 25% random subset (1,059 liganded entries) trained successfully. On dev, this model was worse than stock P2Rank (−5.2 with the `all` sites, −3.5 with `small`), and BURROW v1 led it by +16.5 and +10.1 points and by +21.3 on DCA4. On the test sets (descriptive) it was below stock P2Rank on LIGYSIS (−6.7), LIGYSIS without ions, CryptoBench and the AlphaFold set, and above it on HOLO4K (+2.1). (results/baselines/P2RANK_OURS.md; results/test/v2/DESCRIPTIVE.md.)

**Lesson.** Check the real memory limit of the environment before planning a large training run (the planned 128 GB was in fact about 79 GB). Because only a quarter of the data was used, this control gives a lower bound for P2Rank trained on our data.

## 14. Screened Coulomb potential

**What was tried.** Nothing. The design listed a screened Coulomb potential (ε = 4r, template charges) as an optional point feature, to be kept only if it helped on dev. It was on the list of items to cut if behind schedule, and it was never implemented. Counts of charged atoms within 4–12 Å serve as a cheap proxy. We therefore cannot say whether electrostatics would help.

## 15. Optional deep-learning baselines

**What was tried.** Besides P2Rank and fpocket+PRANK, the baselines agent wrapped eight optional tools: pyKVFinder, DeepPocket, PUResNet v2, GrASP, VN-EGNN, IF-SitePred, Kalasanty and PocketMiner, and ran several of them on the CryptoBench, dimer and AlphaFold sets.

**Evidence and problems.** DeepPocket's official weight links were dead, so a checkpoint from a third-party fork with identical code was used, and the segmentation model could not be found. PUResNet v2 needed its sparse-convolution library replaced by a pure PyTorch port. GrASP needed local patches, and its featurisation scaled roughly quadratically with protein size. The current VN-EGNN code ships no weights, so the older code that matches the released checkpoint was run. Kalasanty had to be ported from TensorFlow 1 and tiled for large proteins, and IF-SitePred's PyMOL step was re-implemented. Most of these tools were trained on scPDB-derived sets that overlap with COACH420 and HOLO4K, so their results were treated as descriptive only. On LIGYSIS, the published per-chain predictions of the deep-learning methods were used instead of re-runs. (baselines/README.md; results/baselines/RUNS.md.)

**Lesson.** Wrapping eight tools took substantial effort for results that could only be descriptive. In the process retrospective, it was named as a branch that should have been given a fixed time limit.

## 16. Speed targets

**What was tried.** The design set a target of at most 70 ms per structure on one core for the default preset, and expected GPUs to beat CPU workers.

**Evidence.** In the final like-for-like timing, the default preset took 96.0 ms (median, excluding parsing; 90th percentile 264.9 ms), so the target was missed on larger chains. Before the speed track in the extension, the GPU path was limited by the host and lost to 24 CPU workers. After that track, three GPUs beat 24 CPU workers for every preset over 10,000 structures, with byte-identical output. On short runs of 2,000 structures, CPU workers still won for the fast preset. (docs/PERF.md.)

**Lesson.** Optimise code that depends on features only after the feature freeze, and measure throughput at the batch sizes that users will run.

## 17. Process approaches that did not work

- Broad process-kill commands (`pkill -f`, or `ps | grep | kill`) on a shared node killed another agent's test job once and an agent's own shell four times. Rule adopted: kill only process IDs that were stored at launch.
- Sub-agents that staged files in git left 169 MB of unreachable objects. Rule adopted: only the PI commits, with explicit paths.
- Timestamps written from memory were 20–50 minutes off. Rule adopted: take times from git or the system clock.
- The builder's own leakage check passed while an independent audit failed. Rule adopted: data and leakage checks are done only when an independent re-implementation agrees.
- About 30% of commits were bookkeeping only, and results were sometimes left uncommitted for hours. Rule adopted: one commit per result, including its note in the research tree.

