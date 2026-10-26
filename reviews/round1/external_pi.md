# Review round 1: external PI (overall strategy)

Reviewer role: external PI (method benchmarking). Reviewed DESIGN.md v1 against PROMPT.md and LITERATURE.md §2.2, §4 and §7. Written 2026-10-26 15:30 CET, about T+0.6 h into the 24 h budget.

## Verdict

The architecture is sound and points at the right headroom: high-recall candidates, then a calibrated ranker, then NMS. Literature §7.8 shows that rescoring alone lifts fpocket from 46.5 to 60.4. It does **not yet** reliably produce a *clear* win, for four reasons:

1. The headline label regime is ambiguous.
2. Novelty is thin: it reads as "P2Rank with more features plus PRANK".
3. The plan is serial, and AF mode, dimers and GPU land late.
4. It has no go/no-go gates and no fallback.

None of the changes below expand the scope.

## MUST

**M1. Fix the headline label regime to match LIGYSIS (ions included).**
- About 40% of LIGYSIS sites are ions, and every published bar (60.4 / 51.9) is on full LIGYSIS.
- §2.2 makes `small-molecule` the default, which would hand away about 40% of the sites.
- The headline model must train on BioLiP2-relevant ligands *including* ions, with a dedicated metal-site candidate source (His Nε / Cys S / Asp/Glu O clusters). This is also where P2Rank's 57% max-recall ceiling sits.
- Report LIGYSIS_NI as co-primary, using the same model plus the NI-trained variant.

**M2. Set the bar as max(published, our re-run) per baseline, and commit to a margin.**
- A paired-bootstrap CI that excludes 0 can be met with +1 point at n ≈ 6,900 sites. That is not "clearly".
- Pre-register: top-N+2 ≥ bar + 3 points **and** paired CI > 0 **and** Precision1K ≥ P2Rank (no-cons) + 5.
- For O2–O6, replace "no significant loss" with explicit non-inferiority margins (e.g. −2 points). Small sets such as COACH420 have too little power for "no significant loss" to mean anything.

**M3. Add gates and a fallback track.**
- *Gate G1* (by about T+7 h, dev set): our candidate generator reaches ≥ 88% max recall at ≤ 40 candidates per protein.
  - If it fails, switch candidates to fpocket output plus our point and pocket features, i.e. a "BURROW-rank on fpocket" track.
- That track is also a *mandatory ablation* in any case. It separates generator gains from ranker gains, and it is the fallback deliverable if our own geometry underperforms.
- *Gate G2* (T+12 h): feature freeze. After it, only AF mode, dimers, GPU and evaluation work proceed.

**M4. Run the work as parallel tracks, not E0→E9 serially.** Start these now, owned by separate agents:
- (a) **Training/dev builder.** BioLiP2 → UniProt-pooled sites → 30% MMseqs2 hold-out against *all* test sets.
- (b) **AFDB single-pass extractor.** The tars are not indexed, so stream each proteome tar once and pull every needed UniProt in that pass: test and training, human and SwissProt. Handle fragmented (F2+) long human models explicitly.
- (c) **Dimer benchmark builder.** Recipe: group BioLiP2 rows by (PDB, ligand chain, ligand resnum). A ligand instance listed on more than one receptor chain is an interface site; classify it homo or hetero by UniProt, then keep it if those chains co-occur in assembly-1 from the mirror. This avoids recomputing contacts from scratch.
- (d) **PocketWatch / CryptoBench prep.** The data is already local.

**M5. AF mode must beat the trivial practice, not only naive input.**
- Add the baseline "naive + discard pockets with mean pLDDT < 70" (HOTPocket / Zillmer & Walther practice).
- The dedicated mode is justified only if it beats that filter on recall at matched hallucination rate.
- Also report the AF→PDB recall gap for the same UniProts. This is the honest measure of what AF mode recovers.
- Stratify by pocket pLDDT band, and by template richness, because AF models of LIGYSIS proteins are holo-like (§4).

**M6. Specify symmetric-copy scoring for dimers.**
- A prediction that hits *any* symmetric copy of a site counts once.
- N is the number of unique sites.
- All methods run on the identical assembly file.

Without this, homodimer copies either inflate the score (HOLO4K-style) or penalise everyone.

## SHOULD

**S1. Sharpen the novelty claim and test each part by ablation.** Three defensible contributions:
- (i) **Confidence-propagated geometry.** pLDDT- and side-chain-weighted soft occlusion inside buriedness, rather than post-hoc filtering. Show that the same mechanism (side-chain tolerance) also helps apo/CryptoBench, not just AF.
- (ii) **Assembly-native detection with calibrated cross-protein probabilities.**
- (iii) **A GPU tree-ensemble plus ray-cast pipeline at more than 10³ structures per minute.**

State these in DESIGN §0 and map each to a pre-registered ablation row.

**S2. Decide on conservation explicitly** (Lit §7.10).
- Without conservation, P2Rank has Precision1K 79.2, while P2Rank-CONS has 90.7. O1b's bar of 90.7 is unrealistic for a pure-structure model.
- Make O1b "beat P2Rank (79.2) and report against P2Rank-CONS".
- An optional small-PLM variant is a COULD, reported separately so the analytical claim stays clean.

**S3. Make the dev set mirror the headline protocol.**
- Dev set: LIGYSIS-style sites (I_rel clustering at 0.5 over UniProt-pooled fingerprints), single-chain input, DCC 12 Å top-N+2.
- Also track DCA 4 Å and one-to-one recall on dev, so we do not tune to the lenient 12 Å criterion.

**S4. Add a runtime budget per stage, and profile at E2.**
- Target ≤ 100 ms per protein per core.
- 1 Å shell grid × 60 rays × 4 lengths is likely to blow this for proteins over 500 residues.
- Use coarse-to-fine: a 2 Å grid with buriedness on an occupancy grid, refined to 1 Å only near candidates.
- If the CPU path exceeds 250 ms, cut features in a pre-agreed order.

**S5. Run the fair-data control at reduced scale.**
- Retraining P2Rank on a 1–2k-chain subsample of our training set is enough to answer "is it just more data?".
- Also run the CHEN11-sized ablation (§7.11): our model trained on about 250 proteins. A strong result there is the best anti-memorisation evidence.

**S6. Cap the review process at 2 rounds, and freeze DESIGN v2 by about T+2 h.** Every hour in review is an hour taken from training iterations.

## COULD / CUT

**Cut to COULD**
- Coulomb potential with template charges and an H-bond "capacity" potential. Keep charged and donor/acceptor atom counts; the potential is unlikely to add much over them.
- The lambdarank ablation.
- DL baselines beyond those that run in under 1 h. LIGYSIS already publishes their numbers; cite them, marked "published".
- PocketMiner 38 pairs.
- PAE: download only for the AF test set, after the pLDDT mechanisms work.

**Keep but shrink**
- The AF-aware ranker. Train it on AF models of about 2k training UniProts, not all of them.
- 3 seeds for the final model only.

**COULD**
- An allosteric stratum, using the AF2BIND allosteric subset or CryptoBench annotations. The brief names allosteric sites, and the design does not cover them; state this as a limitation if it is not done.

## Suggested timeline (CET)

| Window | Work |
|---|---|
| 15:30–17:00 | DESIGN v2. Parallel tracks M4 (a–d) running. Baselines reproduce LIGYSIS published numbers within ±1.5 points. |
| 17:00–22:00 | Candidate generator, then gate G1. First end-to-end dev number by 22:00. |
| 22:00–03:00 | Point and pocket model iterations on dev. G2 feature freeze at 03:00. |
| 03:00–09:00 | Parallel: GPU port of the frozen features plus tree inference; AF mode and its ablations; dimers; fold-disjoint and CHEN11-sized ablations. |
| 09:00–12:00 | Single frozen test evaluation (ledgered), 3 seeds, bootstrap. |
| 12:00–14:30 | REPORT.md and verification by a non-author agent. |
