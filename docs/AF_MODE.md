# BURROW `--af-mode` — AlphaFold-model mode (DESIGN §2.6, §9)

Owner: AF-mode developer. Code: `src/pocketdetect/burrow/af.py` (pruning + soft occlusion used by
the CLI), `af_variants.py` (all ablation/control inputs), `af_rank.py` (AF-aware ranker).
Evidence: `results/dev_af/AF_DEV_RESULTS.md`. Development used only AF *train/dev* models of
Track A entries (Track C2 labels). The AF test set (`data/processed/af`) was never read.

## Final configuration (PI decision D8)

`burrow predict ... --af-mode` runs three steps:

1. **Spaghetti pruning.** Terminal runs with pLDDT < 50 are dropped, and so are internal runs of ≥ 7 residues with pLDDT < 50.
2. **Soft occlusion.** Every remaining atom gets opacity w = clip((pLDDT − 40)/40, 0, 1). Atoms with w < 0.5 do not exclude candidate points.
3. **Standard ranker.** The standard point model and pocket ranker of the frozen model are used unchanged.

Side-chain tolerance stays off (λ = 1). This is exactly `af.af_protein`, which the CLI already calls, so no CLI change is needed.

This is the configuration used for PREREG O5.

**Optional AF ranker: `--af-ranker PATH`.** This is the pocket ranker retrained with pLDDT pocket descriptors on AF models of train proteins plus the PDB train rows (`afpdb_pl`). Use it for proteome screening, where scores are pooled across proteins.

It is a trade-off, not an upgrade (train out-of-fold, 2,844 AF models, cluster-CV):

| | effect of the AF ranker vs the CV standard ranker |
|---|---|
| pooled P1K | **+6.7 to +6.9** |
| P@k (pooled, k = #proteins), proteins > 30 % disordered | 51.0 → **58.7** |
| P@k, proteins 10–30 % disordered | 61.9 → 64.3 |
| top-1 precision, proteins > 30 % disordered | 50.0 → 52.9 |
| DCA4 top-N+2, sites far from disorder | **+1.1 [+0.5, +1.8]** |
| DCC12 / DCA4 top-N+2, sites near low confidence | **−3.0 [−5.2, −0.6]** / −2.8 [−5.3, −0.2] |
| PDB-dev non-inferiority at uniform pLDDT 95 (DCC12) | −0.3 [−1.6, +0.9] on the speed preset (inconclusive at the −1 margin); +0.3 [−0.7, +1.4] on REF |

"Near low confidence" (640 sites) means the site is within 8 Å of a prunable segment, or one of its residues has pLDDT < 70.

The ranker learned the true prior that low-pLDDT pockets are rarely sites. It therefore also demotes the minority of real sites next to disordered segments. A variant with only pLDDT < 50 cues (`h50`) is neutral everywhere: near-disorder −0.6 [−2.4, +1.2], and no calibration gain.

**Honest conclusion.**
- **Recall: AF mode is equivalent to the "naive + drop pockets with mean pLDDT < 70" baseline, not better.**
  - AF dev: +0.7 [−1.4, +2.8].
  - Train out-of-fold: −0.5 [−1.3, +0.2] on the REF model; −0.7 [−1.4, +0.1] on the speed preset.
- **What AF mode does deliver:**
  - It removes hallucinated pockets. On train OOF these are pockets with mean pLDDT < 50 that match no site:
    - all proteins: 6.6 % of pockets and 1.4 % of top-(N+2) → 0.1 % and 0.0 %;
    - proteins more than 30 % disordered: 37.1 % and 14.2 % → 0.2 % and 0.3 %.
  - It does so without discarding the 50–70 pLDDT region, and without hurting confident inputs (byte-identical no-op).
  - It beats P2Rank on the identical AF models. AF dev DCC12 top-N+2: 66.9 vs 56.2 (`-c alphafold`) and 53.0 (default). DCA4: 57.8 vs 36.9.
- **The geometric mechanisms do not rescue sites near disorder.** On that stratum, every geometric variant is at or below the filter:
  - prune + soft: −1.6 [−4.3, +1.2];
  - stronger weight curve (50→90): −3.6;
  - λ = 0.25 on pLDDT < 90 side chains: −4.2;
  - oracle crop at pLDDT ≥ 70: −4.8.
- Matched-random pruning costs 2.5–3.1 points (significant). Pruning low-pLDDT segments is specifically harmless; it is not a free regulariser.
- Only 1 of 607 dev sites and 8 of 6,238 train sites lose ≥ 50 % of their residues to pruning.

**Caveat.** The development sets are holo-biased and mostly confident:
- dev: median fraction of residues with pLDDT < 50 is 1 %;
- only 8 dev and 104 train proteins are more than 30 % disordered.

The low-confidence conclusions rest on the 2,844-model train out-of-fold set. On that set the standard ranker is in-sample, but this applies equally to every geometric variant; the ranker comparison uses cluster-CV rankers.

### Strata (train out-of-fold, 2,844 AF models; REF base model; Δ vs the pLDDT < 70 filter unless noted)

| | naive | filter < 70 | **prune+soft (final)** | soft 30/70 | soft 50/90 | λ 0.5 low | λ 0.25 low | CV std ranker | CV AF ranker (pl) | CV AF ranker (h50) |
|---|---|---|---|---|---|---|---|---|---|---|
| DCC12, near-low sites (640) | 64.7 | 65.2 | 63.6 | 62.8 | 61.6 | 62.0 | 60.9 | 63.3 | 60.3 | 62.7 |
| Δ DCC12 near-low | | | −1.6 [−4.3,+1.2] | −2.3 [−4.9,+0.2] | −3.6 [−6.4,−0.8] | −3.1 [−6.2,0.0] | −4.2 [−7.3,−1.1] | −1.9 [−4.6,+1.1] | −4.8 [−7.7,−1.8] | −2.5 [−5.4,+0.6] |
| DCA4, near-low | 54.2 | 54.7 | 55.5 | 55.5 | 53.4 | 52.7 | 53.0 | 55.5 | 52.7 | 55.9 |
| DCC12, other sites (5,598) | 64.1 | 64.7 | 64.3 | 64.1 | 64.7 | 64.3 | 64.4 | 64.7 | 65.1 | 64.5 |
| DCA4, other sites | 52.8 | 53.1 | 53.1 | 52.8 | 53.5 | 53.0 | 53.2 | 52.4 | 53.5 | 52.9 |
| P@k, < 10 % disordered (2,365) | 83.2 | 83.6 | 83.4 | 83.0 | 83.0 | 83.6 | 82.6 | 83.1 | 84.7 | 83.5 |
| P@k, 10–30 % (375) | 59.2 | 61.3 | 60.8 | 60.8 | 61.9 | 60.0 | 58.9 | 61.9 | 64.3 | 63.5 |
| P@k, > 30 % (104) | 51.0 | 52.9 | 50.0 | 50.0 | 52.9 | 58.7 | 61.5 | 51.0 | 58.7 | 51.0 |
| top-1 precision, > 30 % | 45.2 | 48.1 | 50.0 | 48.1 | 49.0 | 51.9 | 51.0 | 50.0 | 52.9 | 49.0 |
| top-(N+2) precision, > 30 % | 35.3 | 36.2 | 37.1 | 37.3 | 38.6 | 38.1 | 37.8 | 36.5 | 37.6 | 37.8 |
| hallucination % (all / top-N+2), > 30 % | 37.1 / 14.2 | 0 / 0 | 0.2 / 0.3 | 4.8 / 1.5 | 0.1 / 0.0 | 0.2 / 0.0 | 0.1 / 0.0 | 0.1 / 0.0 | 0.2 / 0.3 | 0.1 / 0.3 |

Reading the table:
- The CV-ranker columns all use prune+soft geometry.
- AF ranker vs the CV standard ranker, near-low sites: pl −3.0 [−5.2, −0.6] DCC12 and −2.8 [−5.3, −0.2] DCA4; h50 −0.6 [−2.4, +1.2].
- The dev strata (AF dev: 51 near-low sites, 8 proteins > 30 % disordered) are too small to resolve anything (CIs ±10 pts). They are reported in `results/dev_af/tables_strata.md`.

The decision rule was: adopt any mechanism that helps near-low sites without hurting the rest. No mechanism meets it.

The strata table above uses the REF base model (`burrow_all_s0`, 18:05).

### Frozen model confirmation

Base model: `results/dev/freeze/models/burrow_all_default_s0.json`, sha256 e7d1699e…4373e, tag `final`. AF-dev predictions are identical to the pre-freeze `def0` preview.

**Default configuration on AF dev** (prune+soft, standard ranker):

| | DCC12 top-N+2 | DCA4 top-N+2 | P1K |
|---|---|---|---|
| AF mode | 64.3 | 56.3 | 49.2 |
| naive | 63.1 | 54.5 | 46.9 |
| pLDDT < 70 filter | 63.4 | 54.4 | 47.4 |
| P2Rank `-c alphafold` | 56.2 | 36.9 | 31.3 |
| P2Rank default | 53.0 | 37.9 | 32.9 |

- AF mode vs the filter on AF dev: +0.8 [−1.3, +3.1] DCC12, +2.0 [−0.3, +4.2] DCA4.
- Hallucinated pockets: 5.0 % → 0.2 % of all pockets; 0.6 % → 0.1 % of the top-(N+2).
- No-op check: 579/579 byte-identical.

**Train-OOF, default configuration.**
- vs the filter: −0.7 [−1.4, +0.1] DCC12.
- Near-low-confidence sites vs the filter: −2.2 [−5.4, +1.4].
- Proteins more than 30 % disordered: hallucination 33.8 % → 1.8 % of pockets, and 11.4 % → 1.3 % of the top-(N+2).

**Seeds s1/s2.** AF mode vs the filter, DCC12: −0.2 [−2.6, +2.3] (s1) and +0.2 [−2.5, +2.8] (s2). Hallucinated pockets 4.8–4.9 % → 0.2 %. No-op 579/579 byte-identical for both seeds.

**Optional AF ranker, retrained on the frozen model.**
- File: `results/dev_af/models/afr_final_afpdb_pl.json`, sha256 `935f887131987021f7ba380a05bef5adc8eac7467dac7a86d9fe524a83404a67`.
- Base: frozen `all` default s0. AF geometry: prune_soft. Data: AF train + PDB train rows (8.40 M candidates). Seed `SEEDS[0]`.

Effects on the frozen model (train-OOF, cluster-CV, vs the CV standard ranker):

| metric | effect |
|---|---|
| pooled P1K | +6.7 [+4.6, +9.4] |
| P@k, proteins > 30 % disordered | 49.0 → 55.8 |
| P@k, proteins 10–30 % disordered | 62.1 → 65.3 |
| DCC12 at near-low sites | −0.8 [−3.2, +1.5] |
| DCC12 at other sites | +0.7 [−0.0, +1.4] |

- The recall penalty near disorder is smaller on the frozen model than on REF (−3.0), but the direction is the same.
- PDB-dev non-inferiority at uniform pLDDT 95: −0.3 [−1.6, +0.9] DCC12, which is inconclusive at the −1 margin.
- AF dev: DCC12 62.6 vs 64.3 for the default, P1K 49.9 vs 49.2.
- It remains optional.

## Mechanisms

| # | mechanism | definition | status |
|---|---|---|---|
| 1 | spaghetti pruning | per chain, drop terminal runs with pLDDT < 50 and internal runs of ≥ 7 residues with pLDDT < 50 (AF2BIND convention; a fixed prior, not tuned). Residue pLDDT = CA B-factor | ON |
| 2 | soft occlusion | remaining atoms get opacity w = clip((pLDDT − 40)/40, 0, 1) in both absorption grids (full-atom and backbone+Cβ). Atoms with w < 0.5 do not exclude candidate grid points | ON |
| 3 | side-chain tolerance | side-chain atoms beyond Cβ get w × λ. Scope `all` or only residues with pLDDT < 90. Backbone+Cβ buriedness features are always present | off (λ = 1): λ on all side chains −2 to −16 pts; λ on pLDDT<90 side chains hurts near-low sites |
| 4 | AF-aware ranker | standard pocket descriptors + pocket confidence descriptors (`af_rank.AF_FEATURES`). Trained on AF models of Track A train entries with transferred labels, using cross-fitted point scores; mixed with the PDB train ranker rows at uniform pLDDT 100 | optional (`--af-ranker`), not default |
| 5 | inter-domain surrogate | structure-only (no PAE: HTTP 403, docs/AF_BENCHMARK.md §6). Spectral bisection of the Cα contact graph (8 Å). Pocket span across the two parts, plus inter-part packing density | not adopted (no gain) |

Weight-curve variants (opaque at 70 or at 90) and λ variants were run as ablations
(`af_variants.CONFIGS`).

**No-op guarantee.** If every residue has pLDDT ≥ 80 (≥ 90 for the 50/90 curve and for λ restricted to pLDDT < 90), then:
- nothing is pruned;
- all weights are 1;
- the geometry and the predictions are byte-identical to standard mode.

This is tested on 579/579 PDB-dev chains at uniform pLDDT 95 (`tests/test_af_mode_variants.py`; `scripts/af_dev_run_all.sh` step 2).

## Protocol (dev only)

**AF dev.** AF models (AFDB v6) of the 278 Track A dev entries that have transferred sites; 607 sites.
- Site centre: centroid of the transferred ligand atoms, the same definition as Track A's dev `site_center`.
- Metrics follow the LIGYSIS protocol (`pocketdetect.site_eval`): DCC12 top-N+2, DCA4 top-N+2, pooled P1K.
- Cluster bootstrap over `cluster30`, B = 1000. Δ values are paired.
- Hallucination: a pocket whose mean pLDDT is < 50 on the **unpruned** model and that matches no site at DCC ≤ 12 Å (`pocketdetect.af_eval`).

**AF train, out of fold (supplementary, 10× the power).** The AF models of all 2,844 train entries.
- Each model is scored by the point model of the cross-fitting fold that did not contain its `cluster30`.
- Ranker comparisons use cluster-CV rankers trained on the same folds, so every prediction is fully out of fold.
- `cv_pdb_std` is the CV analogue of the standard ranker.

**Controls.**
- naive + drop pockets with mean pLDDT < 70 (`scripts/af_plddt_filter_baseline.py`)
- matched-random pruning, two variants:
  - same residue count, placed as contiguous runs of the same lengths
  - random single residues, until at least the same number of atoms is removed
- oracle crop to residues with pLDDT ≥ 70
- fraction of true sites lost to pruning
- no-op check
- PDB-dev non-inferiority of the AF ranker (margin −1 pt)
- P2Rank 2.5.1 default and `-c alphafold`, run on full and on our pruned input

## Reproduce
```
scripts/af_dev_run_all.sh <base model json> <tag> [<trainset snapshot>]   # ~50 min, 24 workers
python scripts/af_dev_strata.py --sites results/dev_af/af_dev_sites_all.tsv --methods ... --ref naive_plddt70 --out ...
python scripts/af_dev_report.py --tag <tag>
```

## Deployment notes
- Default `--af-mode` needs no CLI change (`af.af_protein`). For the optional `--af-ranker PATH` (in the method developer's CLI), the CLI calls `af_rank.predict_protein_af(P, model, AFRanker.load(PATH), weights=w)` under `--af-mode`.
- The AF ranker is tied to its base model, because it uses that model's point scores and pocket descriptors. It must be retrained whenever the base model changes (`scripts/af_dev_train_ranker.py`).
- The fast backend needs `af_rank.af_pocket_features` (11 cheap per-pocket reductions) plus the opacity weights. Pruning and the weights are input-side only.
- Runtime is not slower than standard mode: pruning removes atoms. Median per-structure wall time on AF dev was 576 ms (prune+soft) vs 638 ms (naive), on a loaded node.
