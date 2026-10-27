# v3b (track T2) — per-protein / relative output floors

Motivation: results/dev/biolip/NONION_ERRORS.md — 65 % of v2's non-ion dev misses are pockets found but
dropped by the global floor (score < 0.03). Started 2026-10-27 ~09:30.

## Gate (fixed by the PI, RESEARCH_TREE D19)
BioLiP-dev non-ion closest top-N+2 ≥ +2.0 vs v2 with paired CI > 0 AND max recall ≥ v2 + 3 AND DCA4 ≥ v2 − 0.5
AND P1K ≥ v2 − 1, for both rankers, AND apo-dev cryptic ≥ v2 − 1 AND AF-dev ≥ v2 − 1. Default and fast preset
identically (report both).

## Variants (pre-declared, 09:30) — post-filters of the NMS'd full list (`src/pocketdetect/burrow/output_rules.py`)
* (a) relative floor `a<rho>`: keep rank < 3 or score ≥ min(0.03, ρ × best score), ρ ∈ {0.1, 0.2, 0.3}
* (b) size-scaled minimum `b<k>`: keep rank < max(3, k·⌈n_res/150⌉) or score ≥ 0.03, k ∈ {1, 2}
* (c) per region `c<R>_<lo>`: v2 plus every pocket with score ≥ lo whose centre is > R Å from all kept pockets,
  R ∈ {12, 16, 20}, lo ∈ {0.005, 0.01}
* (d) combination `a0.2+c16_<lo>`, lo ∈ {0.005, 0.01}
Sanity: rule `v2` applied to the v1 full list reproduces the frozen v2 prediction files exactly (579/579).

## Selection rule (fixed 09:30, before any result)
Train cluster-CV only (`scripts/burrow_v3b_cv.py`: out-of-fold point scores and rankers, held-out chain inputs),
default preset: among rules with, for BOTH rankers, Δ DCA4 top-N+2 ≥ −0.5, Δ one-to-one P1K ≥ −1 and
Δ max recall ≥ +3 vs rule v2, pick the one with the largest mean Δ non-ion closest top-N+2 (`small` ranker on
non-ion sites and `all` ranker on non-ion sites). Then ONE dev confirmation (default and fast preset).

## Results
### Train cluster-CV (`v3b_cv_<run>_<sites>.csv`; Δ vs rule v2; one-to-one P1K is saturated on the train pool)
| ranker / sites | rule | top-N+2 | Δ top-N+2 | Δ DCA4 | Δ P1K 1:1 | Δ max recall | pockets |
|---|---|---|---|---|---|---|---|
| default small / non-ion | v1 | 66.5 | -5.40 | -0.17 | +0.00 | +18.31 | 41.3 |
| default small / non-ion | v2 | 71.9 | +0.00 | +0.00 | +0.00 | +0.00 | 6.7 |
| default small / non-ion | a0.1 | 71.3 | -0.60 | +0.33 | +0.00 | +5.47 | 9.2 |
| default small / non-ion | a0.2 | 72.2 | +0.31 | +0.49 | +0.00 | +2.73 | 7.5 |
| default small / non-ion | a0.3 | 72.3 | +0.42 | +0.38 | +0.00 | +1.52 | 7.0 |
| default small / non-ion | b1 | 72.0 | +0.04 | +0.01 | +0.00 | +0.06 | 6.7 |
| default small / non-ion | b2 | 72.4 | +0.50 | +0.38 | +0.00 | +1.36 | 7.4 |
| default small / non-ion | c12_0.005 | 72.3 | +0.34 | +1.08 | +0.00 | +14.01 | 19.2 |
| default small / non-ion | c12_0.01 | 72.8 | +0.89 | +1.15 | +0.00 | +10.57 | 13.5 |
| default small / non-ion | c16_0.005 | 74.3 | +2.37 | +0.94 | +0.00 | +8.94 | 13.1 |
| default small / non-ion | c16_0.01 | 74.0 | +2.13 | +0.87 | +0.00 | +6.47 | 10.3 |
| default small / non-ion | c20_0.005 | 73.8 | +1.85 | +0.60 | +0.00 | +4.97 | 10.1 |
| default small / non-ion | c20_0.01 | 73.4 | +1.48 | +0.50 | +0.00 | +3.49 | 8.6 |
| default small / non-ion | a0.2+c16_0.005 | 73.8 | +1.87 | +1.07 | +0.00 | +9.93 | 13.6 |
| default small / non-ion | a0.2+c16_0.01 | 73.6 | +1.72 | +1.01 | +0.00 | +7.72 | 10.9 |
| default all / non-ion | v1 | 64.8 | -6.13 | -0.36 | +0.00 | +16.12 | 41.2 |
| default all / non-ion | v2 | 70.9 | +0.00 | +0.00 | +0.00 | +0.00 | 8.1 |
| default all / non-ion | a0.1 | 70.2 | -0.72 | +0.22 | +0.00 | +3.79 | 9.9 |
| default all / non-ion | a0.2 | 71.0 | +0.07 | +0.29 | +0.00 | +1.68 | 8.7 |
| default all / non-ion | a0.3 | 71.0 | +0.12 | +0.13 | +0.00 | +0.80 | 8.3 |
| default all / non-ion | b1 | 70.9 | -0.02 | -0.01 | +0.00 | +0.01 | 8.1 |
| default all / non-ion | b2 | 71.0 | +0.12 | +0.15 | +0.00 | +0.50 | 8.4 |
| default all / non-ion | c12_0.005 | 70.5 | -0.44 | +0.60 | +0.00 | +12.46 | 21.1 |
| default all / non-ion | c12_0.01 | 71.0 | +0.09 | +0.70 | +0.00 | +9.51 | 15.6 |
| default all / non-ion | c16_0.005 | 72.1 | +1.18 | +0.47 | +0.00 | +7.28 | 14.1 |
| default all / non-ion | c16_0.01 | 72.0 | +1.09 | +0.42 | +0.00 | +5.31 | 11.7 |
| default all / non-ion | c20_0.005 | 71.8 | +0.90 | +0.34 | +0.00 | +3.56 | 11.0 |
| default all / non-ion | c20_0.01 | 71.7 | +0.77 | +0.29 | +0.00 | +2.59 | 9.8 |
| default all / non-ion | a0.2+c16_0.005 | 71.7 | +0.81 | +0.63 | +0.00 | +7.91 | 14.4 |
| default all / non-ion | a0.2+c16_0.01 | 71.7 | +0.81 | +0.60 | +0.00 | +6.15 | 12.0 |
| default all / all sites | v1 | 67.7 | -5.54 | +0.00 | +0.00 | +16.64 | 39.9 |
| default all / all sites | v2 | 73.2 | +0.00 | +0.00 | +0.00 | +0.00 | 7.8 |
| default all / all sites | a0.1 | 72.6 | -0.62 | +0.20 | +0.00 | +3.72 | 9.7 |
| default all / all sites | a0.2 | 73.3 | +0.03 | +0.21 | +0.00 | +1.69 | 8.4 |
| default all / all sites | a0.3 | 73.3 | +0.09 | +0.11 | +0.00 | +0.79 | 8.0 |
| default all / all sites | b1 | 73.3 | +0.01 | +0.00 | +0.00 | +0.02 | 7.8 |
| default all / all sites | b2 | 73.5 | +0.20 | +0.15 | +0.00 | +0.51 | 8.1 |
| default all / all sites | c12_0.005 | 73.0 | -0.26 | +0.64 | +0.00 | +13.22 | 20.4 |
| default all / all sites | c12_0.01 | 73.5 | +0.23 | +0.67 | +0.00 | +9.97 | 15.1 |
| default all / all sites | c16_0.005 | 74.6 | +1.33 | +0.48 | +0.00 | +7.94 | 13.6 |
| default all / all sites | c16_0.01 | 74.5 | +1.22 | +0.40 | +0.00 | +5.64 | 11.2 |
| default all / all sites | c20_0.005 | 74.4 | +1.12 | +0.33 | +0.00 | +3.93 | 10.6 |
| default all / all sites | c20_0.01 | 74.1 | +0.89 | +0.29 | +0.00 | +2.76 | 9.4 |
| default all / all sites | a0.2+c16_0.005 | 74.2 | +0.97 | +0.57 | +0.00 | +8.56 | 13.9 |
| default all / all sites | a0.2+c16_0.01 | 74.2 | +0.91 | +0.52 | +0.00 | +6.41 | 11.6 |
| fast small / non-ion | v1 | 65.7 | -5.49 | +0.44 | +0.00 | +20.73 | 39.7 |
| fast small / non-ion | v2 | 71.2 | +0.00 | +0.00 | +0.00 | +0.00 | 6.1 |
| fast small / non-ion | a0.1 | 70.6 | -0.61 | +0.56 | +0.00 | +6.34 | 8.4 |
| fast small / non-ion | a0.2 | 71.3 | +0.09 | +0.44 | +0.00 | +3.21 | 6.9 |
| fast small / non-ion | a0.3 | 71.6 | +0.38 | +0.27 | +0.00 | +1.76 | 6.4 |
| fast small / non-ion | b1 | 71.3 | +0.14 | +0.05 | +0.00 | +0.14 | 6.1 |
| fast small / non-ion | b2 | 71.7 | +0.55 | +0.36 | +0.00 | +1.77 | 6.9 |
| fast small / non-ion | c12_0.005 | 72.0 | +0.78 | +1.46 | +0.00 | +14.98 | 17.1 |
| fast small / non-ion | c12_0.01 | 72.6 | +1.44 | +1.47 | +0.00 | +11.20 | 11.8 |
| fast small / non-ion | c16_0.005 | 73.7 | +2.54 | +1.11 | +0.00 | +8.92 | 11.4 |
| fast small / non-ion | c16_0.01 | 73.4 | +2.26 | +1.02 | +0.00 | +6.31 | 9.0 |
| fast small / non-ion | c20_0.005 | 73.1 | +1.93 | +0.72 | +0.00 | +4.80 | 8.8 |
| fast small / non-ion | c20_0.01 | 72.8 | +1.66 | +0.66 | +0.00 | +3.40 | 7.5 |
| fast small / non-ion | a0.2+c16_0.005 | 73.0 | +1.79 | +1.33 | +0.00 | +10.22 | 11.9 |
| fast small / non-ion | a0.2+c16_0.01 | 72.8 | +1.58 | +1.25 | +0.00 | +7.91 | 9.5 |
| fast all / non-ion | v1 | 64.4 | -5.88 | +0.09 | +0.00 | +18.51 | 39.6 |
| fast all / non-ion | v2 | 70.3 | +0.00 | +0.00 | +0.00 | +0.00 | 7.3 |
| fast all / non-ion | a0.1 | 69.7 | -0.56 | +0.38 | +0.00 | +4.43 | 9.1 |
| fast all / non-ion | a0.2 | 70.3 | +0.06 | +0.30 | +0.00 | +2.04 | 7.9 |
| fast all / non-ion | a0.3 | 70.6 | +0.33 | +0.30 | +0.00 | +1.14 | 7.5 |
| fast all / non-ion | b1 | 70.4 | +0.14 | +0.06 | +0.00 | +0.15 | 7.3 |
| fast all / non-ion | b2 | 70.5 | +0.24 | +0.17 | +0.00 | +0.99 | 7.7 |
| fast all / non-ion | c12_0.005 | 70.0 | -0.29 | +0.88 | +0.00 | +13.80 | 19.3 |
| fast all / non-ion | c12_0.01 | 70.6 | +0.29 | +0.86 | +0.00 | +10.61 | 14.1 |
| fast all / non-ion | c16_0.005 | 71.8 | +1.50 | +0.64 | +0.00 | +7.61 | 12.6 |
| fast all / non-ion | c16_0.01 | 71.7 | +1.43 | +0.59 | +0.00 | +5.53 | 10.4 |
| fast all / non-ion | c20_0.005 | 71.5 | +1.17 | +0.44 | +0.00 | +3.75 | 9.8 |
| fast all / non-ion | c20_0.01 | 71.3 | +1.01 | +0.41 | +0.00 | +2.78 | 8.7 |
| fast all / non-ion | a0.2+c16_0.005 | 71.4 | +1.10 | +0.74 | +0.00 | +8.50 | 13.0 |
| fast all / non-ion | a0.2+c16_0.01 | 71.4 | +1.08 | +0.71 | +0.00 | +6.58 | 10.8 |
| fast all / all sites | v1 | 67.2 | -5.19 | +0.58 | +0.00 | +18.32 | 38.3 |
| fast all / all sites | v2 | 72.4 | +0.00 | +0.00 | +0.00 | +0.00 | 7.0 |
| fast all / all sites | a0.1 | 71.8 | -0.59 | +0.38 | +0.00 | +4.22 | 9.0 |
| fast all / all sites | a0.2 | 72.4 | +0.00 | +0.23 | +0.00 | +1.96 | 7.6 |
| fast all / all sites | a0.3 | 72.6 | +0.21 | +0.20 | +0.00 | +0.91 | 7.3 |
| fast all / all sites | b1 | 72.4 | +0.09 | +0.04 | +0.00 | +0.10 | 7.0 |
| fast all / all sites | b2 | 72.7 | +0.34 | +0.22 | +0.00 | +0.97 | 7.4 |
| fast all / all sites | c12_0.005 | 72.3 | -0.02 | +1.00 | +0.00 | +14.15 | 18.7 |
| fast all / all sites | c12_0.01 | 73.0 | +0.65 | +0.95 | +0.00 | +10.88 | 13.7 |
| fast all / all sites | c16_0.005 | 74.2 | +1.85 | +0.70 | +0.00 | +8.20 | 12.2 |
| fast all / all sites | c16_0.01 | 74.1 | +1.77 | +0.65 | +0.00 | +5.94 | 10.0 |
| fast all / all sites | c20_0.005 | 73.7 | +1.37 | +0.46 | +0.00 | +3.90 | 9.5 |
| fast all / all sites | c20_0.01 | 73.5 | +1.19 | +0.42 | +0.00 | +2.88 | 8.4 |
| fast all / all sites | a0.2+c16_0.005 | 73.8 | +1.42 | +0.74 | +0.00 | +9.03 | 12.5 |
| fast all / all sites | a0.2+c16_0.01 | 73.7 | +1.36 | +0.70 | +0.00 | +6.94 | 10.4 |

Selection (pre-declared rule, default preset): all region rules satisfy the constraints; mean Δ non-ion top-N+2 over
the two rankers: c16_0.005 +1.78 (small +2.37, all +1.18), c16_0.01 +1.61, c20_0.005 +1.38, a0.2+c16_0.005 +1.34;
relative (a) and size-scaled (b) floors ≤ +0.5. **Selected: c16_0.005.** The fast preset shows the same ordering.

### Single dev confirmation (`v3b/dev_c16_0.005_{default,fast}.csv`; paired vs the same models with the v2 floor)
| preset | ranker | table | top-N+2 v2 → v3b | Δ [95 % CI] | Δ DCA4 | max recall | P1K | pockets/structure |
|---|---|---|---|---|---|---|---|---|
| default | small | biolip_nonion | 71.2 → 72.4 | +1.2 [+0.1,+2.4] | +0.5 [-0.1,+1.3] | 80.1 → 89.7 | 61.3 → 61.3 | 6.9 → 13.6 |
| default | small | apo_cryptic | 71.5 → 70.9 | -0.5 [-1.2,+0.1] | +0.1 [-0.2,+0.4] | 84.7 → 93.1 | 57.9 → 57.9 | 6.2 → 12.9 |
| default | small | apo_noncryptic | 59.2 → 69.2 | +9.9 [+6.6,+13.1] | – | 65.1 → 84.8 | 58.5 → 58.5 | 6.4 → 13.3 |
| default | small | apo_track_dev_small | 67.3 → 70.0 | +2.7 [+0.0,+6.7] | +1.2 [+0.0,+3.4] | 75.4 → 87.7 | 32.3 → 28.2 | 6.3 → 12.7 |
| default | small | af_dev_all | 66.1 → 68.7 | +2.6 [+0.9,+4.9] | +0.5 [+0.0,+1.1] | 73.0 → 85.5 | 45.8 → 45.8 | 6.9 → 14.1 |
| default | all | biolip_nonion | 68.3 → 68.8 | +0.5 [-0.1,+1.2] | +0.1 [+0.0,+0.4] | 82.2 → 89.4 | 58.4 → 58.4 | 8.4 → 14.6 |
| default | all | biolip_all | 70.8 → 71.2 | +0.3 [-0.2,+0.9] | +0.2 [+0.0,+0.4] | 81.6 → 90.2 | 69.7 → 69.7 | 8.1 → 14.1 |
| default | all | apo_cryptic | 68.5 → 68.1 | -0.4 [-0.9,+0.1] | -0.1 [-0.3,+0.0] | 87.4 → 95.2 | 53.8 → 53.8 | 7.5 → 13.9 |
| default | all | apo_noncryptic | 59.0 → 66.9 | +7.9 [+5.1,+10.8] | – | 68.0 → 85.5 | 56.7 → 56.7 | 7.7 → 14.3 |
| default | all | apo_track_dev_all | 59.4 → 61.2 | +1.8 [+0.0,+5.8] | +1.2 [+0.0,+3.8] | 75.4 → 87.1 | 31.9 → 31.9 | 7.6 → 13.1 |
| default | all | af_dev_all | 71.2 → 72.3 | +1.2 [+0.2,+2.3] | +0.8 [+0.2,+1.6] | 78.3 → 87.5 | 49.2 → 49.2 | 8.0 → 15.1 |
| fast | small | biolip_nonion | 69.9 → 71.1 | +1.1 [+0.3,+2.0] | +0.5 [+0.1,+1.0] | 76.4 → 84.2 | 58.6 → 58.6 | 6.2 → 11.5 |
| fast | small | apo_cryptic | 71.4 → 71.3 | -0.1 [-0.8,+0.6] | +0.2 [+0.0,+0.5] | 85.0 → 92.8 | 56.7 → 56.7 | 5.8 → 11.3 |
| fast | small | apo_noncryptic | 61.5 → 71.9 | +10.4 [+7.4,+13.4] | – | 65.7 → 84.2 | 55.0 → 55.0 | 5.9 → 11.6 |
| fast | small | apo_track_dev_small | 64.2 → 68.8 | +4.6 [+0.9,+8.4] | +2.7 [+0.9,+4.6] | 70.8 → 83.8 | 31.3 → 24.7 | 5.6 → 11.1 |
| fast | small | af_dev_all | 64.9 → 68.2 | +3.3 [+0.8,+6.8] | +1.2 [+0.3,+2.1] | 71.5 → 82.9 | 44.3 → 44.3 | 6.1 → 12.2 |
| fast | all | biolip_nonion | 67.7 → 68.2 | +0.5 [-0.2,+1.3] | +0.1 [+0.0,+0.4] | 77.5 → 85.3 | 55.8 → 55.8 | 7.5 → 12.9 |
| fast | all | biolip_all | 70.0 → 70.8 | +0.8 [+0.2,+1.6] | +0.1 [+0.0,+0.3] | 78.0 → 86.4 | 67.4 → 67.4 | 7.3 → 12.5 |
| fast | all | apo_cryptic | 66.8 → 66.7 | -0.1 [-0.7,+0.6] | +0.3 [+0.0,+0.6] | 86.7 → 94.0 | 51.9 → 51.9 | 7.0 → 12.5 |
| fast | all | apo_noncryptic | 61.9 → 70.1 | +8.2 [+5.4,+10.7] | – | 70.0 → 87.0 | 54.2 → 54.2 | 7.1 → 12.9 |
| fast | all | apo_track_dev_all | 61.8 → 64.9 | +3.1 [+0.7,+6.1] | +0.9 [+0.0,+2.0] | 71.1 → 83.4 | 29.4 → 29.4 | 6.8 → 11.8 |
| fast | all | af_dev_all | 70.0 → 71.3 | +1.3 [+0.3,+2.6] | +0.2 [+0.0,+0.6] | 76.4 → 85.3 | 49.7 → 49.7 | 7.0 → 13.3 |

## Decision
**Gate NOT met** (both presets): BioLiP-dev non-ion closest top-N+2 +1.2 [+0.1, +2.4] (`small`) and +0.5 [−0.1, +1.2]
(`all`) for the default preset, +1.1 / +0.5 for the fast preset — below the required +2.0. All guards pass: max recall
+7 to +10 (≥ +3), DCA4 +0.1…+0.5, P1K unchanged (the extra pockets score below the pooled top 1,000), apo-dev
cryptic −0.1…−0.5 (≥ −1), AF dev +1.2…+3.3. No FREEZE_v3; v2 stays final. Dev use in this cycle: one confirmation of
the selected rule (both presets; the fast full lists were produced once for it).

Observation for the report: the per-region rule recovers most of the coverage lost by the v2 floor (max recall
80 → 90 % on non-ion dev, 65 → 85 % on apo non-cryptic sites, ~2× pockets, still ~14 per structure) at no cost in
closest-pocket top-N+2, DCA4 or P1K — and gains +8 to +10 top-N+2 on apo non-cryptic sites (the closest-pocket rule
rewards a pocket near every site). It is implemented (`burrow.output_rules`, `apply_rule("c16_0.005", ...)`) but not
wired into the CLI, because it did not pass the pre-registered gate.
