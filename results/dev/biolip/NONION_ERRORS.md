# Non-ion error analysis — v2 `small` (default preset, seed 0) on BioLiP2-dev (DEV only)

Sites: `results/dev/biolip/dev_sites_small.tsv` (D6-audited dev, 400 chains, **805 non-ion sites**).
Metric: LIGYSIS closest-pocket top-N+2 at DCC ≤ 12 Å (`pocketdetect.site_eval`). Methods on identical
single-chain inputs: BURROW v2 `small` default s0 (= frozen `freeze_v2`), the same model without the output
floor (v1 list), P2Rank 2.5.1 default, fpocket 4.0.3 + P2Rank 2.5.1 rescoring (fpocket+PRANK, run for this
analysis with `baselines/run_fpocket_prank.py`). Per-site table: `nonion_errors_sites.csv`; code:
`scripts/burrow_nonion_errors.py`. No model was changed.

**Overall:** v2 **71.2** · fpocket+PRANK 66.0 · v1 (full list) 65.0 · P2Rank 58.4.

## v2 misses by mechanism (232 of 805 sites)
| miss type | definition | sites | share of misses |
|---|---|---|---|
| removed by output floor | a pocket within 12 Å exists in the full list but scores < 0.03 (dropped) | 151 | 65 % |
| ranking | a kept pocket within 12 Å exists, but only at rank > N+2 | 52 | 22 % |
| closest-pocket rule | a pocket of rank ≤ N+2 is within 12 Å, but a lower-ranked one is closer to the site centre | 20 | 9 % |
| candidate coverage | no pocket within 12 Å even in the full list | 9 | 4 % |

Candidate generation is not the bottleneck (9 coverage failures); **two thirds of the misses are sites whose
nearest pocket is found but scored very low** (floor-removed; median buriedness 0.55 vs
0.82 for hits, median relative WCN 0.89 vs 1.15).

## Stratified hit rates (%) and v2 miss mechanisms (counts)
| ligand class | n | v2 | P2Rank | fpocket+PRANK | v2 misses: floor / ranking / closest-rule / coverage |
|---|---|---|---|---|---|
| additive/fragment | 106 | 56.6 | 50.0 | 51.9 | 36 / 6 / 4 / 0 |
| drug-like/other | 321 | 74.5 | 67.0 | 67.9 | 56 / 17 / 6 / 3 |
| lipid/detergent | 41 | 48.8 | 14.6 | 34.1 | 11 / 7 / 3 / 0 |
| nucleotide/cofactor | 238 | 77.7 | 60.9 | 75.6 | 25 / 16 / 7 / 5 |
| sugar | 99 | 69.7 | 51.5 | 64.6 | 23 / 6 / 0 / 1 |

| ligand heavy atoms | n | v2 | P2Rank | fpocket+PRANK | v2 misses: floor / ranking / closest-rule / coverage |
|---|---|---|---|---|---|
| <10 | 156 | 62.8 | 53.8 | 55.8 | 45 / 9 / 4 / 0 |
| 10-19 | 153 | 68.6 | 62.7 | 64.7 | 33 / 12 / 3 / 0 |
| 20-34 | 146 | 74.0 | 61.6 | 68.5 | 22 / 9 / 7 / 0 |
| ≥35 | 349 | 74.8 | 57.3 | 69.9 | 51 / 22 / 6 / 9 |

| site buriedness (blocked@16 of points ≤4 Å of ligand) | n | v2 | P2Rank | fpocket+PRANK | v2 misses: floor / ranking / closest-rule / coverage |
|---|---|---|---|---|---|
| <0.50 | 99 | 22.2 | 2.0 | 15.2 | 59 / 11 / 1 / 6 |
| 0.50-0.65 | 162 | 48.8 | 19.8 | 39.5 | 58 / 19 / 3 / 3 |
| 0.65-0.80 | 215 | 73.5 | 62.8 | 73.5 | 33 / 14 / 10 / 0 |
| ≥0.80 | 329 | 95.4 | 91.5 | 89.4 | 1 / 8 / 6 / 0 |

| site WCN / chain median (flexibility proxy) | n | v2 | P2Rank | fpocket+PRANK | v2 misses: floor / ranking / closest-rule / coverage |
|---|---|---|---|---|---|
| <0.8 | 70 | 22.9 | 4.3 | 11.4 | 44 / 5 / 0 / 5 |
| 0.8-1.0 | 178 | 46.6 | 30.9 | 44.4 | 66 / 20 / 7 / 2 |
| 1.0-1.2 | 310 | 78.1 | 61.0 | 74.2 | 39 / 19 / 8 / 2 |
| ≥1.2 | 247 | 93.9 | 90.3 | 86.6 | 2 / 8 / 5 / 0 |

| residues from nearest chain terminus | n | v2 | P2Rank | fpocket+PRANK | v2 misses: floor / ranking / closest-rule / coverage |
|---|---|---|---|---|---|
| ≤10 | 138 | 78.3 | 65.2 | 73.9 | 17 / 7 / 6 / 0 |
| 11-30 | 238 | 76.5 | 63.9 | 71.0 | 38 / 12 / 3 / 3 |
| >30 | 429 | 66.0 | 53.1 | 60.6 | 96 / 33 / 11 / 6 |

| sites in the chain (N) | n | v2 | P2Rank | fpocket+PRANK | v2 misses: floor / ranking / closest-rule / coverage |
|---|---|---|---|---|---|
| 1 | 219 | 80.4 | 74.0 | 76.3 | 22 / 12 / 8 / 1 |
| 2 | 190 | 72.1 | 60.5 | 64.7 | 38 / 10 / 4 / 1 |
| 3-4 | 194 | 66.0 | 60.3 | 62.4 | 43 / 20 / 2 / 1 |
| ≥5 | 202 | 65.3 | 37.6 | 59.4 | 48 / 10 / 6 / 6 |

| peptide-adjacent (Track A peptide mask) | n | v2 | P2Rank | fpocket+PRANK | v2 misses: floor / ranking / closest-rule / coverage |
|---|---|---|---|---|---|
| False | 801 | 71.2 | 58.3 | 65.8 | 151 / 52 / 19 / 9 |
| True | 4 | 75.0 | 75.0 | 100.0 | 0 / 0 / 1 / 0 |

## Discordant sites
* v2 misses but P2Rank or fpocket+PRANK hits: **72** sites (removed by floor 35, ranking 22, closest-pocket rule 14, coverage 1).
  Medians — ligand atoms 26, buriedness 0.66, rel. WCN 0.98, sites/chain 3
  (all v2 hits: 31, 0.82, 1.15, 2).
* v2 hits while both baselines miss: **64** sites (median buriedness 0.58, rel. WCN 0.96).

## Findings
1. **Shallow / exposed / flexible sites** are the dominant failure mode: hit rate falls from ~95 % (buriedness ≥ 0.8,
   rel. WCN ≥ 1.2) to ~22 % (< 0.5 / < 0.8) — for every method (P2Rank 2–4 % there); BURROW keeps the largest margin
   there, but most of its misses in these strata are floor removals, i.e. the pocket is found and ranked low.
2. **Ligand class:** additives/fragments and lipids/detergents are hit least (many are crystallisation /
   membrane-mimetic ligands on surfaces); nucleotides/cofactors and drug-like ligands are hit best.
3. **Many sites per chain:** with ≥ 5 sites per chain v2 keeps a large lead over P2Rank, but the top-N+2 budget is
   shared between deep competing pockets; ranking misses concentrate in buried sites (where the floor is irrelevant).
4. The **closest-pocket rule** costs only ~9 % of v2's misses (it was the main issue for v1's long lists).
5. No terminus effect (sites near termini are hit slightly more often), and peptide-adjacent sites are not harder.

## Follow-up ideas (evidence-backed; not tested)
1. **Context-dependent output floor** (largest pool: 151 sites): replace the global floor 0.03 by a floor relative to the
   protein's best pocket or by "≥ 1 pocket per distinct surface region" — floor removals are low-scoring but
   *found* pockets on shallow sites; a relative floor would keep them on proteins whose best pocket is also weak.
   Must be selected on train-CV with the D12 criterion (any-pocket recall should improve, closest-pocket must not drop).
2. **Shallow-site ranker features / calibration:** the ranker's scores collapse on low-buriedness, low-WCN sites;
   add exposure-aware descriptors (local surface curvature, polarity pattern of the pocket rim, hydrophobic
   patch size) or train a separate calibration for shallow candidates (buriedness < 0.65 strata hold ~32 % of the
   sites but ~69 % of the misses).
3. **Label hygiene for `small`:** additives/fragments and detergents/lipids are both weakly predictable and often
   non-biological; report them as a separate stratum and test a `small` ranker trained with them down-weighted
   (expected: better calibration on drug-like sites, explicit trade-off on the additive stratum).
4. **Two-head ranking for multi-site chains:** within-protein LambdaRank improved top-N by +1.0 on train-CV but lost
   pooled P1K (phase 3, V3); a two-head model (pairwise head for the order inside a protein, binary head for the
   cross-protein score/floor) targets the ranking misses in buried, multi-site chains without the P1K loss.
5. **Flexible-region awareness:** low-WCN sites fail for all methods; the backbone-only buriedness already exists as a
   point feature — an AF/apo-style side-chain-tolerant *candidate* pass (not only features) on low-WCN regions could
   raise scores of partially formed shallow sites (H2/H3 did not help as ranker/training changes; this would change
   candidate geometry).
