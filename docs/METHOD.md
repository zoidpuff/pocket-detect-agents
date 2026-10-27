# BURROW — method description (final, FREEZE v2)

BURROW = **B**uriedness-driven **U**ltra-fast **R**anking **O**f p**O**ckets, **W**ith confidence
awareness. Spec: DESIGN.md §2 + §9; decisions D6 (audited data), D8 (AF mode), D12/D13 (output
floor, v2). Code: `src/pocketdetect/burrow/` (CPU reference implementation, this document);
`src/pocketdetect/burrow/fast/` is the fast CPU/GPU backend (separate owner, equivalence-tested
against this reference). Shipped models: FREEZE v2 (`results/dev/FREEZE_v2.md`).

## 1. Usage
```
burrow predict <files|dirs|@list> --out OUT [--ranker all|small|geometry] [--preset default|accurate]
               [--min-score none|<float>] [--af-mode [--af-ranker PATH]] [--model M.json]
               [--threads 1] [--no-points]
```
* Output per structure: `OUT/<stem>.csv` (`rank,score,center_x,center_y,center_z,residues`; the
  project's common prediction format), `OUT/<stem>.points.csv` (`rank,x,y,z` pocket volume
  elements), plus `OUT/timing.csv`.
* `--ranker all` (default; ligands incl. ions/metals) or `small` (non-ion ligands; P2Rank-style
  ligand definitions). `geometry` = learning-free baseline (§8).
* `--preset default|accurate` selects the model JSON (§6). A model JSON carries the point model,
  the pocket ranker, its feature design (`meta.feature_cfg`, `meta.score_floor`), the pocket
  formation / output settings (`formation`, incl. the output floor) and provenance (`meta.provenance`).
* `--min-score none` outputs the full NMS'd pocket list (≤ 50 pockets; = FREEZE v1 behaviour,
  bit-identical); a float overrides the output floor (§5.4).
* `--device` is accepted for compatibility; this reference CLI runs on CPU (GPU: fast backend).

## 2. Input
Protein residues only (standard and modified amino acids such as MSE, heavy atoms, first model,
first altloc): ligands, waters, ions, glycans and nucleic acids are removed, so predictions are
invariant to HETATM records (tested). Per-atom tables (`prep.py`): Bondi radii, 13 atom classes,
7 residue groups, weighted contact number (CA-based Σ 1/d², a flexibility proxy), potential
metal-coordinating atoms, first-hit chemistry class (apolar / polar N,O / charged / aromatic).
B-factors are never used as features (pLDDT is used only by `--af-mode`, §7).

## 3. Candidate points (volume elements) and point features
All geometry is computed in a canonical PCA frame (rotation invariance; tested).
1. **Solvent-shell grid** (1 Å): voxel centres with |x − a| ≥ r_vdW + 1.0 Å for every atom and the
   nearest atom ≤ 6 Å.
2. **Dual absorption grid**: all heavy atoms and backbone+Cβ only, occlusion radius r_vdW + 0.5 Å,
   per-voxel owner class (soft occlusion via per-atom opacity weights is supported, §7).
3. **Cascade**: LIGSITE protein–solvent–protein count (7 directions, 8 Å); points with PSP ≥ 2 are
   kept (volume 1 Å³) plus a coarse sub-lattice of the rest (volume = sub-lattice spacing³).
4. **Metal candidates**: positions 1.95–2.35 Å from a side-chain coordinator with ≥ 2 coordinating
   atoms (His ND1/NE2, Cys SG, Asp/Glu O, Met SD, backbone O) within 2.6 Å, no other atom < 2.4 Å,
   1.5 Å dedup, ≤ 1 per coordinating-atom cluster.
5. **Point features** (84 default / 86 accurate): Fibonacci ray marching through both grids gives
   the blocked fraction at several lengths (full-atom and backbone+Cβ — the latter side-chain
   tolerant), mean first-hit depth, first-hit chemistry fractions and the escape fraction (mouth
   openness); PSP; nearest-atom distance; metal flag, #coordinators ≤ 2.8 Å and their angular
   spread; atom counts at 4/8/10/12 Å; atom-class / residue-group counts at 4 Å and fractions at
   8/12 Å; mean WCN at 8/12 Å; protrusion. Geometric point value b = mean(blocked@10, @16, @longest).

## 4. Point scorer
LightGBM binary classifier, 250 trees × 63 leaves (bagging 0.8, feature fraction 0.8; seeds from
`pocketdetect.seeds.SEEDS`). Labels from Track A's pooled LIGYSIS-style sites (§9): `all` — point
≤ 4 Å of a non-ion ligand atom or ≤ 3 Å of an ion; `small` — ≤ 4 Å of a non-ion ligand atom.
Training sample per structure: 30 % of positives, 3 % of negatives (deterministic per structure
id). Default preset: only points with b ≥ 0.7 (or metal candidates) are scored, the rest get 0
(~45 % of the default point set are scored).

## 5. Pockets
### 5.1 Candidate pockets
Union of a geometric pass (single-linkage components, 2.05 Å, of points with b ≥ 0.40/0.55/0.70/0.85)
and a score pass (point score ≥ 0.15/0.30/0.50); chaining control (components longer than 20 Å are
split by weighted k-means into ~12 Å pieces); multi-scale sub-pockets (components longer than 10 Å
are also split into ~6 Å pieces); minimum volume 10 Å³ (metal candidates exempt); identical point
sets deduplicated. Centre = (volume × value)-weighted centroid of the points.
### 5.2 Pocket ranker
LightGBM (300 trees, 15 leaves) on 76 pocket descriptors (point-score aggregates and quantiles,
volume, buriedness means/quantiles incl. backbone-only, first-hit chemistry, residue groups, PCA
shape and extent, #residues, #chains lining, source level, protein size). Training uses
**cross-fitted** point scores (5 folds of Track A `cluster30`). Graded target with cross-entropy:
q = min DCC over target sites whose ligand atoms lie within 4 Å of the pocket centre;
target = clip(1 − q/12, 0, 1).
### 5.3 Suppression
Greedy in ranker-score order: a pocket is suppressed if its centre is < 10 Å from a kept pocket,
its residue set has Jaccard ≥ 0.75 with one, or its point-overlap coefficient with one is ≥ 0.3
(nested multi-scale duplicates); at most 50 pockets. (10 Å maximised one-to-one P1K and DCA4 on dev;
sensitivity curve in `results/dev/biolip/PHASE2.md`.)
### 5.4 Output score floor (v2; decisions D12/D13)
After suppression, pockets with ranker score < **0.03** are dropped once 3 pockets have been kept
(`formation.min_score = 0.03`, `min_keep = 3`). τ was chosen on train cluster-CV only
(`scripts/burrow_tau_cv.py`). **Trade-off:** the LIGYSIS closest-pocket protocol credits a site only
to the pocket closest to its centre, so long output lists are penalised: low-ranked pockets near a
site "steal" it from the ranked hit. The floor shortens the list from ~42 to 6–8 pockets per
structure and raises closest-pocket top-N+2 by 5–10 points on every dev set (incl. apo/cryptic),
leaves P1K unchanged and slightly raises DCA4 top-N+2, but lowers **coverage**: max recall falls
from ~99 % to 73–87 % (apo non-cryptic 97 → 65 %) and any-pocket top-N+2 by 0.3–4.6 points
(`results/dev/biolip/V2_CYCLE.md`). For exhaustive pocket discovery use `--min-score none`.
**Optional `--output-rule region`** (v3b, not the default; `burrow.output_rules`, rule `c16_0.005`): the v2 floor
plus every pocket with score ≥ 0.005 whose centre is > 16 Å from all pockets kept so far (applied as a post-filter
after either backend). More coverage for exhaustive discovery: on dev, max recall +7 to +10 points at unchanged
closest-pocket top-N+2, DCA4 and P1K (~14 pockets/structure); dev evidence only — it missed the +2.0 non-ion
top-N+2 gate for a new default (`results/dev/biolip/V3b_CYCLE.md`).
### 5.5 Reported residues
Residues with a heavy atom ≤ 4 Å of a pocket point; if none (pockets of distant sub-lattice
points only), residues with any atom ≤ 6 Å of the centre (LIGYSIS rule for point-less methods);
else the nearest atom's residue. Output-only (ranking is unaffected); never empty.

## 6. Presets
| preset | rays | ray lengths (Å) | sub-lattice | point model | scored points | est. fast-backend cost |
|---|---|---|---|---|---|---|
| **default** | 30 | 6/10/16 | 3 Å | 250 × 63 | b ≥ 0.7 or metal (~45 %) | ≈ 80 ms/structure/core |
| accurate | 60 | 6/10/16/24 | 2 Å | 250 × 63 | all | ≈ 200 ms/structure/core |
| fast (additional, not pre-registered) | 12 | 6/10/16 | 1.5 Å grid, sub-lattice 2, shell 5.5 Å, PSP 6 Å | 100 × 31 | b ≥ 0.85 (~20 %) | 30.7 ms measured (speed200 median) |

Everything else is shared, including the output floor (`fast` also scales pocket formation with its 1.5 Å
grid: link 3.08 Å, min volume 3 points; selection and dev numbers in `results/dev/biolip/FAST_PRESET.md`;
the fast GPU path does not support the 1.5 Å grid yet). Choice by the PI's rule (default within
0.5 pt of the best setting on top-N+2 and DCA4 on dev); `accurate` is shipped because it is > 1 pt
better on non-ion sites. Sweep and estimates: `results/dev/biolip/PHASE2.md`.

## 7. AlphaFold mode (`--af-mode`; decision D8, owner: AF-mode developer, `af.py`, `af_*.py`)
Low-pLDDT "spaghetti" pruning (terminal segments and internal stretches ≥ 7 residues with
pLDDT < 50) and soft occlusion (opacity w = clip((pLDDT − 40)/40, 0, 1); atoms with w < 0.5 never
exclude candidate points), then the standard ranker. `--af-ranker PATH` optionally applies an
AF-specific ranker. With uniform pLDDT ≥ 90 the geometry is identical to standard mode (tested).
Details and results: `results/dev_af/`.

## 8. Geometry-only baseline (`--ranker geometry`)
No learning: geometric candidates ranked by buried volume Σ vol · clip((b − 0.7)/0.3, 0, 1)², same
suppression.

## 9. Training data
Track A's leakage-controlled, LIGYSIS-style BioLiP2 sets (docs/TRAINSET.md), audit-filtered per D6:
6,475 train representative chains (+ 1,709 stored assembly-1 inputs) and 579 dev chains; sites =
pooled ligands of up to 20 entries per UniProt superposed on the representative. Model provenance
records the sha256 of the train/dev manifests, of `data/manifests/test_union.tsv` and of the audit
exclusion list, the git commit and the seed. Twelve models per freeze: SEEDS[:3] × {all, small} ×
{default, accurate}. v2 research cycle (AF-model, side-chain-dropout and apo-chain training
augmentation; apo-oriented ranker features) gave no gain and is documented in
`results/dev/biolip/V2_CYCLE.md`; phase-3 ranker variants in `results/dev/biolip/PHASE3.md`.

## 10. Dev results and history
* Final dev numbers (v1 = frozen models without floor; v2 = with floor): `results/dev/biolip/PHASE2.md`,
  `PHASE3.md`, `V2_CYCLE.md`.
* Phase 1 (interim CHEN11/JOINED, gate G1): `results/dev/G1.md` (incl. the BioLiP2 re-measurement:
  G1 not met with 40 geometric candidates; accepted deviation), `results/dev/DEV_RESULTS.md`.

## 11. Timing — reference implementation (ms/structure, 1 core, warm)
`scripts/burrow_timing.py`, 60 audited dev chains (median 1,900 heavy atoms), shared node (load
30–60). Raw: `results/dev/biolip/timing_ref_{default,accurate}.csv`. The fast backend is the speed
deliverable; the reference implementation is for correctness.

| stage (median) | default | accurate |
|---|---|---|
| parse | 15.7 | 14.7 |
| grid points / dual grid + PSP / metal | 13.9 / 14.6 / 18.0 | 13.8 / 14.9 / 16.8 |
| rays | 19.3 | 76.4 |
| neighbour sums / misc | 18.7 / 9.4 | 26.9 / 12.6 |
| point scoring | 26.1 | 108.0 |
| candidate pockets + residues | 90.4 | 110.2 |
| pocket features + ranker + suppression | 43.4 | 42.5 |
| **total excl. parse** | **258.6** | **407.2** |

## 12. Tests
`tests/test_burrow_pipeline.py` (synthetic buried cavity ranked first, rigid-motion and rotation
invariance, HETATM invariance, engineered metal site, AF no-op and pruning, output format),
`tests/test_burrow_biolip.py` (single-chain input extraction, shared-graph single linkage,
vectorised pocket features), `tests/test_burrow_cli_minscore.py` (v2 default floor, `--min-score
none` = v1 bit-identical, floored list is a prefix of the full list).
