# BURROW — method description (phase 2, CPU reference implementation)

Owner: method developer. Spec: DESIGN.md §2 + §9. Code: `src/pocketdetect/burrow/` (reference);
`src/pocketdetect/burrow/fast/` is the fast backend (separate owner, equivalence-tested).
Status: phase 2 — trained on Track A's BioLiP2 LIGYSIS-style train set, tuned on its dev set
(results: `results/dev/biolip/PHASE2.md`). Phase-1 interim results (CHEN11/JOINED) remain in
`results/dev/G1.md`, `results/dev/DEV_RESULTS.md`.

## Usage
```
burrow predict <files|dirs|@list> --out OUT [--af-mode] [--ranker all|small|geometry] \
       [--preset default|accurate] [--model M.json] [--device cpu] [--threads 1]
```
Writes `OUT/<stem>.csv` (`rank,score,center_x,center_y,center_z,residues`), `OUT/<stem>.points.csv`
(`rank,x,y,z`), `OUT/timing.csv`. A model JSON carries its point model, ranker and its feature
design (`meta.feature_cfg`: rays, ray lengths, sub-lattice; `meta.score_floor`), so a preset is
just a model file. Limitations: `--device cuda:*` falls back to CPU in this reference CLI;
`--af-mode` implements only spaghetti pruning + soft occlusion (no AF-trained ranker yet).

## Pipeline (modules)
1. **Input** (`prep.py`): protein residues only (incl. MSE etc.), heavy atoms; ligands, waters,
   ions, glycans, nucleic acids removed ⇒ invariant to HETATM records (tested). B-factor unused
   except as pLDDT in `--af-mode`. Per-atom tables: 13 atom classes, 7 residue groups, WCN
   (CA-based, Σ 1/d²), potential metal-coordinating atoms, first-hit class
   (apolar / polar N,O / charged / aromatic).
2. **Candidates** (`features.py`, `raycast.py`): computed in the canonical PCA frame (rotation
   invariance). 1 Å grid points with |x − a| ≥ r_vdW + 1.0 Å, nearest atom ≤ 6 Å → dual
   absorption grid (all atoms; backbone+Cβ only; per-voxel owner class; occlusion radius
   r_vdW + 0.5 Å) → LIGSITE PSP prefilter (7 dirs, 8 Å): keep PSP ≥ 2 (volume 1 Å³) + 2 Å
   sub-lattice of the rest (volume 8 Å³) → **metal candidates**: positions 1.95–2.35 Å from a
   side-chain coordinator with ≥ 2 coordinating atoms (His ND1/NE2, Cys SG, Asp/Glu O, Met SD,
   backbone O) within 2.6 Å, no other atom < 2.4 Å, 1.5 Å dedup, ≤ 1 per coordinating-atom cluster.
3. **Point features** (86): 60 Fibonacci rays, one march gives blocked fraction at 6/10/16/24 Å for
   full-atom and backbone+Cβ grids, mean first-hit depth, first-hit chemistry fractions + escape
   fraction; PSP; nearest-atom distance; metal flag, #coordinators ≤ 2.8 Å, angular spread;
   atom counts at 4/8/10/12 Å; class/residue-group counts at 4 Å and fractions at 8/12 Å; mean
   WCN at 8/12 Å; protrusion fraction. Screened Coulomb not implemented (SHOULD; cut order).
4. **Point scorer**: LightGBM binary (reference 250 trees × 63 leaves; presets see PHASE2.md;
   bagging 0.8/feature 0.8, seeds from `SEEDS`). Labels from Track A's pooled site ligands:
   `all` = ≤ 4 Å of a non-ion ligand atom or ≤ 3 Å of an ion; `small` = ≤ 4 Å of a non-ion atom.
   Training sample: 30 % of positives, 3 % of negatives per structure (deterministic per id).
   Optional `score_floor`: only points with b ≥ floor (and metal candidates) are scored, others 0.
5. **Pocket formation** (`pockets.py`, `model.py`): candidate union of (a) geometric pass:
   components (single linkage 2.05 Å) of b = mean(bur10,16,24) ≥ .40/.55/.70/.85 and (b) score
   pass: point score ≥ .15/.30/.50; chaining control (extent > 20 Å → weighted k-means, 12 Å);
   multi-scale sub-pockets (extent > 10 Å → ~6 Å k-means pieces); min volume 10 Å³ (metal
   candidates exempt). Centre = (vol·value)-weighted centroid; residues = atoms ≤ 4 Å of points.
6. **Pocket ranker**: LightGBM (300 trees, 15 leaves) on 78 pocket descriptors (point-score
   aggregates, volume, buriedness means/quantiles, chemistry, residue groups, PCA shape, extent,
   #residues, #chains, source level, protein size; vectorised segment reductions). Training
   features use **cross-fitted** point scores (5 folds of Track A `cluster30`; assembly inputs
   share their entry's fold). Target (**graded**, cross-entropy): q = min DCC over target sites
   whose ligand atoms are within DCA ≤ 4 Å of the pocket centre; target = clip(1 − q/12, 0, 1).
   (The binary "DCC ≤ 12 & DCA ≤ 4" target is kept as `_hit` ablation: graded is better on
   top-N/top-N+2/DCA4, binary slightly better on P1K.) Not isotonic-calibrated (monotone
   calibration would not change any rank-based or pooled-precision metric).
   Training inputs: Track A train representative chains + stored assembly-1 inputs.
7. **NMS / output**: greedy by ranker score; suppress if centre < 10 Å from a kept pocket,
   residue Jaccard ≥ 0.75, or point-overlap coefficient ≥ 0.3 (nested multi-scale duplicates);
   ≤ 50 pockets. 10 Å was chosen a priori on CHEN11 cv5-OOF and confirmed on the BioLiP2 dev
   sensitivity curve as the radius maximising DCA4 top-N+2; larger radii raise LIGYSIS top-N+2
   only by spreading pockets (DCA4 and pocket density drop) — curve in PHASE2.md.
8. **Geometry-only baseline** (`--ranker geometry`, no learning): geometric candidates ranked by
   buried volume Σ vol·clip((b − 0.7)/0.3, 0, 1)², same NMS. (Plain Σ vol·b ranks huge convex
   surface patches first: DCA4 top-n 15.9 % on JOINED vs 35.4 %.)

## Dev results
Phase 2 (BioLiP2, audited): `results/dev/biolip/PHASE2.md`. Phase 1 (interim CHEN11/JOINED): `results/dev/G1.md`, `results/dev/DEV_RESULTS.md`.

## Presets (feature design frozen for the fast backend)
| preset | rays | ray lengths (Å) | sub-lattice | point model | scored points | model files |
|---|---|---|---|---|---|---|
| **default** | 30 | 6/10/16 | 3 Å (vol 27) | 250 trees × 63 leaves | b ≥ 0.7 or metal cand. (~45 % of points) | `burrow_{all,small}_default_s{0,1,2}.json` |
| accurate | 60 | 6/10/16/24 | 2 Å (vol 8) | 250 × 63 | all | `burrow_{all,small}_accurate_s{0,1,2}.json` |

Everything else (candidates, 86 → 82 point features — `bur24`/`burbb24` absent in default —, pocket
formation, ranker, NMS 10 Å) is shared. b = mean(blocked@10, @16, @longest). Selection: speed/accuracy
sweep in `results/dev/biolip/PHASE2.md` (PI rule: default ≤ ~100 ms/core estimated in the fast
backend and within 0.5 pt top-N+2 and DCA4 of the best setting).

## Timing — reference implementation (ms/structure, 1 core, warm)
`scripts/burrow_timing.py`, 60 audited dev chains (median 1,900 heavy atoms), shared node (load
30–60). Raw: `results/dev/biolip/timing_ref_{default,accurate}.csv`. The fast backend
(`burrow/fast/`, separate owner) is the speed deliverable; estimated fast-backend cost: default
≈ 80 ms, accurate ≈ 200 ms (scaled from the fast-backend engineer's per-stage measurements).

| stage (median) | default | accurate |
|---|---|---|
| parse | 15.7 | 14.7 |
| grid points / dual grid + PSP / metal | 13.9 / 14.6 / 18.0 | 13.8 / 14.9 / 16.8 |
| rays | 19.3 | 76.4 |
| neighbour sums / misc | 18.7 / 9.4 | 26.9 / 12.6 |
| point scoring | 26.1 | 108.0 |
| candidate pockets + residues | 90.4 | 110.2 |
| pocket features + ranker + NMS | 43.4 | 42.5 |
| **total excl. parse** | **258.6** | **407.2** |

### Phase-1 timing (interim data, before the cheap pocket-stage fixes)
`scripts/burrow_timing.py` (60 random dev structures, median 2,094 heavy atoms, NUMBA/BLAS 1
thread, JIT warm; shared node with load 40–100, so numbers are pessimistic/noisy).
Raw: `results/dev/timing_cpu1.csv`.

| stage | median ms (learned) | median ms (geometry-only) |
|---|---|---|
| parse (gemmi + polymer selection + chemistry tables) | 13.6 | 13.6 |
| solvent-shell grid points | 15.3 | 15.4 |
| dual grid + PSP prefilter | 14.8 | 14.8 |
| metal candidates | 17.4 | 17.5 |
| rays (60 rays × 24 Å, 2 grids, first-hit) | 71.4 | 71.6 |
| neighbour sums | 30.7 | 29.2 |
| misc (metal geometry, assembly) | 12.5 | 12.5 |
| **features subtotal** | 161.7 | 164.3 |
| point scoring (LightGBM 250×63 leaves) | 103.8 | – |
| candidate pockets + residues | 150.8 | 104.6 |
| pocket features + ranker + NMS | 159.9 | 23.3 |
| **total (excl. parse)** | **585.6** (mean 664) | **298.2** (mean 375) |

Far from the ≤ 70 ms target. Known fixes (next): point→atom/residue lists computed once per
structure instead of per candidate; vectorised pocket features (segment reductions) and
numba NMS; fewer candidates (dedupe identical components across thresholds); smaller point model
(≤ 100 trees, 31 leaves) or score only cascade survivors with b ≥ 0.3; 30 rays; metal candidates
via vectorised KD queries; avoid Python loops in metal-geometry features.
