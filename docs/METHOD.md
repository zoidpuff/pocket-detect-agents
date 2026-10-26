# BURROW — method description (phase 1, CPU)

Owner: method developer. Spec: DESIGN.md §2 + §9. Code: `src/pocketdetect/burrow/`.
Status: phase 1 complete on **interim dev data** (CHEN11 + JOINED, P2Rank ligand rules). The
BioLiP2 train/dev sets (Track A) were not final yet; all numbers below must be re-done on them.

## Usage
```
python -m pocketdetect.burrow predict <files|dirs|@list> --out OUT [--af-mode] \
       [--ranker all|small|geometry] [--model M.json] [--device cpu] [--threads 1]
```
Writes `OUT/<stem>.csv` (`rank,score,center_x,center_y,center_z,residues`), `OUT/<stem>.points.csv`
(`rank,x,y,z`), `OUT/timing.csv`. (A console entry point `burrow = pocketdetect.burrow.cli:main`
can be added to pyproject.toml by the PI.) Phase-1 limitations: `--device cuda:*` falls back to
CPU with a warning; `--ranker all` uses the interim no-ion model until the BioLiP2 `all` model is
trained; `--af-mode` implements only spaghetti pruning + soft occlusion (no AF-trained ranker).

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
4. **Point scorer**: LightGBM binary (250 trees, 63 leaves, bagging 0.8/feature 0.8, seeds from
   `SEEDS`), label = point ≤ 4 Å of a relevant ligand atom (ions 3 Å); all positives + 10 % of
   negatives. Top features (gain): escape, bur24, burbb24, protrusion, bur6, n_all12, burbb6, WCN.
5. **Pocket formation** (`pockets.py`, `model.py`): candidate union of (a) geometric pass:
   components (single linkage 2.05 Å) of b = mean(bur10,16,24) ≥ .40/.55/.70/.85 and (b) score
   pass: point score ≥ .15/.30/.50; chaining control (extent > 20 Å → weighted k-means, 12 Å);
   multi-scale sub-pockets (extent > 10 Å → ~6 Å k-means pieces); min volume 10 Å³ (metal
   candidates exempt). Centre = (vol·value)-weighted centroid; residues = atoms ≤ 4 Å of points.
6. **Pocket ranker**: LightGBM binary (300 trees, 15 leaves) on 78 pocket descriptors (point-score
   aggregates, volume, buriedness means/quantiles, chemistry, residue groups, PCA shape, extent,
   #residues, #chains, source level, protein size). Training features use **cross-fitted** point
   scores (5-fold by MMseqs2 30 % groups). Label: pocket centre DCC ≤ 12 Å and DCA ≤ 4 Å to a
   relevant ligand. Not yet isotonic-calibrated.
7. **NMS / output**: greedy by ranker score; suppress if centre < 10 Å from a kept pocket (the
   LIGYSIS 5 Å rule is a special case; 10 Å chosen on CHEN11 cv5-OOF), residue Jaccard ≥ 0.75, or
   point-overlap coefficient ≥ 0.3 (nested multi-scale duplicates); ≤ 50 pockets.
   LDCC-oriented operating point: 12 Å + drop pockets with score < 0.1 (keep ≥ 3).
8. **Geometry-only baseline** (`--ranker geometry`, no learning): geometric candidates ranked by
   buried volume Σ vol·clip((b − 0.7)/0.3, 0, 1)², same NMS. (Plain Σ vol·b ranks huge convex
   surface patches first: DCA4 top-n 15.9 % on JOINED vs 35.4 %.)

## Dev results
See `results/dev/G1.md` (gate G1) and `results/dev/DEV_RESULTS.md` (ranking vs P2Rank/fpocket).

## Timing (ms/structure, 1 core, warm)
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
