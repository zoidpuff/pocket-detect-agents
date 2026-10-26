# Round 1 review — algorithms & compute (DESIGN v1)

## Measurements

I timed the current `geometry/` code on 1 CPU thread (warm, EPYC 9554) with the default config (60 rays, L = 8/12/20 Å). Inputs were single chains of 1hh0, 3ptb and 1a28 (340, 1,629 and 2,019 atoms).

| Points | Count | Time |
|---|---|---|
| SAS | 1.2k–3.3k | 12–53 ms |
| Grid, 1 Å, ≤ 8 Å shell | 29k–60k | 270–660 ms (75% of it in rays) |
| Grid, ≤ 4.5 Å shell | 7k–18k | still > 100 ms |

- **Parsing:** `load_structure` takes ~18 ms warm and ~75 ms on the first call.
- **GPU (one 4090, SAS):** 0.6 ms/structure on 64 copies of 1a28, 0.55 GB peak memory.

**Conclusion.**
- The CPU target (≤ 50–100 ms/core) needs about 3–6k scored points per protein.
- The grid as specified is 10× over that budget.
- The GPU target (10³/min) is met easily. End-to-end throughput will be limited by parsing, not by the GPU.

## Candidate generation

- **MUST — cascade grid points.**
  - Limit the shell to [r_vdw+1.4, ~5 Å].
  - Prefilter with a LIGSITE PSP sweep: 7–13 directions, forward/backward line scans, O(voxels), a few ms on CPU or GPU.
  - Ray-cast only the survivors, plus a 2 Å subsample of the rest so shallow sites stay covered.
  - Gate on dev: points per protein (median and p99) and ligand-atom coverage.
- **MUST — one point process, or volume-normalised aggregates.**
  - In a SAS ∪ grid union, point density depends on which generator covered a region. That biases Σscore and Σscore².
  - Recommend a grid-only design: a thin 1 Å shell is SAS-like.
  - Keep SAS-only as a P2Rank-like ablation.
- **MUST — make candidate-pocket recall a gate.**
  - Measure oracle max recall (DCC 12 / DCA 4) on dev at ≤ 30 pockets per protein.
  - Target ≥ fpocket's ~91%, before any ranker is trained.
- **SHOULD — add alpha-sphere candidates if grid recall falls short.** Delaunay circumspheres (radius 3.2–6.5 Å) take ~5 ms with scipy for 2.5k atoms. This is the generator behind the strongest baseline, fpocket+PRANK.
- **MUST — handle huge inputs.**
  - `grid_candidates` uses 17 B per voxel over the bounding box + 8 Å: ~0.5 GB at 300 Å, > 4 GB for capsids or fibrils, × 64 workers.
  - Switch to float32/int32 and tile in ~96 Å bricks with a halo ≥ max ray length.
  - Batch GPU work by a voxel budget, not by structure count.
  - Add a smoke test on the largest HOLO4K structures and 2,700-aa AFDB models.

## Ray-casting

- **MUST — inflate occlusion radii** by ~0.6–1 Å, tuned on dev. Bare vdW spheres on 1 Å nearest-voxel grids leak rays through sub-probe gaps.
- **MUST — add a rotation-invariance test.**
  - Fixed Fibonacci rays on an axis-aligned grid make results depend on orientation. I measured mean |Δburiedness| 0.03 (p95 0.07) between two random rotations.
  - Test 20 structures × 5 rotations: identical top-3 pockets and score Spearman ≥ 0.95.
  - Smoothing over 2–3 Å neighbours is a cheap fix.
- **SHOULD — use DDA traversal or step = voxel size.** This halves the samples.
- **Bug.** The Triton kernels launch on the current device, so tensors on `cuda:1` fail. Wrap calls in `torch.cuda.device(dev)` or run one process per GPU with `CUDA_VISIBLE_DEVICES`.

## Features

- **SHOULD — cut redundant neighbour features.**
  - Currently ~100 densities (5 radii + 2 Gaussians × ~15 classes).
  - Keep 2 radii (6, 10 Å) and 1 Gaussian, plus ratios (apolar fraction, charge balance).
- **SHOULD — collapse the 20-dim residue composition into ~7 groups.** The full vector acts as a family fingerprint, which risks memorisation (O3).
- **SHOULD — drop convex-hull depth.** It is ill-defined for assemblies and AF models, and ray depth covers it.
- **SHOULD — add evidence-backed pocket descriptors:**
  - √n-saturated size;
  - point-cloud PCA ratios;
  - mouth openness (escaping-ray fraction);
  - depth quantiles;
  - enclosure × hydrophobicity (the SiteScore form);
  - Σq/r with ε = 4r, cut off at 8 Å and reusing neighbour pairs.
- **COULD — add buriedness contrast (b₈ − b₂₀) and ray-escape entropy.** Both come free from the existing ray march.

## Architecture

- **MUST — cross-fit point scores for ranker training.** Use 5-fold out-of-fold predictions grouped by sequence cluster. In-sample GBM scores are overconfident, so the ranker would over-trust Σscore.
- **MUST — fix chaining.**
  - Single linkage at 3 Å on 1 Å points merges whole grooves and interfaces, and the centroid drifts outside the 12 Å DCC window.
  - Use linkage d ≈ 1.5 × spacing.
  - Split any component with Rg > ~8 Å using `cluster.quick_shift` on score-weighted density (mean shift costs ~5× more).
  - Keep both parent and child pockets for the ranker.
  - Add containment NMS (|A∩B| / min ≥ 0.7); the 5 Å / Jaccard rule keeps parent–child pairs.
- **SHOULD — use a score²-weighted pocket centre.**
- **SHOULD — run "geometric candidates + pocket ranker with point-score aggregates" first, in E2.**
  - This is the fpocket+PRANK recipe.
  - Point-threshold clusters become an extra candidate source.
  - Keep binary + isotonic training (needed for P1K); lambdarank stays an ablation.

## GBM inference & batching

- **MUST — use one tree library and one exported format.** Two options:
  - XGBoost throughout: CUDA training, `inplace_predict` on torch via DLPack, `nthread=1` on CPU.
  - LightGBM plus our own Numba/torch traversal over padded node arrays. Then:
    - round thresholds to the largest float32 ≤ the threshold;
    - forbid categorical splits and NaN;
    - assert leaf indices equal `pred_leaf` on 1M dev points.
  - Skip Hummingbird and treelite.
- **SHOULD — size the model for latency.** 300 trees × depth 8 × 5k points is ~10–15 ms/core. Plot dev accuracy against tree count.
- **SHOULD — build the GPU path as producer/consumer.**
  - 12–16 parse workers per GPU feed size-bucketed batches through pinned memory.
  - GBM, clustering and aggregation run as segment ops, with no per-structure Python loop.
  - Report throughput with parsing included and excluded.
- **SHOULD — subsample training negatives.** Keep all positives and ~15% of negatives, stratified by buriedness and reweighted. Without this, 50M points × 80 features is ~16 GB.
- **SHOULD — check CPU/GPU parity at the pocket level.** Criteria: centre Δ < 0.5 Å, score Δ < 1e-3, same top-5, point Jaccard ≥ 0.999. float32 and float64 flip borderline voxels, so exact equality is the wrong test.

## Latency estimate (2k-atom chain, 1 core, warm)

| Stage | Time |
|---|---|
| Parse | 18 ms |
| Cascade | 5 ms |
| Rays (5k points) | 15–25 ms |
| Neighbours | 10 ms |
| GBM | 10 ms |
| Clustering + ranker | 3 ms |
| **Total** | **≈ 60–70 ms** |

- Cold start is ~1–3 s of Numba loading; report it the way P2Rank's JVM start-up is reported.
- GPU: ~1 ms/structure in kernels, so 10⁴/min/GPU is plausible if parsing is parallelised.
