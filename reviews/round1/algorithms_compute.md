# Round 1 review — algorithms & compute (DESIGN v1)

Reviewer focus: candidate generation, features, architecture, GBM inference, batching, latency, memory.
I probed the current `src/pocketdetect/geometry` code on 1 CPU thread (`NUMBA_NUM_THREADS=1`, EPYC 9554, warm JIT). I used single chains of 1hh0, 3ptb and 1a28 (340, 1,629 and 2,019 heavy atoms), with the default config: 60 rays, L = 8/12/20 Å. The GPU probe used one RTX 4090. Scripts are in the session scratchpad and are not committed.

| Path | Points | Point features (ms) |
|---|---|---|
| SAS | 1.2k / 2.5k / 3.3k | 12 / 34 / 53 |
| Grid, 1 Å, shell ≤ 8 Å | 29k / 48k / 60k | 270 / 480 / **660** (rays 75%) |
| Grid, shell ≤ 4.5 Å | 7k / 13k / 18k | ~4× fewer points, still >100 ms |

- **Parsing:** `load_structure` takes ~18 ms warm (gemmi read ~8 ms) and ~75 ms on the first call.
- **GPU, SAS path:** 64 randomly rotated copies of 1a28 took 0.6 ms/structure, with 0.55 GB peak.

**Bottom line.** The 50–100 ms/core target is only reachable with about 3–6k scored points per average protein. The grid candidate set as specified is 10× over budget. The GPU target (≥10³/min) is trivially met by the kernels. End-to-end throughput will be limited by host parsing (~18 ms/core), not by the GPU.

## Candidate generation

- **MUST — cascade the grid candidates; do not ray-cast every shell voxel.**
  - Restrict the shell to [r_vdw+1.4, ~4.5–5 Å] from atom centres. Points further out are almost never within 4 Å of ligand atoms.
  - Prefilter with a LIGSITE PSP sweep: 7–13 line directions, done as forward/backward "last protein voxel seen within L" scans. This is O(voxels), a few ms, and runs on CPU and GPU.
  - Run full multi-scale rays only on survivors (PSP ≥ k), plus a 2 Å subsample of non-survivors so shallow sites still get scored.
  - Gate the design on a dev measurement: points/protein (median and p99) and ligand-atom coverage.
- **MUST — use a single point process or volume-normalised aggregates.**
  - A union of SAS points (density ∝ surface atoms) and 1 Å grid points (density ∝ volume) makes Σscore and Σscore² (P2Rank's ranking statistic) depend on which generator covered a region.
  - Either give each point a volume weight, or drop SAS in favour of the thin grid shell. A grid point at r+1.4 to r+3 Å *is* a SAS-like point.
  - Recommendation: use a grid only (1 Å, cascaded). Keep SAS-only as the P2Rank-like ablation.
- **MUST — make candidate-pocket recall a gating metric.**
  - Measure oracle max recall (DCC ≤ 12 Å, and DCA ≤ 4 Å) of the candidate *pockets*, before ranking, at ≤ 30 pockets/protein, on dev.
  - Target ≥ fpocket's ~91% (LIGYSIS).
  - Train no ranker until this is met: ranking cannot recover missed sites.
- **SHOULD — add alpha-sphere centres (Delaunay) as a candidate source and feature, if grid recall falls short.**
  - `scipy.spatial.Delaunay` on ~2.5k atoms takes ~5 ms. Keep circumspheres with radius 3.2–6.5 Å that do not clash.
  - This is the generator behind the strongest baseline (fpocket+PRANK, 60.4). On the GPU path it can run in the CPU loader workers.
- **MUST — handle huge inputs.**
  - `grid_candidates` allocates dense float64 + int64 + bool arrays: 17 B per voxel over bbox + 8 Å.
  - A 300 Å assembly needs ~0.5 GB; a 600 Å capsid or fibril needs >4 GB per process. 64 such workers exceed 128 GB.
  - Use float32/int32, and add tiling: ~96 Å bricks with a halo ≥ max ray length, or a sparse brick map.
  - Cap or split the GPU batch by a voxel budget, not a structure count.
  - Add an HOLO4K-largest and AFDB 2,700-aa smoke test.

## Ray-casting pitfalls

- **MUST — inflate the occlusion radii for rays** (vdW + ~0.6–1.0 Å, tuned on dev). Bare vdW spheres on a 1 Å nearest-voxel grid leave sub-probe gaps that rays leak through. Buriedness should mean "can a water-sized probe escape?".
- **MUST — add a rotation-invariance test.**
  - A fixed Fibonacci direction set on an axis-aligned grid makes outputs orientation-dependent. I measured mean |Δburiedness| 0.03 (p95 0.07) at 60 rays across two random rotations; 30 rays will be worse.
  - Test on 20 structures × 5 rotations: pocket top-3 identity and score Spearman ≥ 0.95.
  - Smoothing buriedness over 2–3 Å neighbours is a cheap variance reducer.
- **SHOULD — use voxel traversal (Amanatides–Woo DDA) or step = spacing.** This halves the samples compared with 0.5 Å steps at 1 Å voxels, with little loss.
- **Bug found.** The Triton kernels launch on the *current* device, not the tensor's device: `cuda:1` fails with "cannot be accessed from Triton". Either wrap the calls in `torch.cuda.device(dev)` or pin one process per GPU via `CUDA_VISIBLE_DEVICES`.

## Features

- **SHOULD — prune collinear densities.**
  - Currently 5 radii + 2 Gaussians × ~15 classes ≈ 100+ features.
  - Keep counts at 2 radii (6, 10 Å) and 1 Gaussian, plus *ratios* (apolar fraction, charged balance). Ratios generalise across sizes and cost no extra compute.
- **SHOULD — collapse the 20-dim residue composition** to ~7 groups: aromatic, aliphatic, G, P, polar, +, −. A full composition vector is a family fingerprint and a memorisation risk (O3).
- **SHOULD — drop or demote convex-hull depth.** It is ill-defined for assemblies and elongated AF models; ray depth already covers it.
- **SHOULD — add pocket-level shape descriptors with evidence** (SiteMap, DoGSite, Nayal & Honig 2006):
  - √n-saturated size;
  - PCA eigenvalue ratios of the pocket point cloud;
  - mouth openness (fraction of rays escaping at L = 12 Å, averaged over the pocket);
  - depth quantiles;
  - enclosure × hydrophobicity (the SiteScore form);
  - an electrostatics summary: Σq/r over ±8 Å with ε = 4r at a cutoff, reusing the neighbour pairs (no O(P·N) all-pairs).
- **COULD — add point-level DoG buriedness contrast** (b_8 − b_20), which captures sub-pocket structure, and ray-escape direction entropy (channel vs groove). Both are free from the existing march.

## Architecture

- **MUST — cross-fit the point scores used to train the pocket ranker.**
  - In-sample GBM point scores on training proteins are overconfident. The ranker will over-trust Σscore, then miscalibrate on test.
  - Use 5-fold out-of-fold predictions by sequence cluster. This is standard stacking.
- **MUST — fix single-linkage chaining.**
  - Single linkage at d ≈ 3 Å on 1 Å grid points merges whole grooves and interfaces into one blob. Its centroid then drifts outside the 12 Å DCC window.
  - Use d ≈ 1.5 × spacing (26-connectivity). Split any component with Rg > ~8 Å or > N points by quick shift on score-weighted density (`cluster.quick_shift` exists). Mean shift is ~5× costlier for the same effect.
  - Emit both parent and child candidates. The ranker scores both.
  - Change NMS to also suppress on containment (|A∩B| / min(|A|, |B|) ≥ 0.7). Otherwise parent/child pairs survive the 5 Å / Jaccard 0.75 rule.
- **SHOULD — use a score²-weighted pocket centre**, not the geometric centroid. This matters for DCC on large clefts.
- **SHOULD — run "geometric candidates + pocket ranker with point-score aggregates" first in E2.**
  - This is the fpocket+PRANK recipe, the best LIGYSIS baseline.
  - Point-threshold clusters are then an extra candidate source, not the backbone.
  - Keep a binary + isotonic objective (Precision1K needs cross-protein calibration); lambdarank stays an ablation.

## GBM inference & batching

- **MUST — use one tree library and one exported array format.**
  - Either (a) use XGBoost for both training and inference: `device=cuda` training, `inplace_predict` on torch/cupy via DLPack, and the CPU predictor with `nthread=1`.
  - Or (b) keep LightGBM and write ~80 lines of Numba/torch traversal over padded node arrays.
  - Under (b), cast thresholds to the largest float32 ≤ the double threshold (features are float32). Forbid categorical splits and NaN. Verify that leaf indices equal `pred_leaf` on 1M dev points.
  - Skip Hummingbird and treelite: extra dependencies with no gain at this size.
- **SHOULD — size the model for latency.** 300 trees × depth ~8 × 5k points is ~10–15 ms/core. Prefer ≤ 200 trees or shallow trees, and check the cost/accuracy curve on dev.
- **SHOULD — build the GPU pipeline as producer/consumer.**
  - Use 12–16 CPU parse workers per GPU, size-bucketed batches with a voxel/point budget, and pinned-memory transfer.
  - Run GBM, clustering and pocket aggregation as segment ops on the GPU, with no per-structure Python loop. Write JSON in the workers.
  - Report end-to-end structures/min with parsing included and excluded.
- **SHOULD — subsample training negatives.**
  - 5k points × 10k chains × ~80 float32 features is ~16 GB.
  - Keep all positives and ~15% of negatives, stratified by buriedness, with weights corrected.
- **SHOULD — define CPU/GPU parity at the pocket level.** Use centre Δ < 0.5 Å, score Δ < 1e-3, identical top-5, and point-set Jaccard ≥ 0.999. float32 vs float64 clash tests flip borderline voxels, so exact equality is the wrong bar.

## Realistic latency (2,000-atom chain, 1 core, warm)

| Stage | Estimate |
|---|---|
| Parse | ~18 ms |
| Cascaded grid + PSP | ~5 ms |
| Rays (5k × 30–60 rays) | ~15–25 ms |
| Neighbours | ~10 ms |
| GBM | ~10 ms |
| Clustering + ranker | ~3 ms |
| **Total** | **≈ 60–70 ms** |

- This is within target only with the cascade.
- Cold start is dominated by Numba loading (~1–3 s even with `cache=True`). Report it like P2Rank's JVM start-up.
- GPU: ~1 ms/structure in kernels, so ≥ 10⁴/min/GPU is plausible if parsing is parallelised.
