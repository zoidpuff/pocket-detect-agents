# For further development: building a pocket-comparison method on BURROW

This note is for the agent that will build a method to compare pockets (a similarity measure over pocket
descriptors or representations) on top of BURROW. It says what exists, where it is, how to get every pocket
representation from code, and what we learned that matters for comparison. Every command and code block was
run on 2026-10-27 from the root of the main repository with `.venv` active; outputs quoted below are real.

## 1. Orientation

**What BURROW is.** BURROW finds and ranks ligand-binding pockets in protein structures (PDB/mmCIF;
experimental structures, AlphaFold models, multi-chain assemblies).
- It fills the solvent shell around the protein with grid points ("volume elements", 1 Å grid in the default preset).
- It describes each point by ray-cast buriedness, through a full-atom grid and a backbone-only grid, plus first-hit chemistry and neighbourhood composition.
- A small LightGBM model scores the points. Candidate pockets are assembled at several buriedness and score levels.
- A second LightGBM model ranks the pockets. It was trained on a leakage-controlled, LIGYSIS-style BioLiP2 set.
- There are no neural networks. CPU (Numba) and GPU (PyTorch/Triton) give byte-identical output.
- `--af-mode` uses pLDDT to prune low-confidence regions.

**Where things live.**
| place | what |
|---|---|
| this repository (`/cluster/project/beltrao/gmagnus/pocket_detect`) | the full research record: code (`src/pocketdetect/`), scripts, data builds (`data/processed/`, git-ignored), every dev/test result (`results/`), decisions (`RESEARCH_TREE.md`) |
| `release/burrow/` | the clean public repository: same package, models in `models/` (seed 0–2, all presets, `SHA256SUMS`), test predictions in `predictions/`, examples in `examples/`, docs. It does **not** contain the helper `pocketdetect.burrow.export` added for this note. |
| `manuscript/` | the paper (`main.md`, supplementary texts `S1–S3`) and the writing rules (`manuscript/skill/SKILL.md`) |

**Read first, in this order:** `REPORT.md` (results, limitations, negative results §5),
`docs/METHOD.md` (method, presets, output rules), `docs/PERF.md` (speed; "Final like-for-like timing"),
`manuscript/main.md`, `RESEARCH_TREE.md` (all decisions D1–D19, with times), `LESSONS.md` (process rules).

**Final version = FREEZE v2** (`results/dev/FREEZE_v2.md`; models `results/dev/freeze_v2/models/burrow_{all,small}_{default,accurate}_s{0,1,2}.json`).
- **Rankers:** `all` covers ions and organic ligands (LIGYSIS site definition); `small` covers non-ion small molecules (P2Rank definition).
- **Presets:** `default` (pre-registered), `accurate`, and `fast` (added later, `results/dev/FREEZE_fast.md`, `results/dev/freeze_fast/models/`).
- **Output floor:** pockets with score < 0.03 are dropped once 3 are kept.
- **Flags:** `--min-score none` gives the full list (identical to FREEZE v1). `--output-rule region` adds the best pocket of every distant region (§5).
- **AF mode:** `--af-mode`; optional `--af-ranker` (reference backend only).

## 2. Code map

Package `src/pocketdetect/` (reference implementation = the definition; `burrow/fast/` must match it exactly).

| stage | function | file |
|---|---|---|
| read structure (first model, first altloc, author numbering) | `load_structure` | `structure.py` |
| protein-only view + per-atom chemistry tables (13 atom classes, 7 residue groups, WCN, metal-coordinating atoms) | `protein_from_structure`, `load_protein` | `burrow/prep.py` |
| AF mode: prune pLDDT < 50 stretches, soft-occlusion weights | `af_protein`, `prune_mask` | `burrow/af.py` |
| candidate points, cascade, metal candidates, point features | `point_features`, `metal_candidates`, `BurrowConfig` | `burrow/features.py` |
| dual absorption grid, ray marching (buriedness, depth, first hit) | `dual_grid`, `march` | `burrow/raycast.py` |
| grid helpers: shell points, PSP prefilter, neighbour sums, PCA frame | `grid_candidates`, `psp_counts`, `neighbor_features`, `canonical_frame` | `geometry/candidates.py`, `geometry/neighbors.py`, `geometry/frames.py` |
| point scoring | `score_points` | `burrow/model.py` |
| candidate pockets (multi-threshold components, chaining control, sub-pockets) | `candidates`, `generate`, `components` | `burrow/model.py`, `burrow/pockets.py` |
| pocket residues (≤ 4 Å of pocket points) | `assign_residues`; output fallback `output_residues` | `burrow/pockets.py`, `burrow/pipeline.py` |
| 76 pocket descriptors | `pocket_features` | `burrow/model.py` |
| pocket ranking, suppression (NMS), output floor | `rank_predict`, `finalize`, `nms`, `FormationConfig` | `burrow/model.py`, `burrow/pipeline.py`, `burrow/pockets.py` |
| whole pipeline / writer | `predict_protein`, `write_prediction` | `burrow/pipeline.py` |
| optional output rules (region rule) | `apply_rule`, `filter_prediction_files` | `burrow/output_rules.py` |
| **all representations for one structure (new)** | `pocket_representations` | `burrow/export.py` |
| model file (trees + feature design + output settings + provenance) | `BurrowModel` | `burrow/model.py` |
| CLI (routes to the fast backend, falls back to the reference) | `main` | `burrow/cli.py` |
| exact fast CPU/GPU backend, fast parser, drift guard | `fast/pipeline.py`, `fast/io.py`, `fast/run.py`, `fast/guard.py` | `burrow/fast/` |
| training/dev data readers | `read_manifest`, `load_entry`; AF: `scripts/data/build_af_train_io.py`; apo: `read_apo_manifest` | `datasets/trainset.py`, `datasets/apotrain.py` |
| evaluation: ligand-based, site-based (LIGYSIS protocol), pre-registered analysis | `evaluate`, `evaluate_sites`, `summarize_sites`, `compare_site_methods` | `evaluate.py`, `site_eval.py`, `ligysis.py`, `prereg.py` |

Data flow (one structure):
```
file --load_structure--> Structure --protein_from_structure (or af_protein)--> Protein (heavy atoms, chemistry)
  --point_features--> points (input frame + PCA frame) x 84/86 features
  --score_points--> point scores --candidates--> ~100-400 candidate pockets (point index sets)
  --assign_residues + pocket_features--> 76 descriptors per candidate --rank_predict--> pocket scores
  --finalize (NMS 10 Å, Jaccard 0.75, overlap 0.3; floor 0.03)--> ranked pockets --write_prediction--> CSV
```

## 3. Pocket representations

### (a) Per-pocket output (`OUT/<stem>.csv`)
`rank,score,center_x,center_y,center_z,residues`.
- **`score`** is the ranker output. It is probability-like and comparable across proteins: pooled precision (P1K) depends on this property. It is not isotonic-calibrated.
- **Centre** is the volume-weighted (× point value) centroid of the pocket's points, in the input frame.
- **`residues`** are residues with a heavy atom ≤ 4 Å of any pocket point, written `chain:resnum[icode]` (author numbering of the input). If that set is empty, the residues with an atom ≤ 6 Å of the centre are used, else the nearest residue.
- Ranks are 1..n in decreasing score.

### (b) Pocket point cloud (`OUT/<stem>.points.csv`: `rank,x,y,z`)
Each point is a grid point in the solvent shell:
- every atom is at ≥ r_vdW + 1 Å, and the nearest atom is ≤ 6 Å away (5.5 Å in `fast`);
- grid spacing is 1 Å (`default`/`accurate`) or 1.5 Å (`fast`).

Exposed regions are thinned to a coarse sub-lattice: every 3rd point in `default`, every 2nd in `accurate` and `fast`. Such a point stands for `vol` = sub-lattice³ grid voxels, so pocket volume is `Σ vol × spacing³` Å³.
- **Frames:** written coordinates are in the **input frame**. Internally all geometry is computed in a canonical PCA frame (`geometry/frames.py`), which makes outputs rotation-invariant up to round-off (tested in `tests/test_burrow_pipeline.py`).
- **Metal candidates:** they are extra points 1.95–2.35 Å from ≥ 2 coordinating atoms (`vol` = 0).

### (c) Point features (84 in `default`/`fast`, 86 in `accurate`; names from `rep["point_feature_names"]`)
| group | names | meaning | interpretable? |
|---|---|---|---|
| ray buriedness, full atoms | `bur6`, `bur10`, `bur16` (+`bur24` in accurate) | fraction of rays blocked within L Å (Beer–Lambert soft occlusion) | yes |
| backbone+Cβ buriedness | `burbb6`, `burbb10`, `burbb16` (+`burbb24`) | same with side chains removed (side-chain tolerant) | yes |
| depth, mouth | `depth`, `escape` | mean first-hit distance; fraction of rays that escape | yes |
| first-hit chemistry | `hit_apolar`, `hit_polar`, `hit_charged`, `hit_aromatic` | class of the first atom each ray hits | yes |
| enclosure / shell | `psp`, `nearest` | LIGSITE protein–solvent–protein count; distance to nearest atom | yes |
| metal site | `metal_cand`, `n_coord28`, `coord_spread` | metal candidate flag; coordinating atoms ≤ 2.8 Å; their angular spread | yes |
| density | `n_all4`, `n_all8`, `n_all10`, `n_all12`, `prot_frac` | heavy atoms within r Å; protrusion | yes (scale with protein size: `prot_frac`) |
| composition at 4 Å (counts) | `n4_<class>` (C, N, O, S, aromatic, hydrophobic_sc_C, donor, acceptor, backbone, sidechain, positive, negative, coord, 7 residue groups `rg_*`) | atom / residue-group counts | yes |
| composition at 8 and 12 Å (fractions) | `f8_*`, `f12_*` (same classes + `wcn`) | fractions of atoms; mean weighted contact number (flexibility proxy) | yes |

All features are invariant to rotation and translation. B-factors are never features.

### (d) Pocket descriptors (76; names from `rep["descriptor_names"]`)
| group | names | units / scale |
|---|---|---|
| size | `n_pts`, `vol`, `sqrt_vol`, `sat_vol` | points; `vol` in grid voxels (× spacing³ for Å³) |
| learned point scores | `score_sum`, `score_sqrt`, `score_mean`, `score_max`, `score_top10`, `score_top30`, `score_q50`, `score_rank`, `score_rel`, `geo_strength` | 0–1 means; sums scale with size; `score_rank`/`score_rel` are relative to the other candidates of the same protein |
| buriedness | `bur10_mean`, `bur16_mean`, `bur24_mean`, `burbb10_mean`, `burbb16_mean`, `depth_mean`, `psp_mean`, `nearest_mean`, `bur16_q90/q10`, `bur24_q90/q10` | 0–1, Å. In `default`/`fast`, `bur24_*` is computed from the longest ray length (16 Å) |
| chemistry | `hit_*`, `escape`, `f8_*` (21), `n4_rg_*` (7), `n_all8`, `n_all12`, `prot_frac` | means over points |
| metal | `metal_n`, `n_coord_max` | counts |
| shape | `pca1`, `pca2`, `pca3`, `aniso`, `extent` | Å (sqrt of PCA eigenvalues; principal-axis extent) |
| context | `n_res`, `n_chains`, `level`, `is_sub`, `is_score_src`, `n_atoms_prot`, `log_atoms` | candidate source (threshold level, sub-pocket flag); protein size |

`pocket_features(pockets, X, vol, F, names, score, n_atoms)` (`burrow/model.py`) computes them for **any** set
of point-index sets on a structure, so you can describe pockets defined another way (e.g. points within 4 Å of
a ligand) with the same vector.

### (e) Residue sets and pLDDT
Residue sets use author numbering of the input. For AlphaFold models this is UniProt numbering, which makes cross-structure mapping simple.
- With `af_mode=True`, the helper adds `plddt_mean`/`plddt_min` over the lining residues.
- The fast CLI writes the same statistics in compact mode: `python -m pocketdetect.burrow.fast predict ... --compact K`, columns `mean_plddt`, `min_plddt` (see `data/processed/proteome_human/*/pockets.tsv`).

### (f) Getting everything from code
`pocketdetect.burrow.export.pocket_representations` (`src/pocketdetect/burrow/export.py`) runs the reference pipeline and returns points, point features, pocket descriptors and point memberships.
- Its pockets are identical to `predict_protein` (`tests/test_burrow_export.py`, which also checks that default CLI output is unchanged).
- It takes `min_score=None` (full list), `output_rule="region"`, `af_mode=True`, `preset="fast"`.
```python
from pocketdetect.burrow.export import pocket_representations

rep = pocket_representations("release/burrow/examples/1a28_A.pdb", ranker="all", preset="default")
pockets, points = rep["pockets"], rep["points"]
print(pockets[["rank", "score", "n_points", "volume_A3"]].head(3))
top = points.loc[rep["pocket_points"][1]]          # volume elements of the top pocket
print(len(top), top[["x", "y", "z", "bur16", "escape", "point_score"]].mean().round(3).to_dict())
print(len(rep["point_feature_names"]), len(rep["descriptor_names"]))
```
Output: top pocket score 0.626, 312 points (311 Å³), mean `bur16` 0.999, `escape` 0.001; 84 point features, 76
descriptors. A toy comparison (illustration only; not a validated metric):
```python
import numpy as np
from pocketdetect.burrow.export import pocket_representations

# Toy pocket-vs-pocket similarity: cosine of z-scored, size-free descriptors (illustration only).
a = pocket_representations("release/burrow/examples/1a28_A.pdb", min_score=None)["pockets"]
b = pocket_representations("release/burrow/examples/AF-P00918-F1-model_v6.cif.gz", min_score=None, af_mode=True)["pockets"]
cols = [c for c in a.columns if c.startswith(("f8_", "hit_", "bur", "depth", "escape"))]
X = np.vstack([a[cols].to_numpy(), b[cols].to_numpy()])
Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
S = Z[: len(a)] @ Z[len(a):].T                      # (pockets of a) x (pockets of b)
i, j = np.unravel_index(np.argmax(S), S.shape)
print(len(cols), S.shape, f"most similar: 1a28 pocket {a['rank'][i]} vs AF pocket {b['rank'][j]}: {S[i, j]:.2f}")
```
Output: `36 (40, 40) most similar: 1a28 pocket 16 vs AF pocket 18: 0.93`. Low-ranked pockets dominate such raw
matches, which is one of the reasons for §5.

## 4. Running efficiently
```bash
burrow predict release/burrow/examples/1a28_A.pdb --out out/one                      # one structure
burrow predict release/burrow/examples/AF-P00918-F1-model_v6.cif.gz --out out/af --af-mode
burrow predict release/burrow/examples/ --out out/batch --workers 2 --ranker small --preset fast
burrow predict release/burrow/examples/1a28_A.pdb --out out/full --min-score none    # 40 pockets (v2: 6)
burrow predict release/burrow/examples/1a28_A.pdb --out out/region --output-rule region   # 11 pockets
```
- **Backends.** `--backend auto` (default) runs the fast backend and falls back to the reference per input; `--device cuda:0,1 --workers 8` uses GPUs. Outputs are byte-identical across reference, fast CPU and fast GPU (`tests/test_burrow_cli_backends.py`, `scripts/verify_burrow_fast.py`). Exception: `--preset fast` currently has no GPU path (the 1.5 Å grid trips a float32 check), so it falls back to CPU.
- **Speed** (`docs/PERF.md` "Final like-for-like timing", 200 LIGYSIS chains, 1 core, warm, excluding parsing): `fast` 28.4 ms, `default` 96.0 ms, `accurate` 203.0 ms median. Throughput at 24 CPU workers: 371 / 150 / 80 structures/s. Large GPU batches (10,000 structures, 3 GPUs): fast 646/s, default 411/s.
- **Parser.** `burrow/fast/io.py` parses about 3× faster than `load_structure` with identical output.
- **Cold start.** The first Numba compilation takes ~37 s once per installation.
- **Proteome scale** (`results/proteome/HUMAN.md`): the human AFDB (23,586 models) ran at 8,374 structures/min (naive) and 10,996/min (`--af-mode`) on 3 GPUs × 8 workers, with peak RSS 17.5 GB. Use the fast driver with `--compact K` to write one `pockets.tsv` instead of 23k files. Fragmented proteins (> 2,700 aa) were merged by residue Jaccard ≥ 0.5 (`data/processed/proteome_human/run.sh`). Note: that run used FREEZE v1 (no output floor).
- **Memory.** The effective container cap was ~79 GB. The reference featurizer uses one grid per structure (no bricking); all 1,709 training assembly inputs, up to Track A's cap of 150,000 atoms, were featurized without error. Training jobs with full candidate caches OOM-killed when 3 ran at once.
- **Caching.** Point features for the train/dev sets are cached per feature design in `results/dev/cache/feats_biolip/<hash>/<id>.npz` (git-ignored; ~74 GB). Candidate pockets with out-of-fold scores are cached in `results/dev/cache/pockets/<run>/` (`train.pkl` holds descriptors, targets, centres, residues and point indices). These are the fastest source of descriptors for thousands of training pockets.
- **Reproducibility.** Seeds come only from `pocketdetect.seeds.SEEDS`. Every model JSON carries `meta.provenance` (manifest, test-union and exclusion-list sha256, git commit, seed). Re-training reproduces the frozen models byte-for-byte. Frozen test runs used an isolated git worktree. `scripts/reproduce_all.sh` lists every step, and `scripts/reproduce_all.sh smoke` runs a 7 s end-to-end check.

## 5. Learnings that matter for pocket comparison
- **Sites are split into pieces.** Candidates include ~6 Å k-means sub-pockets and multi-threshold components, and suppression keeps pockets 10 Å apart. A large site (e.g. a cofactor plus substrate cleft) often appears as 2–3 output pockets, and one pocket can cover two ligands. For comparison, merge pockets whose point sets touch (or whose residue sets overlap), or compare at the residue-set level.
- **The output list is truncated on purpose.** The v2 floor keeps ~7 pockets per structure. 65 % of v2's non-ion dev misses are pockets that were found but dropped (`results/dev/biolip/NONION_ERRORS.md`). For an exhaustive pocket set use `--min-score none` (≤ 50 pockets) or `--output-rule region` (≈ 14 pockets; max recall +7 to +10 points on dev at no loss in ranking metrics; `results/dev/biolip/V3b_CYCLE.md`).
- **Ion and metal pockets are tiny and numerous.** They carry about 2/3 of the `all` ranker's gain over baselines (REPORT §4.3). Separate them (`metal_n`, `n_coord_max`, small `n_pts`) before computing similarity, or use the `small` ranker. Otherwise metal sites dominate counts and nearest-neighbour lists.
- **The features are redundant.** Dropping any single feature group costs at most 2.5 points (ablation A4, REPORT §4.7), so dimensionality reduction should lose little. Size-dependent descriptors (`vol`, `score_sum`, `n_all*`, `n_atoms_prot`) mix pocket identity with protein size; use fractions and means for similarity.
- **The learned score carries the signal.** With randomly shuffled ranks BURROW falls to the surface-tiling level (14.6 on dev), and geometry-only ranking reaches 53.3 vs 65 for the learned ranker. Weight comparisons by pocket confidence. The score is comparable across proteins (P1K 96.1 on LIGYSIS).
- **AlphaFold and apo caveats.** AF models of well-studied proteins are often holo-like, and pLDDT does not separate binders from non-binders (LITERATURE.md, "Holo-likeness and memorisation"). `--af-mode` changes the atom set (pruned residues) and therefore pocket geometry. It mainly removes hallucinated pockets (5.0 % → 0.2 %). It does not raise recall over a simple pLDDT < 70 filter. Side-chain tolerance was tried and hurt badly (−10.9 points). On apo/cryptic sites BURROW localises well (DCA4) but ranks below P2Rank under the closest-pocket rule.
- **Centres are not unique.** Our centre is a weighted point centroid. LIGYSIS uses the site centre, and P2Rank uses the centroid of SAS points. Under the closest-pocket rule, long output lists lose sites to nearby low-ranked pockets (REPORT §4.2). For pocket matching, use residue-set overlap or point-cloud overlap rather than centre distance alone.
- **Determinism.** Outputs are exact across CPU/GPU and runs. The only randomness is in training, with fixed seeds.
- **Known weak spots.** Shallow and flexible sites have low hit rates for every method (buriedness < 0.5: 22 %; NONION_ERRORS.md). Lipid/detergent and additive sites are frequent and often non-biological. Interface sites in assemblies are at par with P2Rank, not better (V3_CYCLE.md).

## 6. Data assets that a comparison method can reuse
| asset | location / reader | use for comparison |
|---|---|---|
| leakage-controlled LIGYSIS-style train/dev (6,475 / 579 representative chains after the D6 audit; pooled ligands of up to 20 entries per UniProt superposed on one chain; sites = ligand clusters) | `data/processed/{train,dev}/manifest.csv` + `entries/*.npz`; `pocketdetect.datasets.trainset` (`read_manifest`, `load_entry`, `TrainEntry.site_table`, `ligand_atoms`); docs/TRAINSET.md | labelled pockets with ligand identity (CCD codes per site) |
| test union and exclusion lists | `data/manifests/test_union.tsv` (+ `.sha256`), `data/manifests/train_dev_audit_exclusions.tsv`, `data/manifests/test_union_uniprots.txt` | keep any comparison benchmark leakage-free against our tests (MMseqs2 rule in docs/TRAINSET.md §3) |
| AF models of train/dev proteins with transferred sites | `data/processed/af_train/{train,dev}`; `scripts/data/build_af_train_io.py` | the same site in a crystal structure and in an AF model |
| apo chains of train/dev proteins with transferred sites (pocket RMSD; 9.3 % cryptic-like) | `data/processed/apo_train/{train,dev}`; `pocketdetect.datasets.apotrain` | the same site across apo/holo states |
| dimer benchmark (400 assemblies; symmetric copies grouped) | `data/processed/dimer/`; `scripts/eval_dimer.py` | symmetric copies = identical sites (positives) |
| BURROW predictions on all test benchmarks | `release/burrow/predictions/*.csv.gz` (MANIFEST.tsv) | ready-made pocket sets; points are not shipped (re-run for points) |
| human proteome pockets (top-20 per AF model, v1) | `data/processed/proteome_human/{naive,afmode}/pockets.tsv` | large unlabelled pocket set for retrieval tests |
| candidate caches with descriptors (train, out-of-fold) | `results/dev/cache/pockets/p3_*_s0/` | thousands of labelled candidate pockets with descriptors |

**Natural pairs.**
- **Positive pairs:**
  - the same site across apo/holo/AF states (apo_train, af_train);
  - symmetric copies in assemblies (dimer set);
  - pooled ligands of one site across PDB entries;
  - pockets binding the same ligand chemotype in non-homologous proteins (CCD codes in `TrainEntry.arrays["lig_ccd"]`; cluster30 in the manifests).
- **Negatives:** different sites in one protein, and unrelated ligands.
- **Homology leakage:** split by `cluster30`, or stricter by fold; we used Foldseek TM ≥ 0.5 for the fold-disjoint ablation.
- **Ligand-type bias:** cofactors (ATP, NAD, haem) and additives dominate counts. Balance by chemotype, or similarity will learn "is it a nucleotide site".

## 7. Literature pointers (only what is in LITERATURE.md)
- **PocketVec** (*Nat Commun* 2024): pocket descriptors for more than 32k sites in about 20k human domains. Listed under proteome atlases in LITERATURE.md §4.
- **HOTPocket, CavitySpace, AF2BIND:** human proteome pocket atlases (same section). These are possible reference sets.

LITERATURE.md was written for pocket *detection*. It does not review pocket-comparison methods, so a
dedicated literature review on pocket similarity is needed before designing the method.

## 8. Process lessons for the next agent (from LESSONS.md)
1. Inspect mounts, tools and the real resource caps first (container memory ~79 GB, not 128; no Java/nvcc).
2. Run every job longer than ~5 min detached, with a log and a completion marker. Cap workers on the shared node. Kill only PIDs that you stored at launch; never `pkill -f` (it once killed the agent's own shell 4×).
3. Reproduce published numbers before building on an evaluation harness (we reproduced LIGYSIS 33/33 and P2Rank's evaluator to 0.1 %).
4. "Done" for data or leakage work needs an independent re-implementation (an independent audit dropped 186 entries that the builder had passed, D6).
5. Pre-register gates, endpoints and output-list policy, select on train cluster-CV, and touch dev rarely; log every deviation as a dated decision.
6. Every mechanism or negative claim needs a control. Decompose headline gains by site composition before claiming them (ions carried ~2/3 of the `all` gain).
