# Baseline pocket predictors

Wrappers `run_<tool>.py` run third-party pocket predictors and convert their output to **one common format**, so
that the evaluation harness (`src/pocketdetect/...`) never needs tool-specific parsers. Tool binaries, sources, weights
and per-tool virtualenvs live in `baselines/tools/` (git-ignored, ~45 GB); wrapper outputs / benchmark outputs go to
`baselines/outputs/` (git-ignored).

Hardware used for all numbers below: AMD EPYC 9554 (64 cores, 2 sockets-worth of threads exposed as 64 CPUs),
4x NVIDIA RTX 4090 (24 GB), 128 GB RAM; Linux 5.15, Singularity container. **The node is shared: other users' jobs
kept the load average at 25-28 during the benchmarks**, so absolute times are probably 0-30% pessimistic and noisy.

## Common output format

For every input structure `<stem>` (file name without `.pdb/.cif/.ent` and `.gz`, e.g. `1atp`, `1a34-assembly1`),
a wrapper writes into the output directory `OUT`:

* `OUT/<stem>.pockets.csv` - one row per predicted pocket, **ranked best first**, header exactly
  `rank,score,center_x,center_y,center_z,residues`
  * `rank`: 1..n (dense, sorted by the tool's own score, ties broken by tool order);
  * `score`: tool-specific, higher = better; **not comparable across tools** (see table);
  * `center_*`: pocket center in input coordinates (A);
  * `residues`: space separated `chain:resnum[insertion]` (author chain id / author residue number / PDB insertion code,
    e.g. `A:103 H:52a H:100b`), sorted by chain then number. Tool-specific meaning of "pocket residue" - see table.
* `OUT/<stem>.points.csv` (tools that have pocket points), header `rank,x,y,z`: P2Rank = SAS points assigned to the
  pocket, fpocket/DeepPocket = alpha-sphere centers, pyKVFinder = cavity voxels (<=3000 per cavity), others see table.
  Every `rank` in this file exists in `pockets.csv`.
* `OUT/timing.csv` - one row per structure: `stem,status,n_atoms,n_res,n_chains,fmt,clean_s,tool_s,total_s,note`
  (`status` is `ok` or `error`; failures never abort a batch, they only add an error row and no pockets file).
  `clean_s` = our gemmi cleaning, `tool_s` = tool itself (+ conversion), `total_s` = per-structure wall time.
  A structure with zero predicted pockets gets a header-only `pockets.csv`.
* `OUT/raw/<stem>/` with `--keep-raw`: raw tool output.
* Some wrappers add `OUT/<stem>.residues.csv` (per-residue scores: IF-SitePred, PocketMiner).

Format is verified by `baselines/check_outputs.py OUTDIR...` (header/rank/residue-token validation; plus the rank of the
first pocket within 4 A of the ligand for 1atp/3ert/1hsg/4hhb as a smoke test).

## Common input handling (all wrappers, `common.py`)

Inputs: structure files (`.pdb`, `.ent`, `.cif`, `.gz` of those), a directory, or `@listfile` (one path per line).
All wrappers share the same cleaning (`common.clean_structure`, gemmi) unless `--no-clean` is given:

1. first model only (NMR); alternative conformations: first altloc only;
2. waters, ligands, ions and every other non-polymer entity removed (entity-based, so **modified residues inside a
   polymer chain, e.g. MSE, are kept**); hydrogens removed;
3. nucleic-acid polymer residues removed (`--keep-nucleic` keeps them) - benchmarks are protein-only;
4. chain ids / residue numbers are the **author** ones (`auth_asym_id`, `auth_seq_id`, insertion codes preserved);
5. cleaned file written as PDB if representable (<=99,999 atoms, single-character chain ids, resnum < 10000) else mmCIF.
   Chains with multi-character ids (e.g. `A-2`, `SA` in assemblies) are renamed to unique single characters (up to 87
   chains) and **mapped back in the outputs**; the mmCIF fallback is only used by wrappers whose tool can read it
   (P2Rank). The wrappers of the other tools return an `error` row for structures that cannot be written as PDB
   (> 99,999 atoms or > 87 chains after cleaning, e.g. big ribosomes/capsid assemblies) - this is a limitation of
   those tools' PDB-only readers, not of cleaning.

Ligand-binding evaluation should therefore be done against ligands taken from the *original* file, with the cleaned
protein as the only thing the predictors ever saw.

## Calling

```bash
# tools that run in the project venv (CPU tools)
.venv/bin/python baselines/run_p2rank.py   @list.txt -o OUT -j 16 [--config default|alphafold] [--mode batch|per-file]
.venv/bin/python baselines/run_fpocket.py  1atp.pdb 4ake.cif.gz -o OUT -j 16 [--score fpocket|drug]
.venv/bin/python baselines/run_pykvfinder.py DIR -o OUT -j 16
.venv/bin/python baselines/run_fpocket_prank.py @list.txt -o OUT -j 16 [--fpocket-version 4.0.3] [--p2rank-version 2.5.1|2.4.2]
# P2Rank 2.4.2: .venv/bin/python baselines/run_p2rank.py @list -o OUT --version 2.4.2   (fpocket: --version 4.0.3)
# GPU tools (select the GPU with --device N or CUDA_VISIBLE_DEVICES; -j = CPU workers for preprocessing)
baselines/tools/deeppocket/venv/bin/python baselines/run_deeppocket.py @list -o OUT -j 8 --device 1
baselines/tools/puresnet/venv/bin/python   baselines/run_puresnet.py   @list -o OUT -j 8 --device 1
.venv/bin/python baselines/run_grasp.py       @list -o OUT -j 16 --device 2 [--model train_full|coach420_mlig|holo4k_mlig|cv]
.venv/bin/python baselines/run_vnegnn.py      @list -o OUT --device 2 [--split-chains]
CUDA_VISIBLE_DEVICES=3 .venv/bin/python baselines/run_ifsitepred.py  @list -o OUT
CUDA_VISIBLE_DEVICES=3 .venv/bin/python baselines/run_pocketminer.py @list -o OUT
CUDA_VISIBLE_DEVICES=3 .venv/bin/python baselines/run_kalasanty.py   @list -o OUT
```
Common options: `-o/--out`, `-j/--procs`, `--keep-raw`, `--keep-nucleic`, `--no-clean`, `--overwrite` (otherwise structures
with an existing `<stem>.pockets.csv` are skipped, so interrupted runs can be resumed). `baselines/runtime_set.txt` is the
20-structure runtime set (`@baselines/runtime_set.txt`).

## Tools

| tool | version / source | what it is | `score` | `residues` | points | status |
|---|---|---|---|---|---|---|
| **P2Rank** | 2.5.1 (default) and 2.4.2 (`--version 2.4.2`, for the LIGYSIS reproduction gate; `tools/p2rank_2.4.2`) (GitHub release tarballs) + Temurin JDK 17.0.21 (`tools/jdk`) | RF on SAS-point features, clustered | P2Rank pocket score (sum of point scores) | P2Rank's pocket residues (surface-atom neighbours) | SAS points of the pocket | works; configs `default`, `alphafold` |
| **fpocket** | 4.2.3 (`tools/fpocket_src`, default) and 4.0.3 (`tools/fpocket_4.0.3`, `--version 4.0.3`; **only works when built `-O0`**: the stock `-O2 -pg` build segfaults) - both built with gcc | Voronoi alpha spheres, clustering, heuristic score | fpocket `Score` (`--score drug`: Druggability Score) | residues of pocket atoms (`pocketN_atm.pdb`) | alpha-sphere centers; center = mean of them | works |
| **fpocket + PRANK** | `run_fpocket_prank.py`: fpocket (4.0.3 default) then P2Rank `rescore` (2.5.1 default, or `--p2rank-version 2.4.2`; `--rescore-config rescore_2024`) | fpocket pockets re-scored by the PRANK RF model | PRANK score | P2Rank residues of rescored pockets | SAS points | works |
| **pyKVFinder** | 0.9.5 (PyPI) | geometric cavity detection (no learning, no ranking) | cavity volume A^3 (`--score area`) | pyKVFinder interface residues (see note) | cavity voxels (stride-subsampled) | works |
| **DeepPocket** | devalab/DeepPocket @04ba9f5, torch 2.3, libmolgrid 0.5.3 | fpocket candidates re-ranked by 3D-CNN | CNN probability | fpocket residues (segmentation stage not run) | fpocket alpha spheres | works (weights from 3rd-party mirror, see notes) |
| **PUResNet v2** | jivankandel/PUResNetV2.0 @6588018 + pure-torch sparse-conv port (`pv2_torch.py`) | sparse 3D ResNet atom classifier, DBSCAN | sum of atom probabilities per cluster | residues of cluster atoms | cluster atoms | works (MinkowskiEngine replaced) |
| **GrASP** | tiwarylab/GrASP @0a01251 (py3.7 micromamba env), weights in repo | GNN on surface atoms, average-linkage clustering | sum p^2 of site atoms | residues of site atoms | site atoms | works (2 local patches) |
| **VN-EGNN** | ml-jku/vnegnn v1 code @1407eec + HF checkpoint `fses91/VNEGNN-MODEL` | E(3)-equivariant GNN with virtual nodes + ESM2-650M | mean cluster confidence | residues with any atom <=6 A of center (derived) | virtual nodes | works (`--split-chains` for big assemblies) |
| **IF-SitePred** | annacarbery/binding-sites @fb9d778 | ESM-IF1 embeddings + 40 LightGBM | cluster size (point count) | residues of the cluster's binding residues | cluster grid points | works (PyMOL step re-implemented) |
| **Kalasanty** | cheminfIBB/kalasanty @aa91525 (TF 2.15 port) | 3D U-Net voxel segmentation | probability mass of pocket voxels | residues <=4.5 A from pocket voxels | pocket voxels | works (patched) |
| **PocketMiner** | Mickdub/gvp branch `pocket_pred` @187062d (TF 2.15) | GVP-GNN per-residue *cryptic* pocket probability | sum of residue probabilities | residues of the cluster (p>=0.5, 8 A single linkage) | none | works; not a ligand-site predictor |

Notes
* pyKVFinder reports residues as chain+number without insertion codes; the wrapper lists all insertion-code variants of that number.
* Scores are **not** calibrated between tools; use rank only unless a tool's score semantics are stated.
* GrASP, VN-EGNN, PUResNet, DeepPocket, Kalasanty and IF-SitePred need the GPU (GrASP/Kalasanty/DeepPocket/PUResNet also spend
  most time on CPU featurisation, so give them `-j`).
* None of the tools except P2Rank's `alphafold` profile has an AlphaFold/pLDDT mode; P2Rank `default` uses the B-factor as a
  feature (**do not use `default` on AF models, whose B-factor column is pLDDT; use `--config alphafold`**).
  PocketMiner/IF-SitePred/VN-EGNN/GrASP/Kalasanty/DeepPocket ignore B-factors.

### Tool-specific details, quirks and deviations (recorded for the fairness section of the report)

**P2Rank 2.5.1** - official release; JVM called directly (`java -cp bin/p2rank.jar:bin/lib/* cz.siret.prank.program.Main
predict`, so that `-Xmx` is ours). Batch mode = one JVM, dataset file, `-threads N`; per-file mode = one JVM per structure.
Reads PDB and mmCIF (also `.gz`); SAS points taken from `visualizations/data/<file>_points.pdb.gz` (resSeq = pocket rank;
P2Rank's pseudo-PDB is not strictly fixed width, so the wrapper parses it tolerant of rank >= 100). Default config = `default.groovy`
(model `default`, features incl. B-factor). `--config alphafold` = `-c alphafold` (model `alphafold`; features `chem,volsite,protrusion`, no B-factor).
Conservation configs are not used (they need HMM files). 2.4.2 differences: no `vis_renderers` option (wrapper omits it); its `rescore` writes only `<file>_rescored.csv`, so `run_fpocket_prank.py` takes pocket geometry from the fpocket output and score/rank from the rescored csv. P2Rank output residue ids keep insertion codes (`H_52a` -> `H:52a`).
Training set (`models/readme.md`): both `default` and `alphafold` models were trained on **CHEN11** (`chen11-fpocket.ds`, Chen et al. 2011, 251 holo
proteins); P2Rank is evaluated in its paper on COACH420, HOLO4K, CHEN11 (CV), JOINED. Leakage w.r.t. new benchmarks: CHEN11 PDB ids are old (<2010); check by id/UniProt.

**fpocket 4.2.3** - built with `make` (gcc). Defaults (`-m 3.4 -M 6.2 -D 2.4 -i 15`); the wrapper passes `-w p` (PDB output) and
accepts extra args via `--fpocket-args`. Single-threaded; run time grows super-linearly with atom count (qhull). No training set (heuristic score).
fpocket keeps HETATM groups as potential ligands, so removing them (our cleaning) matters. mmCIF input only works with gemmi-written files
whose chains are single characters (VMD pdbx plugin), hence the wrapper writes PDB.

**pyKVFinder 0.9.5** - `pyKVFinder.run_workflow` with defaults (step 0.6 A, probe_in 1.4, probe_out 4.0, removal distance 2.4, volume cutoff 5 A^3, SES).
Produces *all* cavities (hundreds on large complexes), ranked by volume - it is not a ligand-site predictor, only a geometric baseline. No training set.

**DeepPocket** - Steps of upstream `predict.py`: clean (+ keep only 20 standard amino acids) -> fpocket 4.2.3 -> candidate centres (alpha-sphere barycentre,
radius weighted) -> libmolgrid gninatypes voxelisation (16 A, 0.5 A) -> CNN (`first_model_fold1_best_test_auc_85001.pth.tar`, fold-1 classifier). The official OneDrive weight links
are dead; the classification checkpoint was taken from the third-party fork `D4-course/DeepPocket` whose code is byte-identical to upstream (sha256 of the weights file:
d166c8a36b2475297193dbe31e5a88ff9b853e343281ef21df537029ba053f9a). The segmentation checkpoint could not be found, so only re-ranking is provided. Training set: **scPDB (v.2017, ~17k entries), 5-fold CV** (fold 1 model),
paper tests COACH420 and HOLO4K; no explicit homology filtering vs these known -> treat as potentially leaky.

**PUResNet v2** - Weights from the repo (MIT). MinkowskiEngine cannot be built without nvcc: sparse convolutions re-implemented in pure torch (`pv2_torch.py`, strict state-dict load;
atom-level AUC 0.93 on the 12 repo example proteins, not bit-exact vs ME). openbabel-wheel needs libX11/Xrender/Xext (absent): stub libs in `tools/puresnet/stubs`
(wrapper sets `LD_LIBRARY_PATH` via re-exec). Residue definitions (CCD) are pre-fetched with curl into `tools/puresnet/residues` (tool's own downloader uses `wget`).
No native ranking: score = sum of atom probabilities per DBSCAN cluster (eps 6 A, p>=0.34). May return 0 pockets. Training set: **PDB-wide, 49,353 train + 12,338 val
entries (61,691 PDB ids up to 9xia; ids in `puresnet/data/{training,validation}_data.npy`), split by id only - very high leakage risk** for any PDB-derived benchmark.

**GrASP** - original repo, py3.7 env (OpenBabel 2.4.1, torch 1.13.1, PyG 2.2.0), Cython extension rebuilt. Local patches in `tools/grasp/src/featurize_protein.py`
(hybridisation `SP`, bond order 3) because 4hhb crashed otherwise. Upstream "production" mode only emits per-atom probabilities; the wrapper adds the paper's clustering
(average linkage 15 A, p>0.3, site score sum p^2). Default weights `train_full` = **all of sc-PDB (~15.9k ids)**, no similarity filtering vs COACH420/HOLO4K/LIGYSIS.
Leakage-controlled weights shipped by the authors: `--model coach420_mlig` (scPDB filtered by UniProt vs COACH420), `--model holo4k_mlig`, `--model cv` (10-fold CV fold 0); id lists
in `tools/grasp/src/splits/`. CPU featurisation is ~quadratic in atom count (single thread per structure); structures > ~35k atoms are impractical.

**VN-EGNN** - current ml-jku/vnegnn master ships no weights and has a different architecture; the released checkpoint belongs to the v1 code (commit 1407eec), which is what
we run (+ ESM2-650M, 8 virtual nodes, MeanShift). Stochastic (random initial rotation): `--seed 0`. Residues are derived by us (6 A). Whole-structure mode degrades on large assemblies
(virtual-node sphere radius ~ complex size); `--split-chains` follows the authors' HOLO4K protocol (per-chain prediction, pooled and ranked by confidence). Training set:
**PUResNet's scPDB subset, 4498 train + 520 valid complexes** (lists in `tools/vnegnn/src_v1/data/scpdb/`); checkpoint selected on val DCC; no similarity filtering vs test sets documented.

**IF-SitePred** - Binding-MOAD-family-leader trained (6,550 families, filtered to <25% seq identity to the authors' test sets HAP/HOLO4K-AF2; `src/data/train.fasta` lists PDB ids).
ESM-IF1 (142M) per-residue embeddings of single chains + 40 LightGBM models; official code needs PyMOL for the point-cloud step, re-implemented with scipy (residue predictions verified
identical to the official script on 12as; clustering step not cross-checked). Deviation: all chains are processed and all clusters reported (official: first chain only, top 3 sites; `--first-chain-only`).

**Kalasanty** - TF1-era 3D U-Net ported to TF 2.15 / OpenBabel 3 (tfbio patched, weights `model_scpdb2017.hdf` loaded by topology). 36^3 voxel box (72 A) centred on the protein centroid;
larger proteins are tiled by the wrapper (`--no-tile` = official behaviour, truncation). Inputs are protonated at pH 7.4 with OpenBabel. The tool has no score; ours = probability mass.
Training set: **sc-PDB 2017 (~16.2k entries) excluding 328** (full set for the shipped model; 10-fold fold lists in the repo) -> heavy overlap with scPDB-derived benchmarks.

**PocketMiner** - per-residue cryptic-pocket probability (GVP-GNN, TF 2.15, checkpoint in repo). Pocket extraction is ours (residues p>=0.5, single linkage 8 A, score = sum p, center = mean CA);
often yields no pockets on holo-like structures (3ert: none). Reproduction check: pooled test ROC-AUC 0.79 on the repo's labelled test residues (paper reports 0.87; unresolved).
Training set: MD-simulation-derived labels of **37 proteins**; test = 38 apo-holo pairs (<55% seq id to training).

## Runtime benchmark

Set: `baselines/runtime_set.txt` (20 structures: 327 to 93,415 atoms after cleaning, 1-78 chains, incl. two PDB
biological assemblies `1a34-assembly1` (60 chains) and `4hhb-assembly1`, one PDB-format entry `4ake`, a CA-only 1ffk, and
multi-chain asymmetric units such as 1aon, 5xnl, 7k00, 6zj3). Times are **per-structure wall seconds including our gemmi cleaning**
(`timing.csv: total_s`), raw logs in `baselines/outputs/bench/<run>/timing.csv`; tables made with `summarize_timing.py`.
Measured 26 Oct 2026 on the shared node (load average 22-60 from other users' jobs, so +-30% noise and some inflation;
e.g. fpocket 4.2.3 on 6zj3 took 815 s in a heavily loaded run vs 294 s for fpocket 4.0.3 in a lighter one).

**CPU tools, "single thread" = each structure handled by one single-threaded process (16 such processes concurrently).**
P2Rank `perfile` = one JVM per structure (`-threads 1`, includes ~4-7 s JVM start + model load under load);
`batch_j1` = one JVM, one thread, all 20 structures sequentially: the number shown is total/20 (amortised, warm), real total = 71.4 s (2.5.1) / 130.2 s (2.4.2).
Run names: `fpocket_j16` = fpocket 4.2.3, `fpocket403` = fpocket 4.0.3 (needs -O0 build), `fpocket403_prank` = fpocket 4.0.3 + P2Rank 2.5.1 `rescore` (default_rescore).

| structure | atoms | chains | p2rank_perfile_j16 | p2rank_batch_j1 | p2rank242_perfile_j16 | p2rank242_batch_j1 | fpocket_j16 | fpocket403_j16 | fpocket403_prank_j16 | pykvfinder_j16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1crn | 327 | 1 | 7.50 | 3.45 | 8.59 | 6.44 | 0.10 | 0.10 | 1.37 | 0.25 |
| 1ubq | 602 | 1 | 7.36 | 3.43 | 8.62 | 6.43 | 0.16 | 0.16 | 1.41 | 0.40 |
| 2lzm | 1309 | 1 | 7.64 | 3.43 | 9.20 | 6.43 | 0.29 | 0.28 | 1.58 | 0.86 |
| 1hsg | 1514 | 2 | 7.69 | 3.43 | 9.36 | 6.43 | 0.33 | 0.36 | 1.65 | 0.95 |
| 3ert | 1932 | 1 | 7.86 | 4.07 | 9.46 | 6.43 | 0.65 | 0.48 | 1.77 | 2.19 |
| 3pqr | 2681 | 2 | 7.91 | 3.44 | 9.23 | 6.43 | 0.80 | 0.76 | 2.05 | 1.47 |
| 1atp | 2934 | 2 | 7.94 | 3.43 | 10.32 | 6.43 | 0.71 | 0.63 | 1.85 | 1.32 |
| 4ake | 3312 | 2 | 4.09 | 3.43 | 10.85 | 6.43 | 1.06 | 0.87 | 2.11 | 1.73 |
| 1ffk | 3656 | 27 | 7.20 | 3.54 | 13.79 | 6.52 | 5.13 | 4.15 | 5.32 | 34.65 |
| 1tim | 3740 | 2 | 8.11 | 3.43 | 9.78 | 6.43 | 1.15 | 1.26 | 2.50 | 1.95 |
| 5tzo | 4258 | 3 | 8.29 | 3.44 | 10.80 | 6.45 | 1.11 | 0.95 | 2.18 | 2.47 |
| 4hhb | 4384 | 4 | 8.49 | 3.44 | 10.41 | 6.44 | 1.37 | 1.31 | 2.56 | 2.12 |
| 4hhb-assembly1 | 4384 | 4 | 4.54 | 3.43 | 10.89 | 6.43 | 1.36 | 1.34 | 2.57 | 1.85 |
| 6vxx | 22812 | 3 | 10.54 | 3.77 | 15.56 | 6.47 | 23.22 | 17.31 | 18.71 | 13.18 |
| 6cvm | 32772 | 4 | 11.19 | 3.50 | 15.96 | 6.49 | 28.79 | 30.83 | 31.26 | 12.82 |
| 7k00 | 43768 | 49 | 15.87 | 3.72 | 25.08 | 6.68 | 85.98 | 62.49 | 59.40 | 60.39 |
| 1aon | 58674 | 21 | 18.45 | 3.55 | 27.09 | 6.54 | 199.85 | 99.55 | 88.76 | 26.64 |
| 1a34-assembly1 | 68220 | 60 | 10.90 | 3.84 | 24.39 | 6.84 | 100.78 | 80.29 | 81.25 | 37.50 |
| 5xnl | 72688 | 56 | 20.81 | 3.62 | 32.77 | 6.61 | 457.05 | 203.87 | 204.26 | 43.49 |
| 6zj3 | 93415 | 78 | 28.88 | 3.96 | 44.26 | 6.87 | 815.09 | 294.42 | 350.96 | 126.86 |
| sum of per-structure s | | | 211.3 | 71.4 | 316.4 | 130.2 | 1725.0 | 801.4 | 863.5 | 373.1 |

Whole-set wall clock using many cores (throughput): P2Rank 2.5.1 batch one JVM `-threads 64`: **25.5 s** (alphafold config, 16 threads: 27.5 s; per-file j16: 29.1 s);
P2Rank 2.4.2 per-file j16: 44.6 s; fpocket 4.2.3 j64: 586 s (j16: 816 s, heavily loaded); fpocket 4.0.3 j16: 294.5 s; fpocket4.0.3+PRANK j16: 377 s;
pyKVFinder j16 / j64: 128 s (both bounded by the single largest structure 6zj3 = 127 s on one thread; `--threads N` OpenMP option exists but was not benchmarked).
fpocket and pyKVFinder are single-threaded per structure and super-linear in atoms, so large assemblies dominate; for P2Rank the whole set is ~20-30 s with all cores, the largest
structure (6zj3) takes ~29 s as a single JVM call. P2Rank parallelises over structures, a single structure uses 1-few cores.

**GPU tools** (RTX 4090; GPU 3 for IF-SitePred/PocketMiner/VN-EGNN/PUResNet, GPU 2 for the rest; model loaded once per run, load time (5-11 s) excluded from per-structure times,
included in whole-run wall below). CPU preprocessing parallelised with `-j` (4 for the first group, 16 for DeepPocket, 15 for GrASP, 4 for Kalasanty), so per-structure times of the
CPU-bound tools (DeepPocket = fpocket, GrASP/Kalasanty/PUResNet = python featurisation) are single-core latencies; the GPU part alone is 0.05-0.5 s per structure.
GrASP was run only on the 15 structures <= 33k atoms (its featurisation is ~quadratic; 1aon/5xnl/7k00/6zj3/1a34 estimated >15 min each, skipped). PocketMiner fails on the CA-only 1ffk by design.

| structure | atoms | chains | deeppocket_gpu | puresnet_gpu | grasp_gpu | vnegnn_gpu | vnegnn_split_gpu | ifsitepred_gpu | pocketminer_gpu | kalasanty_gpu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1crn | 327 | 1 | 0.14 | 0.30 | 14.86 | 0.36 | 0.40 | 0.94 | 9.45 | 1.87 |
| 1ubq | 602 | 1 | 0.22 | 0.55 | 15.35 | 0.36 | 0.42 | 0.33 | 0.28 | 0.16 |
| 2lzm | 1309 | 1 | 0.32 | 0.55 | 17.30 | 0.37 | 0.37 | 0.53 | 0.25 | 0.22 |
| 1hsg | 1514 | 2 | 0.38 | 0.61 | 17.75 | 0.36 | 0.37 | 0.40 | 0.16 | 0.21 |
| 3ert | 1932 | 1 | 0.86 | 0.75 | 19.80 | 0.10 | 0.09 | 0.74 | 0.13 | 0.23 |
| 3pqr | 2681 | 2 | 0.92 | 0.84 | 25.58 | 0.13 | 0.14 | 0.88 | 0.31 | 0.46 |
| 1atp | 2934 | 2 | 0.87 | 0.87 | 31.79 | 0.11 | 0.23 | 0.76 | 0.17 | 0.38 |
| 4ake | 3312 | 2 | 1.24 | 1.27 | 27.68 | 0.07 | 0.15 | 1.19 | 0.13 | 0.47 |
| 1ffk | 3656 | 27 | 5.28 | 5.56 | 30.19 | 1.68 | 1.95 | 3.12 | FAIL | 7.39 |
| 1tim | 3740 | 2 | 1.25 | 1.09 | 30.47 | 0.08 | 0.12 | 0.58 | 0.27 | 0.44 |
| 5tzo | 4258 | 3 | 1.26 | 1.04 | 35.87 | 0.16 | 0.27 | 0.97 | 0.24 | 0.81 |
| 4hhb | 4384 | 4 | 1.56 | 1.25 | 36.45 | 0.15 | 0.23 | 1.03 | 0.29 | 0.56 |
| 4hhb-assembly1 | 4384 | 4 | 1.55 | 1.44 | 35.91 | 0.09 | 0.19 | 0.46 | 0.20 | 0.57 |
| 6vxx | 22812 | 3 | 24.43 | 5.63 | 153.29 | 1.62 | 0.82 | 3.24 | 0.23 | 16.03 |
| 6cvm | 32772 | 4 | 31.91 | 8.26 | 261.69 | 1.54 | 0.95 | 5.15 | 0.59 | 36.58 |
| 7k00 | 43768 | 49 | 85.57 | 18.68 | - | 3.70 | 3.46 | 37.27 | 0.59 | 88.80 |
| 1aon | 58674 | 21 | 214.60 | 13.24 | - | 6.45 | 2.22 | 15.59 | 0.41 | 171.30 |
| 1a34-assembly1 | 68220 | 60 | 113.64 | 26.37 | - | 6.79 | 3.06 | 24.04 | 0.81 | 325.20 |
| 5xnl | 72688 | 56 | 541.55 | 23.10 | - | 8.28 | 3.93 | 28.02 | 0.86 | 370.00 |
| 6zj3 | 93415 | 78 | 713.30 | 26.16 | - | 11.07 | 4.74 | 20.95 | 0.96 | 700.25 |
| sum of per-structure s | | | 1740.9 | 137.6 | 754.0 | 43.5 | 24.1 | 146.2 | 16.3 | 1721.9 |

Whole-run wall (20 structures unless noted): VN-EGNN 87.6 s (whole-structure; 19.4 s with `--split-chains`), IF-SitePred 195.5 s, PocketMiner 49.4 s, PUResNet v2 61.1 s,
Kalasanty 1748 s (tiling of large boxes; 6zj3 700 s), DeepPocket 788 s, GrASP (15 structures) 340 s.

**Smoke test** (`check_outputs.py`: rank of first pocket within 4 A of the bound ligand / center-to-ligand distance of rank 1 / #pockets) on 1atp (ATP), 3ert (OHT), 1hsg (MK1), 4hhb (HEM):
P2Rank 2.5.1/2.4.2 r1,r1,r1,r1; P2Rank alphafold r1 for all; fpocket r1,r1,r2,r2; fpocket+PRANK r1,r1,r1,r1 (re-ranking works); pyKVFinder r1,r1,r1,r2; DeepPocket r1 x4; GrASP r1 x4;
Kalasanty r1 x4; IF-SitePred r1 x4; PUResNet r1,r1,r1,miss; VN-EGNN r1,r1,r1,miss (split-chains r4,r1,r1,r1); PocketMiner (cryptic) r1,none,miss,miss. Output format validated for all runs (no FORMAT PROBLEM lines).


## Maintenance notes
* Recreate a tool: see the per-tool sections above and `baselines/tools/<tool>/`; versions are pinned in the venvs (`uv pip freeze`).
* `common.py` is the only shared module; wrappers re-exec into their tool venv (`C.reexec_in_venv`) where the tool needs one.
* Re-run the format check after changing any wrapper: `.venv/bin/python baselines/check_outputs.py baselines/outputs/<run>`.
