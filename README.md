# BURROW — fast protein ligand-binding pocket detection and ranking

**BURROW** (*Buriedness-driven Ultra-fast Ranking Of pOckets, With confidence awareness*) finds and
ranks ligand-binding pockets in protein structures (PDB/mmCIF, experimental or predicted). It fills
the solvent shell around the protein with 1 Å volume elements, describes each by multi-scale ray-cast
buriedness (full-atom and backbone-only), first-hit chemistry and neighbourhood composition, scores
them with a small gradient-boosted tree model, assembles candidate pockets at several buriedness
and score levels, and ranks them with a second tree model trained on leakage-controlled,
LIGYSIS-style BioLiP2 data. It runs on CPU (Numba) and GPU (PyTorch/Triton) with identical output,
has a dedicated AlphaFold mode (`--af-mode`) and handles multimeric assemblies natively.
Method details: [docs/METHOD.md](docs/METHOD.md). Results: [REPORT.md](REPORT.md).

## Install
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e .
uv pip install numpy scipy pandas numba gemmi lightgbm scikit-learn   # tested: numpy 2.5, scipy 1.18,
                                                                      # pandas 3.1, numba 0.68, gemmi 0.7.5, lightgbm 4.7
uv pip install torch triton            # optional, GPU backend (tested: torch 2.6 + CUDA 12.4, triton 3.2)
```
This installs the `burrow` command (equivalently `python -m pocketdetect.burrow`). The frozen models
are the JSON files in `results/dev/freeze_v2/models/` (FREEZE v2, see below).

## Quickstart
```bash
# one structure (protein only is used: ligands, waters, ions are ignored)
burrow predict 1abc.cif.gz --out out/

# a directory or a list of files, 16 CPU worker processes
burrow predict structures/ --out out/ --workers 16
burrow predict @my_files.txt --out out/ --workers 16

# AlphaFold / predicted model (pLDDT in the B-factor column): prune low-confidence spaghetti, soft occlusion
burrow predict AF-P12345-F1-model_v6.cif.gz --out out_af/ --af-mode

# multimeric assembly (biological assembly file; interface pockets are found across chains)
burrow predict 1abc-assembly1.cif.gz --out out_asm/

# high-throughput GPU batch (one process per GPU, CPU workers per GPU for parsing / post-processing)
burrow predict proteome_dir/ --out out_gpu/ --device cuda:0,1 --workers 8

# exhaustive pocket list (no output score floor) and the slower "accurate" preset
burrow predict 1abc.cif.gz --out out_all/ --min-score none --preset accurate

# ligand definition without ions (P2Rank-style "small molecules"); learning-free baseline
burrow predict 1abc.cif.gz --out out_small/ --ranker small
burrow predict 1abc.cif.gz --out out_geo/ --ranker geometry
```
Backends: `--backend auto` (default) uses the fast backend and falls back automatically to the
reference implementation; `--backend reference` forces the reference implementation. All
backends produce byte-identical files (tested).

## Output
Per input `<stem>`:
* `out/<stem>.csv` — one row per pocket, best first:
  `rank,score,center_x,center_y,center_z,residues` (score = ranker probability-like score, comparable
  across proteins; residues = `chain:resnum[icode]`, author numbering of the input, space-separated).
* `out/<stem>.points.csv` — the pocket's volume elements: `rank,x,y,z` (omit with `--no-points`).
* `out/timing.csv` — status and per-structure timings.

## Presets and the output floor
| preset | what | typical cost (fast backend, 1 core, excl. parsing) |
|---|---|---|
| `default` | 30 rays, rays up to 16 Å, 3 Å coarse sub-lattice, only buried points scored | ≈ 80 ms / structure (estimate) |
| `accurate` | 60 rays up to 24 Å, 2 Å sub-lattice, all points scored (≈ +1–2 pt on non-ion sites) | ≈ 200 ms / structure (estimate) |
| `fast` | 1.5 Å grid, 12 rays, small point model (≈ −1 pt top-N+2, −1.5 pt DCA4 vs default on dev; CPU only for now) | ≈ 31 ms / structure (measured, speed200) |

**Output score floor (`--min-score`, default 0.03 in FREEZE v2).** Pockets with score < 0.03 are
dropped once 3 pockets are listed. This shortens the list from ~40 to ~6–8 pockets per structure and
improves "closest-pocket" benchmark metrics (e.g. LIGYSIS top-N+2) by 5–10 points, but lowers
coverage: some true sites are only found by low-scoring pockets (max recall ~99 % → 73–87 %).
For exhaustive pocket discovery use `--min-score none` (identical to FREEZE v1 output).

**Optional `--output-rule region`** (not the default): the v2 floor plus the best pocket (score ≥ 0.005) of every
surface region more than 16 Å away from the pockets already kept. More coverage for exhaustive discovery
(~14 instead of ~7 pockets per structure; dev max recall +7 to +10 points with unchanged top-N+2, DCA4 and P1K —
dev evidence only, it did not pass the gate for a new default: `results/dev/biolip/V3b_CYCLE.md`).

## Models
* `results/dev/freeze_v2/models/burrow_{all,small}_{default,accurate}_s{0,1,2}.json` — FREEZE v2
  (paths and sha256: [results/dev/FREEZE_v2.md](results/dev/FREEZE_v2.md)); the CLI uses seed 0.
  Each JSON contains the point model, the pocket ranker, the feature design, the output settings and
  provenance (training-manifest, test-union and exclusion-list sha256, git commit, seed).
* `results/dev/freeze/models/` — FREEZE v1 (same models, no output floor).
* `results/dev/freeze_fast/models/burrow_{all,small}_fast_s{0,1,2}.json` — additional `--preset fast`
  ([results/dev/FREEZE_fast.md](results/dev/FREEZE_fast.md); not pre-registered).

## Data and citation
Every dataset used (PDB, AFDB, BioLiP2, LIGYSIS, COACH420, HOLO4K, CryptoBench, PocketMiner,
PocketWatch, scPDB, P2Rank datasets, …) is listed with source, version, checksum and license in
[DATA.md](DATA.md) — please cite the original datasets from there. Training/dev data construction:
[docs/TRAINSET.md](docs/TRAINSET.md); evaluation protocol: [docs/EVALUATION.md](docs/EVALUATION.md);
pre-registered claims: [docs/PREREG.md](docs/PREREG.md).

## Results
Headline test results, comparisons with P2Rank, fpocket(+PRANK) and deep-learning baselines,
runtimes and limitations: [REPORT.md](REPORT.md). Full reproduction: `scripts/reproduce_all.sh`
(`scripts/reproduce_all.sh smoke` runs a 2-minute end-to-end check).
