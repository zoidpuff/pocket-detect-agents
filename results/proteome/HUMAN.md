# BURROW on the human proteome (AFDB v6, UP000005640) — descriptive

23586 AFDB models = 20550 UniProt entries; 215 proteins > 2,700 aa are split by AFDB into overlapping fragments (3251 fragment models), whose pockets are mapped to UniProt numbering and merged (greedy by score, residue Jaccard >= 0.5). Model `burrow_all_default_s0` (frozen v1, default preset), fast backend, 3 GPUs (0,1,3) x 8 CPU workers; outputs `data/processed/proteome_human/` (top-20 pockets per structure; git-ignored). **naive** = standard mode on the full model; **afmode** = `--af-mode` (pLDDT<50 spaghetti pruning + soft occlusion). Shared node (see load) — timings are pessimistic. Code: frozen v1 reference (git worktree at b6a896b) + the fast-module patch of 2026-10-26 22:20 (bounded GPU driver, `--compact`, fp64 thresholds in GPU AF mode; `src/pocketdetect/burrow/fast/`, file-set md5 4be956ffc31b); run script `data/processed/proteome_human/run.sh`, summary `scripts/proteome_summary.py`.

## Throughput and output

| | naive | afmode |
|---|---|---|
| structures ok / error | 23586 / 0 | 23586 / 0 |
| wall-clock run (s) | 169.0 | 128.7 |
| structures/min (run only) | 8373.7 | 10995.8 |
| structures/min incl. tar extraction (49.5 s, once) | 6476.7 | 7941.4 |
| node load (1/5/15 min, at end) | 52.96 62.40 72.33 | 61.54 62.17 70.91 |
| pockets per protein after NMS (median / mean; NMS keeps <= 50 per model, summed over fragments) | 50 / 54.08 | 50 / 49.44 |
| proteins without any pocket | 0 | 129 |

Peak RSS of the whole job (driver + 3 GPU processes + 24 workers): 17.5 GB.


## Hallucination proxy: pockets in low-confidence regions

Fraction of top-ranked pockets whose mean residue pLDDT is < 50 / < 70 (pocket residues = residues with a heavy atom <= 4 Å of a pocket point; pLDDT from the full model).

| pockets | naive <50 | naive <70 | afmode <50 | afmode <70 |
|---|---|---|---|---|
| top-1 (n = 20550 / 20421) | 0.0781 | 0.1794 | 0.0091 | 0.113 |
| top-3 (n = 61650 / 61099) | 0.0797 | 0.1911 | 0.0087 | 0.1204 |

### By protein disorder (fraction of residues with pLDDT < 50)

| disorder | proteins | median pockets (naive / af) | top-3 mean pLDDT<50: naive / af | top-3 <70: naive / af |
|---|---|---|---|---|
| [0,0.1] | 7409 | 50 / 50 | 0.006 / 0.0006 | 0.0635 / 0.0551 |
| [0.1,0.3] | 7170 | 50 / 50 | 0.0431 / 0.0042 | 0.1634 / 0.1011 |
| [0.3,0.5] | 3471 | 50 / 50 | 0.1183 / 0.0166 | 0.2927 / 0.1802 |
| [0.5,1.0] | 2500 | 50 / 49 | 0.3493 / 0.0363 | 0.5073 / 0.2994 |

## Overlap with known (BioLiP2-derived) pockets from HOTPocket

Descriptive only: many of these proteins / sites are in our training and test sets. Known sites = HOTPocket `known_pockets.csv` (BioLiP2 binding-site residues) mapped to UniProt numbering by local alignment (`scripts/proteome_known_pockets.py`) and de-duplicated per protein (Jaccard >= 0.5). A site counts as recovered within the top-k if one of the protein's top-k pockets contains >= 50 % of its residues.

| site category | sites | proteins | mode | top-1 | top-3 | top-5 | top-10 | top-20 |
|---|---|---|---|---|---|---|---|---|
| all | 9754 | 4247 | naive | 0.2232 | 0.3814 | 0.4502 | 0.53 | 0.5964 |
| all | 9754 | 4247 | afmode | 0.2322 | 0.3953 | 0.4625 | 0.5353 | 0.6045 |
| ion | 3440 | 1626 | naive | 0.3125 | 0.5427 | 0.6485 | 0.7544 | 0.8203 |
| ion | 3440 | 1626 | afmode | 0.3186 | 0.552 | 0.659 | 0.7549 | 0.8294 |
| nucleic | 1429 | 794 | naive | 0.0189 | 0.0595 | 0.0917 | 0.1414 | 0.1966 |
| nucleic | 1429 | 794 | afmode | 0.0259 | 0.0588 | 0.0931 | 0.1407 | 0.1959 |
| peptide | 973 | 769 | naive | 0.0761 | 0.1346 | 0.1799 | 0.2343 | 0.296 |
| peptide | 973 | 769 | afmode | 0.0689 | 0.1408 | 0.1757 | 0.2292 | 0.2857 |
| small_molecule | 3912 | 2240 | naive | 0.2559 | 0.4185 | 0.4739 | 0.5483 | 0.6201 |
| small_molecule | 3912 | 2240 | afmode | 0.2722 | 0.4438 | 0.4959 | 0.5624 | 0.6352 |

## P2Rank `-c alphafold` vs BURROW on 500 random proteins

500 random non-fragmented human AFDB proteins (seed 20261026). Wall-clock includes parsing, JVM / process start-up and writing; same node, run after the proteome jobs. Pocket pLDDT = mean residue pLDDT of the pocket's residues (P2Rank: its residue list; BURROW: residues <= 4 Å of pocket points).

| method | wall s | structures/min | top-3 mean pLDDT<50 | top-3 <70 | pockets/protein (median) |
|---|---|---|---|---|---|
| BURROW naive (3 GPUs x 8 workers) | 10.8 | 2782.6 | 0.09 | 0.2027 | 50.0 |
| BURROW afmode (3 GPUs x 8 workers) | 10.4 | 2895.0 | 0.0102 | 0.1253 | 50.0 |
| P2Rank 2.5.1 -c alphafold (one JVM, 24 threads) | 31.2 | 960.5 | 0.0161 | 0.0985 | 3.0 |
