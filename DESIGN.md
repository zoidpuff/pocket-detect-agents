# DESIGN — v1 (draft for review round 1)

Working name: **BURROW** (Buriedness-driven Ultra-fast Ranking Of pOckets, With confidence awareness). Package: `pocketdetect`.

Grounding: LITERATURE.md (esp. §2.2 LIGYSIS, §4 AF, §7 implications). Time budget: 24 h wall clock from 2026-10-26 14:54 CET; 4×RTX 4090, 64 cores.

## 1. Objectives and success criteria (fixed before development)

| ID | Claim | Benchmark | Primary metric | Bar to clearly beat |
|---|---|---|---|---|
| O1 | Better detection+ranking than P2Rank & co. | LIGYSIS (2,775 chains, LBS-comparison protocol), also LIGYSIS_NI | top-N+2 recall, DCC ≤ 12 Å | fpocket+PRANK 60.4; P2Rank 51.9; P2Rank-CONS 53.9 (published; we re-run P2Rank/fpocket ourselves) |
| O1b | Calibrated cross-protein scores | LIGYSIS | Precision1K, #TP@100FP | P2Rank-CONS P1K 90.7 |
| O2 | Legacy comparability | COACH420(mlig), HOLO4K(mlig) | DCA 4 Å top-n, top-(n+2) | P2Rank re-run by us (published 72.0 / 68.6 top-n) |
| O3 | Generalization to novel sites/folds | (a) temporal PocketWatch test (≥2024 sites) if license allows, else BioLiP2 date-cut; (b) LIGYSIS stratified by max seq-id / Foldseek TM to our training set | top-N+2 recall per stratum | P2Rank per stratum |
| O4 | Apo & cryptic | CryptoBench test (222 apo), PocketMiner 38 pairs | pocket top-k DCA/DCC on apo; residue AUPRC | P2Rank (AUPRC 0.21 published), fpocket |
| O5 | AF mode | LIGYSIS UniProts on AFDB v6 models (sites by UniProt residue numbers) | top-N+2 recall; hallucination rate (pockets with mean pLDDT<50 matching no site) | naive mode, P2Rank `-c alphafold`, fpocket on same models |
| O6 | Dimers | New: interface-site benchmark from PDB biological assemblies (single-chain / homo-interface / hetero-interface sites) | top-N+2 recall per site class, isolated chain vs assembly input | P2Rank & fpocket on the same inputs |
| O7 | Speed | all of the above | ms/structure CPU (1 core, warm+cold), structures/s GPU batched | P2Rank (~0.5–0.9 s), fpocket (~0.2 s) |

"Clearly beat" = paired bootstrap 95% CI of the difference vs the best baseline excludes 0 on the primary metric of O1, and no significant loss on O2–O6.

## 2. Method

### 2.1 Pipeline
1. **Parse** heavy atoms (gemmi → struct-of-arrays). All chains of the given file are used (single chain, assembly, or AF model). Per-atom occlusion weight `w_a` = 1 (standard) or pLDDT/side-chain-dependent (AF/flex mode, §2.4).
2. **Candidate points** (union, deduplicated at 1 Å):
   - (a) grid points (1.0 Å spacing) in the solvent shell (≥ vdW+probe 1.4 Å from atoms, ≤ 8 Å from protein) with ray-cast buriedness ≥ β₀ (β₀ chosen on dev set for ≥ 95% ligand-atom coverage at minimal point count) — LIGSITE/cavity-style, high recall for enclosed sites;
   - (b) SAS points (P2Rank-style, probe 1.6 Å) — recall for shallow sites.
3. **Point features** (all analytical, CPU Numba + GPU torch):
   - multi-scale ray-cast buriedness (K=30/60 rays; L = 6, 10, 16, 24 Å) with soft occlusion; also backbone+Cβ-only buriedness (side-chain-tolerant);
   - P2Rank-like protrusion (atom count within 10 Å) and atom-class densities within 4/6/8/12 Å (C, N, O, S, aromatic C, hydrophobic side-chain C, donors, acceptors, +/− charged, backbone, metal-coordinating atoms His N/Cys S/Asp-Glu O);
   - residue-type composition within 8 Å (20-dim, Gaussian-weighted) — conservation-free proxy for propensity;
   - physics: Coulomb potential with distance-dependent dielectric (ε=4r) from template partial charges; hydrophobic potential (Fauchère–Pliska atom-projected, Gaussian decay); H-bond capacity;
   - shape: distance to nearest atom, local point density, PCA anisotropy of atoms within 8 Å, depth = distance to convex-hull/“envelope” surface;
   - B-factor is **not** used (absent/unreliable; pLDDT in AF models).
4. **Point scorer**: LightGBM binary classifier, label = point within 4 Å of a relevant ligand heavy atom (labels pooled over all holo structures of the same UniProt, LIGYSIS-style, to reduce false negatives). Small model (≤ 300 trees, ≤ 63 leaves). Inference on CPU (LightGBM / treelite-style numba traversal) and GPU (tree tensors in torch, or XGBoost `device=cuda` if we switch to XGBoost).
5. **Pocket formation**: points with score ≥ τ → single-linkage clustering (d ≈ 3 Å) + a recall-oriented second pass: geometric pockets from buriedness-connected components (fpocket-like coverage; LIGYSIS shows max recall 57% for P2Rank vs 91% for fpocket). Non-max suppression: drop a pocket if centroid within 5 Å or residue Jaccard ≥ 0.75 to a higher-ranked one (LIGYSIS redundancy definition).
6. **Pocket ranker**: LightGBM on pocket-level features (Σ score, Σ score², max, top-10 mean, #points, volume estimate, buriedness quantiles, apolar fraction, polar/charged counts, residue composition, anisotropy, #chains lining the pocket, pLDDT stats in AF mode). Objective: binary "pocket hits a site (DCC ≤ 12 Å **and** DCA ≤ 4 Å)" with isotonic calibration on dev → cross-protein-comparable probability (for Precision1K). Ablation: lambdarank objective.
7. **Output**: ranked pockets: center, calibrated score, residues (atoms within 4 Å of pocket points), points; JSON/CSV.

### 2.2 Training data (leakage-controlled)
- Source: BioLiP2 (current weekly release) relevant ligands, holo chains from the local PDB mirror; resolution ≤ 3 Å X-ray/EM.
- Two label regimes: `small-molecule` (exclude metal ions / single atoms; default mode) and `all` (incl. ions; reported on full LIGYSIS).
- **Leakage control**: MMseqs2 clustering at 30% identity; remove every training cluster containing any chain ≥ 30% identical (cov ≥ 0.5) to any chain in any test set (LIGYSIS, COACH420, HOLO4K, CryptoBench test, PocketMiner test, PocketWatch test, dimer test). Foldseek: compute max TM to training for each test protein → novelty strata. Ablation model **fold-disjoint**: additionally remove training chains with Foldseek TM ≥ 0.5 to any test chain.
- One representative chain per 30% cluster (ligand labels pooled over the cluster's same-UniProt members), target ~5–10k training chains. Dev split: 10% of clusters (held out by cluster) — **all tuning on dev only**.
- Training inputs as in deployment: single chain AND (for a 30% subset) biological assembly, so the model sees interfaces.

### 2.3 Dimers
- Our method runs on any multi-chain input natively (features span chains; residues reported as chain:resnum).
- Benchmark built from PDB assembly-1 files of test-cluster proteins (disjoint from training at 30%): sites = relevant-ligand contacts (≤ 4 Å), class = single-chain / homo-interface / hetero-interface (≥ 3 contacting residues on each of ≥ 2 chains); symmetric copies collapsed (one site per unique set of entity-residue pairs). Inputs: isolated chain vs full assembly. Metric: top-N+2 (DCA 4 Å and DCC 12 Å) per class.

### 2.4 AF mode (`--af-mode`)
Mechanisms (each individually ablated):
1. **Spaghetti pruning**: remove terminal segments with pLDDT < 50 and internal stretches ≥ 7 residues with pLDDT < 50 (AF2BIND precedent); remaining low-pLDDT atoms get soft occlusion weight `w = clip((pLDDT − 40)/40, 0, 1)` in buriedness (transparent ghosts cannot create enclosure).
2. **Side-chain tolerance**: in AF mode side-chain atoms beyond Cβ have occlusion weight × 0.5 (and backbone+Cβ buriedness features are always present), so apo-like rotamers don't close clefts.
3. **Confidence features in the pocket ranker**: mean/min pLDDT of lining residues, fraction of points near atoms with pLDDT < 70; trained on AF models (see below).
4. **PAE (optional)**: if a PAE file is given, pockets whose lining residues span residue pairs with median PAE > 10 Å get a feature/penalty (inter-domain fake cleft). PAE JSONs downloaded from AFDB for test proteins only (not in local tars).
- **AF-aware training**: AF-mode ranker trained on AFDB models of *training* UniProts with BioLiP sites mapped by UniProt numbering (SIFTS) — learns how real sites look on AF models.
- Evaluation: O5; ablation table naive vs each mechanism.

### 2.5 Backends
- CPU: NumPy + Numba (`parallel=True`), per-structure latency and 64-process throughput.
- GPU: PyTorch tensor ops for grid/raycast/neighbour features, batched across structures (concatenate with batch index), tree-ensemble inference on GPU; multi-GPU via one process per device.
- Identical outputs (tolerance-checked) between backends — tested.

## 3. Evaluation protocol & integrity
- Metrics code frozen with toy tests before development (docs/EVALUATION.md). LIGYSIS evaluated with the authors' LBS-comparison code **and** our harness (must agree on published baseline predictions).
- Baselines re-run by us with pinned versions on identical inputs: P2Rank 2.5 default/alphafold, fpocket 4, fpocket+PRANK, plus DL methods that install (GrASP, DeepPocket, PUResNet, VN-EGNN) with training-overlap notes. **Fair-data control**: P2Rank retrained on our training set.
- Test sets evaluated only on frozen model versions; every test evaluation is logged in `results/TEST_LEDGER.md` (version, commit, date) — no silent re-runs.
- Seeds: fixed list `SEEDS = [0, 1, 2]` for model training; report mean ± sd across seeds for the final model. Bootstrap (1,000 resamples over proteins; seed 0) for CIs; paired bootstrap vs baselines.
- Controls: random-surface-point pockets (negative), ligand-centroid oracle (positive), shuffled-label model (negative; should collapse to geometric prior).
- Runtime: warm/cold, I/O included and excluded, hardware stated.

## 4. Experiment plan (ordered)
E0 harness + baselines on all test sets → E1 geometric-only scorer (buriedness ranking; no learning) → E2 point GBM + clustering (P2Rank-like with our features) → E3 + recall pass + NMS + pocket ranker → E4 feature ablations (physics, side-chain-tolerant, composition) → E5 AF mode & ablations → E6 dimers → E7 GPU backend & throughput → E8 fold-disjoint model & novelty strata → E9 final frozen evaluation (3 seeds) → REPORT.md.

## 5. Risks
- BioLiP-based training data still near-duplicates test sites at <30% identity (same fold, same cofactor) → novelty strata + fold-disjoint ablation.
- LIGYSIS 12 Å DCC is lenient: big pockets can game it → report DCA 4 Å, one-to-one matching, and pocket size distributions.
- Time: 24 h. DL baselines optional; P2Rank + fpocket mandatory.
