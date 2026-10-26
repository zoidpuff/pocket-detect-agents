# DESIGN — v2 (after review round 1)

Working name: **BURROW** (Buriedness-driven Ultra-fast Ranking Of pOckets, With confidence awareness). Package: `pocketdetect`.
Grounding: LITERATURE.md (§2.2 LIGYSIS, §4 AF, §7). Reviews: reviews/round1/*.md (6 reviewers); change log in §8.
Budget: 24 h from 2026-10-26 14:54 CET (T0). Hardware: 64 cores EPYC 9554, 4×RTX 4090, ≤128 GB RAM, ≤500 GB data.

## 1. Pre-registered claims (frozen before any test-set run; mirrored in docs/EVALUATION.md)

**Primary claim (C1).** On LIGYSIS (full, ions included; LBS-comparison protocol), BURROW-`all` top-N+2 recall at DCC ≤ 12 Å exceeds the best baseline B* = fpocket+PRANK, using bar = max(published 60.4, our re-run), by **≥ 2.0 points**, with the paired cluster-bootstrap 95% CI of the difference excluding 0, **for each of the 3 training seeds**. Paired against the authors' per-chain predictions on the identical chain set (denominators matched; failures/empty outputs = misses).
**C1-strict.** The same comparison under one stricter criterion — DCA ≤ 4 Å to any site ligand atom (top-N+2) — must have CI excluding 0 in our favour (no magnitude requirement).
**C2 (precision).** Precision1K (recomputed inside each bootstrap resample) ≥ best baseline − 3 points; also report one-to-one (Hungarian) Precision1K.
**Secondary (Holm-adjusted, fixed order; non-inferiority margin −2 points vs best of P2Rank/fpocket+PRANK re-run on identical inputs; a CI too wide to decide = "inconclusive", not "pass"):**
O2 COACH420(mlig)/HOLO4K(mlig) DCA 4 Å top-n, top-(n+2) (BURROW-`small`) → O3 PocketWatch temporal test (MIT; follow its scoring mask) → O4 CryptoBench test (apo, 222) → O5 AF mode on LIGYSIS UniProts → O6 dimer benchmark. Everything else (strata, PocketMiner-38, residue metrics) is descriptive.
**Speed (O7).** ms/structure on 1 core (warm, cold; parse included and excluded) and structures/s on 1 and 4 GPUs, vs P2Rank and fpocket measured on the same machine.

## 2. Method

### 2.1 Input policy
Polymer heavy atoms only (standard + common modified amino acids). Waters, ligands, metals, glycans, hydrogens removed — identically for BURROW and every baseline (wrapper-enforced). Unit test: predictions invariant to presence of HETATM ligand records. B-factor never used as a feature.

### 2.2 Candidates (single point set)
- One grid (1.0 Å) point set in a solvent shell, built with a **cascade**: (i) shell points ≥ r_vdW + 1.0 Å and ≤ 6 Å from atoms; (ii) cheap prefilter (LIGSITE-style axis sweeps / 14-ray buriedness); (iii) full multi-scale rays only on survivors. Point = volume element ⇒ pocket aggregates are volume-weighted (no SAS∪grid double counting).
- **Metal-site candidates**: additional points at 1.8–2.6 Å from ≥ 2 potential coordinating atoms (His ND1/NE2, Cys SG, Asp/Glu carboxylate O, backbone O, Met SD), which the solvent-shell rule would exclude.
- Ray occlusion uses inflated radii (no leaking through gaps); deterministic Fibonacci directions; rotation-invariance test (tolerance documented).
- **Gate G1 (dev, by T+7 h):** max recall (DCC ≤ 12 Å and DCA ≤ 4 Å) ≥ 88% with ≤ 40 pockets/protein before NMS ranking. If missed: fall back to fpocket candidates + our ranker. "fpocket candidates + BURROW ranker" is a mandatory ablation regardless.

### 2.3 Point features (pruned to evidence-backed, cheap)
Multi-scale ray buriedness (L = 6/10/16/24 Å; full-atom and backbone+Cβ-only = side-chain-tolerant); **first-hit ray chemistry** (fraction of rays whose first hit is apolar C / polar N,O / charged / aromatic — enclosure-by-chemistry, a desolvation proxy); atom-class densities at 4/8/12 Å (C, N, O, S, aromatic, donors, acceptors, ±charged; pruned for redundancy); residue-group composition (7 groups) within 8 Å; protrusion (atoms within 10 Å); metal-coordination counts/geometry (#coordinating atoms within 2.8 Å, angular spread); **weighted contact number** (flexibility proxy usable on PDB/apo/AF); distance to nearest atom; screened Coulomb potential (ε = 4r, template charges) — SHOULD, kept only if it helps on dev.

### 2.4 Scoring and pocket formation
1. **Point scorer**: one tree library (LightGBM; float32-safe thresholds; export to a numba/torch traversal for CPU/GPU inference). Label: point within 4 Å (ions: 3 Å) of a pooled site ligand atom. Small model (≤ 300 trees).
2. **Pocket formation**: points with score ≥ τ → single-linkage (d ≈ 2–3 Å) with chaining control (split clusters whose extent > 20 Å via k-means/density peaks; suppress nested pockets) + geometric recall pass (buriedness-connected components) → NMS (centroid < 5 Å or residue Jaccard ≥ 0.75 to a higher-ranked pocket — LIGYSIS redundancy definition).
3. **Pocket ranker** (LightGBM, binary, isotonic-calibrated on dev): pocket features = aggregates of **cross-fitted** point scores (5-fold by cluster; avoids train-score optimism), size (√n-saturated), volume, buriedness quantiles, first-hit chemistry, residue groups, shape descriptors (PCA anisotropy, mouth openness = fraction of unblocked rays), #chains lining, pLDDT stats (AF). Label: pocket hits a site (DCC ≤ 12 Å and DCA ≤ 4 Å).
4. **Two pre-registered rankers** on the same point features: `all` (labels incl. ions; used for LIGYSIS full, dimers, AF) and `small` (ions excluded; used for LIGYSIS_NI, COACH420/HOLO4K, CryptoBench, PocketWatch). Mapping fixed by each benchmark's ligand definition, not by results.
5. Output: center, calibrated score, residues (atoms ≤ 4 Å of pocket points), points.

### 2.5 Training/dev data (leakage-controlled, LIGYSIS-style)
- Source: BioLiP2 snapshot 2025-11-12 (local), structures from local PDB mirror; X-ray/EM ≤ 3.0 Å.
- **Leakage removal (domain-aware)**: MMseqs2 search of all candidate training chains vs all test chains (LIGYSIS, COACH420, HOLO4K, CryptoBench all splits used as test, PocketMiner, PocketWatch test, AF set, dimer set): remove any training chain with ≥ 30% identity over an alignment covering ≥ 50% of either chain **or** ≥ 80 aligned residues (local, domain-level). Report removed fraction. Independent agent re-checks (seq-id + PDB-ID + UniProt overlap).
- **LIGYSIS-style pooling**: per UniProt, pick representative chain (best resolution, most complete); pool relevant ligands from up to 20 other holo entries by sequence-anchored superposition onto the representative (keep if local pocket Cα RMSD ≤ 2 Å; record RMSD); sites = ligand clusters (LIGYSIS fingerprint clustering). 3D point labels from pooled ligand atoms.
- One representative per 30% cluster; inputs as deployed: single representative chain (+ for ~30% of entries, assembly-1 to expose interface sites).
- **Dev**: 10% of clusters, built identically (LIGYSIS-like); all tuning/gates on dev only. Training seeds: `pocketdetect.seeds.SEEDS[:3]`; bootstrap seed `BOOTSTRAP_SEED` (single source of truth: src/pocketdetect/seeds.py).
- Ablations: fold-disjoint model (also remove Foldseek TM ≥ 0.5 to any test chain); CHEN11-only-trained model (memorisation control); shuffled-label negative control.

### 2.6 AF mode (`--af-mode`)
Mechanisms (each ablated): (1) **spaghetti pruning** — drop terminal segments with pLDDT < 50 and internal stretches ≥ 7 residues < 50 (AF2BIND convention, a prior not a tuned value); (2) **soft occlusion** — remaining atoms get occlusion weight w = clip((pLDDT−40)/40, 0, 1); low-weight atoms never exclude candidate points; (3) **side-chain tolerance** — side-chain atoms beyond Cβ get occlusion × λ (λ chosen on dev); backbone-only buriedness features always present; (4) **confidence-aware ranker** trained on AFDB models of *training* UniProts, labels transferred by local superposition (record local RMSD); (5) structure-only inter-domain surrogate (pocket lining spans two contact-defined domains with weak inter-domain packing); **PAE only if available for both train and test** (COULD).
Validation controls (MUST): naive vs AF mode; **"naive + drop pockets with mean pLDDT < 70"** baseline that AF mode must beat; matched-random pruning (same #atoms removed); baselines run on our pruned input; fraction of true sites lost to pruning; hallucination rate (pockets with mean pLDDT < 50 matching no site) computed on unpruned input; no-op check (all-high-pLDDT/PDB input ⇒ identical to standard mode); same metric on PDB chains of the same proteins. AF test set: LIGYSIS UniProts on AFDB v6 human models; residue-type match check before site mapping; fragmented models (>2,700 aa) excluded and counted.

### 2.7 Dimers
Benchmark from PDB assembly-1 (only `author_and_software_defined_assembly` to avoid crystal contacts) of test-cluster proteins: site classes single-chain / homo-interface / hetero-interface (≥ 3 contacting residues on each of ≥ 2 chains). **Symmetric copies**: sites with identical entity-level residue sets collapse to one unique site with several acceptable copies; a prediction hits if it matches any copy; N = number of unique sites; predictions matching an already-matched copy are not counted as false positives. Inputs: isolated chain vs full assembly; all methods get identical inputs.

### 2.8 Backends
CPU: NumPy + Numba (parallel). GPU: PyTorch (+Triton kernels already written; device-guarded), batched over structures; one process per GPU. Output agreement CPU vs GPU tested.

## 3. Evaluation & integrity protocol
- Metrics frozen with toy tests (docs/EVALUATION.md); scoring/labeling code checksum recorded in every result file.
- **Reproduce first**: our harness on the authors' LIGYSIS predictions must reproduce the published table; our P2Rank 2.4.2/fpocket 4.0 re-runs must match the published rows (then also run P2Rank 2.5.x latest).
- DL baselines on LIGYSIS: authors' published per-chain predictions (paired). Our own runs of DL tools only where needed (CryptoBench, AF, dimers) and feasible.
- Statistics: cluster bootstrap (30% sequence clusters; UniProt as sensitivity), 1,000 resamples, failures = misses, P1K recomputed per resample. Pre-defined strata: fold novelty (max Foldseek TM to training: < 0.5 / 0.5–0.7 / > 0.7), ligand class (ion / nucleotide-cofactor / other small molecule), pocket pLDDT band (AF); strata < 100 sites merged.
- **Test gating**: `scripts/run_test_eval.py` refuses dirty trees, records commit + model checksum, appends `results/TEST_LEDGER.md` automatically; ≤ 3 model versions evaluated on test before the final frozen run; training code cannot read test labels (path guard). Headline numbers recomputed by a fresh agent; `scripts/check_report_numbers.py` checks every number in REPORT.md against committed results files.
- Controls on every evaluation: oracle (ligand centroids), random surface points, surface tiling (uniform pockets every ~12 Å), protein centroid.

## 4. Gates & schedule (T0 = 14:54)
- T+3 h: training/dev sets built, harness reproduces LIGYSIS table, baselines run.
- T+7 h (G1): candidate recall gate on dev.
- T+10 h: first full model on dev; test v1 run.
- T+12 h: **feature freeze**. Then AF mode, dimers, GPU throughput, ablations.
- T+20 h: final frozen model, 3 seeds, test runs; T+22 h: REPORT.md.

## 5. Novelty claims (each tied to an ablation)
N1 Volume-element candidates with cascaded multi-scale ray buriedness and first-hit chemistry, giving fpocket-level candidate recall with P2Rank-level ranking (ablation: fpocket candidates + our ranker; SAS candidates). N2 Confidence-aware geometry for AF models — soft occlusion + side-chain tolerance + AF-trained ranker (ablation per mechanism, vs pruning-only and pLDDT-filter baselines). N3 Native multi-chain/interface-aware ranking with a symmetric-copy-aware dimer benchmark. Plus speed: ≤ 70 ms/structure/core and ≥ 10³ structures/min/GPU.

## 6. Parallel tracks (start now)
A training/dev builder · B baselines on all test sets + LIGYSIS reproduction · C AF test set + AFDB extraction (one pass over the human tar) · D dimer benchmark · E geometry cascade (in progress) · F evaluation harness (in progress).

## 7. Risks
Leakage at <30% via shared folds/cofactors (fold strata, fold-disjoint ablation); 12 Å leniency (C1-strict, size distributions, controls); AF memorisation of holo pockets (stratify by template richness; report PDB-chain metric on the same proteins); time (gates + fallback).

## 8. Changes from v1 (by reviewer)
- External PI: ions in headline model + metal candidates; quantified "clearly beat"; gates/fallback; parallel tracks; AF must beat pLDDT-filter baseline; novelty claims; cut lambdarank; DL baselines via published predictions.
- Statistics: pre-registered single primary claim; non-inferiority margins + Holm; cluster bootstrap; failures = misses; per-resample P1K; test ledger cap; strata fixed; seeds from seeds.py.
- Algorithms/compute: cascade candidates; single volume-element point set; inflated ray radii + rotation test; cross-fitted point scores; chaining control; one tree library; Triton device bug forwarded.
- Data/prior work: BioLiP2 snapshot dated; PocketWatch MIT, follow its mask; domain-aware 30% exclusion; 3D pooled labels by superposition; LIGYSIS reproduction with P2Rank 2.4.2/fpocket 4.0; AF residue checks + fragments; dimer crystal-contact filter; allosteric gap → descriptive allosteric stratum if an annotated set (e.g. ASBench/ASD) is obtainable (COULD).
- Biophysics/AF: polymer-only input + invariance test; metal sites; WCN flexibility; soft-occlusion candidate rule; hallucination on unpruned input; AF controls; PAE train+test-or-surrogate; superposition-based AF labels; symmetric-copy rule.
- Agent failure modes: gated test script; ledger; frozen scoring code checksum; reproduce-first; no silent drops; controls each run; provenance tags (re-run vs published) on every number; fresh-agent recomputation; file ownership (each agent owns listed paths; PI commits explicit paths only).
- Not adopted: carving a LIGYSIS dev slice (we use a BioLiP2 LIGYSIS-like dev instead, keeping all 2,775 chains for comparability with the published table).
