# Research Tree — ultra-fast pocket detection & ranking

Legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding

## Status summary (latest first)

### 2026-10-26 15:00 — M0 kickoff
Project started 2026-10-26 14:54 CET; review due ~2026-10-27 14:30 (24 h budget). Hardware: 64-core EPYC 9554, 4×RTX 4090 (24 GB), budget 128 GB RAM (node shows 375 GB; we cap at 128), 1.7 TB local /tmp NVMe, project storage 9 TB free (we cap our data at 500 GB). Shared read-only: AFDB v6 (48 proteome tars + SwissProt tar, 157 GB) and a full PDB rsync mirror (mmCIF divided, assemblies, biounit, derived_data). No Java, no nvcc, no conda; uv available, internet via proxy works.

Plan: parallel kick-off of (a) literature review → LITERATURE.md, (b) benchmark dataset acquisition → DATA.md, (c) baseline tool installation (P2Rank, fpocket, others), (d) core I/O + evaluation harness with toy-data verification. Then DESIGN.md → up to 3 review rounds → implementation and experiments.

### 2026-10-26 20:15 — M2 first dev results (T+5.3 h)
BURROW (volume-element grid candidates with cascaded multi-scale ray buriedness and first-hit chemistry → LightGBM point scorer → multi-scale pockets → cross-fitted LightGBM pocket ranker → 10 Å NMS) trained on the leakage-audited BioLiP2 LIGYSIS-style set beats P2Rank 2.5.1 on the BioLiP2 dev set by +10 pts top-N+2 (DCC12), +20 pts DCA4 and +24 pts P1K (`all` ranker); the ion-free `small` ranker is ahead but borderline on one seed. On AF dev models it scores 65.2 vs P2Rank-alphafold 56.2. All infrastructure is in place and validated: harness reproduces P2Rank's evaluator (COACH420/HOLO4K) and the published LIGYSIS table (33/33 variants), test sets (LIGYSIS, COACH420/HOLO4K, CryptoBench, PocketWatch, AF, dimer) are built, baselines wrapped; fast backend is bit-identical to the reference and does ~3.3k structures/min/GPU.

Open: (i) G1 candidate-recall gate failed on BioLiP dev as specified (58.6% at ≤40 geometric candidates; fpocket 47.3%) — fallback not taken; (ii) CPU latency ~80 ms/core (target ≤100) pending port of the frozen features; (iii) AF-mode geometry does not beat a pLDDT<70 filter on recall — its value is hallucination removal + precision; (iv) independent review of dev numbers running; (v) baseline test-set runs in progress. Next: phase-3 ranker improvements → feature freeze 23:00 → AF ranker retrain + fast-backend port → gated test run v1 ≈01:00 → ablations (fold-disjoint, CHEN11-only, shuffled labels) → final 3-seed runs → REPORT.md.

### 2026-10-26 15:50 — M1 design v2
Literature done; six round-1 reviews consolidated into DESIGN v2 (pre-registered primary claim C1: LIGYSIS full top-N+2 DCC≤12 Å ≥ fpocket+PRANK bar + 2 pts with cluster-bootstrap CI>0 for all 3 seeds; C1-strict at DCA 4 Å; P1K within 3 pts of best). Key decisions: ions included in headline (`all` ranker) with a pre-registered `small` ranker for ion-free benchmarks; LIGYSIS-style pooled training labels from BioLiP2 with domain-aware 30% leakage removal; gated test runs with ledger (≤3 versions before final); candidate-recall gate G1 at T+7 h with fpocket-candidate fallback.

Running: round-2 review (2 reviewers), data acquisition, baselines, eval harness, geometry cascade, Track A (train/dev), Track C (AF set), Track D (dimer set). Next: reproduce the LIGYSIS published table with our harness, then first point/pocket models on dev.

## Decisions log
- D7 (20:15): Accept G1 deviation (fallback worse than primary); NMS by one-to-one P1K; phase-3 selection on train-CV to limit dev optimism; freeze 23:00.
- D6 (18:40): Leakage audit fix is conservative — drop every entry flagged under any tier (186 entries, ~2.6%) rather than only the 8 hard failures; manifests filtered by scripts/data/apply_audit_exclusions.py.
- D5 (17:15): Speed budget — default preset ≤ ~100 ms/core (≥5× faster than P2Rank) within 0.5 pts of best dev setting; optional `--preset accurate` if >1 pt better; pre-registered claims use the default. Feature design freeze ~23:00 (backend port before 01:00 freeze). Fast backend post-feature path already bit-identical (79 vs 252 ms).
- D4 (16:20): Session restart interrupted 5 agents; all resumed via SendMessage from on-disk state (no work lost). Node is SHARED with other users (foreign processes, GPU 2 partly used) → prefer GPUs 0,1,3; respect 128 GB RAM.
- D3 (15:55): Start method pipeline before train set exists (dev on P2Rank's CHEN11/JOINED training sets — not test sets) to de-risk G1. Note: timestamps in D1/D2/PREREG header were estimated ~20 min late; git commit times are authoritative.
- D2 (16:10): DESIGN v3 final; PREREG frozen: fixed-sequence C1→C1-strict→C2, Holm over O2a/O2b/O3–O6, no test run before feature freeze (T+10 h ≈ 01:00), test-union frozen before training build, label-free symmetric collapse in dimer eval for all methods.
- D1 (15:45): DESIGN v2 — see DESIGN §8 for reviewer→change map. Not adopted: LIGYSIS dev slice (use BioLiP2 LIGYSIS-like dev instead to keep all 2,775 chains comparable to published table).
- D0 (14:55): Single uv venv at `.venv` shared by all agents; restricted/large data kept out of git under `data/raw`, `data/processed`; manifests + checksums committed. Git identity set locally to gmagnusson <gmagnusson@ethz.ch> (no preset identity found in container).

## Tree
- [~] 0. Infrastructure
  - [x] 0.1 Hardware/data inspection
  - [x] 0.2 Python env (.venv: numpy 2.5/scipy/numba 0.68/gemmi 0.7.5/torch 2.6-cu124 4 GPUs OK/lightgbm/xgboost)
  - [x] 0.3 Eval harness — [!] reproduces P2Rank 2.5.1 evaluator on COACH420/HOLO4K(+mlig) to 0.1% (rules `p2rank_strict` = O2 headline)
  - [!] 0.4a LIGYSIS reproduction gate PASSED: our harness on authors' per-chain predictions reproduces published top-N/top-N+2/max recall for all 28 method variants within ≤0.05 pts (DeepPocket-Seg 0.38); 6,721 NR scored sites, 2,775 chains (results/validation/ligysis_reproduction.md)
  - [x] 0.4 LIGYSIS protocol (33/33 variants reproduce; authors' code agrees), cluster bootstrap (30% MMseqs2 over test union), gated run_test_eval.py + ledger, CryptoBench (222/270 sites) + PocketWatch (1,566/1,573) GT
  - [x] 0.5 LIGYSIS ligand coords (6,387/6,721 sites; PREREG A2), AF (2,526 models) + dimer in test runner, AF utils
- [x] 1. Literature & benchmarks (LITERATURE.md) — [!] LIGYSIS top-N+2 bar: fpocket+PRANK 60.4, P2Rank 51.9, P2Rank-CONS 53.9 (P1K 90.7); headroom = candidate recall (P2Rank max recall 57% vs fpocket 91%) + calibrated ranking; no dimer benchmark exists → build one
- [x] 2. Data acquisition (DATA.md) — 48 GB raw: P2Rank sets, LIGYSIS (+11 methods' preds), PocketWatch (MIT, 1,573 test sites ≥2024), CryptoBench, PocketMiner, scPDB (no license: academic only), AF2BIND, HOTPocket (catalogue, not benchmark), BioLiP2 2025-11-12; PDB mirror snapshot ~2026-10-16; mmseqs/foldseek installed
  - [x] 2.A Train/dev: 6,644 train / 596 dev entries (5,303/513 clusters; 14,073/1,261 sites; ~37% ion-only sites); 77.4% of 268k BioLiP2 chains removed by leakage rule; test-union sha 27c3d793; 60% of train fold-disjoint (TM<0.5); verify PASS
  - [x] 2.A' Independent leakage audit: marginal FAIL (7 assembly-partner inputs missed by builder; 85 borderline 0.30–0.35 id hits via observed-vs-canonical seq; 1 UniProt overlap) → D6: drop all flagged entries (train 6,475, dev 579)
  - [x] 2.C AF test set: 2,527 proteins/6,731 sites (LBS subset, primary O5); holo-biased (median local RMSD 0.38 Å), only 95 sites pLDDT<70; PAE unavailable (403) → dropped
  - [x] 2.D Dimer benchmark: 400 complexes (274 homo/126 hetero), 807 unique sites, 243 interface (177 homo, 66 hetero); oracle ceiling 1.0 (J≥0.5)
- [x] 3. Baselines wrapped: P2Rank 2.5.1/2.4.2 (Temurin 17), fpocket 4.2.3/4.0.3 (-O0), fpocket+PRANK; optional: pyKVFinder, DeepPocket, PUResNet v2 (torch port), GrASP, VN-EGNN, IF-SitePred, Kalasanty, PocketMiner
  - [~] 3.1 Run mandatory baselines on all test sets; LIGYSIS re-run reproduction gate (±2 pts); P2Rank batch-mode timing
- [~] 4. DESIGN.md + review rounds
  - [x] 4.1 DESIGN v1 draft (BURROW: grid-buriedness+SAS candidates → analytical point feats → GBM point scorer → clustering + geometric recall pass + NMS → calibrated GBM pocket ranker; AF mode = pruning + soft occlusion + side-chain tolerance + AF-trained ranker)
  - [x] 4.2 Review round 1: external PI, data/prior work, statistics, algorithms/compute, agent failure modes, biophysics/AF → DESIGN v2
  - [x] 4.3 Review round 2 (external PI+stats; data+agent risks) → DESIGN v3 + docs/PREREG.md frozen (sha256 582b8b56349cb2ab762d0f59a495738a938ee1d592c01e2517dbf581e049f8d6). Reviews capped at 2 rounds (time; round-2 changes were refinements, no structural objections).
- [~] 5. Method development
  - [x] 5.1 BURROW v0 (CPU end-to-end): 1 Å shell grid + prefilter + metal candidates → 86 feats (60 rays, first-hit chemistry) → LGBM point scorer → multi-scale pockets → LGBM ranker (cross-fitted) → NMS
    - [!] G1 PASSED: candidate recall 90.97% (≤40 pockets) vs fpocket 77.5%, P2Rank-all 83.4% (CHEN11+JOINED, 974 ligands)
    - [x] Trained on CHEN11 only → JOINED: DCA4 top-n 71.1–71.8 vs P2Rank 73.5; top-n+2 82.4–83.2 vs 79.7; LDCC12 top-n+2 89–91 vs 88.0 (unpaired). Geometry-only: DCA4 top-n 35.4.
    - [-] Speed: 586 ms/structure (target 70) — Python loops; perf pass after feature stabilisation
  - [x] 5.2 Phase 2 (BioLiP2 audited train 6,475 / dev 579; 3 seeds; paired cluster bootstrap vs P2Rank 2.5.1)
    - [!] `all` default: top-N+2 DCC12 65.0 vs 54.1 (Δ +10.2…+11.4, CI lo ≥ +5.8); DCA4 top-N+2 55.7 vs 36.0; P1K 70.2 vs 46.3; top-N 55.7 vs 47.7
    - [~] `small` (non-ion): 64.2 vs 58.4 (one seed CI lo < 0); DCA4 +12; P1K +21
    - [x] Presets: default (30 rays, L 6/10/16, 3 Å sub-lattice, score buriedness ≥0.7 points, 250×63) ≈80 ms/core; accurate (60 rays) +1.2 top-N+2 on small → shipped as `--preset accurate`
    - [x] NMS 10 Å chosen by one-to-one P1K (raw P1K rewards duplicates; 14 Å would inflate top-N+2 to 72.6 by spreading — rejected)
    - [-] fpocket candidates + BURROW ranker: top-N+2 64.8 but DCA4 39.7 vs 56.2, P1K 59.4 vs 72.1 → own candidates matter for precision/localisation
    - [-] G1 on BioLiP dev: 58.6% at ≤40 geometric candidates (fpocket 47.3%) → gate failed as specified but fallback would be worse; kept BURROW candidates (deviation, G1.md addendum). Full pre-ranking pool 93.6%.
  - [x] 5.3 Independent review: CONFIRMED (own scorer + harness; CLI 30/30 bit-identical; gains hold at DCC≤4 Å +16.4 and one-to-one +24.8; BURROW hits median DCC 3.3 Å; surface tiling 15–38%; largest gain on fold-novel chains TM<0.5 +19.4). [!] Caveat: ion-only sites (34%) carry ~2/3 of `all` gain; non-ion ≈ +5 (CI touches 0 on 2/3 seeds). Bug: 8% top pockets empty residue list.
  - [~] 5.4 Phase 3: `small` ranker + within-protein ranking improvements (selection by train cluster-CV; dev once); FREEZE 23:00
- [~] 6. AF mode
  - [x] 6.1 Track C2: AF train/dev labels (2,844/278 entries with sites; 45% coverage of train in local AFDB; median local RMSD 0.23 Å; only 57 true sites pLDDT<70); AF stress set (300 test proteins ≥37.5% residues pLDDT<50; descriptive)
  - [~] 6.2 AF-mode developer (interim, AF dev 278 entries/607 sites + 10× train-OOF set):
    - [!] BURROW naive on AF dev DCC12 top-N+2 65.2 vs P2Rank -c alphafold 56.2 (DCA4 56.0 vs 36.9; P1K 51.5 vs 31.3)
    - [-] Geometric mechanisms (prune, soft occlusion) do not raise recall (train-OOF +0.0 [−0.8,+0.8]); do not beat the pLDDT<70 filter on recall (−0.5 [−1.3,+0.2])
    - [x] They remove hallucinated pockets (6.6% → 0.1%) without discarding 50–70 pLDDT pockets; random pruning costs −2.5 to −3.1 (control OK); no-op byte-identical; 1/607 sites lost to pruning
    - [x] AF ranker (pLDDT features, AF+PDB rows): P1K +6.9 [+4.5,+9.9], DCA4 +0.7 [+0.1,+1.3]; PDB-dev non-inferior on DCC12
    - [-] Side-chain tolerance λ on all side chains hurts badly (λ0.25: −16 pts); domain surrogate no help
    - [~] Stratify by proximity to low-pLDDT regions + disorder bins
- [~] 7. CPU/GPU backends + throughput
  - [x] 7.1 Geometry primitives CPU+GPU: cascade candidates 39–48 ms/core (300-res), canonical PCA frame → rotation-invariant (|Δb| 0.025→1.5e-6), Triton device fix, tiling (3j3q 2.4M atoms: 33 s CPU16 / 3.4 s GPU); GPU ~530 structs/s (excl. parsing)
  - [x] 7.2 BURROW fast backend: exact Numba/Triton ports, 60/60 dev structures identical CPU & GPU; [!] GPU 3,324 structs/min (1 GPU, incl. parsing), 6,542/min (3 GPUs), CPU-32 workers 4,845/min; CPU 1-core 166 ms (features 105 ms dominate) → needs method-side cuts (D5)
- [ ] 8. Final benchmark + REPORT.md
