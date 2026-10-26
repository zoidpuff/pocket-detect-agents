# Research Tree — ultra-fast pocket detection & ranking

Legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding

## Status summary (latest first)

### 2026-10-26 15:00 — M0 kickoff
Project started 2026-10-26 14:54 CET; review due ~2026-10-27 14:30 (24 h budget). Hardware: 64-core EPYC 9554, 4×RTX 4090 (24 GB), budget 128 GB RAM (node shows 375 GB; we cap at 128), 1.7 TB local /tmp NVMe, project storage 9 TB free (we cap our data at 500 GB). Shared read-only: AFDB v6 (48 proteome tars + SwissProt tar, 157 GB) and a full PDB rsync mirror (mmCIF divided, assemblies, biounit, derived_data). No Java, no nvcc, no conda; uv available, internet via proxy works.

Plan: parallel kick-off of (a) literature review → LITERATURE.md, (b) benchmark dataset acquisition → DATA.md, (c) baseline tool installation (P2Rank, fpocket, others), (d) core I/O + evaluation harness with toy-data verification. Then DESIGN.md → up to 3 review rounds → implementation and experiments.

### 2026-10-26 15:50 — M1 design v2
Literature done; six round-1 reviews consolidated into DESIGN v2 (pre-registered primary claim C1: LIGYSIS full top-N+2 DCC≤12 Å ≥ fpocket+PRANK bar + 2 pts with cluster-bootstrap CI>0 for all 3 seeds; C1-strict at DCA 4 Å; P1K within 3 pts of best). Key decisions: ions included in headline (`all` ranker) with a pre-registered `small` ranker for ion-free benchmarks; LIGYSIS-style pooled training labels from BioLiP2 with domain-aware 30% leakage removal; gated test runs with ledger (≤3 versions before final); candidate-recall gate G1 at T+7 h with fpocket-candidate fallback.

Running: round-2 review (2 reviewers), data acquisition, baselines, eval harness, geometry cascade, Track A (train/dev), Track C (AF set), Track D (dimer set). Next: reproduce the LIGYSIS published table with our harness, then first point/pocket models on dev.

## Decisions log
- D3 (15:55): Start method pipeline before train set exists (dev on P2Rank's CHEN11/JOINED training sets — not test sets) to de-risk G1. Note: timestamps in D1/D2/PREREG header were estimated ~20 min late; git commit times are authoritative.
- D2 (16:10): DESIGN v3 final; PREREG frozen: fixed-sequence C1→C1-strict→C2, Holm over O2a/O2b/O3–O6, no test run before feature freeze (T+10 h ≈ 01:00), test-union frozen before training build, label-free symmetric collapse in dimer eval for all methods.
- D1 (15:45): DESIGN v2 — see DESIGN §8 for reviewer→change map. Not adopted: LIGYSIS dev slice (use BioLiP2 LIGYSIS-like dev instead to keep all 2,775 chains comparable to published table).
- D0 (14:55): Single uv venv at `.venv` shared by all agents; restricted/large data kept out of git under `data/raw`, `data/processed`; manifests + checksums committed. Git identity set locally to gmagnusson <gmagnusson@ethz.ch> (no preset identity found in container).

## Tree
- [~] 0. Infrastructure
  - [x] 0.1 Hardware/data inspection
  - [x] 0.2 Python env (.venv: numpy 2.5/scipy/numba 0.68/gemmi 0.7.5/torch 2.6-cu124 4 GPUs OK/lightgbm/xgboost)
  - [x] 0.3 Eval harness — [!] reproduces P2Rank 2.5.1 evaluator on COACH420/HOLO4K(+mlig) to 0.1% (rules `p2rank_strict` = O2 headline)
  - [~] 0.4 LIGYSIS protocol + reproduction gate, cluster bootstrap, gated test runner, CryptoBench/PocketWatch GT (DCA/DCC/top-n, top-n+2, LIGYSIS-style metrics) + toy tests
- [x] 1. Literature & benchmarks (LITERATURE.md) — [!] LIGYSIS top-N+2 bar: fpocket+PRANK 60.4, P2Rank 51.9, P2Rank-CONS 53.9 (P1K 90.7); headroom = candidate recall (P2Rank max recall 57% vs fpocket 91%) + calibrated ranking; no dimer benchmark exists → build one
- [x] 2. Data acquisition (DATA.md) — 48 GB raw: P2Rank sets, LIGYSIS (+11 methods' preds), PocketWatch (MIT, 1,573 test sites ≥2024), CryptoBench, PocketMiner, scPDB (no license: academic only), AF2BIND, HOTPocket (catalogue, not benchmark), BioLiP2 2025-11-12; PDB mirror snapshot ~2026-10-16; mmseqs/foldseek installed
  - [~] 2.A Train/dev builder (BioLiP2, LIGYSIS-style pooled labels, leakage removal)
  - [~] 2.C AF test set (LIGYSIS UniProts on AFDB v6)
  - [x] 2.D Dimer benchmark: 400 complexes (274 homo/126 hetero), 807 unique sites, 243 interface (177 homo, 66 hetero); oracle ceiling 1.0 (J≥0.5)
- [~] 3. Baselines (P2Rank, fpocket, …)
- [~] 4. DESIGN.md + review rounds
  - [x] 4.1 DESIGN v1 draft (BURROW: grid-buriedness+SAS candidates → analytical point feats → GBM point scorer → clustering + geometric recall pass + NMS → calibrated GBM pocket ranker; AF mode = pruning + soft occlusion + side-chain tolerance + AF-trained ranker)
  - [x] 4.2 Review round 1: external PI, data/prior work, statistics, algorithms/compute, agent failure modes, biophysics/AF → DESIGN v2
  - [x] 4.3 Review round 2 (external PI+stats; data+agent risks) → DESIGN v3 + docs/PREREG.md frozen (sha256 582b8b56349cb2ab762d0f59a495738a938ee1d592c01e2517dbf581e049f8d6). Reviews capped at 2 rounds (time; round-2 changes were refinements, no structural objections).
- [~] 5. Method development
  - [~] 5.1 BURROW pipeline v0 (CPU end-to-end), G1 candidate-recall gate, geometry-only ranking, first LightGBM point+pocket model (dev: CHEN11/JOINED until Track A dev arrives)
- [ ] 6. AF mode
- [~] 7. CPU/GPU backends + throughput
  - [~] 7.1 Geometry primitives (SAS pts, ray-cast buriedness w/ soft occlusion, neighbor feats, clustering) CPU+GPU — started early as design-independent
- [ ] 8. Final benchmark + REPORT.md
