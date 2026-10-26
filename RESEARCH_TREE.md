# Research Tree — ultra-fast pocket detection & ranking

Legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding

## Status summary (latest first)

### 2026-10-26 15:00 — M0 kickoff
Project started 2026-10-26 14:54 CET; review due ~2026-10-27 14:30 (24 h budget). Hardware: 64-core EPYC 9554, 4×RTX 4090 (24 GB), budget 128 GB RAM (node shows 375 GB; we cap at 128), 1.7 TB local /tmp NVMe, project storage 9 TB free (we cap our data at 500 GB). Shared read-only: AFDB v6 (48 proteome tars + SwissProt tar, 157 GB) and a full PDB rsync mirror (mmCIF divided, assemblies, biounit, derived_data). No Java, no nvcc, no conda; uv available, internet via proxy works.

Plan: parallel kick-off of (a) literature review → LITERATURE.md, (b) benchmark dataset acquisition → DATA.md, (c) baseline tool installation (P2Rank, fpocket, others), (d) core I/O + evaluation harness with toy-data verification. Then DESIGN.md → up to 3 review rounds → implementation and experiments.

## Decisions log
- D0 (14:55): Single uv venv at `.venv` shared by all agents; restricted/large data kept out of git under `data/raw`, `data/processed`; manifests + checksums committed. Git identity set locally to gmagnusson <gmagnusson@ethz.ch> (no preset identity found in container).

## Tree
- [~] 0. Infrastructure
  - [x] 0.1 Hardware/data inspection
  - [~] 0.2 Python env (.venv: numpy/scipy/numba/gemmi/torch-cu124/lightgbm)
  - [ ] 0.3 Eval harness (DCA/DCC/top-n, top-n+2, LIGYSIS-style metrics) + toy tests
- [~] 1. Literature & benchmarks (LITERATURE.md)
- [~] 2. Data acquisition (DATA.md)
- [~] 3. Baselines (P2Rank, fpocket, …)
- [ ] 4. DESIGN.md + review rounds
- [ ] 5. Method development
- [ ] 6. AF mode
- [ ] 7. CPU/GPU backends + throughput
- [ ] 8. Final benchmark + REPORT.md
