# Baseline speed on a fixed 200-chain LIGYSIS subset

Subset: `baselines/ligysis_speed200.txt` (200 random LIGYSIS chains, seed 20261026; authors' ligand-free trans.pdb chains, median ~300 residues).
Script: `baselines/outputs/speed.sh` (raw log `baselines/outputs/speed200/speed.log`, run 26 Oct 2026 18:29-19:13 CET).
Hardware: AMD EPYC 9554 (64 CPUs), shared node. **Node load average during the runs was 20-68 (other users' jobs; our own jobs were the only ones of this project running)**, so
single-thread numbers are inflated by an unknown amount (hyper-thread / memory contention); multi-thread numbers more so.
Wall seconds include our gemmi cleaning (~0.02 s/structure, parallelisable) and, for P2Rank, JVM start + model load (~2-4 s, once per JVM).

| tool | mode | procs/threads | wall for 200 (s) | ms / structure (amortised) |
|---|---|---|---|---|
| P2Rank 2.5.1 | batch, **one JVM** (intended high-throughput use) | 1 | 59.0 | 295 |
| P2Rank 2.5.1 | batch, one JVM | 8 | 9.8 | 49 |
| P2Rank 2.5.1 | batch, one JVM | 32 | 9.5 | 47 (JVM start/IO bound) |
| P2Rank 2.5.1 | one JVM per structure, sequential | 1 | 654.7 | 3273 (JVM start dominates; *not* the fair number) |
| P2Rank 2.4.2 | batch, one JVM | 1 | 117.6 | 588 |
| P2Rank 2.4.2 | batch, one JVM | 8 / 32 | 24.6 / 18.8 | 123 / 94 |
| P2Rank 2.4.2 | one JVM per structure | 1 | 1344.5 | 6723 |
| fpocket 4.2.3 | single thread, sequential | 1 | 124.0 | 620 |
| fpocket 4.2.3 | 32 parallel processes | 32 | 7.7 | 39 |
| fpocket 4.0.3 (-O0 build) | 1 / 32 | 1 / 32 | 128.1 / 5.6 | 641 / 28 |
| fpocket 4.2.3 + PRANK 2.5.1 rescore | fpocket sequential + one rescore JVM (1 thread) | 1 | 154.7 | 774 |

Take-aways: P2Rank in batch mode is ~2x *faster* than single-thread fpocket per structure (0.3 s vs 0.6 s), and with 8 threads ~0.05 s/structure; the commonly seen 3-7 s/structure for P2Rank is JVM start-up of per-file calls.
Our method's timing comparison should therefore use P2Rank batch mode (one JVM) at equal thread counts, and fpocket with one process per core.
