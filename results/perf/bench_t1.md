## Track T1: GPU path and per-structure latency, before / after (same session, 2026-10-27 12:01 -> 2026-10-27 12:12)

Before = commit d57fdf6; after = c7da4db + uncommitted fast-backend changes. Same node (eu-g6-048), same 200 LIGYSIS speed chains, `scripts/bench_final.py` run back to back for both trees (CPU workers <= 24; GPUs 0,1,3, 8 CPU workers per GPU). Outputs are byte-identical between the two trees (see the equivalence section). Throughput rows include process start-up, parsing and writing.

| preset | measurement | before | after | after / before |
|---|---|---|---|---|
| fast | 1 core warm, excl. parsing, median / p90 ms | 31.4 / 75.1 | 28.4 / 69.1 | 1.11x |
| fast | 1 core warm, incl. parsing, median / p90 ms | 41.6 / 100.2 | 31.5 / 77.7 | 1.32x |
| fast | parse (median ms) | 10.5 | 3.1 | 3.39x |
| fast | CLI, 1 worker, 200 chains: wall s (ms / structure) | 12.6 (63) | 9.7 (48) | 1.30x |
| fast | CLI, 24 CPU workers, 2,000 structures: structures/s | 214.2 | 371.3 | 1.73x |
| fast | CLI, 1 GPU (8 workers), 2,000 structures: structures/s | 129.1 | 202.9 | 1.57x |
| fast | CLI, 3 GPUs (8 workers each), 2,000 structures: structures/s | 177.3 | 336.8 | 1.90x |
| fast | CLI, 24 CPU workers, 10,000 structures: structures/s | 406.7 | 544.7 | 1.34x |
| fast | CLI, 3 GPUs (8 workers each), 10,000 structures: structures/s | 436.4 | 646.3 | 1.48x |
| default | 1 core warm, excl. parsing, median / p90 ms | 99.3 / 271.3 | 96.0 / 264.9 | 1.03x |
| default | 1 core warm, incl. parsing, median / p90 ms | 110.6 / 296.8 | 99.2 / 273.0 | 1.11x |
| default | parse (median ms) | 11.3 | 3.3 | 3.42x |
| default | CLI, 1 worker, 200 chains: wall s (ms / structure) | 31.3 (156) | 28.4 (142) | 1.10x |
| default | CLI, 24 CPU workers, 2,000 structures: structures/s | 123.2 | 149.6 | 1.21x |
| default | CLI, 1 GPU (8 workers), 2,000 structures: structures/s | 96.6 | 129.4 | 1.34x |
| default | CLI, 3 GPUs (8 workers each), 2,000 structures: structures/s | 162.4 | 236.6 | 1.46x |
| default | CLI, 24 CPU workers, 10,000 structures: structures/s | 155.3 | 171.0 | 1.10x |
| default | CLI, 3 GPUs (8 workers each), 10,000 structures: structures/s | 326.9 | 411.4 | 1.26x |
| accurate | 1 core warm, excl. parsing, median / p90 ms | 205.7 / 500.4 | 203.0 / 493.4 | 1.01x |
| accurate | 1 core warm, incl. parsing, median / p90 ms | 216.8 / 528.2 | 206.4 / 501.5 | 1.05x |
| accurate | parse (median ms) | 11.3 | 3.2 | 3.53x |
| accurate | CLI, 1 worker, 200 chains: wall s (ms / structure) | 57.6 (288) | 54.2 (271) | 1.06x |
| accurate | CLI, 24 CPU workers, 2,000 structures: structures/s | 72.6 | 79.5 | 1.10x |
| accurate | CLI, 1 GPU (8 workers), 2,000 structures: structures/s | 77.5 | 107.7 | 1.39x |
| accurate | CLI, 3 GPUs (8 workers each), 2,000 structures: structures/s | 141.0 | 195.9 | 1.39x |
| accurate | CLI, 24 CPU workers, 10,000 structures: structures/s | 83.2 | 88.2 | 1.06x |
| accurate | CLI, 3 GPUs (8 workers each), 10,000 structures: structures/s | 269.2 | 338.1 | 1.26x |

Node load (1/5/15 min) before run: 25.23 26.21 27.75 -> 38.11 35.78 32.07; after run: 38.11 35.78 32.07 -> 37.67 33.53 30.87. *after / before*: speed-up factor (time ratio for latency rows, throughput ratio for throughput rows).

