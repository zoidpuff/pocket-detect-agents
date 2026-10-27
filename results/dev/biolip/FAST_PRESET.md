# Additional `--preset fast` (not pre-registered; default/accurate models unchanged)

Target (PI, 2026-10-27 ~08:30): ≤ 35 ms/structure/core median warm, excluding parsing, on the speed200
LIGYSIS chains (`baselines/ligysis_speed200.txt`, the set of docs/PERF.md), fast backend; ≤ 2.0 pt loss in
BioLiP-dev top-N+2 (closest rule) and DCA4 top-N+2 vs v2 default, for `all` and `small`; v2 output rule kept.
Constraint chosen here: **no reference-code change** (only `BurrowConfig` / `FormationConfig` values and model
size), so the fast backend's validated stages stay valid (guard hashes unchanged).

## 1. Timing screen (fast featurizer only, 1 core, warm, speed200; `scripts/burrow_fast_preset_timing.py`)
| config | grid Å | rays | sub-lattice | shell Å | PSP Å | metal | features ms (median / p90) | points (median) |
|---|---|---|---|---|---|---|---|---|
| C0 (= default) | 1.0 | 30 | 3 | 6 | 8 | on | 70.0 / 176.8 | 4,372 |
| C1 | 1.0 | 14 | 4 | 6 | 8 | on | 56.9 / 146.1 | 3,678 |
| C2 | 1.0 | 14 | 4 | 5 | 6 | off | 36.5 / 91.7 | 2,558 |
| C3 | 1.25 | 14 | 3 | 5.5 | 6 | on | 30.4 / 74.6 | 1,884 |
| C4 | 1.5 | 14 | 2 | 6 | 6 | on | 26.0 / 59.7 | 2,112 |
| C5 | 1.25 | 12 | 3 | 5.5 | 6 | off | 24.6 / 60.9 | 1,846 |
| C6 | 1.5 | 12 | 2 | 5.5 | 6 | off | 18.4 / 44.3 | 1,854 |
Ray lengths stay 6/10/16 Å and neighbour radii 4/8/10/12 Å (both are referenced by name in the frozen
reference code). Pocket formation scaled with the grid: link = 2.05 × spacing, min volume = 10 / spacing³ points.

## 2. Trained candidates (seed 0; point model 100 trees × 31 leaves, points scored if b ≥ 0.85)
Train cluster-CV at the v2 output rule (`scripts/burrow_tau_cv.py --grid v2`; out-of-fold point scores and
rankers), Δ vs v2 default (small: top-N 60.1 / top-N+2 71.9 / DCA4 61.5; all: 62.4 / 73.2 / 57.2):

| config | ranker | top-N | top-N+2 | DCA4 top-N+2 | fast-backend ms (speed200 median, excl. parsing) |
|---|---|---|---|---|---|
| C5 | small | 60.0 | 71.6 (−0.3) | 59.9 (−1.7) | 34.7 |
| C5 | all | 56.3 | 67.4 (−5.9) | 43.9 (−13.2) | – |
| C6 | small | 60.0 | 71.1 (−0.8) | 59.5 (−2.0) | 25.7 |
| C6 | all | 56.3 | 67.1 (−6.2) | 43.8 (−13.3) | – |
| **C6m** (= C6 + metal candidates) | small | 60.1 | 71.2 (−0.7) | 59.8 (−1.8) | **30.6** |
| **C6m** | all | 62.2 | 72.4 (−0.9) | 55.6 (−1.6) | **30.7** |

Disabling metal candidates destroyed ion-site localisation for the `all` ranker (DCA4 −13); C6m keeps them.
Selection rule (fixed before the `all` results): most accurate config with ≤ 35 ms (with the v2 floor) and
train-CV losses ≤ 2.0 for both rankers → **C6m** (C5 + metal would exceed 35 ms).

## 3. Timing of the frozen fast models (fast backend, 1 core, warm, speed200, `scripts/burrow_fast_preset_bench.py`)
| model | total excl. parsing, median / p90 ms | parse | features | point score | pockets | rank | pockets/structure |
|---|---|---|---|---|---|---|---|
| v2 default all (same run) | 99.1 / 270.2 | 10.6 | 70.3 | 11.4 | 10.1 | 6.7 | 7 |
| **fast all** | **30.7 / 72.9** | 12.4 | 23.5 | 0.6 | 3.5 | 3.0 | 7 |
| **fast small** | **30.6 / 72.5** | 12.2 | 23.4 | 0.6 | 3.5 | 3.0 | 5 |
≈ 3.2× faster than v2 default; shared node (load ~18). Fast backend == reference: `scripts/verify_burrow_fast.py`
on 40 random dev structures, CPU: exact 40/40 for both models; CLI `--preset fast` reference == fast CPU on
10/10 structures (tests/test_burrow_cli_backends.py). **GPU: not supported yet** — the fast GPU kernels require
float32-exact scalars and 1/1.5 Å is not (ValueError in verify_burrow_fast.py --device cuda:1); `burrow predict
--preset fast --device cuda:N` therefore falls back per input to the reference implementation. Fix belongs to
the fast-backend owner (pass the inverse spacing as an fp64 tensor, as done for the AF thresholds).

## 4. Single dev confirmation (seed 0, paired vs v2 default s0; `scripts/burrow_v2_eval.py`, rows `fast_*` in v2_eval.csv)
| set | ranker | top-N+2 | Δ [95 % CI] | DCA4 top-N+2 | Δ | P1K | Δ |
|---|---|---|---|---|---|---|---|
| BioLiP dev (all sites) | all | 70.0 | −0.8 [−2.9, +1.5] | 54.4 | −1.7 | 67.4 | −2.3 |
| BioLiP dev (non-ion) | small | 69.9 | −1.2 [−4.2, +1.5] | 60.6 | −1.5 | 58.6 | −2.7 |
| apo-dev cryptic | all / small | 66.8 / 71.4 | −1.6 / −0.1 | 52.6 / 56.1 | −2.0 / −1.1 | | |
| apo-dev non-cryptic | all / small | 61.9 / 61.5 | +2.9 / +2.3 | – | | | |
| apo-track-dev | all / small | 61.8 / 64.2 | +2.5 / −3.1 | 42.8 / 47.7 | +0.6 / −2.7 | | |
| AF dev (prune_soft) | all / small | 70.0 / 64.9 | −1.2 / −1.2 | 52.4 / 46.6 | **−5.3** / −1.3 | | |

**Gate met** on BioLiP dev for both rankers (top-N+2 −0.8 / −1.2, DCA4 −1.7 / −1.5; ≤ 2.0). Caveats: P1K drops
2.3–2.7 pt; on AF models the `all` ranker loses 5.3 pt DCA4 (ion/metal sites in AF models are localised less
precisely with the coarser grid) — use the default preset for AF/metal work.

## 5. Frozen models
`results/dev/FREEZE_fast.md`: `results/dev/freeze_fast/models/burrow_{all,small}_fast_s{0,1,2}.json` (seeds
SEEDS[:3]; dev numbers above are seed 0). CLI: `burrow predict --preset fast` (seed 0).
