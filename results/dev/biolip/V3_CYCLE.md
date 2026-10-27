# BURROW v3 cycle (dev-only, multimer inputs) — NEGATIVE, no FREEZE_v3

Started 2026-10-27 01:15 after test v1/v2 were scored (results/test/v{1,2}/SUMMARY.md read for
context only; nothing tuned on test). Motivation: O6 (dimer interface sites, full-assembly input)
was the weakest test endpoint (v1 −2.1, v2 −3.7 vs P2Rank, inconclusive).

## Gate (fixed by the PI before any v3 result)
Interface top-N+2 (dimer-benchmark rules) on the TRAIN assembly inputs, out-of-fold, **+3 pt with
CI > 0 vs v2**; no loss beyond −2 on the dev assembly inputs; BioLiP-dev top-N+2/DCA4 ≥ v2 − 0.5,
P1K ≥ v2 − 1, apo-dev ≥ v2 − 1; single-chain behaviour unchanged where possible.

## Dev signal construction
* Inputs: Track A stored assembly-1 inputs of the audited train set (1,709) and dev set (141).
* Ground truth (`scripts/burrow_asm_gt.py` → results/dev/biolip/asm/gt/): dimer-benchmark JSON per
  input — one copy per assembly ligand (Track A `asm_lig_*`, per entry, not pooled), binding residues
  ≤ 4 Å, entity = identical chain sequence, site classes as Track D (≥ 3 contact residues on each of
  ≥ 2 chains → homo/hetero interface).
* Scoring (`scripts/burrow_asm_eval.py`): Track D's own functions (`scripts/eval_dimer.py`: j50
  symmetric-copy grouping, label-free prediction collapse, any kept prediction of rank ≤ N+2 hitting
  any copy, DCC 12 Å / DCA 4 Å); paired cluster bootstrap over Track A cluster30 (B = 1000).
* BURROW out-of-fold on train (`scripts/burrow_asm_oof.py`): cached candidates of `p3_all_default_s0`
  (out-of-fold point scores, 5 folds of cluster30), ranker retrained per fold on the other folds;
  dev: frozen v2 `all` default s0 (`scripts/burrow_asm_dev_predict.py`). P2Rank 2.5.1 default on the
  same assembly files (baseline wrapper).

## Step 1 — does the dev signal reproduce the test weakness? (yes, weakly)
| set | method | interface sites | interface top-N+2 DCC12 | Δ vs P2Rank | interface DCA4 top-N+2 | all-site top-N+2 | Δ all vs P2Rank | pockets kept |
|---|---|---|---|---|---|---|---|---|
| train OOF (1,709 inputs) | P2Rank 2.5.1 | 319 | 84.0 | – | 70.2 | 66.8 | – | 8.1 |
| | BURROW v1 | 319 | 83.7 | −0.3 [−6.0, +5.2] | 76.2 | 81.1 | +14.2 | 40.9 |
| | BURROW v2 | 319 | 81.8 | −2.2 [−9.4, +4.1] | 74.9 | 79.6 | +12.8 | 9.5 |
| dev (141 inputs) | P2Rank 2.5.1 | 17 | 94.1 | – | 64.7 | 64.6 | – | 7.2 |
| | BURROW v1 / v2 | 17 | 94.1 / 94.1 | +0.0 / +0.0 | 88.2 / 88.2 | 79.1 / 78.6 | +14.6 / +14.0 | 41.1 / 9.0 |

The train-OOF signal reproduces the direction of the test result (v2 slightly behind P2Rank on
interface DCC12; the v1 → v2 drop of 1.9 pt mirrors the test drop of 1.6 pt), while BURROW is
ahead on interface DCA4 and far ahead on all assembly sites. The dev assembly set has only 17
interface sites and cannot discriminate.

## Step 2 — hypotheses (train OOF, interface sites, paired vs v2)
| variant | interface top-N+2 DCC12 | Δ vs v2 [95 % CI] | interface DCA4 Δ | all-site top-N+2 Δ | pockets kept |
|---|---|---|---|---|---|
| v2 (reference) | 81.8 | – | – | – | 9.5 |
| (a) min_keep = 3 × #chains (floor 0.03) | 83.1 | +1.3 [+0.0, +3.9] | +1.3 | +0.4 | 9.9 |
| (b) interface-aware ranker features (`model.IF_NAMES`) | 82.4 | +0.6 [−0.9, +2.3] | −0.3 | +0.0 | 9.4 |
| (b) + (a) | 83.7 | +1.9 [−0.6, +5.4] | +0.6 | +0.3 | 9.8 |
| (c) assembly inputs weighted ×3 in ranker training | 82.1 | +0.3 [−1.4, +2.2] | −0.9 | −0.1 | 9.5 |
| (c) + (a) | 83.1 | +1.3 [−1.3, +5.1] | +0.0 | +0.1 | 9.9 |

(b)'s second element — label-free collapse of symmetric duplicates in BURROW's own output — was not
built: the O6 evaluator already collapses symmetric duplicates before taking the top-N+2, so it
cannot free top-N+2 slots under this metric. Note also that removing the output floor entirely (v1)
recovers only +1.9 on this signal, so (a) alone could not reach +3.

## Decision
**Gate not met** (best: (b)+(a) +1.9 [−0.6, +5.4]; required ≥ +3 with CI > 0). No dev confirmation
was run, no FREEZE_v3 is written; v2 stays final. Dev-set use in this cycle: the reproduction check
on the 141 dev assembly inputs (v1/v2/P2Rank) only; all hypothesis comparisons used train
out-of-fold predictions. Tables: results/dev/biolip/asm/asm_eval_repro.csv, asm_eval_v3.csv.

Interpretation for the report: on multimer inputs BURROW localises interface ligands more precisely
(DCA4 +4.7 vs P2Rank on train OOF) and finds far more of all assembly sites (+12.8), but its DCC12
interface top-N+2 is at par / slightly below P2Rank; the shorter v2 output list costs ~1–2 pt there.
The code hooks for (a) (`FormationConfig.min_keep`), (b) (`meta.if_feats`) stay in the reference
implementation, unused by the frozen models.
