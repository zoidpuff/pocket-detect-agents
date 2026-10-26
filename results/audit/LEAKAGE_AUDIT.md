# Independent leakage audit (train/dev vs test sets)

Auditor re-implemented everything (scripts/audit/a1..a5). Rule: E<1e-3 AND fident>=0.30 AND (qcov>=0.5 OR tcov>=0.5 OR alnlen>=80);
MMseqs2 easy-search -s 7.5 --max-seqs 5000 -e 1e-3. Builder scripts not used (builder test_union.tsv used only in the diagnostic a4).

## Inputs extracted
- Train: 6,644 rep-chain inputs + 4,209 assembly-1 chains (1,744 entries with stored assembly) = 8,277 unique seqs (observed residues from structure files, MSE->M).
- Dev: 596 rep chains + 274 assembly chains. Train+dev unique 8,277 / train 7,601 / dev 676.
- Test (own extraction with gemmi, observed + SEQRES for LIGYSIS): 21,036 unique sequences:
  LIGYSIS 3,448 chains (1 chain, 7uw7_A, has no mirror file), COACH420 (420 entries), HOLO4K (4,009 entries, 7,455 chains), CryptoBench (all 5,005 cif entries, all chains),
  PocketMiner (all-structures + apo-structures, 357+ files), PocketWatch test (1,566 UniProt seqs), AF (3,473 models), dimer (all 2,124 chains).
- "Primary" tier = chains actually evaluated (LIGYSIS listed chains; CryptoBench apo/holo chains of dataset.json; PocketMiner test+val apo chains; all chains for COACH/HOLO/dimer/AF/PW). "Broad" = every chain of the entry/file.

## (3) PDB-ID and UniProt overlap
| test set | #test PDB ids | train PDB overlap | dev PDB overlap | train UniProt overlap (primary) | dev UniProt overlap (primary) |
|---|---|---|---|---|---|
| ligysis | 3,376 | 0 | 0 | 1 (Q96Q15__6z3r_A) | 0 |
| coach420 | 420 | 0 | 0 | 0 | 0 |
| holo4k | 4,009 | 0 | 0 | 0 | 0 |
| cryptobench | 5,005 | 0 | 0 | 0 | 0 |
| pocketminer | 185 | 0 | 0 | 0 | 0 |
| pocketwatch test | 8,036 | 0 | 0 | 0 | 0 |
| af | - | - | - | 1 (same entry, Q96Q15) | 0 |
| dimer | 400 | 0 | 0 | 0 | 0 |

Broad (entry-wide SIFTS UniProt of every chain of test entries vs rep-chain UniProts): 86 train/dev entries share a UniProt with some non-primary chain of a test entry
(e.g. other chains of CryptoBench cif files, PocketMiner non-test files, LIGYSIS partner chains); list in uniprot_overlap_strict.tsv. Not counted as violations.

## (2) Sequence-rule violations, train/dev vs test (violations_vs_test.tsv, violations_primary.tsv)
Primary-tier violating inputs (92 total; dev 11, train 81):
| test set | violating inputs (primary) | broad |
|---|---|---|
| ligysis | 31 | 31 |
| coach420 | 6 | 6 |
| holo4k | 21 | 21 |
| cryptobench | 29 | 157 |
| pocketminer | 0 | 14 |
| pocketwatch | 2 | 2 |
| af | 1 | 1 |
| dimer | 13 | 13 |
(an input can hit several sets). By kind: train rep 45, train asm 36, dev rep 9, dev asm 2.
Identity of the strongest hit per violator: 88 of 92 are 0.30-0.35 (borderline); 2 in 0.35-0.5; 1 at 0.53 (7dco_u vs ligysis 7qtt_J, tcov 0.94, alnlen 68).
Exception for PocketMiner/CryptoBench non-evaluated chains: 141 broad-only violators (e.g. train 2cgj_A identical to PocketMiner-all-structures 2cgl/2cgk; 1pp9_O identical to CryptoBench 1ppj chain O) are NOT
in the evaluated chain set but are in test-set PDB entries (whole-entry sequences).

Diagnostic (a4): re-searching the 92 primary violators against the BUILDER's canonical test_union (data only): 85 no longer violate (difference = observed-residue vs canonical-SEQRES
sequence representation, hits sit at the 0.30 identity cliff); **7 still violate the builder's own union**, all train assembly-partner chains:
`A3MTM6__5fb3_F` (asm chains B,E vs holo4k 1o2d, id 0.31-0.32), `P58502__2z30_B` (asm A, id 0.41, 312 aln vs subtilisins in HOLO4K/CryptoBench),
`Q6CK86__4m3o_B` (asm A, 0.32 vs LIGYSIS 7b7u_M / AF Q8IXW5), `Q1QD55__8e62_A` (asm B, 0.32 vs PocketWatch Q6Q627), `Q5PXQ6__1tmx_A` (asm B, 0.31 vs holo4k 1eob), `Q9I194__4wks_C` (asm A, 0.31-0.32 vs holo4k 1jw0/1pnl).
These assembly-1 inputs survive in the builder's manifests, i.e. the assembly-partner check missed them.

## (4) Train vs dev
0 rule hits (611 raw hits at E<1e-3 all fail identity/coverage rule). PDB-ID overlap train/dev: 42 dev entries share a PDB id with a train entry (different chains of the same
complex, e.g. ribosomes/respiratory complexes; sequence-dissimilar, allowed by the rule). UniProt overlap train/dev: 0.

## (5) Spot-check of 20 random train entries (seed 20261026; spotcheck20.tsv)
All 20: structure file exists, chain length equals manifest n_res; npz rep_seq equals the structure sequence (4 differ only by MSE->M vs X); pooled ligand atoms lie
within 3.1-6.2 A (min) of the chain CA atoms; ligand CCD identities plausible (ZN/CA/MG/FE/CU1 ions, GSH, SAM, ADP, TYR, IPE, B7N, ...); resolutions 1.13-2.8 A, X-ray or EM.
No anomalies found.

## Verdict: **FAIL (marginal)**
- Hard failures: 7 train assembly-1 inputs (listed above) violate the stated rule against the builder's own union; 1 train entry (Q96Q15__6z3r_A) shares a UniProt with a LIGYSIS/AF-set protein.
- Soft findings: using observed-residue test sequences (what the model sees) yields 85 more borderline (0.30-0.35 id) violators; builder's union of canonical sequences misses these because
  of the identity cliff. 141 further train/dev inputs match non-evaluated chains in CryptoBench/PocketMiner test entries.
- Clean: PDB-ID overlap 0 everywhere; train-vs-dev rule 0; no sequence-identical (>=0.55 id) leak to any primary-evaluated chain.
Recommended fix: drop the 7 assembly inputs (assembly input -> chain only) and the Q96Q15 entry; optionally re-run with observed-sequence test set and a 0.25 identity margin.
