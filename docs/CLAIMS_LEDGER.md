# Claims ledger

Every number in the README, figures or paper has a row here and a file under `results/`.

**Tiers:** `T1-HW` measured on the U55C via XRT from the C++ host only; `T2-POSTROUTE`
post-place-and-route reports; `T3-HLS` Vitis HLS csynth estimates; `T4-SIM` software simulation or
analytic model. `PROJECTED` may appear here but never in the abstract, title, conclusions or a
results figure.

**Exactness:** `exact` means integer, rational or bit-identical comparison. Weight equality is
decided at a stated number of significant digits, recorded in the row.

| Id | Claim | Value | Tier | Exact or digits | Evidence | Status |
|---|---|---|---|---|---|---|
| E-001 | GraphCrunch brute-force self test on this host | 1,258 graphs, 0 failures | T4-SIM | exact | `results/env.json` | supported |
| E-002 | Regression table reproduced, NVIDIA d=13 T=13 p=0.003 basis X | 51 of 52 rows (bases X and Z) match; 1 differs (D-1) | T4-SIM | exact (counts), 9 digits (weights) | `results/phase1_reproduction.json` | supported |
| E-003 | Decoding graph and bare group, d=13 basis X | 2101 nodes, 10872 edges; order 2; 1555 vertex orbits (546 of size 2); 8058 edge orbits; no twins | T4-SIM | exact | `results/phase1_reproduction.json` | supported |
| E-004 | σ-swapped pairs with unequal weight, d=13 basis X | 2814 pairs, 469 unequal at 9 digits (683 at exact equality); 445 touch rounds 0/12, 24 rounds 1/11, 0 bulk only | T4-SIM | 9 digits | `results/phase1_reproduction.json` | supported |
| E-005 | Weight-table classes, d=13 basis X | 10,872 edges -> 2664 (translation) -> 1994 (translation and σ); 64 distinct weights at 9 digits, 96 exact | T4-SIM | 9 digits | `results/phase1_reproduction.json` | supported |
| E-006 | Periodic bulk graph (8 layers) weighted group | order 256, sector orders {'X': 16, 'Z': 16}; 17 bulk weights | T4-SIM | exact (order), 9 digits (weights) | `results/phase1_reproduction.json` | supported |
| E-007 | Fault graph d=13 basis X | 121239 faults, 123340 vertices, 339920 edges -> 42174 after twin contraction (40073 mechanisms); log10|K| = 46581; 81166 transpositions | T4-SIM | exact | `results/phase1_reproduction.json` | supported |
| E-008 | H1: twin quotient equals Stim's merged DEM | exact symptom-set equality in all 24 configurations (2 families, d=3..13, both bases); max relative probability difference 4.8e-16 | T4-SIM | exact (sets), 1e-15 relative (probabilities) | `results/symmetry_sweep.json` | supported |
| E-009 | H2 NVIDIA circuits: σ weight breaking confined to rounds 0, 1, T-2, T-1 | depth 2 at each end for every d = 5..13, both bases | T4-SIM | 9 digits | `results/symmetry_sweep.json` | supported |
| E-010 | H2 Stim generator: σ weight breaking confined to first and last layer; the other sector's reflection is an exact weighted automorphism | depth 1 at each end for every d = 5..13, both bases; weighted group order 2 | T4-SIM | 9 digits | `results/symmetry_sweep.json` | supported |
| E-011 | σ on the halved-coordinate grid is an exact flip i -> d - i (t -> T-1-t) | all moved detectors, every d = 5..13, both bases, both families | T4-SIM | exact | `results/symmetry_sweep.json` | supported |
| E-012 | σ on NVIDIA's CNN grid, d=13 basis X | 936 cells (row, col) -> (11-row, 11-col), 156 cells (row, col) -> (11-row, 12-col); t -> 12-t | T4-SIM | exact | `results/phase1_reproduction.json` | supported |
| E-013 | Weight compression (translation and σ), NVIDIA basis X | d=5: 1.65x; d=7: 2.68x; d=9: 3.61x; d=11: 4.53x; d=13: 5.45x | T4-SIM | 9 digits | `results/symmetry_sweep.json` | supported |
| E-014 | GraphCrunch d=5 fault graph: contract+solve+lift vs plain nauty (this host) | 0.08 s vs 65.5 s, identical partition | T4-SIM | timing, single run | `results/phase1_reproduction.json` | supported |
| E-015 | GraphCrunch d=13 fault graph | contract+solve+lift 2.38 s; plain run fails (exit after 28.4 s) | T4-SIM | timing, single run | `results/phase1_reproduction.json` | supported |
| E-016 | Colour-aware twin contraction validated by brute force | 38,178 exhaustive coloured graphs (n <= 5) plus random, blow-up and nauty-path cases, 0 failures; GraphCrunch patch: 1,298 coloured graphs, 0 failures | T4-SIM | exact | `tests/test_coloured.py`, `patches/graphcrunch/README.md` | supported |
| E-017 | Metamorphic invariants under σ, symmetrised weights, NVIDIA d=13 basis X, 5,000 shots per p | weight (1e-9 relative) and logical class equal in 100% at p = 0.003, 0.005, 0.007; pulled-back correction explains s in 100%; edge sets equal in 76.5% [75.3, 77.7], 52.9% [51.5, 54.2], 36.9% [35.6, 38.2] | T4-SIM | exact counts; weight at 1e-9 | `results/phase2_metamorphic.json` | supported |
| E-018 | Metamorphic invariants on periodic windows (P = 8, 16), translation and both reflections, both families | weight equal in 100% of 2,000 shots for every element; edge sets equal in 37% to 80% | T4-SIM | weight at 1e-9 | `results/phase2_metamorphic.json` | supported |
| E-019 | σ-twin disagreement under true weights, NVIDIA basis X, d=5..13, p=0.002..0.008 | 0.0001% to 0.77% of shots; 34 of 35 points conclusive (>= 100 disagreeing shots) | T4-SIM | Wilson 95% in file | `results/phase2_sigma_twin.json` | supported |
| E-020 | P(MWPM fails | σ-twin disagrees) vs P(fails | agrees) | 0.38 to 0.46 vs 1.3e-05 to 1.0e-01 | T4-SIM | Wilson 95% in file | `results/phase2_sigma_twin.json` | supported |
| E-021 | Recall of the σ-twin flag P(disagree | MWPM fails) | 0.017 to 0.053; falls with d | T4-SIM | Wilson 95% in file | `results/phase2_sigma_twin.json` | supported |
| E-022 | Automorphism ensemble decoding (order 2, true-weight rule) vs plain MWPM | identical failures at all 9 points (d = 5, 9, 13; p = 0.003, 0.005, 0.007; 3.84 M shots each); twin never lighter in 66050 disagreeing shots (45 ties) | T4-SIM | exact counts | `results/phase2_ensemble.json` | supported |
| E-023 | Phase 3, no pre-decoder arm beats PyMatching alone (Stim rotated memory Z, p = 0.003, sdr50 operating point, 3 seeds, 200k shots d = 5, 7, 9, 100k d = 11, 13) | LER ratio to PyMatching alone 1.004 to 1.014 (d = 5), 1.16 to 1.22 (d = 7), 1.27 to 1.33 (d = 9), 2.0 to 2.06 (d = 11), 2.7 to 3.1 (d = 13); best grid threshold (0.99) gives PyMatching's count within 2 failures | T4-SIM | exact counts, Wilson 95% in file | `results/phase3/report.json`, `results/phase3/summary.json` | supported |
| E-024 | H3 paired verdicts against the same-seed baseline (rule of 2026-10-21) | augmented: inconclusive (d = 5; d = 9, 11, 13); equivariant: inconclusive (both; d = 5 pooled -3.0e-5, CI [-5.3e-5, -6.9e-6], but seed 0 not lower); baseline + TTS and augmented + TTS: beat baseline at d = 5 only (-2.5e-5, -2.3e-5) | T4-SIM | exact paired counts | `results/phase3/report.json` | supported |
| E-025 | Equivariance after training, equivariant arm, 1,000 validation shots | max logit deviation 8.6e-6 to 1.1e-5 in FP32 (above the 1e-6 target), 1.4e-14 to 1.8e-14 in FP64 | T4-SIM | float | `results/phase3/eval/equivariant_s*.json` | supported |
| E-026 | Parameters, width 64 depth 2 | baseline 118,983; equivariant 118,535 stored, 61,959 free | T4-SIM | exact | `results/phase3/eval/*.json` | supported |
| E-027 | Phase 4 integer golden model, sdr50, 3 seeds pooled, failures (d = 5 / d = 9, 200k shots each; PyMatching 615 / 128) | baseline FP32 1871 / 497, INT8 1861 / 501, INT4 1910 / 531, ternary 2499 / 793; equivariant FP32 1853 / 505, INT8 1857 / 502, INT4 1897 / 582, ternary 3042 / 972 | T4-SIM | exact counts | `results/phase3/summary.json`, `results/phase4/eval/*.json` | supported |
| E-028 | Integer σ-equivariance (flip check, 200 validation shots, every accumulator) | equivariant: 0 differing accumulators at INT8, INT4, ternary, all seeds; baseline: 60,110 to 60,200 differ | T4-SIM | exact | `results/phase4/eval/*.json` | supported |
| E-029 | Fake-quantized FP64 vs integer golden decisions, 1,000 validation shots, t = 0.5 and 0.9 | 0 mismatches in 35 of 36 (model, threshold) cases; 1 in baseline s1 ternary at t = 0.5, an exact tie (integer accumulator 0 = threshold, FP64 logit 2.2e-16) | T4-SIM | exact | `results/phase4/eval/baseline_s1_ternary.json` | supported |
