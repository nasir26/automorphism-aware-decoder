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
