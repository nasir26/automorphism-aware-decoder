# automorphism-aware-decoder

**Status: Phases 0 to 2 done (symmetry discovery, reproduction, software decoders). Phases 3 to 8
(equivariant pre-decoder, quantization, HLS weight memory, verification and gating, card runs,
paper) not started; Phase 3 waits for approval of its training budget. Every number below is
software simulation (tier T4); nothing has run on an FPGA yet.**

Exact symmetry analysis of circuit-level surface code decoding problems, by twin contraction and
nauty (GraphCrunch, Prakash 2026), and tests of whether that symmetry is useful to a decoder.

Author: Nasir Ali, Embedded Systems Division, C-DAC Noida. The twin-quotient theorem and GraphCrunch
are by Vaibhav N. Prakash (Mahindra University).

## Headline results so far

| Result | Value | Source |
|---|---|---|
| Regression table (NVIDIA memory circuit, d = 13, T = 13, p = 0.003) | 51 of 52 rows reproduced; the count of distinct exact weights differs (96 measured, 78 in the table) | `results/phase1_reproduction.json`, `docs/DECISIONS.md` (D-1) |
| H1: twin contraction of the fault graph gives Stim's merged DEM | exact in 24 of 24 configurations (NVIDIA and Stim circuits, d = 3 to 13, both bases); probabilities equal to 5e-16 | `results/symmetry_sweep.json` |
| H1: the contraction preserves the group | plain nauty on the uncontracted fault graph gives |K| x |A| with identical orbits wherever it finished (d = 3, 5) | `results/faultgraph_group.json` |
| Decoding graph group (d >= 5) | order 2 for NVIDIA's circuit (a space-time point reflection σ of one stabilizer sector), order 4 for Stim's generator | `results/symmetry_sweep.json` |
| H2: where noise weights break σ | rounds 0, 1, T-2, T-1 for NVIDIA's circuit and the first and last layer for Stim's, at every d; the bulk keeps time translation and σ exactly | `results/symmetry_sweep.json` |
| σ on the halved-coordinate grid of `fpga-neural-predecoder` | an exact flip i -> d - i, t -> T - 1 - t, for both circuit families; on NVIDIA's CNN grid it is not a flip | `results/symmetry_sweep.json`, `results/phase1_reproduction.json` |
| Weight table compression by translation and σ | 1.65x (d = 5) to 5.45x (d = 13) | `results/symmetry_sweep.json` |
| GraphCrunch vs plain nauty, d = 5 fault graph | 0.08 s vs 65.5 s, identical partition; at d = 13, 2.4 s while the plain run crashes | `results/phase1_reproduction.json` |
| Metamorphic invariants of MWPM under σ (symmetrised weights) | weight and logical class invariant in 100% of shots; edge sets equal in only 37% to 77% (ties) | `results/phase2_metamorphic.json` |
| σ-twin disagreement as a failure flag (true weights) | P(fail | disagree) 0.38 to 0.46 against P(fail | agree) near the plain LER; recall only 1.7% to 5.3% | `results/phase2_sigma_twin.json` |
| Automorphism ensemble decoding with σ | no change at all: for an exact matcher the twin can never be lighter (0 of 66,050 disagreeing shots) | `results/phase2_ensemble.json` |

## Key limitations

- All results are T4 (software). No FPGA results yet.
- H1 checks twin contraction against Stim's own error analysis, not against an independent one.
- Weight equality is decided at 9 significant digits.
- The group available to a decoder is small (order 2 at d >= 5), which bounds what H3 to H6 can gain.
- See `docs/LIMITATIONS.md`, phase by phase.

## Reproduce

```
# third-party trees beside the repository: third_party/README.md
python3.11 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt
make tools        # builds build/nauty_aut against GraphCrunch's nauty tree
make test         # 41 tests, about 20 s
make phase1       # reproduction, sweep, fault-graph group check, exports, figures (about 45 min)
../.venv/bin/python bench/phase2_metamorphic.py
../.venv/bin/python bench/phase2_sigma_twin.py --workers 192       # about 1 h on 192 cores
../.venv/bin/python bench/phase2_sigma_twin.py --workers 192 --distances 5 9 13 --ps 0.003 0.005 0.007 \
    --batch 20000 --min-disagree 1 --max-shots 1 --out results/phase2_ensemble.json
../.venv/bin/python bench/plot_phase2.py
```

## Layout

| Path | Content |
|---|---|
| `src/autsym/` | circuits and detector geometry, fault graph, coloured twin contraction, nauty and GraphCrunch wrappers, symmetry checks, periodic bulk graphs, orbit tables |
| `src/qec/` | detector error models, decoding graph, PyMatching wrapper |
| `src/verify/` | metamorphic (symmetry) checks |
| `tools/nauty_aut.c` | nauty driver that writes or only counts generators |
| `bench/` | one script per experiment |
| `results/` | JSON outputs, orbit tables, GraphML exports, figures |
| `automorph/` | edge lists and colour files for Automorph |
| `patches/graphcrunch/` | proposed GraphCrunch change (vertex colourings, group order), not sent upstream |
| `docs/` | claims ledger, decisions, limitations, hypotheses, prior art, blockers |

## Licence

Apache-2.0. NVIDIA's Ising-Decoding (Apache-2.0) is imported from a clone outside the repository;
GraphCrunch (MIT) and nauty (Apache-2.0) are run from a clone outside it. See `NOTICE` and
`THIRD_PARTY.md`.
