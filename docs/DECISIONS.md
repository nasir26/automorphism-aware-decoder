# Decisions

Each entry: date, decision, reason, alternatives considered.

## 2026-10-21: workspace layout

- **Decision:** the repository lives at `automorphism/automorphism-aware-decoder/`; read-only clones
  of other repositories live beside it in `automorphism/third_party/`; the project venv is
  `automorphism/.venv`. Nothing from `third_party/` is copied into the repository.
- **Reason:** the prompt asks for third-party code outside the repository; a sibling directory keeps
  the relative path `../third_party/` used in the documentation.

## 2026-10-21: what is reused from each repository (Phase 0 reconnaissance)

| Repository (commit read) | Licence | Reused | Re-derived | Not used |
|---|---|---|---|---|
| `vaibh-pra/GraphCrunch` (`0d96131`) | MIT | invoked as an external tool: `src/orbits_serial.py` and the `automorph_serial_col` solver, built in `../third_party/GraphCrunch` against nauty 2.8.8 | colour-aware twin contraction is written here and, if it validates, offered back as `patches/graphcrunch/` | not vendored. The solver prints vertex and edge orbits only; it does not print generators or the group order, which Phase 1 needs. That is a candidate patch, not a local edit |
| `NVIDIA/Ising-Decoding` (`33acb15`, local mirror) | Apache-2.0 | `code/qec/surface_code/memory_circuit.py` (`MemoryCircuit`) and `code/qec/noise_model.py` (`NoiseModel.from_single_p`) imported at analysis time from `../third_party/Ising-Decoding/code`; their only dependencies are stim and numpy | nothing | no code copied; no checkpoints fetched |
| `nasir26/fpga-neural-predecoder` (`8ba71fd`) | Apache-2.0 | conventions: grid `(T, H, W)` with `T = rounds + 1`, `H = W = d + 1` from halved Stim coordinates; per-local-edge output head; XOR-parity targets of local fired DEM errors; QAT recipe and golden-model rule ("zero tolerance" judged on the decision) | `PreDecoder3D`, edge catalogue, shift-and-XOR application and observable fold-in are rewritten in `src/nn` and `src/qec`, with regression tests for its three documented bugs (per-voxel parity, dropped observable effect, double-counted even-parity observable target; its `docs/DECISIONS.md` entries of 2026-08-16) | trained checkpoints are not reused, so every result here is trained here |
| `nasir26/surface-code-hierarchical-decoder` (`7666e77`) | Apache-2.0 | T1 to T4 provenance tags; the `scripts/env.sh` pattern for Vitis and XRT | the confidence gate (`src/pipeline/hierarchical.py`), because that repository has only Phase 0 complete and no gate implementation | its PROMPT is not followed here |
| `nasir26/qec-fpga-revision` (`a2e2757`) | none stated on GitHub | claims ledger layout and the rule that only the C++ XRT host produces timing claims | the Union-Find kernel would be re-derived if Phase 5 option B is chosen | Steane and Shor LUT kernels |
| `vaibh-pra/Dequantization-Certifier` (`1b7b3a4`) | MIT | discipline only: separate exact from numerical claims; report inconclusive rather than guess | nothing | no code |
| `nasir26/ising-decoder-graphml` | n/a | does not exist (GitHub 404 on 2026-10-21) | the GraphML exports are regenerated in Phase 1 | n/a |

## 2026-10-21: host differs from the one in the prompt

- **Finding:** this host is a 2 socket AMD EPYC 7742 (256 logical CPUs, 504 GB) with four U55C cards
  at `0000:01:00.1`, `0000:21:00.1`, `0000:41:00.1`, `0000:a1:00.1`, XRT 2.16.204, Vitis 2023.2, and
  **no GPU** (`results/env.json`). The prompt describes the hierarchical repository's host (one U55C
  at `0000:8c:00.1`, XRT 2.15.225, RTX A4000).
- **Decision:** all training in this project is CPU training on this host unless you move it; Phase 7
  card addresses are this host's.

## 2026-10-21: nauty obtained from the official mirror without curl

- **Decision:** `scripts/get_nauty.sh` needs `curl`, which is not installed. The same tarball was
  fetched with Python `urllib` from the first mirror the script lists
  (`users.cecs.anu.edu.au/~bdm/nauty/nauty2_8_8.tar.gz`, 2,302,360 bytes, sha256
  `159d2156...29ce`), then the unmodified script built it. The pynauty fallback was not needed.
- **Result:** `make test` in GraphCrunch: 1,258 graphs checked, 0 failures.

## 2026-10-21: Python environment resolved but not installed

- **Decision:** venv at `../.venv` created from Python 3.11.16 (the interpreter of an existing local
  environment, so no system install). The pinned set resolves to 31 wheels, 313 MB, of which torch
  2.14.1+cpu is 196 MB (CPU wheel from the PyTorch index; no CUDA libraries). Install is waiting for
  approval, as the prompt requires.
- **Pins chosen:** stim 1.16.0, pymatching 2.4.0, numpy 2.4.6, scipy 1.17.1, networkx 3.6.1,
  torch 2.14.1+cpu; tooling unpinned until resolution, then frozen into `docs/env.lock`.

## 2026-10-21: co-author

- **Decision:** on 2026-10-21 you asked for a submittable package with V. N. Prakash as co-author.
  He will be on the paper's author list. The prompt's rule still holds for the LICENSE, the commit
  authorship and `CITATION.cff`, which will not name him unless you say so.

## 2026-10-21: Phase 0 gate approved

- **Decision:** you approved ("i agree and continue") the novelty statement in `docs/PRIOR_ART.md`,
  the three hypothesis adjustments proposed there, the venv install, pushing each phase to
  `nasir26/automorphism-aware-decoder`, and continuing. Runs over 30 minutes, `v++ -t hw` builds and
  card runs are still asked about separately.
- **Hypothesis changes made (approved):** H5 gains a fourth test policy, Shen and Zhong's
  weight-preserving relabelling relation; H6 gains a control, a second decode with randomly perturbed
  weights of equal cost; H3 is recorded as expected null, so its experiment is powered for a small
  paired effect.
- **Install:** 31 wheels as resolved, frozen in `docs/env.lock`; tooling versions pinned in
  `requirements.txt` from the lock.

## 2026-10-21: Phase 1 conventions (each measured, then fixed in code)

- **NVIDIA circuit variant:** `MemoryCircuit(..., noise_model=NoiseModel.from_single_p(p))` with
  `add_boundary_detectors=False` (its default). With boundary detectors on, the circuit has 2,184
  detectors instead of the regression table's 2,100. NVIDIA's evaluation pipeline uses `True`; the
  regression table, and so this repository, uses `False`.
- **DEM options:** `approximate_disjoint_errors=True` everywhere; Stim refuses NVIDIA's
  PAULI_CHANNEL_2 without it. The raw DEM is built without `flatten_loops` and flattened afterwards:
  with `flatten_loops=True` Stim already merges repeated lines (40,073 lines instead of 45,392).
- **Detector geometry:** NVIDIA's circuit has no detector coordinates. Each detector is traced
  through its measurement records to the stabilizer ancilla it reads; (x, y) are NVIDIA's ancilla
  coordinates and t counts earlier measurements of that ancilla (plus one for final-data detectors).
  For Stim's circuits the same tracing is cross-checked against Stim's own detector coordinates at
  every run (exact agreement). Time layers: NVIDIA's `n_rounds` counts preparation and final rounds,
  so T = d gives d layers; Stim's `rounds = d` gives d + 1 layers.
- **Fault probabilities:** measured against Stim with isolating circuits (`tests/test_faultgraph.py`),
  not assumed. With `approximate_disjoint_errors`, Stim converts PAULI_CHANNEL_1 and DEPOLARIZE1/2
  exactly into independent components, but takes PAULI_CHANNEL_2 components at their listed values,
  and components of one PAULI_CHANNEL_2 application with the same symptom add (disjoint events).
  The merged class probabilities then agree with Stim's merged DEM to 5e-16 relative.
- **Name clash:** NVIDIA's package and this repository's `src/qec` are both called `qec`. NVIDIA's
  modules are imported once with ours set aside in `sys.modules`, then removed again
  (`autsym.circuits._import_nvidia`, tested). The prompt's layout is kept.
- **Edge templates:** an edge is anchored at the earliest layer of its detector endpoints; its
  descriptor is the sorted pair (stabilizer, x, y, t - anchor) or boundary, plus its weight at 9
  significant digits. This definition reproduces the table's 912 / 11 / 731. The weight-table classes
  (2,664 and 1,994) are reproduced when edges anchored in the bulk rounds share a class with their
  translates and every edge anchored in a non-bulk round is its own class.
- **σ is constructed, then checked:** `autsym.orbits.reflection` builds the point reflection of one
  stabilizer sector through the centre of that sector's coordinate box; `is_automorphism` decides.
  In the sweep σ means the reflection of the basis sector; every other reflection that is an
  automorphism is reported too.
- **Group orders:** from this repository's `tools/nauty_aut.c`, which writes generators to disk as
  nauty finds them (or only counts them, `nogens`) instead of keeping them in memory.
- **SI1000:** not swept; it is not one of `stim.Circuit.generated`'s noise options, and building it
  by hand would be a third circuit family with its own correctness risk.

## 2026-10-21: correction to the Phase 0 note on GraphCrunch

- Phase 0 recorded that GraphCrunch's solver prints orbits only. That was wrong: its output file
  also has an "Automorphism Group" section with the order and the generators in cycle notation. What
  is missing is in the Python front end, which neither parses the order nor reports |Aut(G)| =
  |K| |A|. `patches/graphcrunch/coloured_twins.patch` adds that, together with vertex colourings.

## 2026-10-21: regression difference D-1, distinct exact weights

- **Measured:** the d = 13 basis X decoding graph has 96 distinct weights as exact binary64 values
  (98 distinct probabilities), and 64 at 9 significant digits. The table expects 64 at 9 digits
  (matches) and 78 exact (does not).
- **Checked, not changed:** no rounding level gives 78 (17 digits: 96, 16: 93, 15: 71, 14 down to
  10: 64; float32: 64); basis Z gives 102 exact; bulk subsets give 35 or 60. The 78 may come from a
  different weight representation in the original analysis (for example GraphML text). No code was
  changed to chase it; every row that depends on weight equality uses 9 digits, which matches.
  Reported to you for a decision.

## 2026-10-21: Phase 2 conventions

- **Decoder interface:** `qec.matching.MatchingDecoder.decode(syndrome) -> (edges, weight, logical)`
  over PyMatching 2.4.0, with weights summed by `math.fsum` (order independent). Its logical
  prediction equals PyMatching's own on every tested shot (`tests/test_phase2.py`).
- **Symmetrised weights:** w'(e) = (w(e) + w(σe)) / 2, which makes σ an exact weighted automorphism.
  Weight equality is judged at 1e-9 relative; exact float equality is also recorded (it fails in a
  few shots because different tied corrections sum different floats).
- **Periodic windows:** translations and sector reflections are exact automorphisms only on the
  periodic bulk graphs (P = 8, 16 layers); the finite-window metamorphic tests are run there, with
  errors sampled edge by edge from the template probabilities. These graphs have no observable.
- **σ-twin batching:** the pull-back's logical prediction is read from a second matcher whose
  observable mask on edge e is that of σ^-1(e), so both decodes run in PyMatching's batched C++ path.
  `tests/test_phase2.py` checks it against the explicit per-shot pull-back.
- **Shot budget:** 192 workers x 200,000 shots per round (38.4 M), repeated until 100 disagreeing
  shots or 40 M shots. One point (d = 13, p = 0.002) stayed below 100 and is marked inconclusive.

## 2026-10-21: bug in the first ensemble baseline, found and fixed

- **What happened:** the first σ-twin run implemented automorphism ensemble decoding as "keep the
  decode with the smaller weight as PyMatching reports it". For the twin that weight is measured on
  σ s with the true weights, which is the pulled-back correction's weight under σ-permuted weights,
  not its true weight. The prompt's rule is the minimum true weight.
- **Fix:** the pulled-back correction's true weight is now computed from its edges, on the shots
  where the two predictions disagree (elsewhere the choice cannot change the prediction).
- **Result kept, relabelled:** the completed 35-point run's columns were renamed
  `*_lighter_own_weight` with a note in the file; their values were not changed. The corrected
  ensemble is in `results/phase2_ensemble.json` (9 points, 3.84 M shots each).
- **Why the corrected ensemble can only tie:** decode(s) is a minimum-weight correction for s under
  the true weights, so no other correction of s, the pulled-back twin included, can be lighter.
  Measured: across 66,050 disagreeing shots the twin was never lighter (45 exact ties, which keep
  the plain decode), and the ensemble's failures equal plain MWPM's at all 9 points.

## 2026-10-21: run time over the 30 minute rule

- The 35-point σ-twin sweep took 62 minutes on 192 workers; it was estimated at about 5 minutes and
  started without asking. The estimate missed that the first round already samples 38.4 M shots at
  every point. The rule was broken; later runs are sized with a timing probe first.
