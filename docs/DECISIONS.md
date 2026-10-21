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
