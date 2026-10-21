# Third-party components

None of these is vendored. `third_party/README.md` gives the commands that place them beside the
repository.

| Component | Version used | Licence | How it is used |
|---|---|---|---|
| NVIDIA/Ising-Decoding | commit 33acb15 | Apache-2.0 | `MemoryCircuit`, `NoiseModel` imported at analysis time to rebuild the reference circuits |
| vaibh-pra/GraphCrunch | commit 0d96131 | MIT | external tool for twin contraction and orbit lifting |
| nauty | 2.8.8 | Apache-2.0 | linked into the GraphCrunch solver |
| stim, pymatching, numpy, scipy, networkx, torch | see `requirements.txt`, `docs/env.lock` | Apache-2.0, Apache-2.0, BSD, BSD, BSD, BSD | Python dependencies |
