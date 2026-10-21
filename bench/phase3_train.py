"""Phase 3: train one arm with one seed at the fixed budget (src/nn/train.py).

Budget approved 2026-10-21: d = rounds = 5, basis Z, Stim circuit, p = 0.006 (noise upscaling of
nasir26/fpga-neural-predecoder), depth 2 (R = 5), width 64, 20 epochs of 2,000,000 fresh shots,
batch 256, Adam 1e-3. Three arms x three seeds, each run on 28 threads.
usage: python bench/phase3_train.py ARM SEED [--epochs N --shots N --threads N --out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nn.train import ARMS, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=ARMS)
    ap.add_argument("seed", type=int)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--shots", type=int, default=2_000_000)
    ap.add_argument("--threads", type=int, default=28)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else REPO / "results" / "phase3" / "models" / f"{a.arm}_s{a.seed}"
    train(a.arm, a.seed, out, epochs=a.epochs, shots_per_epoch=a.shots, threads=a.threads, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
