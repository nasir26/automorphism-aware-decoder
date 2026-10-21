"""Phase 4: quantization-aware fine-tuning of one trained Phase 3 model at one precision.

Recipe of nasir26/fpga-neural-predecoder: 3 epochs x 500,000 fresh shots, batch 256, Adam 1e-4,
from the FP32 checkpoint, at the training point (d = 5, p = 0.006). Arms: baseline and equivariant.
usage: python bench/phase4_qat.py ARM SEED PRECISION [--threads N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nn.quant import PRECISIONS, fine_tune  # noqa: E402
from nn.train import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["baseline", "equivariant"])
    ap.add_argument("seed", type=int)
    ap.add_argument("precision", choices=PRECISIONS)
    ap.add_argument("--threads", type=int, default=28)
    a = ap.parse_args()
    model, cfg = load(REPO / "results" / "phase3" / "models" / f"{a.arm}_s{a.seed}" / "model.pt")
    model.train()
    out = REPO / "results" / "phase4" / "models" / f"{a.arm}_s{a.seed}_{a.precision}"
    fine_tune(model, a.precision, out, a.seed, threads=a.threads, train_meta=cfg, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
