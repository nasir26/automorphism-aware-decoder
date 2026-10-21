"""Phase 3 evaluation of one trained model (plain, and with test-time symmetrisation for the two
unconstrained arms). Rules fixed before any full-budget result was seen (docs/DECISIONS.md,
2026-10-21 "Phase 3 evaluation protocol"):

* validation: d = 5, p = 0.003, 200,000 circuit shots, seed 5_000_001; test shots never seen there;
* operating points per model variant:
  - ``sdr50`` (primary): the threshold at which the validation shots lose 50% of their sector
    detection events (bisection, 30 steps);
  - ``t0.5``: the fixed threshold 0.5;
  - ``best``: the grid threshold with the fewest validation failures (ties: higher threshold);
* test: p = 0.003, d = 5, 7, 9, 11, 13, circuit shots with seed 6_000_000 + d, the same shots for
  every model; every grid threshold is also recorded;
* per-edge-type correction profile: at each operating point, the mean number of applied edges per
  shot in every edge channel (channel 0 = boundary edges, then the offset catalogue);
* equivariance after training (equivariant arm): the largest |logit(flipped volume) at σe - logit at
  e| over 1,000 validation shots, in FP32 (the prompt asks for below 1e-6) and in FP64 (separates
  rounding from a construction error); and the number of free
  parameters of the model and of its unconstrained counterpart.

Per-shot failure bits are stored (packed) so the paired comparisons in bench/phase3_report.py are
exact. Writes results/phase3/eval/<arm>_s<seed>.json and .npz (T4-SIM).
usage: python bench/phase3_eval.py ARM SEED [--threads N --shots N --distances ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nn.evaluate import Evaluator  # noqa: E402
from nn.grid import reflect_volume, sector_grid, to_volume  # noqa: E402
from nn.train import load  # noqa: E402

GRID = (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99)
P_EVAL = 0.003
VAL_SHOTS, VAL_SEED = 200_000, 5_000_001


def sdr_threshold(ev: Evaluator, prob: np.ndarray, target: float = 0.5, steps: int = 30) -> tuple[float, float]:
    lo, hi = 0.0, 1.0  # SDR decreases with the threshold
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        _, info = ev.fails(prob, mid)
        if info["syndrome_density_reduction"] > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return t, ev.fails(prob, t)[1]["syndrome_density_reduction"]


def profile(sg, prob: np.ndarray, threshold: float) -> list[float]:
    applied = prob > threshold
    return [float(applied[:, sg.edge_channel == c].sum(1).mean()) for c in range(sg.n_channels)]


def effective_parameters(model) -> int:
    """Free parameters after the constraints: a point-symmetric 3x3x3 kernel has 14 free entries per
    (out, in) pair, and the tied head has no separate w2."""
    n = sum(q.numel() for q in model.parameters())
    if model.equivariant:
        for conv in model.convs:
            n -= conv.v.numel() - conv.v.shape[0] * conv.v.shape[1] * 14
    return n


def equivariance_deviation(model, sg, dets: np.ndarray, dtype=torch.float32) -> float:
    import copy
    model = copy.deepcopy(model).to(dtype)
    ef = torch.from_numpy(sg.edge_flat)
    with torch.no_grad():
        x = torch.from_numpy(to_volume(sg, dets)).to(dtype)
        a = model.edge_logits(x, ef)
        b = model.edge_logits(reflect_volume(x), ef)[:, torch.from_numpy(sg.sigma_edge)]
    return float((a - b).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("seed", type=int)
    ap.add_argument("--threads", type=int, default=28)
    ap.add_argument("--shots", type=int, nargs="+", default=[200_000, 200_000, 200_000, 100_000, 100_000])
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 7, 9, 11, 13])
    ap.add_argument("--models", default=str(REPO / "results" / "phase3" / "models"))
    ap.add_argument("--out", default=str(REPO / "results" / "phase3" / "eval"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, cfg = load(Path(a.models) / f"{a.arm}_s{a.seed}" / "model.pt")
    variants = [False] if cfg["model"]["equivariant"] else [False, True]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sgv = sector_grid(5, P_EVAL, offsets=model.offsets)
    ev = Evaluator(sgv, VAL_SHOTS, VAL_SEED)
    ops = {}
    val = {"shots": VAL_SHOTS, "seed": VAL_SEED, "mwpm_fails": int(ev.fail_mwpm.sum()), "variants": {},
           "parameters": sum(q.numel() for q in model.parameters()), "effective_parameters": effective_parameters(model),
           "equivariance_max_deviation_fp32": equivariance_deviation(model, sgv, ev.dets[:1000]),
           "equivariance_max_deviation_fp64": equivariance_deviation(model, sgv, ev.dets[:1000], torch.float64)}
    print(a.arm, a.seed, "equivariance deviation", val["equivariance_max_deviation_fp32"], flush=True)
    for tts in variants:
        name = "tts" if tts else "plain"
        prob = ev.probabilities(model, tts)
        grid = {}
        for th in GRID:
            f, info = ev.fails(prob, th)
            grid[str(th)] = {"fails": int(f.sum()), **info}
        best = min(GRID, key=lambda th: (grid[str(th)]["fails"], -th))
        t50, s50 = sdr_threshold(ev, prob)
        ops[name] = {"sdr50": t50, "t0.5": 0.5, "best": best}
        val["variants"][name] = {"grid": grid, "operating_points": ops[name], "sdr50_validation_sdr": s50}
        print(a.arm, a.seed, name, "val", ops[name], flush=True)

    res = {"channels": [[0, 0, 0]] + [list(o) for o in model.offsets],
           "arm": a.arm, "seed": a.seed, "tier": "T4-SIM", "p": P_EVAL, "train": cfg, "validation": val, "test": []}
    bits = {}
    for d, n in zip(a.distances, a.shots):
        sg = sector_grid(d, P_EVAL, offsets=model.offsets)
        ev = Evaluator(sg, n, 6_000_000 + d)
        row = {"distance": d, "rounds": d, "shots": n, "seed": 6_000_000 + d, "mwpm_fails": int(ev.fail_mwpm.sum()),
               "variants": {}}
        bits[f"d{d}_mwpm"] = np.packbits(ev.fail_mwpm)
        for tts in variants:
            name = "tts" if tts else "plain"
            t1 = time.time()
            prob = ev.probabilities(model, tts)
            v = {"forward_seconds": round(time.time() - t1, 1), "grid": {}, "operating_points": {}}
            for th in GRID:
                f, info = ev.fails(prob, th)
                v["grid"][str(th)] = {"fails": int(f.sum()), **info}
            for op, th in ops[name].items():
                f, info = ev.fails(prob, th)
                v["operating_points"][op] = {"threshold": th, "fails": int(f.sum()), **info,
                                             "applied_per_channel": profile(sg, prob, th)}
                bits[f"d{d}_{name}_{op}"] = np.packbits(f)
            row["variants"][name] = v
            print(a.arm, a.seed, d, name, {k: x["fails"] for k, x in v["operating_points"].items()}, "mwpm", row["mwpm_fails"], flush=True)
        res["test"].append(row)
        res["seconds"] = round(time.time() - t0, 1)
        (out / f"{a.arm}_s{a.seed}.json").write_text(json.dumps(res, indent=1))
        np.savez_compressed(out / f"{a.arm}_s{a.seed}.npz", **bits)


if __name__ == "__main__":
    main()
