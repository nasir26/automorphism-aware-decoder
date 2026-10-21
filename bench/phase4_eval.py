"""Phase 4 evaluation of one quantized model through the integer golden model (nn.quant.GoldenTorch,
bit-equal to the NumPy reference by tests/test_quant.py).

Same protocol as Phase 3 (docs/DECISIONS.md): validation d = 5, p = 0.003, 200,000 shots, seed
5,000,001, operating points sdr50, t0.5 and best chosen there on the integer accumulators; test at
p = 0.003, d = 5 and 9, 200,000 shots each, seed 6,000,000 + d (the Phase 3 test shots).
Also recorded: the integer flip check on 1,000 validation shots (differing edge accumulators between
a shot and its flipped volume at the σ image; 0 means bit-exact equivariance), the agreement of the
fake-quantized float model's decisions with the golden decisions on those shots, and the integer
ranges (for HLS sizing). Writes results/phase4/eval/<arm>_s<seed>_<precision>.json and .npz (T4-SIM).
usage: python bench/phase4_eval.py ARM SEED PRECISION [--threads N]
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
from nn.grid import sector_grid, to_volume  # noqa: E402
from nn.quant import GoldenTorch, export_golden, flip_check, load_quant  # noqa: E402

GRID = (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99)
P_EVAL, VAL_SHOTS, VAL_SEED = 0.003, 200_000, 5_000_001


class IntDecider:
    """Accumulators for every local edge; 'probabilities' are replaced by a monotone score so the
    Evaluator's threshold logic can be reused: applied <=> acc > thr_int(t)."""

    def __init__(self, g: GoldenTorch, sg):
        self.g, self.sg = g, sg

    def acc(self, dets, batch=1024):
        ef = torch.from_numpy(self.sg.edge_flat)
        out = np.empty((len(dets), self.sg.n_local), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(dets), batch):
                out[s:s + batch] = self.g.edge_acc(torch.from_numpy(to_volume(self.sg, dets[s:s + batch])), ef).numpy()
        return out


def fails_at(ev, acc, net, sg, t):
    thr = net.thresholds(t)[sg.edge_channel]
    return ev.fails((acc > thr).astype(np.float32), 0.5)


def sdr_threshold(ev, acc, net, sg, target=0.5, steps=30):
    lo, hi = 1e-9, 1 - 1e-9
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if fails_at(ev, acc, net, sg, mid)[1]["syndrome_density_reduction"] > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return t, fails_at(ev, acc, net, sg, t)[1]["syndrome_density_reduction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("seed", type=int)
    ap.add_argument("precision")
    ap.add_argument("--threads", type=int, default=28)
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 9])
    ap.add_argument("--shots", type=int, nargs="+", default=[200_000, 200_000])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    name = f"{a.arm}_s{a.seed}_{a.precision}"
    q, cfg = load_quant(REPO / "results" / "phase4" / "models" / name / "model.pt")
    net = export_golden(q)
    g = GoldenTorch(net)
    out = REPO / "results" / "phase4" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sgv = sector_grid(5, P_EVAL, offsets=q.offsets)
    ev = Evaluator(sgv, VAL_SHOTS, VAL_SEED)
    acc = IntDecider(g, sgv).acc(ev.dets)
    x1k = to_volume(sgv, ev.dets[:1000])
    ef = torch.from_numpy(sgv.edge_flat)
    with torch.no_grad():
        logit = q.double().edge_logits(torch.from_numpy(x1k).double(), ef).numpy()
    q.float()
    agree = {str(t): int((np.asarray(logit > np.log(t / (1 - t))) != (acc[:1000] > net.thresholds(t)[sgv.edge_channel])).sum())
             for t in (0.5, 0.9)}
    val = {"shots": VAL_SHOTS, "seed": VAL_SEED, "mwpm_fails": int(ev.fail_mwpm.sum()), "grid": {},
           "flip_check_differing_accumulators": flip_check(net, x1k[:200], sgv),
           "fake_quant_fp64_vs_golden_decision_mismatches_1000_shots": agree,
           "ranges": {"conv_weight_max_abs": [int(np.abs(c.w).max()) for c in net.convs],
                      "head_w_max_abs": int(max(np.abs(net.w1).max(), np.abs(net.w2).max())), "head_u_max_abs": int(np.abs(net.u).max()),
                      "head_m_max_abs": int(np.abs(net.m).max()), "head_acc_max_abs_validation": float(np.abs(acc).max())}}
    for th in GRID:
        f, info = fails_at(ev, acc, net, sgv, th)
        val["grid"][str(th)] = {"fails": int(f.sum()), **info}
    best = min(GRID, key=lambda th: (val["grid"][str(th)]["fails"], -th))
    t50, s50 = sdr_threshold(ev, acc, net, sgv)
    ops = {"sdr50": t50, "t0.5": 0.5, "best": best}
    val.update(operating_points=ops, sdr50_validation_sdr=s50)
    print(name, "val", ops, "flip", val["flip_check_differing_accumulators"], "mismatch", agree, flush=True)

    res = {"arm": a.arm, "seed": a.seed, "precision": a.precision, "tier": "T4-SIM", "p": P_EVAL, "train": cfg,
           "golden": "nn.quant.GoldenTorch (integer datapath, float64 kernels, bit-equal to golden_forward)",
           "validation": val, "test": []}
    bits = {}
    for d, n in zip(a.distances, a.shots):
        sg = sector_grid(d, P_EVAL, offsets=q.offsets)
        ev = Evaluator(sg, n, 6_000_000 + d)
        acc = IntDecider(g, sg).acc(ev.dets)
        row = {"distance": d, "shots": n, "seed": 6_000_000 + d, "mwpm_fails": int(ev.fail_mwpm.sum()), "grid": {}, "operating_points": {}}
        for th in GRID:
            f, info = fails_at(ev, acc, net, sg, th)
            row["grid"][str(th)] = {"fails": int(f.sum()), **info}
        for op, th in ops.items():
            f, info = fails_at(ev, acc, net, sg, th)
            row["operating_points"][op] = {"threshold": th, "fails": int(f.sum()), **info}
            bits[f"d{d}_{op}"] = np.packbits(f)
        print(name, d, {k: v["fails"] for k, v in row["operating_points"].items()}, "mwpm", row["mwpm_fails"], flush=True)
        res["test"].append(row)
        res["seconds"] = round(time.time() - t0, 1)
        (out / f"{name}.json").write_text(json.dumps(res, indent=1))
        np.savez_compressed(out / f"{name}.npz", **bits)
    (out / f"{name}.golden.json").write_text(json.dumps(net.to_json()))


if __name__ == "__main__":
    main()
