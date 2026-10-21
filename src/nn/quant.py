"""Quantization-aware fine-tuning and the integer golden model of the pre-decoder (Phase 4).

Recipe of ``nasir26/fpga-neural-predecoder`` (its src/model/quantize.py, qat.py, golden.py),
re-derived for this architecture:

* weights: symmetric per output channel, INT8 in [-127, 127], INT4 in [-7, 7], or ternary in
  {-1, 0, 1} (threshold 0.7 mean |w| per channel, scale = mean |w| above it);
* activations after ReLU: unsigned 8 bit, scale = running max / 255 (momentum 0.9) per layer;
  the input is exact 0/1;
* biases: integers at the accumulator scale (weight scale x input scale);
* rounding round half up, floor(x + 0.5); saturation by clipping;
* requantization between layers in float64 from the exact integer accumulator, as in the earlier
  golden model;
* the decision sigmoid(logit) > t is taken on the integer accumulator, acc > floor(logit(t) / scale),
  never through a sigmoid.

Edge head in integers. With h the last 8-bit activation (scale s), for channel c with offset o:

    A = sum_f w1[c, f] h_f(v) + sum_f w2[c, f] h_f(v + o)      (w1, w2 share one scale sw_c)
    B = sum_f u[c, f] h_f(v) h_f(v + o)                          (u has its own scale su_c)
    acc = 2^16 (A + bias_int) + m_c B,   m_c = round(su_c s / sw_c 2^16)
    logit = acc sw_c s / 2^16

so the bilinear term enters through one 16-bit fixed-point multiplier per channel.

Equivariance in integers: point-symmetric float kernels quantize to point-symmetric integer kernels
(rounding is applied entrywise to equal values), and the equivariant head stores one tensor for
w1 = w2, so the integer network commutes with the volume flip bit for bit (tests/test_quant.py).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.grid import TrainingSampler, reflect_volume, sector_grid, to_volume
from nn.model import PreDecoder, shift

PRECISIONS = ("int8", "int4", "ternary")
QMAX = {"int8": 127, "int4": 7}
TERNARY = 0.7
MULT_BITS = 16


def rhu(x):
    return torch.floor(x + 0.5) if isinstance(x, torch.Tensor) else np.floor(x + 0.5)


def quantize_per_channel(w: torch.Tensor, precision: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(integer values as float tensor, per-channel scale) along axis 0; no gradient."""
    with torch.no_grad():
        flat = w.reshape(w.shape[0], -1)
        if precision == "ternary":
            a = flat.abs()
            mask = a > TERNARY * a.mean(1, keepdim=True)
            cnt = mask.sum(1, keepdim=True)
            scale = torch.where(cnt > 0, (a * mask).sum(1, keepdim=True) / cnt.clamp(min=1), torch.ones_like(cnt, dtype=w.dtype))
            q = torch.where(mask, torch.sign(flat), torch.zeros_like(flat))
        else:
            qm = QMAX[precision]
            scale = flat.abs().max(1, keepdim=True).values.clamp(min=1e-12) / qm
            q = rhu(flat / scale).clamp(-qm, qm)
        return q.reshape(w.shape), scale.reshape(-1)


def ste(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    return x + (xq - x).detach()


def _bc(scale: torch.Tensor, ndim: int) -> torch.Tensor:
    return scale.view(-1, *([1] * (ndim - 1)))


class QuantPreDecoder(nn.Module):
    """Fake-quantized forward around a PreDecoder (shares its parameters)."""

    def __init__(self, base: PreDecoder, precision: str, momentum: float = 0.9):
        super().__init__()
        if precision not in PRECISIONS:
            raise ValueError(precision)
        self.base, self.precision, self.momentum = base, precision, momentum
        self.register_buffer("act_max", torch.ones(base.depth))

    @property
    def offsets(self):
        return self.base.offsets

    def calibrate(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            h = x
            for i, conv in enumerate(self.base.convs):
                h = F.relu(conv(h))
                self.act_max[i] = h.max().clamp(min=1e-6)

    def head_weights(self):
        b = self.base
        C, Fd = b.w1.shape
        w2 = b.w1 if b.w2 is None else b.w2
        w12 = torch.cat([b.w1, w2], 1)
        q12, sw = quantize_per_channel(w12, self.precision)
        qu, su = quantize_per_channel(b.u, self.precision)
        return q12[:, :Fd], q12[:, Fd:], sw, qu, su

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s_in = torch.tensor(1.0, dtype=x.dtype)
        h = x
        for i, conv in enumerate(self.base.convs):
            w = conv.weight()
            q, sw = quantize_per_channel(w, self.precision)
            wq = ste(w, q * _bc(sw, 5))
            acc_scale = sw * s_in
            bq = ste(conv.bias, rhu(conv.bias.detach() / acc_scale) * acc_scale)
            y = F.relu(F.conv3d(h, wq, bq, padding=1))
            if self.training:
                with torch.no_grad():
                    self.act_max[i].mul_(self.momentum).add_(y.max().clamp(min=1e-6) * (1 - self.momentum))
            s_out = (self.act_max[i] / 255).clamp(min=1e-9).detach()
            h = ste(y, rhu(y.detach() / s_out).clamp(0, 255) * s_out)
            s_in = s_out
        b = self.base
        q1, q2, sw, qu, su = self.head_weights()
        s = s_in
        w1q = ste(b.w1, q1 * sw[:, None])
        w2 = b.w1 if b.w2 is None else b.w2
        w2q = ste(w2, q2 * sw[:, None])
        m = rhu(su * s / sw * 2 ** MULT_BITS)
        ueff = ste(b.u, (qu * m[:, None]) * (sw / (2 ** MULT_BITS * s))[:, None])
        acc_scale = sw * s
        bias = ste(b.head_bias, rhu(b.head_bias.detach() / acc_scale) * acc_scale)
        C, Fd = b.w1.shape
        lin1 = F.conv3d(h, w1q.view(C, Fd, 1, 1, 1), bias)
        lin2 = F.conv3d(h, w2q.view(C, Fd, 1, 1, 1))
        out = []
        for c, o in enumerate([(0, 0, 0)] + b.offsets):
            if o == (0, 0, 0):
                hb, l2 = h, lin2[:, c]
            else:
                hb, l2 = shift(h, o), shift(lin2[:, c:c + 1], o)[:, 0]
            out.append(lin1[:, c] + l2 + (h * hb * ueff[c].view(1, Fd, 1, 1, 1)).sum(1))
        return torch.stack(out, 1)

    def edge_logits(self, x: torch.Tensor, edge_flat: torch.Tensor) -> torch.Tensor:
        return self(x).flatten(1)[:, edge_flat]


# ----------------------------------------------------------------------------------------------
# integer golden model

@dataclass
class GoldenConv:
    w: np.ndarray       # int64 (out, in, 3, 3, 3)
    sw: np.ndarray      # float64 (out,)
    b: np.ndarray       # int64 (out,)
    s_in: float
    s_out: float


@dataclass
class GoldenNet:
    convs: list
    w1: np.ndarray      # int64 (C, F)
    w2: np.ndarray      # int64 (C, F)
    u: np.ndarray       # int64 (C, F)
    m: np.ndarray       # int64 (C,)
    b: np.ndarray       # int64 (C,)
    sw: np.ndarray      # float64 (C,)
    s: float
    offsets: list
    precision: str

    def thresholds(self, t: float) -> np.ndarray:
        """Integer threshold per channel: sigmoid(logit) > t  <=>  acc > thr."""
        L = float(np.log(t / (1 - t)))
        return np.floor(L * 2 ** MULT_BITS / (self.sw * self.s)).astype(np.int64)

    def to_json(self) -> dict:
        return {"precision": self.precision, "offsets": [list(o) for o in self.offsets], "s": self.s, "mult_bits": MULT_BITS,
                "convs": [{"w": c.w.tolist(), "sw": c.sw.tolist(), "b": c.b.tolist(), "s_in": c.s_in, "s_out": c.s_out}
                          for c in self.convs],
                "head": {"w1": self.w1.tolist(), "w2": self.w2.tolist(), "u": self.u.tolist(), "m": self.m.tolist(),
                         "b": self.b.tolist(), "sw": self.sw.tolist()}}


def export_golden(q: QuantPreDecoder) -> GoldenNet:
    convs = []
    s_in = 1.0
    with torch.no_grad():
        for i, conv in enumerate(q.base.convs):
            w = conv.weight().double()
            wi, sw = quantize_per_channel(w, q.precision)
            sw = sw.numpy().astype(np.float64)
            s_out = float((q.act_max[i].double() / 255).clamp(min=1e-9))
            b = rhu(conv.bias.double().numpy() / (sw * s_in)).astype(np.int64)
            convs.append(GoldenConv(w=wi.numpy().astype(np.int64), sw=sw, b=b, s_in=s_in, s_out=s_out))
            s_in = s_out
        bq = q.base
        b2 = bq.w1 if bq.w2 is None else bq.w2
        q12, sw = quantize_per_channel(torch.cat([bq.w1, b2], 1).double(), q.precision)
        qu, su = quantize_per_channel(bq.u.double(), q.precision)
        Fd = bq.w1.shape[1]
        sw = sw.numpy()
        m = rhu(su.numpy() * s_in / sw * 2 ** MULT_BITS).astype(np.int64)
        b = rhu(bq.head_bias.double().numpy() / (sw * s_in)).astype(np.int64)
    return GoldenNet(convs=convs, w1=q12[:, :Fd].numpy().astype(np.int64), w2=q12[:, Fd:].numpy().astype(np.int64),
                     u=qu.numpy().astype(np.int64), m=m, b=b, sw=sw, s=s_in, offsets=list(bq.offsets), precision=q.precision)


def _shift_np(h: np.ndarray, o) -> np.ndarray:
    out = np.zeros_like(h)
    T, H, W = h.shape[-3:]
    src = [slice(max(k, 0), n + min(k, 0)) for k, n in zip(o, (T, H, W))]
    dst = [slice(max(-k, 0), n - max(k, 0)) for k, n in zip(o, (T, H, W))]
    out[..., dst[0], dst[1], dst[2]] = h[..., src[0], src[1], src[2]]
    return out


def golden_forward(net: GoldenNet, x: np.ndarray) -> np.ndarray:
    """Reference integer datapath in NumPy int64. x: (B, 4, T, H, W) in {0, 1}. Returns the head
    accumulators, int64 (B, C, T, H, W)."""
    h = x.astype(np.int64)
    for c in net.convs:
        B, Ci, T, H, W = h.shape
        hp = np.zeros((B, Ci, T + 2, H + 2, W + 2), dtype=np.int64)
        hp[:, :, 1:-1, 1:-1, 1:-1] = h
        acc = np.zeros((B, c.w.shape[0], T, H, W), dtype=np.int64)
        for dt in range(3):
            for dy in range(3):
                for dx in range(3):
                    acc += np.einsum("oi,bithw->bothw", c.w[:, :, dt, dy, dx], hp[:, :, dt:dt + T, dy:dy + H, dx:dx + W])
        acc = np.maximum(acc + c.b[None, :, None, None, None], 0)
        val = acc.astype(np.float64) * (c.sw * c.s_in)[None, :, None, None, None] / c.s_out
        h = np.clip(np.floor(val + 0.5), 0, 255).astype(np.int64)
    out = []
    for k, o in enumerate([(0, 0, 0)] + list(net.offsets)):
        hb = h if tuple(o) == (0, 0, 0) else _shift_np(h, o)
        A = np.einsum("f,bfthw->bthw", net.w1[k], h) + np.einsum("f,bfthw->bthw", net.w2[k], hb)
        Bt = np.einsum("f,bfthw->bthw", net.u[k], h * hb)
        out.append((A + net.b[k]) * 2 ** MULT_BITS + net.m[k] * Bt)
    return np.stack(out, 1)


class GoldenTorch:
    """The same integer datapath with float64 torch kernels, used for large evaluations. Every
    intermediate is an integer below 2^53, so float64 sums are exact; tests check bit equality
    with golden_forward."""

    def __init__(self, net: GoldenNet):
        self.net = net
        self.w = [torch.from_numpy(c.w.astype(np.float64)) for c in net.convs]
        self.b = [torch.from_numpy(c.b.astype(np.float64)) for c in net.convs]
        self.req = [torch.from_numpy(c.sw * c.s_in).view(1, -1, 1, 1, 1) for c in net.convs]
        self.s_out = [c.s_out for c in net.convs]
        C, Fd = net.w1.shape
        self.w1 = torch.from_numpy(net.w1.astype(np.float64)).view(C, Fd, 1, 1, 1)
        self.w2 = torch.from_numpy(net.w2.astype(np.float64)).view(C, Fd, 1, 1, 1)
        self.u = torch.from_numpy(net.u.astype(np.float64))
        self.m = net.m.astype(np.float64)
        self.hb = torch.from_numpy(net.b.astype(np.float64))
        bound = 2.0 ** 53
        worst = (255 * 255 * np.abs(net.u).sum(1).max() * np.abs(net.m).max()
                 + 2 ** MULT_BITS * (255 * np.abs(net.w1).sum(1).max() * 2 + np.abs(net.b).max()))
        if worst >= bound:
            raise AssertionError("head accumulator may exceed float64's exact integer range")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = x.double()
        for w, b, r, so in zip(self.w, self.b, self.req, self.s_out):
            acc = F.relu(F.conv3d(h, w, b, padding=1))
            h = torch.clamp(torch.floor(acc * r / so + 0.5), 0, 255)
        A1 = F.conv3d(h, self.w1)
        A2 = F.conv3d(h, self.w2)
        out = []
        for k, o in enumerate([(0, 0, 0)] + list(self.net.offsets)):
            if tuple(o) == (0, 0, 0):
                hb, a2 = h, A2[:, k]
            else:
                hb, a2 = shift(h, tuple(o)), shift(A2[:, k:k + 1], tuple(o))[:, 0]
            Bt = (h * hb * self.u[k].view(1, -1, 1, 1, 1)).sum(1)
            out.append((A1[:, k] + a2 + self.hb[k]) * 2 ** MULT_BITS + self.m[k] * Bt)
        return torch.stack(out, 1)

    def edge_acc(self, x: torch.Tensor, edge_flat: torch.Tensor) -> torch.Tensor:
        return self(x).flatten(1)[:, edge_flat]


# ----------------------------------------------------------------------------------------------
# fine-tuning

def fine_tune(fp_model: PreDecoder, precision: str, out: Path, seed: int, d: int = 5, p: float = 0.006, epochs: int = 3,
              shots_per_epoch: int = 500_000, batch: int = 256, chunk: int = 100_000, lr: float = 1e-4, threads: int | None = None,
              log=print, train_meta: dict | None = None) -> QuantPreDecoder:
    """QAT fine-tuning from a trained FP32 model (earlier recipe: 3 epochs x 500,000 shots, lr 1e-4).
    The arm's own training rule is kept: no augmentation here (baseline and equivariant only)."""
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sg = sector_grid(d, p, offsets=fp_model.offsets)
    sampler = TrainingSampler(sg)
    q = QuantPreDecoder(fp_model, precision)
    cal, _, _ = sampler.sample(4096, 10_000_000 + seed)
    q.calibrate(torch.from_numpy(to_volume(sg, cal)))
    q.train()
    opt = torch.optim.Adam(q.base.parameters(), lr=lr)
    ef = torch.from_numpy(sg.edge_flat)
    out.mkdir(parents=True, exist_ok=True)
    hist = []
    chunk_seed = 20_000_000 + 1_000 * seed + 100 * PRECISIONS.index(precision)
    cfg = {"precision": precision, "seed": seed, "d": d, "p": p, "epochs": epochs, "shots_per_epoch": shots_per_epoch, "batch": batch,
           "lr": lr, "model": fp_model.config(), "fp32": train_meta or {}, "tier": "T4-SIM"}
    for ep in range(epochs):
        t0 = time.time()
        losses, done = [], 0
        while done < shots_per_epoch:
            n = min(chunk, shots_per_epoch - done)
            chunk_seed += 1
            dets, _, y = sampler.sample(n, chunk_seed)
            x = torch.from_numpy(to_volume(sg, dets))
            yt = torch.from_numpy(y).float()
            perm = torch.from_numpy(rng.permutation(n))
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(q.edge_logits(x[idx], ef), yt[idx])
                loss.backward()
                opt.step()
                losses.append(loss.item())
            done += n
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)})
        log(f"qat {precision} seed {seed} epoch {ep} loss {hist[-1]['loss']:.6f} {hist[-1]['seconds']} s")
        q.eval()
        torch.save({"config": fp_model.config(), "state": fp_model.state_dict(), "act_max": q.act_max, "precision": precision,
                    "train": cfg}, out / "model.pt")
        (out / "train.json").write_text(json.dumps({**cfg, "history": hist}, indent=1))
        q.train()
    q.eval()
    return q


def load_quant(path: Path) -> tuple[QuantPreDecoder, dict]:
    from nn.model import from_config
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = from_config(ck["config"])
    m.load_state_dict(ck["state"])
    q = QuantPreDecoder(m, ck["precision"])
    q.act_max.copy_(ck["act_max"])
    q.eval()
    return q, ck["train"]


def flip_check(net: GoldenNet, x: np.ndarray, sg) -> int:
    """Number of edge accumulators that differ between a shot and its flipped volume at the σ image
    (0 for an exactly equivariant integer network)."""
    a = golden_forward(net, x).reshape(len(x), -1)[:, sg.edge_flat]
    b = golden_forward(net, reflect_volume(x)).reshape(len(x), -1)[:, sg.edge_flat][:, sg.sigma_edge]
    return int((a != b).sum())
