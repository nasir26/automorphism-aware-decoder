"""Fully convolutional 3D pre-decoder with a per-local-edge head, in a plain and a σ-equivariant form.

Both forms share one architecture: ``depth`` hidden 3x3x3 convolutions (same padding, ReLU), so the
receptive field of a hidden feature is R = 2 depth + 1, then an edge head. For edge channel c with
offset o_c the logit at anchor v is

    g_c(a, b) = w1_c . a + w2_c . b + u_c . (a * b) + bias_c,   a = h(v), b = h(v + o_c)

with h the last hidden feature map (zero outside the grid). Channel 0 (boundary edge) has o = 0.

The equivariant form constrains the parameters so that the network commutes with σ, the flip of all
three grid axes of the input volume (``nn.grid.reflect_volume``):

* every hidden kernel is point symmetric, W[o, i, k] = W[o, i, -k]; the constraint is imposed by
  projecting free parameters, W = (V + V[..., ::-1, ::-1, ::-1]) / 2;
* the head is symmetric in its two endpoints, w1_c = w2_c.

Then logit(σx) at the image of an edge equals logit(x) at the edge, to rounding (tests/test_nn.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.grid import INPUT_CHANNELS


class Conv3x3x3(nn.Module):
    def __init__(self, c_in: int, c_out: int, equivariant: bool):
        super().__init__()
        self.v = nn.Parameter(torch.empty(c_out, c_in, 3, 3, 3))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.kaiming_uniform_(self.v, a=5 ** 0.5)
        bound = 1 / (c_in * 27) ** 0.5
        nn.init.uniform_(self.bias, -bound, bound)
        self.equivariant = equivariant

    def weight(self) -> torch.Tensor:
        if not self.equivariant:
            return self.v
        return 0.5 * (self.v + self.v.flip(2, 3, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self.weight(), self.bias, padding=1)


def shift(h: torch.Tensor, o: tuple[int, int, int]) -> torch.Tensor:
    """out[..., v] = h[..., v + o], zero where v + o is outside the grid."""
    T, H, W = h.shape[-3:]
    pads = []
    for size, k in zip((W, H, T), (o[2], o[1], o[0])):
        pads += [max(-k, 0), max(k, 0)]
    hp = F.pad(h, pads)
    st = [max(-k, 0) + k for k in o]
    return hp[..., st[0]:st[0] + T, st[1]:st[1] + H, st[2]:st[2] + W]


class PreDecoder(nn.Module):
    def __init__(self, offsets: list[tuple[int, int, int]], depth: int = 2, width: int = 64, equivariant: bool = False):
        super().__init__()
        self.offsets = [tuple(int(x) for x in o) for o in offsets]
        self.depth, self.width, self.equivariant = depth, width, equivariant
        layers = []
        c_in = INPUT_CHANNELS
        for _ in range(depth):
            layers.append(Conv3x3x3(c_in, width, equivariant))
            c_in = width
        self.convs = nn.ModuleList(layers)
        C = 1 + len(self.offsets)
        self.w1 = nn.Parameter(torch.randn(C, width) / width ** 0.5)
        self.w2 = None if equivariant else nn.Parameter(torch.randn(C, width) / width ** 0.5)
        self.u = nn.Parameter(torch.zeros(C, width))
        self.head_bias = nn.Parameter(torch.full((C,), -4.0))

    @property
    def R(self) -> int:
        return 2 * self.depth + 1

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, T, H, W) -> logits (B, C, T, H, W)."""
        h = self.features(x)
        C, F_ = self.w1.shape
        w2 = self.w1 if self.w2 is None else self.w2
        lin1 = F.conv3d(h, self.w1.view(C, F_, 1, 1, 1), self.head_bias)
        lin2 = F.conv3d(h, w2.view(C, F_, 1, 1, 1))
        out = []
        for c, o in enumerate([(0, 0, 0)] + self.offsets):
            if o == (0, 0, 0):
                b, l2 = h, lin2[:, c]
            else:
                b, l2 = shift(h, o), shift(lin2[:, c:c + 1], o)[:, 0]
            out.append(lin1[:, c] + l2 + (h * b * self.u[c].view(1, F_, 1, 1, 1)).sum(1))
        return torch.stack(out, 1)

    def edge_logits(self, x: torch.Tensor, edge_flat: torch.Tensor) -> torch.Tensor:
        """Logits gathered at the local edges: (B, E)."""
        return self(x).flatten(1)[:, edge_flat]

    def config(self) -> dict:
        return {"offsets": self.offsets, "depth": self.depth, "width": self.width, "equivariant": self.equivariant}


def from_config(cfg: dict) -> PreDecoder:
    return PreDecoder([tuple(o) for o in cfg["offsets"]], cfg["depth"], cfg["width"], cfg["equivariant"])
