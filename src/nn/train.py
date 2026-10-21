"""Training of the three Phase 3 arms at one fixed budget.

* ``baseline``: unconstrained PreDecoder;
* ``augmented``: unconstrained PreDecoder, each training shot replaced by its σ image with
  probability 1/2 (volume flipped with every input channel, labels moved by σ);
* ``equivariant``: σ-equivariant PreDecoder (nn.model).

Shots are fresh DEM-level samples in chunks, never reused, with a deterministic seed per chunk, as
in ``nasir26/fpga-neural-predecoder``. Loss: binary cross entropy on the logits of the local edges.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nn.grid import TrainingSampler, reflect_volume, sector_grid, to_volume
from nn.model import PreDecoder

ARMS = ("baseline", "augmented", "equivariant")


def train(arm: str, seed: int, out: Path, d: int = 5, p: float = 0.006, depth: int = 2, width: int = 64, epochs: int = 20,
          shots_per_epoch: int = 2_000_000, batch: int = 256, chunk: int = 100_000, lr: float = 1e-3, threads: int | None = None,
          log=print) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sg = sector_grid(d, p)
    sampler = TrainingSampler(sg)
    model = PreDecoder(sg.offsets, depth, width, equivariant=(arm == "equivariant"))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ef = torch.from_numpy(sg.edge_flat)
    sig = torch.from_numpy(sg.sigma_edge)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"arm": arm, "seed": seed, "d": d, "rounds": d, "p": p, "basis": "Z", "family": "stim", "epochs": epochs,
           "shots_per_epoch": shots_per_epoch, "batch": batch, "chunk": chunk, "lr": lr, "torch_threads": torch.get_num_threads(),
           "model": model.config(), "n_parameters_free": sum(q.numel() for q in model.parameters()),
           "tier": "T4-SIM"}
    hist = []
    chunk_seed = 1_000_000 * (seed + 1) + 1000 * ARMS.index(arm)
    t_all = time.time()
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
                xb, yb = x[idx], yt[idx]
                if arm == "augmented":
                    flip = torch.from_numpy(rng.random(len(idx)) < 0.5)
                    if flip.any():
                        xb = xb.clone()
                        yb = yb.clone()
                        xb[flip] = reflect_volume(xb[flip])
                        r = torch.empty_like(yb[flip])
                        r[:, sig] = yb[flip]
                        yb[flip] = r
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model.edge_logits(xb, ef), yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            done += n
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "shots": done, "seconds": round(time.time() - t0, 1)})
        log(f"{arm} seed {seed} epoch {ep} loss {hist[-1]['loss']:.6f} {hist[-1]['seconds']} s")
        torch.save({"config": model.config(), "state": model.state_dict(), "train": cfg}, out / "model.pt")
        (out / "train.json").write_text(json.dumps({**cfg, "history": hist, "seconds_total": round(time.time() - t_all, 1)}, indent=1))
    return {**cfg, "history": hist}


def load(path: Path) -> tuple[PreDecoder, dict]:
    from nn.model import from_config
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = from_config(ck["config"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["train"]
