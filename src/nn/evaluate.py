"""Residual decoding: pre-decoder, then PyMatching on what is left.

For every shot the model's local edges with sigmoid(logit) > threshold are applied as whole edges:
both endpoints are flipped in the syndrome and the edge's observable bit is XORed into the
prediction. PyMatching (the full decomposed DEM, both sectors) decodes the residual syndrome. Any
set of applied edges gives a valid correction, so the result is never inconsistent, only possibly
worse than PyMatching alone, which is computed on the same shots.

Test-time symmetrisation (``tts=True``): the logit of each local edge is the mean of the model's
logit on the shot and its logit on the σ image of the shot (flipped volume) at the σ image of the
edge.
"""
from __future__ import annotations

import numpy as np
import pymatching
import scipy.sparse
import torch

from nn.grid import SectorGrid, reflect_volume, to_volume
from qec.dem import decomposed_dem


def edge_probabilities(model, sg: SectorGrid, dets: np.ndarray, tts: bool = False, batch: int = 2048) -> np.ndarray:
    ef = torch.from_numpy(sg.edge_flat)
    sig = torch.from_numpy(sg.sigma_edge)
    out = np.empty((len(dets), sg.n_local), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(dets), batch):
            x = torch.from_numpy(to_volume(sg, dets[s:s + batch]))
            z = model.edge_logits(x, ef)
            if tts:
                z = 0.5 * (z + model.edge_logits(reflect_volume(x), ef)[:, sig])
            out[s:s + batch] = torch.sigmoid(z).numpy()
    return out


def residual(sg: SectorGrid, dets: np.ndarray, applied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(residual detection events, observable parity of the applied edges)."""
    a = scipy.sparse.csr_matrix(applied.astype(np.int32))
    flips = (a @ sg.incidence.astype(np.int32)).toarray() % 2
    res = dets.astype(np.uint8) ^ flips.astype(np.uint8)
    par = ((a @ sg.edge_obs.astype(np.int32)) % 2).astype(np.uint8)
    return res, par


class Evaluator:
    """Fixed shots at one (d, p): PyMatching alone, and any (model, threshold, tts) on the same shots."""

    def __init__(self, sg: SectorGrid, shots: int, seed: int):
        self.sg = sg
        self.matcher = pymatching.Matching.from_detector_error_model(decomposed_dem(sg.exp.circuit))
        d, o = sg.exp.circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        self.dets = d.astype(np.uint8)
        self.obs = o[:, 0].astype(np.uint8)
        self.fail_mwpm = (self.matcher.decode_batch(self.dets)[:, 0] != self.obs)

    def probabilities(self, model, tts: bool = False) -> np.ndarray:
        return edge_probabilities(model, self.sg, self.dets, tts)

    def fails(self, prob: np.ndarray, threshold: float) -> tuple[np.ndarray, dict]:
        applied = prob > threshold
        res, par = residual(self.sg, self.dets, applied)
        pred = self.matcher.decode_batch(res)[:, 0] ^ par
        sd = self.sg.sector_dets
        before = float(self.dets[:, sd].mean())
        after = float(res[:, sd].mean())
        return pred != self.obs, {"applied_edges_per_shot": float(applied.sum(1).mean()), "sector_density_before": before,
                                  "sector_density_after": after,
                                  "syndrome_density_reduction": 1 - after / before if before else None}


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return [c - h, c + h]


def paired(fa: np.ndarray, fb: np.ndarray, z: float = 1.96) -> dict:
    """Paired comparison of two failure vectors on the same shots: discordant counts, LER difference
    (a - b) with a normal interval on the paired difference, and an exact two-sided sign test."""
    from scipy.stats import binomtest
    n = len(fa)
    a_only = int((fa & ~fb).sum())
    b_only = int((fb & ~fa).sum())
    diff = (a_only - b_only) / n
    var = ((a_only + b_only) / n - diff ** 2) / n
    pval = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"a_only": a_only, "b_only": b_only, "ler_diff": diff, "ler_diff_ci95": [diff - z * var ** 0.5, diff + z * var ** 0.5],
            "sign_test_p": pval}
