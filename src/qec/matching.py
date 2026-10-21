"""PyMatching behind the interface every decoder in this project exposes:

    decode(syndrome) -> (edges, weight, logical)

``edges`` are indices into the DecodingGraph's edge arrays (the correction), ``weight`` is their total
weight under the decoder's weights (``math.fsum``, so it does not depend on summation order), and
``logical`` is the XOR of the observable masks of those edges. The weights may differ from the
graph's own (symmetrised or perturbed weights); the observable masks never do.
"""
from __future__ import annotations

import math

import numpy as np
import pymatching

from qec.dem import DecodingGraph


class MatchingDecoder:
    def __init__(self, g: DecodingGraph, weights: np.ndarray | None = None):
        self.g = g
        self.w = np.asarray(g.weight if weights is None else weights, dtype=np.float64)
        m = pymatching.Matching()
        for i, (a, b) in enumerate(zip(g.u.tolist(), g.v.tolist())):
            mask = int(g.obs[i])
            fid = {k for k in range(64) if mask >> k & 1}
            p = float(min(max(g.prob[i], 1e-300), 0.5))
            if b == g.n_det:
                m.add_boundary_edge(a, weight=float(self.w[i]), fault_ids=fid, error_probability=p)
            else:
                m.add_edge(a, b, weight=float(self.w[i]), fault_ids=fid, error_probability=p)
        if m.num_detectors < g.n_det:
            m.set_boundary_nodes(set())  # keep the detector count even if trailing detectors have no edges
        self.m = m
        self._idx = g.edge_index()
        self._obs = g.obs.astype(np.int64)

    def edges_from_pairs(self, pairs: np.ndarray) -> np.ndarray:
        nd = self.g.n_det
        out = np.empty(len(pairs), dtype=np.int64)
        for k, (a, b) in enumerate(pairs.tolist()):
            b = nd if b == -1 else b
            a = nd if a == -1 else a
            out[k] = self._idx[(min(a, b), max(a, b))]
        return out

    def decode(self, syndrome: np.ndarray) -> tuple[np.ndarray, float, int]:
        pairs = self.m.decode_to_edges_array(np.asarray(syndrome, dtype=np.uint8))
        e = self.edges_from_pairs(pairs)
        return e, math.fsum(self.w[e].tolist()), self.logical(e)

    def logical(self, edges: np.ndarray) -> int:
        x = 0
        for v in self._obs[edges].tolist():
            x ^= v
        return x

    def weight_of(self, edges: np.ndarray, weights: np.ndarray | None = None) -> float:
        w = self.w if weights is None else weights
        return math.fsum(np.asarray(w)[edges].tolist())
