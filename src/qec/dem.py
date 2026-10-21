"""Detector error models and the matching (decoding) graph.

The decoding graph is the graph PyMatching builds from the decomposed DEM: one vertex per detector
plus one boundary vertex (index ``n_det``), one edge per merged graphlike mechanism, with its
probability, its weight ``log((1 - p) / p)`` and the observables it flips.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymatching
import stim

DEM_OPTIONS = dict(approximate_disjoint_errors=True)


@dataclass
class DecodingGraph:
    n_det: int
    u: np.ndarray  # int64, u < v
    v: np.ndarray  # int64, boundary edges have v == n_det
    prob: np.ndarray
    weight: np.ndarray
    obs: np.ndarray  # uint64 bit mask of flipped observables

    @property
    def boundary(self) -> int:
        return self.n_det

    @property
    def n_nodes(self) -> int:
        return self.n_det + 1

    @property
    def n_edges(self) -> int:
        return len(self.u)

    def edge_index(self) -> dict[tuple[int, int], int]:
        return {(int(a), int(b)): i for i, (a, b) in enumerate(zip(self.u, self.v))}


def decomposed_dem(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    return circuit.detector_error_model(decompose_errors=True, **DEM_OPTIONS)


def raw_dem(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """Undecomposed, unmerged DEM as Stim writes it, loops flattened."""
    return circuit.detector_error_model(decompose_errors=False, **DEM_OPTIONS)


def matching(circuit: stim.Circuit) -> pymatching.Matching:
    return pymatching.Matching.from_detector_error_model(decomposed_dem(circuit))


def decoding_graph(circuit: stim.Circuit) -> DecodingGraph:
    m = matching(circuit)
    nd = circuit.num_detectors
    rows = []
    for a, b, attr in m.edges():
        b = nd if b is None else b
        a, b = min(a, b), max(a, b)
        mask = 0
        for f in attr.get("fault_ids", set()):
            mask |= 1 << int(f)
        rows.append((a, b, attr["error_probability"], attr["weight"], mask))
    rows.sort()
    u = np.array([r[0] for r in rows], dtype=np.int64)
    v = np.array([r[1] for r in rows], dtype=np.int64)
    if len(set(zip(u.tolist(), v.tolist()))) != len(rows):
        raise AssertionError("parallel edges in the matching graph")
    return DecodingGraph(
        n_det=nd, u=u, v=v,
        prob=np.array([r[2] for r in rows], dtype=np.float64),
        weight=np.array([r[3] for r in rows], dtype=np.float64),
        obs=np.array([r[4] for r in rows], dtype=np.uint64),
    )


def merged_raw_dem(circuit: stim.Circuit) -> dict[tuple[tuple[int, ...], int], float]:
    """Symptom set (sorted detectors, observable mask) -> merged probability, merging repeated
    lines of the raw DEM as independent mechanisms: p = (1 - prod(1 - 2 p_i)) / 2."""
    acc: dict[tuple[tuple[int, ...], int], float] = {}
    for ins in raw_dem(circuit).flattened():
        if ins.type != "error":
            continue
        dets, mask = [], 0
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                mask ^= 1 << t.val
        key = (tuple(sorted(dets)), mask)
        q = acc.get(key, 0.0)
        p = ins.args_copy()[0]
        acc[key] = q * (1 - p) + p * (1 - q)
    return acc


def raw_dem_line_count(circuit: stim.Circuit) -> int:
    return sum(1 for ins in raw_dem(circuit).flattened() if ins.type == "error")
