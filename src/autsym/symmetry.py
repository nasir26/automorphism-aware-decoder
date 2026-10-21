"""Group elements of a decoding graph as detector permutations, edge permutations and tensor maps.

A group element is stored as a permutation ``perm`` of the decoding graph's nodes (detectors and the
boundary, length ``n_det + 1``). Everything else is derived from it and checked, never assumed:
the induced edge permutation, whether weights and observable flips are preserved, and how the
element acts on each tensor layout.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from autsym.coloured import significant
from qec.dem import DecodingGraph


def edge_permutation(g: DecodingGraph, perm: np.ndarray) -> np.ndarray | None:
    """Edge index image of every edge, or None if ``perm`` does not map edges onto edges."""
    idx = g.edge_index()
    pu, pv = perm[g.u], perm[g.v]
    out = np.empty(g.n_edges, dtype=np.int64)
    for i, (a, b) in enumerate(zip(pu.tolist(), pv.tolist())):
        j = idx.get((min(a, b), max(a, b)))
        if j is None:
            return None
        out[i] = j
    if len(set(out.tolist())) != g.n_edges:
        return None
    return out


def is_automorphism(g: DecodingGraph, perm: np.ndarray, weighted: bool = False, digits: int | None = 9,
                    observables: bool = False) -> bool:
    """True if ``perm`` (on detectors plus boundary) is an automorphism of the decoding graph.

    weighted: edge weights must also be preserved, compared at ``digits`` significant digits
    (None: exact binary64 equality). observables: each edge's observable mask must be preserved."""
    perm = np.asarray(perm, dtype=np.int64)
    if perm.shape != (g.n_nodes,) or sorted(perm.tolist()) != list(range(g.n_nodes)):
        return False
    if perm[g.boundary] != g.boundary:
        return False
    ep = edge_permutation(g, perm)
    if ep is None:
        return False
    if weighted:
        w = significant(g.weight, digits)
        if not np.array_equal(w[ep], w):
            return False
    if observables and not np.array_equal(g.obs[ep], g.obs):
        return False
    return True


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a o b)[i] = a[b[i]]."""
    return np.asarray(a)[np.asarray(b)]


def inverse(a: np.ndarray) -> np.ndarray:
    inv = np.empty_like(a)
    inv[a] = np.arange(len(a))
    return inv


@dataclass
class TensorMap:
    """How a detector permutation acts on a grid layout: for every moved cell, the sums
    (coordinate + image coordinate) per axis. A reflection i -> c - i shows as one constant sum c."""

    layout: str
    sums: Counter  # (sum axis 0, sum axis 1, sum axis 2) -> number of detectors
    fixed_axes: Counter  # tuple of axes left unchanged -> number of detectors
    cells: int

    def as_dict(self) -> dict:
        return {"layout": self.layout, "cells": self.cells,
                "coordinate_sums": {str(k): v for k, v in self.sums.most_common()},
                "unchanged_axes": {str(k): v for k, v in self.fixed_axes.most_common()}}


def tensor_map(perm: np.ndarray, coords: np.ndarray, layout: str, mask: np.ndarray | None = None) -> TensorMap:
    """coords: (n_det, k) grid coordinates of each detector (e.g. (t, row, col)). Only detectors in
    ``mask`` (default: all moved detectors) are summarised."""
    n = len(coords)
    p = np.asarray(perm[:n])
    sel = np.arange(n) if mask is None else np.flatnonzero(mask)
    sums, fixed = Counter(), Counter()
    for i in sel.tolist():
        j = int(p[i])
        a, b = coords[i], coords[j]
        sums[tuple(int(x) for x in a + b)] += 1
        fixed[tuple(k for k in range(len(a)) if a[k] == b[k])] += 1
    return TensorMap(layout=layout, sums=sums, fixed_axes=fixed, cells=len(sel))
