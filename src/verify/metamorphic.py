"""Symmetry (metamorphic) checks for any decoder with decode(syndrome) -> (edges, weight, logical).

For a group element g (a node permutation of the decoding graph, boundary fixed) and its induced
edge permutation, a minimum-weight decoder with g-invariant weights must satisfy, for every
syndrome s:

* weight(decode(s)) == weight(decode(g s));
* the correction g^-1 decode(g s) explains s, and its logical class equals that of decode(s)
  whenever the two corrections have equal weight and the observable is g-equivalent (measured, not
  assumed: see ``check``);
* the edge sets may differ when minimum-weight corrections are tied; that rate is reported apart.

The logical class of the pulled-back correction is evaluated with the graph's own observable masks
on the pulled-back edges, which is the prediction a decoder of s would make with that correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from autsym.symmetry import inverse
from qec.dem import DecodingGraph


def act_on_syndrome(perm: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(g s)[g(i)] = s[i] for detectors i."""
    n = len(s)
    out = np.zeros_like(s)
    out[perm[:n]] = s
    return out


def explains(g: DecodingGraph, edges: np.ndarray, s: np.ndarray) -> bool:
    par = np.zeros(g.n_det + 1, dtype=np.int64)
    np.add.at(par, g.u[edges], 1)
    np.add.at(par, g.v[edges], 1)
    return bool(np.array_equal(par[: g.n_det] % 2, s.astype(np.int64)))


@dataclass
class MetamorphicTally:
    shots: int = 0
    weight_equal_exact: int = 0
    weight_equal_rel1e9: int = 0
    logical_equal: int = 0
    edges_equal: int = 0
    pulled_back_explains: int = 0
    max_rel_weight_diff: float = 0.0
    failures: list = field(default_factory=list)  # first few shot indices failing an invariant

    def as_dict(self) -> dict:
        n = max(self.shots, 1)
        return {"shots": self.shots,
                "weight_equal_exact": self.weight_equal_exact, "weight_equal_rel1e9": self.weight_equal_rel1e9,
                "logical_equal": self.logical_equal, "edges_equal": self.edges_equal,
                "pulled_back_explains": self.pulled_back_explains,
                "rate_weight_equal_rel1e9": self.weight_equal_rel1e9 / n, "rate_logical_equal": self.logical_equal / n,
                "rate_edges_equal": self.edges_equal / n, "max_rel_weight_diff": self.max_rel_weight_diff,
                "first_failures": self.failures[:10]}


def check(decoder, g: DecodingGraph, perm: np.ndarray, edge_perm: np.ndarray, syndromes: np.ndarray,
          tally: MetamorphicTally | None = None) -> MetamorphicTally:
    t = tally or MetamorphicTally()
    einv = inverse(edge_perm)
    for k, s in enumerate(syndromes):
        e1, w1, l1 = decoder.decode(s)
        e2, w2, _ = decoder.decode(act_on_syndrome(perm, s))
        back = einv[e2]
        l2 = decoder.logical(back)
        t.shots += 1
        exact = w1 == w2
        rel = abs(w1 - w2) / max(abs(w1), abs(w2), 1e-300)
        t.max_rel_weight_diff = max(t.max_rel_weight_diff, rel)
        t.weight_equal_exact += exact
        t.weight_equal_rel1e9 += rel <= 1e-9
        t.logical_equal += l1 == l2
        t.edges_equal += set(e1.tolist()) == set(back.tolist())
        ok = explains(g, back, s)
        t.pulled_back_explains += ok
        if not (rel <= 1e-9 and l1 == l2 and ok):
            t.failures.append(k)
    return t
