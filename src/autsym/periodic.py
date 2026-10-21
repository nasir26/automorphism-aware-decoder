"""Per-round edge templates, translation classes and the periodic bulk graph.

Every decoding-graph edge is anchored at the earliest time layer among its detector endpoints. Its
template descriptor is the sorted pair of endpoint descriptors (stabilizer, x, y, t - anchor), with
the boundary written as (-1,), together with its weight at a stated number of significant digits.
Round r's template is the set of descriptors anchored at r. The periodic bulk graph repeats one
bulk template around P time layers with wrap-around, so time translation by one layer is an
automorphism by construction and the question is what else its weighted group contains.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from autsym.circuits import DetectorInfo
from autsym.coloured import significant, subdivided, weight_colours
from qec.dem import DecodingGraph

BOUNDARY_DESC = (-1,)


def anchors(g: DecodingGraph, det: DetectorInfo) -> np.ndarray:
    tu = det.t[np.minimum(g.u, g.n_det - 1)]
    tv = np.where(g.v < g.n_det, det.t[np.minimum(g.v, g.n_det - 1)], np.iinfo(np.int64).max)
    return np.minimum(tu, tv)


def descriptor(g: DecodingGraph, det: DetectorInfo, i: int, anchor: int) -> tuple:
    out = []
    for a in (int(g.u[i]), int(g.v[i])):
        if a == g.n_det:
            out.append(BOUNDARY_DESC)
        else:
            out.append((int(det.stab[a]), int(det.x[a]), int(det.y[a]), int(det.t[a]) - anchor))
    return tuple(sorted(out))


@dataclass
class RoundTemplates:
    by_round: dict[int, set]
    reference_round: int

    def overlap(self) -> dict[int, dict]:
        ref = self.by_round[self.reference_round]
        return {r: {"edges": len(s), "match_reference": len(s & ref)} for r, s in sorted(self.by_round.items())}

    def bulk_rounds(self) -> list[int]:
        """Rounds whose template equals the reference template exactly (weights included)."""
        ref = self.by_round[self.reference_round]
        return [r for r, s in sorted(self.by_round.items()) if s == ref]


def round_templates(g: DecodingGraph, det: DetectorInfo, digits: int | None = 9, reference_round: int | None = None) -> RoundTemplates:
    a = anchors(g, det)
    w = significant(g.weight, digits)
    by = defaultdict(set)
    for i in range(g.n_edges):
        by[int(a[i])].add((descriptor(g, det, i, int(a[i])), float(w[i])))
    if reference_round is None:
        reference_round = int(np.max(det.t)) // 2
    return RoundTemplates(by_round=dict(by), reference_round=reference_round)


def translation_classes(g: DecodingGraph, det: DetectorInfo, bulk_rounds: list[int], digits: int | None = 9) -> np.ndarray:
    """Class id per edge: edges anchored in ``bulk_rounds`` with equal descriptor and weight share a
    class (time translation); every other edge is its own class."""
    a = anchors(g, det)
    w = significant(g.weight, digits)
    bulk = set(bulk_rounds)
    key: dict = {}
    out = np.empty(g.n_edges, dtype=np.int64)
    for i in range(g.n_edges):
        k = (descriptor(g, det, i, int(a[i])), float(w[i])) if int(a[i]) in bulk else ("edge", i)
        out[i] = key.setdefault(k, len(key))
    return out


def merge_classes(cls: np.ndarray, edge_perm: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Merge classes related by an edge permutation wherever the two edges' weights are equal."""
    parent = list(range(int(cls.max()) + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(cls)):
        j = int(edge_perm[i])
        if w[i] == w[j]:
            a, b = find(int(cls[i])), find(int(cls[j]))
            if a != b:
                parent[a] = b
    roots = {}
    return np.array([roots.setdefault(find(int(c)), len(roots)) for c in cls], dtype=np.int64)


@dataclass
class PeriodicGraph:
    period: int
    nodes: list[tuple]  # (stab, x, y, t mod P); the last node is the boundary
    u: np.ndarray
    v: np.ndarray
    weight: np.ndarray

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def boundary(self) -> int:
        return self.n - 1


def periodic_bulk_graph(template: set, period: int) -> PeriodicGraph:
    """Repeat a bulk round template around ``period`` layers. Template entries are
    (descriptor, weight) with endpoint times relative to the anchor round."""
    span = max(d[3] for desc, _ in template for d in desc if d != BOUNDARY_DESC)
    if span >= period:
        raise ValueError("period too short for the template's time span")
    nodes: dict = {}
    rows = []
    for r in range(period):
        for desc, w in template:
            ends = []
            for d in desc:
                if d == BOUNDARY_DESC:
                    ends.append("B")
                else:
                    ends.append(nodes.setdefault((d[0], d[1], d[2], (r + d[3]) % period), len(nodes)))
            rows.append((ends, w))
    nb = len(nodes)
    node_list = sorted(nodes, key=nodes.get) + [BOUNDARY_DESC]
    u, v, w = [], [], []
    seen = set()
    for (a, b), wt in rows:
        a = nb if a == "B" else a
        b = nb if b == "B" else b
        e = (min(a, b), max(a, b))
        if e in seen:
            raise ValueError("wrapped template produced a parallel edge")
        seen.add(e)
        u.append(e[0])
        v.append(e[1])
        w.append(wt)
    return PeriodicGraph(period=period, nodes=node_list, u=np.array(u), v=np.array(v), weight=np.array(w))


def periodic_coloured(pg: PeriodicGraph, digits: int | None = 9):
    cols, vals = weight_colours(pg.weight, digits)
    n, e, col = subdivided(pg.n, pg.boundary, pg.u, pg.v, cols)
    return n, e, col, vals


def periodic_translation(pg: PeriodicGraph, k: int = 1) -> np.ndarray:
    """Node permutation shifting every layer by k (boundary fixed)."""
    idx = {nd: i for i, nd in enumerate(pg.nodes)}
    perm = np.arange(pg.n, dtype=np.int64)
    for i, nd in enumerate(pg.nodes[:-1]):
        perm[i] = idx[(nd[0], nd[1], nd[2], (nd[3] + k) % pg.period)]
    return perm


def periodic_reflection(pg: PeriodicGraph, sector: int) -> np.ndarray | None:
    """Point reflection (x, y, t) -> (cx - x, cy - y, -t mod P) of one sector, others fixed."""
    idx = {nd: i for i, nd in enumerate(pg.nodes)}
    pts = [nd for nd in pg.nodes[:-1] if nd[0] == sector]
    if not pts:
        return None
    cx = min(p[1] for p in pts) + max(p[1] for p in pts)
    cy = min(p[2] for p in pts) + max(p[2] for p in pts)
    perm = np.arange(pg.n, dtype=np.int64)
    for i, nd in enumerate(pg.nodes[:-1]):
        if nd[0] == sector:
            j = idx.get((sector, cx - nd[1], cy - nd[2], (-nd[3]) % pg.period))
            if j is None:
                return None
            perm[i] = j
    return perm


def periodic_is_automorphism(pg: PeriodicGraph, perm: np.ndarray, digits: int | None = 9) -> bool:
    """Weighted automorphism test on the periodic graph (weights compared at ``digits``)."""
    w = significant(pg.weight, digits)
    edges = {(int(a), int(b)): float(x) for a, b, x in zip(pg.u, pg.v, w)}
    for (a, b), x in edges.items():
        pa, pb = int(perm[a]), int(perm[b])
        if edges.get((min(pa, pb), max(pa, pb))) != x:
            return False
    return True
