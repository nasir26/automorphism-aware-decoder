"""Colour-aware twin contraction and weight colouring.

Twin contraction (Prakash, doi:10.20944/preprints202609.1249.v1; implemented for uncoloured graphs in
GraphCrunch) extended to vertex-coloured graphs: two vertices are coloured twins when they have the
same colour and the same open neighbourhood (false twins) or the same closed neighbourhood (true
twins). The quotient vertex of a class is coloured by (original colour, class size, clique type).

Why the decomposition carries over (sketch, checked by brute force in tests/test_coloured.py): the
transposition of two coloured twins is an automorphism of the coloured graph, because it preserves
adjacency (twins) and colour (same colour). So K = prod Sym(C) over coloured twin classes lies in
Aut(G, c). Coloured twin classes are permuted by every colour-preserving automorphism (it preserves
neighbourhoods and colours). The induced map on the quotient preserves (colour, size, clique type)
and quotient adjacency; conversely every automorphism of the coloured quotient lifts by any choice
of bijections between classes, exactly as in the uncoloured proof, because classes mapped to each
other have equal size, type and original colour. Hence Aut(G, c) = K ⋊ A with A the automorphism
group of the coloured quotient, and the vertex and edge orbit lifting formulas apply unchanged.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from autsym.graphcrunch import AutResult, nauty_aut


@dataclass
class Contraction:
    n: int
    classes: list[list[int]]
    is_clique: list[bool]
    rep: np.ndarray  # class index of each vertex
    q_edges: np.ndarray  # quotient edges (class indices), a < b
    q_colour: np.ndarray  # quotient colour index
    q_colour_key: list[tuple]  # (original colour, size, clique) of each quotient colour index

    @property
    def m(self) -> int:
        return len(self.classes)

    @property
    def log10_K(self) -> float:
        return sum(math.lgamma(len(c) + 1) for c in self.classes) / math.log(10)

    @property
    def num_transpositions(self) -> int:
        return sum(len(c) - 1 for c in self.classes)


def adjacency(n: int, edges) -> list[set[int]]:
    adj = [set() for _ in range(n)]
    for a, b in np.asarray(edges, dtype=np.int64).reshape(-1, 2).tolist():
        if a == b:
            raise ValueError("self loop")
        adj[a].add(b)
        adj[b].add(a)
    return adj


def twin_classes(n: int, adj, colour=None) -> tuple[list[list[int]], list[bool]]:
    """Coloured twin classes, mirroring GraphCrunch's ``twin_classes`` with the colour added to the
    key. Without colours it gives GraphCrunch's classes exactly."""
    col = [0] * n if colour is None else [int(c) for c in colour]
    byo, byc = defaultdict(list), defaultdict(list)
    for v in range(n):
        byo[(col[v], frozenset(adj[v]))].append(v)
        byc[(col[v], frozenset(adj[v] | {v}))].append(v)
    seen, cls, clq = set(), [], []
    for tbl, flag in ((byo, False), (byc, True)):
        for g in tbl.values():
            g = [v for v in g if v not in seen]
            if len(g) > 1:
                cls.append(sorted(g))
                clq.append(flag)
                seen.update(g)
    for v in range(n):
        if v not in seen:
            cls.append([v])
            clq.append(False)
    return cls, clq


def contract(n: int, edges, colour=None) -> Contraction:
    adj = adjacency(n, edges)
    cls, clq = twin_classes(n, adj, colour)
    rep = np.empty(n, dtype=np.int64)
    for i, c in enumerate(cls):
        rep[c] = i
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    qa, qb = rep[e[:, 0]], rep[e[:, 1]]
    keep = qa != qb
    q = np.unique(np.stack([np.minimum(qa, qb)[keep], np.maximum(qa, qb)[keep]], 1), axis=0)
    col = np.zeros(n, dtype=np.int64) if colour is None else np.asarray(colour, dtype=np.int64)
    key = [(int(col[c[0]]), len(c), bool(clq[i])) for i, c in enumerate(cls)]
    keys = sorted(set(key))
    kid = {k: i for i, k in enumerate(keys)}
    return Contraction(n=n, classes=cls, is_clique=clq, rep=rep, q_edges=q.reshape(-1, 2),
                       q_colour=np.array([kid[k] for k in key], dtype=np.int64), q_colour_key=keys)


@dataclass
class ColouredOrbits:
    contraction: Contraction
    quotient_aut: AutResult | None
    vertex_orbits: list[list[int]]
    edge_orbits: list[list[tuple[int, int]]]
    log10_order: float  # log10 |K| + log10 |A|


def _quotient_edge_orbits(q_edges: np.ndarray, gens: list[np.ndarray]) -> list[list[tuple[int, int]]]:
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(q_edges.tolist())}
    parent = list(range(len(idx)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in gens:
        for (a, b), i in idx.items():
            ga, gb = int(g[a]), int(g[b])
            j = idx[(min(ga, gb), max(ga, gb))]
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for e, i in idx.items():
        groups[find(i)].append(e)
    return list(groups.values())


def lift(c: Contraction, q_vertex_orbits: list[list[int]], q_edge_orbits: list[list[tuple[int, int]]]):
    """Vertex and edge orbits of G from those of the coloured quotient (two edge families)."""
    vo = [sorted(v for i in o for v in c.classes[i]) for o in q_vertex_orbits]
    eo = []
    for qo in q_edge_orbits:
        grp = {(min(a, b), max(a, b)) for (i, j) in qo for a in c.classes[i] for b in c.classes[j]}
        if grp:
            eo.append(sorted(grp))
    for o in q_vertex_orbits:
        grp = set()
        for i in o:
            m = c.classes[i]
            if c.is_clique[i] and len(m) > 1:
                grp.update((m[x], m[y]) for x in range(len(m)) for y in range(x + 1, len(m)))
        if grp:
            eo.append(sorted(grp))
    return vo, eo


def coloured_orbits(n: int, edges, colour=None, quotient_solver=None) -> ColouredOrbits:
    """Contract coloured twins, solve the coloured quotient, lift. ``quotient_solver(m, edges,
    colours)`` must return (vertex_orbits, generators, log10_order); default is nauty."""
    c = contract(n, edges, colour)
    if quotient_solver is None:
        res = nauty_aut(c.m, c.q_edges, c.q_colour)
        qvo, gens, lA = res.orbit_partition(), res.generators, res.log10_order
    else:
        res = None
        qvo, gens, lA = quotient_solver(c.m, c.q_edges, c.q_colour)
    qeo = _quotient_edge_orbits(c.q_edges, gens)
    vo, eo = lift(c, qvo, qeo)
    return ColouredOrbits(contraction=c, quotient_aut=res, vertex_orbits=vo, edge_orbits=eo,
                          log10_order=c.log10_K + lA)


def significant(x: np.ndarray, digits: int | None) -> np.ndarray:
    """Round to ``digits`` significant digits (None: exact binary64 values)."""
    x = np.asarray(x, dtype=np.float64)
    if digits is None:
        return x.copy()
    return np.array([float(f"{v:.{digits}g}") for v in x.tolist()])


def weight_colours(w: np.ndarray, digits: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Colour index per edge from its weight bucketed at ``digits`` significant digits, and the
    sorted distinct bucket values."""
    r = significant(w, digits)
    vals, inv = np.unique(r, return_inverse=True)
    return inv.astype(np.int64), vals


def subdivided(n_nodes: int, boundary: int | None, u, v, edge_colour=None):
    """Subdivide every edge with a midpoint vertex so edge colours become vertex colours.

    Vertices 0..n_nodes-1 keep their ids (colour 0, boundary its own colour 1); midpoint of edge i
    is n_nodes + i with colour 2 + edge_colour[i]. Returns (n, edges, colours)."""
    m = len(u)
    mids = n_nodes + np.arange(m, dtype=np.int64)
    edges = np.concatenate([np.stack([np.asarray(u), mids], 1), np.stack([np.asarray(v), mids], 1)])
    col = np.zeros(n_nodes + m, dtype=np.int64)
    if boundary is not None:
        col[boundary] = 1
    col[n_nodes:] = 2 + (np.zeros(m, dtype=np.int64) if edge_colour is None else np.asarray(edge_colour))
    return n_nodes + m, edges, col
