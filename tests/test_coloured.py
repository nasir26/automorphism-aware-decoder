"""Brute-force validation of colour-aware twin contraction and orbit lifting.

Mirrors GraphCrunch's tests/test_orbits.py with vertex colours added: the lifted vertex orbits, edge
orbits and group order must equal those of the full colour-preserving automorphism group found by
enumerating every permutation. The exhaustive families use a brute-force quotient solver (fast);
a random sample also goes through nauty, so the production path is covered too.
"""
import itertools
import math
import random

import pytest

from autsym.coloured import coloured_orbits, contract


def brute_group(n, edges, colour):
    es = {frozenset(e) for e in edges}
    col = list(colour)
    group = []
    for p in itertools.permutations(range(n)):
        if any(col[p[v]] != col[v] for v in range(n)):
            continue
        if all(frozenset((p[a], p[b])) in es for a, b in edges):
            group.append(p)
    return group


def orbits_of(group, n, edges):
    vo = {frozenset(p[v] for p in group) for v in range(n)}
    eo = {frozenset(tuple(sorted((p[a], p[b]))) for p in group) for a, b in edges}
    return vo, eo


def brute_quotient_solver(m, q_edges, q_colour):
    edges = [tuple(e) for e in q_edges.tolist()]
    g = brute_group(m, edges, q_colour.tolist())
    vo, _ = orbits_of(g, m, edges)
    return [sorted(o) for o in vo], [list(p) for p in g], math.log10(len(g))


def check(n, edges, colour, solver=brute_quotient_solver):
    edges = [tuple(sorted(e)) for e in edges]
    g = brute_group(n, edges, colour)
    vo, eo = orbits_of(g, n, edges)
    r = coloured_orbits(n, edges if edges else [], colour, quotient_solver=solver)
    assert {frozenset(o) for o in r.vertex_orbits} == vo
    assert {frozenset(o) for o in r.edge_orbits} == eo
    assert abs(r.log10_order - math.log10(len(g))) < 1e-9


def all_graphs(n):
    pairs = list(itertools.combinations(range(n), 2))
    for mask in range(1 << len(pairs)):
        yield [pairs[i] for i in range(len(pairs)) if mask >> i & 1]


@pytest.mark.parametrize("n,ncol", [(1, 2), (2, 2), (3, 3), (4, 3), (5, 2)])
def test_all_coloured_graphs_small(n, ncol):
    count = 0
    for edges in all_graphs(n):
        for colour in itertools.product(range(ncol), repeat=n):
            check(n, edges, colour)
            count += 1
    assert count == (2 ** (n * (n - 1) // 2)) * ncol ** n


def test_random_coloured_6_7():
    rng = random.Random(7)
    for _ in range(150):
        n = rng.choice([6, 7])
        edges = [e for e in itertools.combinations(range(n), 2) if rng.random() < rng.choice([0.3, 0.5, 0.7])]
        colour = [rng.randrange(rng.choice([1, 2, 3])) for _ in range(n)]
        check(n, edges, colour)


def blowup(base_edges, base_n, sizes, clique, colours):
    """Lexicographic blow-up: vertex i of the base becomes a block of sizes[i] vertices, a clique if
    clique[i], all with colour colours[i]; base edges become complete bipartite joins."""
    off, blocks = 0, []
    for s in sizes:
        blocks.append(list(range(off, off + s)))
        off += s
    edges = []
    for i, b in enumerate(blocks):
        if clique[i]:
            edges += list(itertools.combinations(b, 2))
    for a, b in base_edges:
        edges += [(x, y) for x in blocks[a] for y in blocks[b]]
    colour = [colours[i] for i, b in enumerate(blocks) for _ in b]
    return off, edges, colour


def test_coloured_blowups():
    rng = random.Random(11)
    done = 0
    while done < 60:
        bn = rng.choice([2, 3])
        base = [e for e in itertools.combinations(range(bn), 2) if rng.random() < 0.6]
        sizes = [rng.choice([1, 2, 3]) for _ in range(bn)]
        if sum(sizes) > 7:
            continue
        n, edges, colour = blowup(base, bn, sizes, [rng.random() < 0.5 for _ in range(bn)],
                                  [rng.randrange(2) for _ in range(bn)])
        check(n, edges, colour)
        done += 1


def test_colour_splits_twins():
    # two false twins with different colours are not coloured twins
    c = contract(3, [(0, 2), (1, 2)], [0, 1, 0])
    assert sorted(map(len, c.classes)) == [1, 1, 1]
    c = contract(3, [(0, 2), (1, 2)], [0, 0, 1])
    assert sorted(map(len, c.classes)) == [1, 2]


def test_size_and_type_colouring_is_needed():
    # independent set of 3 and triangle of 3, joined to a common hub: same size, different type
    edges = [(0, 6), (1, 6), (2, 6), (3, 4), (3, 5), (4, 5), (3, 6), (4, 6), (5, 6)]
    check(7, edges, [0] * 7)


def test_nauty_path_random_sample():
    rng = random.Random(3)
    for _ in range(40):
        n = rng.choice([5, 6, 7])
        edges = [e for e in itertools.combinations(range(n), 2) if rng.random() < 0.5]
        colour = [rng.randrange(2) for _ in range(n)]
        check(n, edges, colour, solver=None)
