"""Decoding graph symmetry: sigma, weights, templates, periodic bulk graph, tensor maps."""
import numpy as np
import pytest

from autsym.circuits import X_STAB, memory
from autsym.coloured import significant
from autsym.graphcrunch import nauty_aut
from autsym.orbits import reflection_candidates
from autsym.periodic import periodic_bulk_graph, periodic_coloured, round_templates
from autsym.symmetry import compose, edge_permutation, inverse, is_automorphism, tensor_map
from qec.dem import decoding_graph


@pytest.fixture(scope="module")
def nv13():
    e = memory("nvidia", 13, 13, "X", 0.003)
    return e, decoding_graph(e.circuit)


def test_regression_bare_group(nv13):
    e, g = nv13
    r = nauty_aut(g.n_nodes, np.stack([g.u, g.v], 1))
    assert (g.n_nodes, g.n_edges, r.order, r.num_orbits) == (2101, 10872, 2, 1555)
    (s,) = r.generators
    assert np.array_equal(s, reflection_candidates(e.det, g.n_nodes)["reflect_X"])


def test_sigma_properties(nv13):
    e, g = nv13
    s = reflection_candidates(e.det, g.n_nodes)["reflect_X"]
    assert is_automorphism(g, s)
    assert not is_automorphism(g, s, weighted=True, digits=9)
    assert not is_automorphism(g, s, observables=True)
    assert np.array_equal(compose(s, s), np.arange(g.n_nodes))
    assert np.array_equal(inverse(s), s)
    ep = edge_permutation(g, s)
    flip = np.flatnonzero(g.obs != 0)
    assert len(flip) == 91 and np.all(g.v[flip] == g.boundary) and np.all(g.obs[ep[flip]] == 0)


def test_identity_and_a_non_automorphism(nv13):
    _, g = nv13
    ident = np.arange(g.n_nodes)
    assert is_automorphism(g, ident, weighted=True, digits=None, observables=True)
    swap = ident.copy()
    swap[[0, 1]] = [1, 0]
    assert not is_automorphism(g, swap)


def test_sigma_is_exact_flip_on_halved_grid(nv13):
    e, g = nv13
    s = reflection_candidates(e.det, g.n_nodes)["reflect_X"]
    half = np.stack([e.det.t, e.det.x // 2, e.det.y // 2], 1)
    tm = tensor_map(s, half, "halved", e.det.stab == X_STAB)
    assert dict(tm.sums) == {(12, 13, 13): 1092}


def test_sigma_on_nvidia_grid_is_not_a_flip(nv13):
    e, g = nv13
    s = reflection_candidates(e.det, g.n_nodes)["reflect_X"]
    c = np.stack([e.det.t, e.nvidia_cell[:, 1], e.nvidia_cell[:, 2]], 1)
    tm = tensor_map(s, c, "nvidia", e.det.stab == X_STAB)
    assert dict(tm.sums) == {(12, 11, 11): 936, (12, 11, 12): 156}


def test_templates_and_periodic_group(nv13):
    e, g = nv13
    rt = round_templates(g, e.det, 9, reference_round=6)
    assert rt.bulk_rounds() == list(range(2, 12))
    ov = rt.overlap()
    assert (ov[6]["edges"], ov[0]["match_reference"], ov[1]["match_reference"]) == (912, 11, 731)
    pg = periodic_bulk_graph(rt.by_round[6], 8)
    n, E, col, vals = periodic_coloured(pg, 9)
    assert len(vals) == 17
    assert nauty_aut(n, E, col).order == 256


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("d", [3, 5])
def test_stim_generator_reflections_are_flips_on_halved_grid(basis, d):
    e = memory("stim", d, d, basis, 0.003)
    g = decoding_graph(e.circuit)
    half = np.stack([e.det.t, e.det.x // 2, e.det.y // 2], 1)
    for name, p in reflection_candidates(e.det, g.n_nodes).items():
        if name == "reflect_both" or not is_automorphism(g, p):
            continue
        moved = p[: g.n_det] != np.arange(g.n_det)
        tm = tensor_map(p, half, "halved", moved)
        assert list(tm.sums) == [(d, d, d)]


def test_significant_digits():
    x = np.array([1.23456789012, 1.23456789049, 2.0])
    assert len(np.unique(significant(x, 9))) == 2
    assert len(np.unique(significant(x, None))) == 3
