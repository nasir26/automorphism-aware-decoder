"""Decoder wrapper, metamorphic invariants and the batched σ-twin path."""
import importlib.util
from pathlib import Path

import numpy as np
import pymatching
import pytest

from autsym.circuits import memory
from autsym.orbits import reflection_candidates
from autsym.symmetry import edge_permutation, inverse
from qec.dem import decoding_graph
from qec.matching import MatchingDecoder
from verify.metamorphic import act_on_syndrome, check, explains


@pytest.fixture(scope="module")
def d5():
    e = memory("nvidia", 5, 5, "X", 0.005)
    g = decoding_graph(e.circuit)
    sig = reflection_candidates(e.det, g.n_nodes)["reflect_X"]
    dets, obs = e.circuit.compile_detector_sampler(seed=5).sample(400, separate_observables=True)
    return e, g, sig, edge_permutation(g, sig), dets, obs


def test_wrapper_matches_pymatching(d5):
    e, g, _, _, dets, _ = d5
    ref = pymatching.Matching.from_detector_error_model(
        e.circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True))
    dec = MatchingDecoder(g)
    for s in dets:
        edges, w, logical = dec.decode(s)
        assert logical == int(ref.decode(s)[0])
        assert explains(g, edges, s)


def test_act_on_syndrome(d5):
    _, g, sig, _, dets, _ = d5
    s = dets[0]
    gs = act_on_syndrome(sig, s)
    assert np.array_equal(gs[sig[: g.n_det]], s)
    assert np.array_equal(act_on_syndrome(inverse(sig), gs), s)


def test_metamorphic_invariants_with_symmetrised_weights(d5):
    _, g, sig, ep, dets, _ = d5
    dec = MatchingDecoder(g, (g.weight + g.weight[ep]) / 2)
    t = check(dec, g, sig, ep, dets)
    assert t.weight_equal_rel1e9 == t.logical_equal == t.pulled_back_explains == t.shots == len(dets)


def test_true_weights_can_break_the_invariant(d5):
    # with the true (unsymmetrised) weights the weight invariant is not guaranteed; this records that
    # the check is able to see a difference rather than asserting one
    _, g, sig, ep, dets, _ = d5
    t = check(MatchingDecoder(g), g, sig, ep, dets)
    assert t.pulled_back_explains == t.shots


def test_batched_twin_equals_explicit_pullback(d5):
    e, g, sig, ep, dets, obs = d5
    spec = importlib.util.spec_from_file_location("twin", Path(__file__).resolve().parents[1] / "bench" / "phase2_sigma_twin.py")
    twin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twin)
    _, m1, m2, sinv, *_ = twin._setup(("nvidia", 5, "X", 0.005))
    p2 = m2.decode_batch(dets[:, sinv])[:, 0]
    dec = MatchingDecoder(g)
    einv = inverse(ep)
    for k, s in enumerate(dets):
        e2, _, _ = dec.decode(act_on_syndrome(sig, s))
        assert dec.logical(einv[e2]) == p2[k]
