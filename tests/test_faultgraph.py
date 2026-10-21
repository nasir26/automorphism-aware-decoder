"""Fault graph against Stim's own detector error model."""
import collections

import pytest
import stim

from autsym.circuits import memory
from autsym.faultgraph import _independent_components, fault_graph, merged_probability
from qec.dem import merged_raw_dem, raw_dem_line_count


def _bell_pc1(args):
    c = stim.Circuit(f"""R 0 1
H 0
CX 0 1
PAULI_CHANNEL_1({','.join(map(str, args))}) 0
MPP X0*X1 Z0*Z1
DETECTOR rec[-2]
DETECTOR rec[-1]""")
    dem = c.detector_error_model(approximate_disjoint_errors=True)
    got = {tuple(sorted(t.val for t in i.targets_copy())): i.args_copy()[0] for i in dem.flattened() if i.type == "error"}
    return got[(1,)], got[(0, 1)], got[(0,)]  # X flips ZZ only, Y both, Z flips XX only


@pytest.mark.parametrize("args", [(0.001, 0.001, 0.001), (0.001, 0.002, 0.0005), (0.0002, 0.0001, 0.0007)])
def test_pauli_channel_1_is_converted_exactly(args):
    assert _bell_pc1(args) == pytest.approx(_independent_components(args), rel=1e-12)


def test_pauli_channel_2_components_are_listed_values_and_add_within_one_application():
    c = stim.Circuit("""R 0 1
PAULI_CHANNEL_2(0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002) 0 1
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1]""")
    dem = c.detector_error_model(approximate_disjoint_errors=True)
    # four disjoint components flip D0 alone (X or Y on qubit 0, I or Z on qubit 1): they add
    vals = [i.args_copy()[0] for i in dem.flattened() if i.type == "error"]
    assert len(vals) == 3 and all(v == pytest.approx(0.0008, rel=1e-12) for v in vals)


@pytest.mark.parametrize("family,d,basis", [("nvidia", 3, "X"), ("nvidia", 5, "Z"), ("stim", 3, "Z"), ("stim", 5, "X")])
def test_fault_classes_equal_merged_dem(family, d, basis):
    e = memory(family, d, d, basis, 0.003)
    fg = fault_graph(e.circuit)
    groups = collections.defaultdict(list)
    for i, k in enumerate(fg.footprint):
        groups[k].append(i)
    ref = merged_raw_dem(e.circuit)
    assert set(groups) == set(ref)
    for k, members in groups.items():
        assert merged_probability(fg, members) == pytest.approx(ref[k], rel=1e-12)
    assert raw_dem_line_count(e.circuit) >= len(ref)


def test_nvidia_d13_counts():
    e = memory("nvidia", 13, 13, "X", 0.003)
    fg = fault_graph(e.circuit)
    assert (fg.n_fault, fg.n, len(fg.edges())) == (121239, 123340, 339920)
    assert len(set(fg.footprint)) == 40073
    assert raw_dem_line_count(e.circuit) == 45392
