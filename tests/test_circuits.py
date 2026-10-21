"""Circuit builders and detector geometry."""
import numpy as np
import pytest

from autsym.circuits import X_STAB, Z_STAB, memory


@pytest.mark.parametrize("family", ["nvidia", "stim"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_geometry_is_injective_and_complete(family, basis):
    e = memory(family, 5, 5, basis, 0.003)
    d = e.det
    keys = set(zip(d.stab.tolist(), d.x.tolist(), d.y.tolist(), d.t.tolist()))
    assert len(keys) == d.n == e.circuit.num_detectors
    # halved coordinates fit the (d + 1) x (d + 1) grid of fpga-neural-predecoder
    assert d.x.min() >= 0 and d.x.max() // 2 <= 5 and d.y.max() // 2 <= 5


def test_nvidia_d13_layers():
    e = memory("nvidia", 13, 13, "X", 0.003)
    d = e.det
    assert e.circuit.num_detectors == 2100
    assert sorted(set(d.t[d.stab == X_STAB].tolist())) == list(range(13))
    assert sorted(set(d.t[d.stab == Z_STAB].tolist())) == list(range(1, 13))
    assert (d.x[d.stab == X_STAB].min(), d.x[d.stab == X_STAB].max()) == (2, 24)
    assert (d.y[d.stab == X_STAB].min(), d.y[d.stab == X_STAB].max()) == (0, 26)
    assert np.all(np.bincount(d.t[d.stab == X_STAB]) == 84)


def test_nvidia_import_leaves_our_qec_package():
    import qec.dem  # noqa: F401  this repository's package

    memory("nvidia", 3, 3, "X", 0.003)
    import qec

    assert hasattr(qec, "__path__") and any("automorphism-aware-decoder" in p for p in qec.__path__)
