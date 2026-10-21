"""Phase 4: fake-quantized forward, integer golden model, integer equivariance."""
import numpy as np
import pytest
import torch

from nn.grid import TrainingSampler, reflect_volume, sector_grid, to_volume
from nn.model import PreDecoder
from nn.quant import PRECISIONS, GoldenTorch, QuantPreDecoder, export_golden, fine_tune, flip_check, golden_forward


@pytest.fixture(scope="module")
def data():
    sg = sector_grid(5, 0.006)
    dets, _, _ = TrainingSampler(sg).sample(256, 31)
    return sg, to_volume(sg, dets)


def make(sg, x, equivariant, precision, seed=0):
    torch.manual_seed(seed)
    m = PreDecoder(sg.offsets, 2, 8, equivariant)
    with torch.no_grad():
        m.u.normal_(0, 0.1)
    q = QuantPreDecoder(m, precision)
    q.calibrate(torch.from_numpy(x))
    q.eval()
    return q


@pytest.mark.parametrize("precision", PRECISIONS)
def test_golden_numpy_equals_golden_torch(data, precision):
    sg, x = data
    net = export_golden(make(sg, x, False, precision))
    a = golden_forward(net, x[:16])
    b = GoldenTorch(net)(torch.from_numpy(x[:16])).numpy()
    assert np.array_equal(a, b.astype(np.int64)) and np.all(b == np.round(b))


@pytest.mark.parametrize("precision", PRECISIONS)
def test_fake_quant_decisions_match_golden(data, precision):
    # zero tolerance judged on the decision: the float64 fake-quantized model and the integer
    # golden model agree on every edge decision at threshold 0.5 and 0.9
    sg, x = data
    q = make(sg, x, True, precision).double()
    net = export_golden(q)
    ef = torch.from_numpy(sg.edge_flat)
    with torch.no_grad():
        logit = q.edge_logits(torch.from_numpy(x).double(), ef).numpy()
    acc = golden_forward(net, x).reshape(len(x), -1)[:, sg.edge_flat]
    ch = sg.edge_channel
    for t in (0.5, 0.9):
        thr = net.thresholds(t)[ch]
        assert np.array_equal(logit > np.log(t / (1 - t)), acc > thr)
    scale = (net.sw * net.s / 2 ** 16)[ch]
    np.testing.assert_allclose(acc * scale, logit, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_integer_equivariance_is_bit_exact(data, precision):
    sg, x = data
    assert flip_check(export_golden(make(sg, x, True, precision)), x[:32], sg) == 0
    assert flip_check(export_golden(make(sg, x, False, precision)), x[:32], sg) > 0


def test_integer_kernels_are_point_symmetric(data):
    sg, x = data
    net = export_golden(make(sg, x, True, "int8"))
    for c in net.convs:
        assert np.array_equal(c.w, c.w[:, :, ::-1, ::-1, ::-1])
    assert np.array_equal(net.w1, net.w2)


def test_golden_torch_on_flipped_volume(data):
    sg, x = data
    g = GoldenTorch(export_golden(make(sg, x, True, "int4")))
    ef = torch.from_numpy(sg.edge_flat)
    a = g.edge_acc(torch.from_numpy(x), ef)
    b = g.edge_acc(torch.from_numpy(reflect_volume(x)), ef)[:, torch.from_numpy(sg.sigma_edge)]
    assert torch.equal(a, b)


def test_fine_tune_smoke(tmp_path):
    sg = sector_grid(5, 0.006)
    m = PreDecoder(sg.offsets, 2, 8, True)
    q = fine_tune(m, "ternary", tmp_path, 0, epochs=1, shots_per_epoch=512, batch=128, chunk=256, threads=2, log=lambda s: None)
    assert (tmp_path / "model.pt").exists()
    assert flip_check(export_golden(q), to_volume(sg, TrainingSampler(sg).sample(8, 1)[0]), sg) == 0
