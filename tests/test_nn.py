"""Phase 3 building blocks: sector grid, targets, σ action, equivariance, residual decoding."""
import numpy as np
import pytest
import torch

from nn.evaluate import Evaluator, edge_probabilities, paired, residual
from nn.grid import TrainingSampler, offset_catalogue, reflect_shots, reflect_volume, sector_grid, to_volume
from nn.model import PreDecoder, shift
from nn.train import train


@pytest.fixture(scope="module")
def sg5():
    return sector_grid(5, 0.006)


@pytest.fixture(scope="module")
def shots(sg5):
    return TrainingSampler(sg5).sample(2000, 7)


def test_catalogue_is_the_same_at_every_distance(sg5):
    for d in (3, 7, 9):
        s = sector_grid(d, 0.003)
        assert offset_catalogue(s.g, s.cell, s.sector) == sg5.offsets


def test_targets_reproduce_syndrome_and_observable(sg5, shots):
    dets, obs, y = shots
    syn = (y.astype(np.int64) @ sg5.incidence.astype(np.int64).toarray()) % 2
    assert np.array_equal(syn[:, sg5.sector_dets], dets[:, sg5.sector_dets])
    assert np.array_equal((y.astype(np.int64) @ sg5.edge_obs) % 2, obs)
    # applying the true edges removes every sector detection event and gives the observable
    res, par = residual(sg5, dets, y.astype(bool))
    assert not res[:, sg5.sector_dets].any()
    assert np.array_equal(par, obs)


def test_sigma_is_an_involution_on_edges_and_moves_volumes(sg5, shots):
    assert np.array_equal(sg5.sigma_edge[sg5.sigma_edge], np.arange(sg5.n_local))
    dets, _, y = shots
    d2, y2 = reflect_shots(sg5, dets[:50], y[:50])
    x, x2 = to_volume(sg5, dets[:50]), to_volume(sg5, d2)
    assert np.array_equal(x2[:, :2], reflect_volume(x)[:, :2])
    # reflected labels still explain the reflected syndrome
    syn = (y2.astype(np.int64) @ sg5.incidence.astype(np.int64).toarray()) % 2
    assert np.array_equal(syn[:, sg5.sector_dets], d2[:, sg5.sector_dets])


def test_shift():
    h = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(1, 2, 3, 4, 5)
    s = shift(h, (1, -1, 2))
    assert s[0, 1, 0, 1, 0] == h[0, 1, 1, 0, 2]
    assert s[0, 1, 2, 1, 0] == 0 and s[0, 1, 0, 0, 0] == 0


@pytest.mark.parametrize("equivariant", [True, False])
def test_equivariance(sg5, shots, equivariant):
    torch.manual_seed(1)
    m = PreDecoder(sg5.offsets, 2, 8, equivariant).double()
    ef = torch.from_numpy(sg5.edge_flat)
    x = torch.from_numpy(to_volume(sg5, shots[0][:16])).double()
    a = m.edge_logits(x, ef)
    b = m.edge_logits(reflect_volume(x), ef)[:, torch.from_numpy(sg5.sigma_edge)]
    err = (a - b).abs().max().item()
    assert (err < 1e-10) == equivariant


def test_equivariant_model_breaks_only_near_time_boundaries():
    # on the real problem (indicators not flipped) outputs whose receptive field misses both
    # indicator layers are σ-equivariant
    sg = sector_grid(13, 0.003)
    torch.manual_seed(2)
    m = PreDecoder(sg.offsets, 2, 8, True).double()
    dets, _, _ = TrainingSampler(sg).sample(4, 3)
    ef = torch.from_numpy(sg.edge_flat)
    a = m.edge_logits(torch.from_numpy(to_volume(sg, dets)).double(), ef)
    b = m.edge_logits(torch.from_numpy(to_volume(sg, reflect_shots(sg, dets))).double(), ef)[:, torch.from_numpy(sg.sigma_edge)]
    off = np.array([(0, 0, 0)] + sg.offsets)[sg.edge_channel]
    t_lo = np.minimum(sg.edge_anchor[:, 0], sg.edge_anchor[:, 0] + off[:, 0])
    t_hi = np.maximum(sg.edge_anchor[:, 0], sg.edge_anchor[:, 0] + off[:, 0])
    T = sg.shape[0]
    bulk = torch.from_numpy((t_lo > 2) & (t_hi < T - 3))
    assert bulk.any() and (~bulk).any()
    assert (a - b)[:, bulk].abs().max().item() < 1e-10
    assert (a - b)[:, ~bulk].abs().max().item() > 1e-6


def test_tts_is_exactly_equivariant(sg5, shots):
    torch.manual_seed(3)
    m = PreDecoder(sg5.offsets, 2, 8, False).double()
    m.float()
    dets = shots[0][:64]
    p1 = edge_probabilities(m, sg5, dets, tts=True)
    d2 = reflect_shots(sg5, dets)
    x2 = reflect_volume(to_volume(sg5, dets))
    assert np.array_equal(x2[:, 0], to_volume(sg5, d2)[:, 0])
    # σ image under tts: evaluate on the flipped volume directly
    ef = torch.from_numpy(sg5.edge_flat)
    sig = torch.from_numpy(sg5.sigma_edge)
    with torch.no_grad():
        x = torch.from_numpy(x2)
        z = 0.5 * (m.edge_logits(x, ef) + m.edge_logits(reflect_volume(x), ef)[:, sig])
    np.testing.assert_allclose(torch.sigmoid(z).numpy()[:, sg5.sigma_edge], p1, atol=1e-6)


def test_evaluator_without_corrections_is_pymatching():
    sg = sector_grid(5, 0.003)
    ev = Evaluator(sg, 3000, 11)
    f, info = ev.fails(np.zeros((3000, sg.n_local)), 0.5)
    assert np.array_equal(f, ev.fail_mwpm) and info["applied_edges_per_shot"] == 0
    r = paired(f, ev.fail_mwpm)
    assert r["a_only"] == r["b_only"] == 0


def test_short_training_runs(tmp_path):
    r = train("augmented", 0, tmp_path, epochs=1, shots_per_epoch=512, batch=128, chunk=256, width=8, threads=2, log=lambda s: None)
    assert len(r["history"]) == 1 and np.isfinite(r["history"][0]["loss"])
    assert (tmp_path / "model.pt").exists()


# Regression tests for the three bugs documented in nasir26/fpga-neural-predecoder docs/DECISIONS.md
# (2026-08-16), re-derived here.

def test_regression_per_voxel_parity(sg5):
    # bug 1: thresholding voxels independently could cancel one endpoint of a pair. Here every
    # applied edge flips both of its endpoints (one for a boundary edge), never just one.
    rng = np.random.default_rng(0)
    for k in rng.choice(sg5.n_local, 50, replace=False):
        a = np.zeros((1, sg5.n_local), dtype=bool)
        a[0, k] = True
        res, _ = residual(sg5, np.zeros((1, sg5.g.n_det), dtype=np.uint8), a)
        e = sg5.edge_id[k]
        want = {int(sg5.g.u[e])} | ({int(sg5.g.v[e])} if sg5.g.v[e] != sg5.g.n_det else set())
        assert set(np.flatnonzero(res[0]).tolist()) == want


def test_regression_dropped_observable_effect(sg5):
    # bug 2: corrections that flip the observable must be folded into the prediction. Applying any
    # subset of the true fired edges and decoding the rest with PyMatching gives the same logical
    # prediction as folding the subset's parity into PyMatching's decode of the residual.
    dets, obs, y = TrainingSampler(sg5).sample(3000, 21)
    assert sg5.edge_obs.any()
    import pymatching

    from qec.dem import decomposed_dem
    m = pymatching.Matching.from_detector_error_model(decomposed_dem(sg5.exp.circuit))
    rng = np.random.default_rng(1)
    sub = y.astype(bool) & (rng.random(y.shape) < 0.5)
    res, par = residual(sg5, dets, sub)
    pred = m.decode_batch(res)[:, 0] ^ par
    dropped = m.decode_batch(res)[:, 0]
    # the fold-in matters on some shots, and with it the error rate is not worse than without
    assert (pred != dropped).any()
    assert (pred != obs).sum() < (dropped != obs).sum()


def test_regression_even_parity_observable_not_double_counted(sg5):
    # bug 3: a DEM line listing the same observable (or the same edge) twice has no net effect.
    m = __import__("nn.grid", fromlist=["error_to_edge_matrix"]).error_to_edge_matrix(sg5)
    assert set(np.unique(m.data).tolist()) <= {1}
    # observable bits come from the merged graph masks, one bit per edge
    assert set(np.unique(sg5.edge_obs).tolist()) <= {0, 1}
    dets, obs, y = TrainingSampler(sg5).sample(10_000, 22)
    assert np.array_equal((y.astype(np.int64) @ sg5.edge_obs) % 2, obs)


def test_augmentation_moves_corrections_and_residuals_together(sg5):
    # the prompt's check: permuted corrections applied to the transformed syndrome give the
    # transformed residual exactly, on 10,000 random shots with random correction sets
    dets, _, y = TrainingSampler(sg5).sample(10_000, 23)
    rng = np.random.default_rng(2)
    a = rng.random(y.shape) < 0.05
    res, _ = residual(sg5, dets, a)
    d2, a2 = reflect_shots(sg5, dets, a)
    res2, _ = residual(sg5, d2, a2)
    assert np.array_equal(res2, reflect_shots(sg5, res))
    # observable bits are not moved by σ (σ maps the observable's support to a different but
    # equivalent logical representative), so only the syndrome part is compared
