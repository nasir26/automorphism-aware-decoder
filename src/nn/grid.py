"""The basis sector of a Stim memory experiment as a dense (T, H, W) grid, its local-edge channels,
the action of σ on both, and training-shot generation.

Grid: ``t`` is the detector's time layer, row = y / 2, col = x / 2 (the halved Stim coordinates of
``nasir26/fpga-neural-predecoder``), so T = rounds + 1 and H = W = d + 1. Only the detectors of the
memory basis are placed on the grid: in Stim's rotated memory circuits no decoding-graph edge joins
the two sectors and no edge of the other sector flips the observable (checked in ``SectorGrid``),
so the other sector cannot change the logical prediction.

Channels: channel 0 is the boundary edge of a detector ("singleton"); channel c >= 1 is the edge from
anchor v to v + o_c, where o_c is the lexicographically smaller of the two directed offsets between
the endpoints. The offset catalogue is a property of the bulk and is passed in from the training
distance, so a model trained at one d runs at any other; edges whose offset is not in the catalogue
are left to the matcher.

σ is the point reflection of the sector through the centre of its box, which on this grid is the flip
of all three axes (Phase 1). It maps the edge (c, v) to (c, C - v - o_c), with C = (T-1, H-1, W-1).

Network input channels: detection events, site mask (1 where a sector detector exists), first-layer
indicator, last-layer indicator. The equivariant network commutes with the flip of the whole input
volume, indicators included; on the real problem the indicators are not flipped (the first layer
stays first), so the network still tells the two time boundaries apart, and σ-equivariance holds
exactly only for outputs whose receptive field misses both indicator layers. This is boundary
policy (b), docs/DECISIONS.md. (Swapping the two indicator channels under σ would make them a
σ-invariant pair and give policy (a) instead.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse
import stim

from autsym.circuits import X_STAB, Z_STAB, MemoryExperiment, memory
from autsym.orbits import reflection_candidates
from autsym.symmetry import edge_permutation
from qec.dem import DecodingGraph, decoding_graph, decomposed_dem

INPUT_CHANNELS = 4

Offset = tuple[int, int, int]


def canonical(a: np.ndarray, b: np.ndarray) -> tuple[Offset, int]:
    """(canonical offset, 0 if a is the anchor else 1)."""
    ab = tuple(int(x) for x in (b - a))
    ba = tuple(-x for x in ab)
    return (ab, 0) if ab <= ba else (ba, 1)


def offset_catalogue(g: DecodingGraph, cell: np.ndarray, sector: np.ndarray) -> list[Offset]:
    out = set()
    for a, b in zip(g.u.tolist(), g.v.tolist()):
        if b != g.n_det and sector[a]:
            out.add(canonical(cell[a], cell[b])[0])
    return sorted(out)


@dataclass
class SectorGrid:
    exp: MemoryExperiment
    g: DecodingGraph
    shape: tuple[int, int, int]
    cell: np.ndarray        # (n_det, 3) grid cell of every detector (other sector: -1)
    sector: np.ndarray      # bool (n_det,), detector belongs to the basis sector
    sector_dets: np.ndarray  # detector ids of the sector
    offsets: list[Offset]
    # local edges: graph edge id, channel, anchor cell
    edge_id: np.ndarray
    edge_channel: np.ndarray
    edge_anchor: np.ndarray  # (E, 3)
    edge_flat: np.ndarray    # flat index into (C, T, H, W)
    edge_obs: np.ndarray     # uint8, the edge flips observable 0
    incidence: scipy.sparse.csr_matrix  # (E, n_det) endpoints of each local edge
    sigma_edge: np.ndarray   # local edge k -> local edge index of its σ image
    sigma_det: np.ndarray    # detector permutation of σ (graph node ids, boundary fixed)
    static: np.ndarray       # (3, T, H, W) float32: site mask, first, last indicators

    @property
    def n_channels(self) -> int:
        return 1 + len(self.offsets)

    @property
    def n_local(self) -> int:
        return len(self.edge_id)


def sector_grid(d: int, p: float, basis: str = "Z", offsets: list[Offset] | None = None, rounds: int | None = None) -> SectorGrid:
    rounds = d if rounds is None else rounds
    exp = memory("stim", d, rounds, basis, p)
    g = decoding_graph(exp.circuit)
    det = exp.det
    s = {"X": X_STAB, "Z": Z_STAB}[basis]
    sector = det.stab == s
    if not np.all(sector[g.u[g.v < g.n_det]] == sector[g.v[g.v < g.n_det]]):
        raise AssertionError("an edge joins the two sectors")
    if np.any(g.obs[~sector[g.u]] != 0):
        raise AssertionError("an edge of the other sector flips the observable")
    T, H, W = rounds + 1, d + 1, d + 1
    cell = np.full((g.n_det, 3), -1, dtype=np.int64)
    cell[sector, 0] = det.t[sector]
    cell[sector, 1] = det.y[sector] // 2
    cell[sector, 2] = det.x[sector] // 2
    sd = np.flatnonzero(sector)
    if len({tuple(c) for c in cell[sd].tolist()}) != len(sd):
        raise AssertionError("two sector detectors share a grid cell")
    if cell[sd].min() < 0 or cell[sd, 0].max() >= T or cell[sd, 1].max() >= H or cell[sd, 2].max() >= W:
        raise AssertionError("sector detector outside the grid")

    if offsets is None:
        offsets = offset_catalogue(g, cell, sector)
    ch_of = {o: i + 1 for i, o in enumerate(offsets)}
    ids, chans, anchors, rows, cols = [], [], [], [], []
    for e, (a, b) in enumerate(zip(g.u.tolist(), g.v.tolist())):
        if not sector[a]:
            continue
        if b == g.n_det:
            c, anc = 0, a
        else:
            o, which = canonical(cell[a], cell[b])
            if o not in ch_of:
                continue
            c, anc = ch_of[o], (a if which == 0 else b)
        k = len(ids)
        ids.append(e)
        chans.append(c)
        anchors.append(cell[anc])
        rows += [k] if b == g.n_det else [k, k]
        cols += [a] if b == g.n_det else [a, b]
    edge_id = np.array(ids, dtype=np.int64)
    edge_channel = np.array(chans, dtype=np.int64)
    edge_anchor = np.array(anchors, dtype=np.int64).reshape(-1, 3)
    edge_flat = np.ravel_multi_index((edge_channel, edge_anchor[:, 0], edge_anchor[:, 1], edge_anchor[:, 2]),
                                     (1 + len(offsets), T, H, W))
    incidence = scipy.sparse.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(ids), g.n_det))

    sig = reflection_candidates(det, g.n_nodes)[f"reflect_{basis}"]
    ep = edge_permutation(g, sig)
    local_of = -np.ones(g.n_edges, dtype=np.int64)
    local_of[edge_id] = np.arange(len(edge_id))
    sigma_edge = local_of[ep[edge_id]]
    if np.any(sigma_edge < 0):
        raise AssertionError("σ maps a local edge outside the catalogue")
    # σ on the grid is the flip of all three axes
    C = np.array([T - 1, H - 1, W - 1])
    if not np.array_equal(cell[sig[sd]], C - cell[sd]):
        raise AssertionError("σ is not the axis flip on this grid")

    static = np.zeros((3, T, H, W), dtype=np.float32)
    static[0][tuple(cell[sd].T)] = 1
    static[1, 0] = 1
    static[2, T - 1] = 1
    return SectorGrid(exp=exp, g=g, shape=(T, H, W), cell=cell, sector=sector, sector_dets=sd, offsets=offsets,
                      edge_id=edge_id, edge_channel=edge_channel, edge_anchor=edge_anchor, edge_flat=edge_flat,
                      edge_obs=(g.obs[edge_id] & 1).astype(np.uint8), incidence=incidence, sigma_edge=sigma_edge,
                      sigma_det=sig[: g.n_det], static=static)


def to_volume(sg: SectorGrid, dets: np.ndarray) -> np.ndarray:
    """(shots, n_det) detection events -> (shots, 4, T, H, W) float32 network input."""
    n = len(dets)
    x = np.zeros((n, INPUT_CHANNELS, *sg.shape), dtype=np.float32)
    c = sg.cell[sg.sector_dets]
    x[:, 0, c[:, 0], c[:, 1], c[:, 2]] = dets[:, sg.sector_dets]
    x[:, 1:] = sg.static
    return x


def reflect_shots(sg: SectorGrid, dets: np.ndarray, target: np.ndarray | None = None):
    """σ applied to shots: (σ s)[σ j] = s[j]; local edge labels move the same way. The network input
    of the reflected problem is ``reflect_volume(to_volume(sg, dets))``, not ``to_volume`` of these."""
    out = np.empty_like(dets)
    out[:, sg.sigma_det] = dets
    if target is None:
        return out
    t = np.empty_like(target)
    t[:, sg.sigma_edge] = target
    return out, t


def reflect_volume(x):
    """Flip of all three grid axes of a (B, 4, T, H, W) array or tensor, every channel included."""
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x[:, :, ::-1, ::-1, ::-1])
    return x.flip(2, 3, 4)


def error_to_edge_matrix(sg: SectorGrid) -> scipy.sparse.csr_matrix:
    """(n_errors, n_local) over GF(2): for every line of the decomposed DEM, the local edges its
    graphlike components land on (components of the other sector or outside the catalogue are
    dropped)."""
    dem = decomposed_dem(sg.exp.circuit)
    idx = sg.g.edge_index()
    local_of = {int(e): k for k, e in enumerate(sg.edge_id.tolist())}
    nd = sg.g.n_det
    rows, cols = [], []
    k = 0
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        comp: list[int] = []
        for t in ins.targets_copy() + [stim.DemTarget.separator()]:
            if t.is_separator():
                if comp:
                    a, b = (comp[0], nd) if len(comp) == 1 else (min(comp), max(comp))
                    if len(comp) > 2:
                        raise AssertionError("component with more than two detectors")
                    e = idx.get((a, b))
                    if e is not None and e in local_of:
                        rows.append(k)
                        cols.append(local_of[e])
                comp = []
            elif t.is_relative_detector_id():
                comp.append(t.val)
        k += 1
    m = scipy.sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(k, sg.n_local))
    m.sum_duplicates()
    m.data %= 2
    m.eliminate_zeros()
    return m


class TrainingSampler:
    """Fresh DEM-level shots with per-local-edge XOR-parity targets."""

    def __init__(self, sg: SectorGrid):
        self.sg = sg
        self.dem = decomposed_dem(sg.exp.circuit)
        self.m = error_to_edge_matrix(sg)

    def sample(self, shots: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dets, obs, err = self.dem.compile_sampler(seed=seed).sample(shots, return_errors=True)
        y = (scipy.sparse.csr_matrix(err.astype(np.int32)) @ self.m).toarray() % 2
        return dets.astype(np.uint8), obs[:, 0].astype(np.uint8), y.astype(np.uint8)
