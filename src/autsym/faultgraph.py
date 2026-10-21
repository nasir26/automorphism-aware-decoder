"""Fault and detector (and observable) bipartite graph of a circuit.

Vertices, in this order: one per single circuit fault (every Pauli component of every noise
channel, and every measurement flip, that Stim reports as having a nonempty symptom), then one per
detector, then one per observable. A fault is adjacent to the detectors it flips and to the
observables it flips. Footprints come from
``circuit.explain_detector_error_model_errors(reduce_to_one_representative_error=False)`` on the
flattened circuit, so they are Stim's undecomposed symptoms.

Each fault also carries the probability Stim's error analysis assigns to it, read from its
instruction: the listed component for PAULI_CHANNEL_1/2 and X/Y/Z_ERROR, and the independent
per-channel probability for DEPOLARIZE1/2, the conversion Stim applies to those channels.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np
import stim

PAULI = {"I": 0, "X": 1, "Y": 2, "Z": 3}


@dataclass
class FaultGraph:
    n_fault: int
    n_det: int
    n_obs: int
    footprint: list[tuple[tuple[int, ...], int]]  # per fault: (detectors, observable mask)
    prob: np.ndarray  # per fault
    instruction: list[str] = field(default_factory=list)  # per fault, gate name
    application: list[tuple[int, int]] = field(default_factory=list)  # per fault, (instruction, first target)

    @property
    def n(self) -> int:
        return self.n_fault + self.n_det + self.n_obs

    def det_vertex(self, d: int) -> int:
        return self.n_fault + d

    def obs_vertex(self, k: int) -> int:
        return self.n_fault + self.n_det + k

    def edges(self) -> np.ndarray:
        rows = []
        for f, (dets, mask) in enumerate(self.footprint):
            rows.extend((f, self.det_vertex(d)) for d in dets)
            k = 0
            while mask >> k:
                if mask >> k & 1:
                    rows.append((f, self.obs_vertex(k)))
                k += 1
        return np.array(rows, dtype=np.int64).reshape(-1, 2)

    def adjacency(self) -> list[set[int]]:
        adj = [set() for _ in range(self.n)]
        for a, b in self.edges():
            adj[a].add(int(b))
            adj[b].add(int(a))
        return adj


def _depolarize1_component(p: float) -> float:
    return 0.5 - 0.5 * (1 - 4 * p / 3) ** 0.5


def _depolarize2_component(p: float) -> float:
    return 0.5 - 0.5 * (1 - 16 * p / 15) ** 0.125


def _component_probability(ins: stim.CircuitInstruction, loc) -> float:
    name, args = ins.name, ins.gate_args_copy()
    if name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        return args[0]
    if name == "DEPOLARIZE1":
        return _depolarize1_component(args[0])
    if name == "DEPOLARIZE2":
        return _depolarize2_component(args[0])
    if name in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"):
        return args[0]
    paulis = {p.gate_target.value: _pauli_name(p.gate_target) for p in loc.flipped_pauli_product}
    tgts = [t.gate_target.value for t in loc.instruction_targets.targets_in_range]
    if name == "PAULI_CHANNEL_1":
        (q,) = tgts
        return _independent_components(tuple(args))[PAULI[paulis[q]] - 1]
    if name == "PAULI_CHANNEL_2":
        a, b = tgts
        idx = 4 * PAULI[paulis.get(a, "I")] + PAULI[paulis.get(b, "I")] - 1
        # Stim (approximate_disjoint_errors) takes PAULI_CHANNEL_2 components at their listed
        # values as independent events, unlike PAULI_CHANNEL_1, which it converts exactly; measured
        # in tests/test_faultgraph.py
        return args[idx]
    raise NotImplementedError(f"noise instruction {name}")


def _symplectic(index: int, n: int) -> np.ndarray:
    """Pauli number 1..4^n - 1 in Stim's argument order (I, X, Y, Z per qubit, first qubit most
    significant) as an (x | z) bit vector."""
    v = np.zeros(2 * n, dtype=np.int64)
    for k in range(n):
        p = index // 4 ** (n - 1 - k) % 4
        v[k] = p in (1, 2)
        v[n + k] = p in (2, 3)
    return v


@functools.lru_cache(maxsize=None)
def _independent_components(args: tuple[float, ...]) -> tuple[float, ...]:
    """Exact conversion of a disjoint Pauli channel (one probability per non-identity Pauli) into
    independent Pauli components q_k with the same output distribution.

    For every non-identity Pauli s, the expectation of its sign character is
    F(s) = 1 - 2 * sum of the probabilities of the Paulis anticommuting with s for the disjoint
    channel, and prod over anticommuting k of (1 - 2 q_k) for the independent one. Taking logs gives a
    linear system with the (invertible) anticommutation matrix."""
    m = len(args)
    n = {3: 1, 15: 2}[m]
    vec = [_symplectic(i + 1, n) for i in range(m)]
    anti = np.array([[(a[:n] @ b[n:] + a[n:] @ b[:n]) % 2 for b in vec] for a in vec], dtype=float)
    f = 1 - 2 * anti @ np.array(args, dtype=float)
    if np.any(f <= 0):
        raise ValueError("Pauli channel has no independent decomposition")
    logs = np.linalg.solve(anti, np.log(f))
    return tuple(float(x) for x in (1 - np.exp(logs)) / 2)


def _pauli_name(gt: stim.GateTarget) -> str:
    if gt.is_x_target:
        return "X"
    if gt.is_y_target:
        return "Y"
    if gt.is_z_target:
        return "Z"
    raise ValueError(gt)


def fault_graph(circuit: stim.Circuit) -> FaultGraph:
    flat = circuit.flattened()
    instrs = list(flat)
    explained = flat.explain_detector_error_model_errors(reduce_to_one_representative_error=False)
    footprint, prob, names, apps = [], [], [], []
    for e in explained:
        dets, mask = [], 0
        for t in e.dem_error_terms:
            dt = t.dem_target
            if dt.is_relative_detector_id():
                dets.append(dt.val)
            elif dt.is_logical_observable_id():
                mask ^= 1 << dt.val
            # separators do not occur in undecomposed explanations
        key = (tuple(sorted(dets)), mask)
        for loc in e.circuit_error_locations:
            (frame,) = loc.stack_frames
            ins = instrs[frame.instruction_offset]
            footprint.append(key)
            prob.append(_component_probability(ins, loc))
            names.append(ins.name)
            apps.append((frame.instruction_offset, loc.instruction_targets.target_range_start))
    return FaultGraph(
        n_fault=len(footprint), n_det=circuit.num_detectors, n_obs=circuit.num_observables,
        footprint=footprint, prob=np.array(prob, dtype=np.float64), instruction=names, application=apps,
    )


def merged_probability(fg: FaultGraph, members) -> float:
    """Probability of the merged mechanism of a set of faults with one footprint, combined as Stim's
    error analysis combines them: PAULI_CHANNEL_2 components of one application that give the same
    symptom are disjoint events and add; every other component is an independent event after the
    conversion in _component_probability; the per-application totals XOR-combine."""
    per_app: dict = {}
    for f in members:
        key = fg.application[f] if fg.instruction[f] == "PAULI_CHANNEL_2" else ("indep", f)
        per_app[key] = per_app.get(key, 0.0) + float(fg.prob[f])
    return xor_combine(per_app.values())


def xor_combine(ps) -> float:
    q = 0.0
    for p in ps:
        q = q * (1 - p) + p * (1 - q)
    return q


def log10_factorial(k: int) -> float:
    return math.lgamma(k + 1) / math.log(10)
