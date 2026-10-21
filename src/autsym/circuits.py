"""Circuit builders and detector geometry.

Two circuit families:

* ``nvidia``: NVIDIA's ``MemoryCircuit`` (Ising-Decoding, Apache-2.0), imported from a clone outside
  this repository (``$ISING_DECODING_DIR``, default ``../third_party/Ising-Decoding``), with
  ``NoiseModel.from_single_p(p)``. Nothing from that code is copied here.
* ``stim``: ``stim.Circuit.generated("surface_code:rotated_memory_<basis>")`` with one ``p`` on its four
  noise knobs, the uniform depolarizing generator of ``nasir26/fpga-neural-predecoder``.

For every detector the geometry is derived from the circuit itself, by following the detector's
measurement records back to the stabilizer ancilla they measure: stabilizer type, ancilla
coordinates ``(x, y)`` and time layer ``t``. ``t`` counts how many earlier measurements of that ancilla
the detector's latest ancilla record has, plus one for detectors that also read final data
measurements.
"""
from __future__ import annotations

import collections
import functools
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

REPO = Path(__file__).resolve().parents[2]
ISING_DIR = Path(os.environ.get("ISING_DECODING_DIR", REPO.parent / "third_party" / "Ising-Decoding"))

X_STAB, Z_STAB = 0, 1
MEASURE_OPS = {"M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY"}


@dataclass(frozen=True)
class DetectorInfo:
    """Per-detector geometry. Arrays are indexed by detector id."""

    stab: np.ndarray  # X_STAB or Z_STAB
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    ancilla: np.ndarray  # qubit id of the measured stabilizer ancilla

    @property
    def n(self) -> int:
        return len(self.stab)


@dataclass(frozen=True)
class MemoryExperiment:
    family: str
    distance: int
    rounds: int
    basis: str
    p: float
    circuit: stim.Circuit
    det: DetectorInfo
    nvidia_cell: np.ndarray | None = None  # per detector (channel 0 = X / 1 = Z, row, col) of NVIDIA's grid

    @property
    def tag(self) -> str:
        return f"{self.family}_d{self.distance}_r{self.rounds}_{self.basis}_p{self.p:g}"


def _trace_detectors(circuit: stim.Circuit, ancilla: dict[int, tuple[int, tuple[int, int]]]) -> DetectorInfo:
    """Map each detector to (stabilizer type, ancilla coordinates, time layer)."""
    meas: list[tuple[int, int]] = []
    count: collections.Counter = collections.Counter()
    rows = []
    for ins in circuit.flattened():
        if ins.name in MEASURE_OPS:
            for tg in ins.targets_copy():
                q = tg.value
                meas.append((q, count[q]))
                count[q] += 1
        elif ins.name == "DETECTOR":
            recs = [meas[len(meas) + tg.value] for tg in ins.targets_copy()]
            anc = {q for q, _ in recs if q in ancilla}
            if len(anc) != 1:
                raise ValueError(f"detector {len(rows)} reads {len(anc)} ancillas, expected 1")
            q = anc.pop()
            k = max(k for qq, k in recs if qq == q)
            final = any(qq not in ancilla for qq, _ in recs)
            s, (x, y) = ancilla[q]
            rows.append((s, x, y, k + int(final), q))
    a = np.array(rows, dtype=np.int64).reshape(-1, 5)
    if len(a) != circuit.num_detectors:
        raise ValueError("detector count mismatch")
    return DetectorInfo(stab=a[:, 0], x=a[:, 1], y=a[:, 2], t=a[:, 3], ancilla=a[:, 4])


@functools.lru_cache(maxsize=1)
def _import_nvidia():
    """Import NVIDIA's MemoryCircuit and NoiseModel. Their package is also named ``qec``, like this
    repository's ``src/qec``, so it is imported with ours set aside and then removed again from
    ``sys.modules``; the two classes keep references to the modules they need."""
    code = ISING_DIR / "code"
    if not code.is_dir():
        raise FileNotFoundError(f"NVIDIA Ising-Decoding not found at {ISING_DIR}; see third_party/README.md")
    ours = {k: v for k, v in sys.modules.items() if k == "qec" or k.startswith("qec.")}
    for k in ours:
        del sys.modules[k]
    sys.path.insert(0, str(code))
    try:
        noise_model = importlib.import_module("qec.noise_model")
        memory_circuit = importlib.import_module("qec.surface_code.memory_circuit")
        data_mapping = importlib.import_module("qec.surface_code.data_mapping")
    finally:
        sys.path.remove(str(code))
        for k in [k for k in sys.modules if k == "qec" or k.startswith("qec.")]:
            del sys.modules[k]
        sys.modules.update(ours)
    return memory_circuit.MemoryCircuit, noise_model.NoiseModel, data_mapping


def nvidia_memory(distance: int, rounds: int, basis: str, p: float, rotation: str = "XV") -> MemoryExperiment:
    """NVIDIA's memory circuit with NoiseModel.from_single_p(p), boundary detectors off."""
    MemoryCircuit, NoiseModel, data_mapping = _import_nvidia()
    mc = MemoryCircuit(
        distance=distance, idle_error=p, sqgate_error=p, tqgate_error=p, spam_error=2 * p / 3,
        n_rounds=rounds, basis=basis, code_rotation=rotation, noise_model=NoiseModel.from_single_p(p),
    )
    mc.set_error_rates()
    circuit = stim.Circuit(mc.circuit)
    cd = mc.code.code_dict
    anc = {}
    for key, s in (("syndrome_X", X_STAB), ("syndrome_Z", Z_STAB)):
        for q, v in cd[key].items():
            anc[int(q)] = (s, (int(v["coord"][0]), int(v["coord"][1])))
    det = _trace_detectors(circuit, anc)
    # NVIDIA's CNN layout: stabilizer tensor index -> anchor data qubit -> (row, col) of a d x d grid
    nx = len(cd["syndrome_X"])
    xmap = np.asarray(data_mapping.compute_stabX_to_data_index_map(distance, rotation)).astype(np.int64)
    zmap = np.asarray(data_mapping.compute_stabZ_to_data_index_map(distance, rotation)).astype(np.int64)
    cell = np.empty((det.n, 3), dtype=np.int64)
    for i in range(det.n):
        q = int(det.ancilla[i])
        if det.stab[i] == X_STAB:
            dq = xmap[q - distance ** 2]
        else:
            dq = zmap[q - distance ** 2 - nx]
        cell[i] = (det.stab[i], dq // distance, dq % distance)
    return MemoryExperiment("nvidia", distance, rounds, basis, p, circuit, det, cell)


def stim_memory(distance: int, rounds: int, basis: str, p: float) -> MemoryExperiment:
    """Stim's generated rotated memory circuit, uniform depolarizing p on all four knobs."""
    circuit = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}", distance=distance, rounds=rounds,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p,
    )
    qc = circuit.get_final_qubit_coordinates()
    measured = collections.Counter()
    hadamard = set()
    for ins in circuit.flattened():
        if ins.name in MEASURE_OPS:
            measured.update(t.value for t in ins.targets_copy())
        elif ins.name == "H":
            hadamard.update(t.value for t in ins.targets_copy())
    # ancillas are measured every round; data qubits once, at the end
    if rounds < 2:
        raise ValueError("rounds >= 2 needed to tell ancillas from data qubits")
    anc = {q: (X_STAB if q in hadamard else Z_STAB, (int(qc[q][0]), int(qc[q][1])))
           for q, c in measured.items() if c > 1}
    det = _trace_detectors(circuit, anc)
    # cross-check against Stim's own detector coordinates (x, y, t)
    sc = circuit.get_detector_coordinates()
    for i in range(det.n):
        x, y, t = sc[i][:3]
        if (int(x), int(y)) != (det.x[i], det.y[i]) or int(t) != det.t[i]:
            raise AssertionError(f"detector {i}: traced {(det.x[i], det.y[i], det.t[i])}, stim {sc[i]}")
    return MemoryExperiment("stim", distance, rounds, basis, p, circuit, det)


def memory(family: str, distance: int, rounds: int, basis: str, p: float) -> MemoryExperiment:
    if family == "nvidia":
        return nvidia_memory(distance, rounds, basis, p)
    if family == "stim":
        return stim_memory(distance, rounds, basis, p)
    raise ValueError(family)
