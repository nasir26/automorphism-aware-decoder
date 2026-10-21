"""Thin wrappers around the external solvers.

* ``nauty_aut``: this repository's ``tools/nauty_aut.c`` (built into ``build/nauty_aut`` against the
  nauty tree inside ``../third_party/GraphCrunch/nauty``). Returns the group order, orbits and
  generators of a vertex-coloured graph.
* ``graphcrunch_structure`` and ``graphcrunch_cli``: GraphCrunch itself (vaibh-pra/GraphCrunch, MIT),
  run from its clone. Its solver prints orbits only, so group orders always come from ``nauty_aut``.
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GRAPHCRUNCH = Path(os.environ.get("GRAPHCRUNCH_DIR", REPO.parent / "third_party" / "GraphCrunch"))
NAUTY_AUT = Path(os.environ.get("NAUTY_AUT_BIN", REPO / "build" / "nauty_aut"))


@dataclass
class AutResult:
    n: int
    mantissa: float
    exponent: int
    orbits: np.ndarray  # orbit representative per vertex
    generators: list[np.ndarray]
    seconds: float
    num_generators: int | None = None

    @property
    def log10_order(self) -> float:
        return math.log10(self.mantissa) + self.exponent

    @property
    def order(self) -> int:
        """Exact only while the order fits in a double without rounding (below 2^53)."""
        v = self.mantissa * 10.0 ** self.exponent
        if v >= 2 ** 53:
            raise OverflowError("group order too large for an exact integer here; use log10_order")
        return int(round(v))

    @property
    def num_orbits(self) -> int:
        return len(set(self.orbits.tolist()))

    def orbit_partition(self) -> list[list[int]]:
        out: dict[int, list[int]] = {}
        for v, r in enumerate(self.orbits.tolist()):
            out.setdefault(r, []).append(v)
        return sorted(out.values())


def write_graph(path: Path, n: int, edges) -> None:
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    with open(path, "w") as f:
        f.write(f"{n} {len(e)}\n")
        np.savetxt(f, e, fmt="%d")


def nauty_aut(n: int, edges, colours=None, workdir: Path | None = None, timeout: float | None = None,
              generators: bool = True) -> AutResult:
    """generators=False counts generators without writing them (very large groups)."""
    if not NAUTY_AUT.is_file():
        raise FileNotFoundError(f"{NAUTY_AUT} missing; run `make tools`")
    with tempfile.TemporaryDirectory(prefix="nauty-", dir=workdir) as d:
        d = Path(d)
        write_graph(d / "g.el", n, edges)
        col = ""
        if colours is not None:
            col = str(d / "g.col")
            np.savetxt(col, np.asarray(colours, dtype=np.int64), fmt="%d")
        t0 = time.time()
        r = subprocess.run([str(NAUTY_AUT), str(d / "g.el"), col, str(d / "out")] + ([] if generators else ["nogens"]),
                           capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        if r.returncode != 0:
            raise RuntimeError(f"nauty_aut failed ({r.returncode}): {r.stderr[:300]}")
        gens, mant, expo, orbits, ngen = [], None, None, None, None
        with open(d / "out") as f:
            for line in f:
                if line.startswith("gen "):
                    gens.append(np.array(line.split()[1:], dtype=np.int64))
                elif line.startswith("group_size"):
                    _, a, b = line.split()
                    mant, expo = float(a), int(b)
                elif line.startswith("num_generators"):
                    ngen = int(line.split()[1])
                elif line.startswith("orbits"):
                    orbits = np.array(line.split()[1:], dtype=np.int64)
    return AutResult(n=n, mantissa=mant, exponent=expo, orbits=orbits, generators=gens, seconds=dt,
                     num_generators=ngen)


def _load_orbits_serial():
    src = GRAPHCRUNCH / "src" / "orbits_serial.py"
    if not src.is_file():
        raise FileNotFoundError(f"GraphCrunch not found at {GRAPHCRUNCH}; see third_party/README.md")
    spec = importlib.util.spec_from_file_location("graphcrunch_orbits_serial", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def graphcrunch_binary() -> Path:
    b = GRAPHCRUNCH / "automorph_serial_col"
    if not b.is_file():
        raise FileNotFoundError(f"{b} missing; build GraphCrunch (third_party/README.md)")
    return b


def graphcrunch_structure(n: int, edges):
    """GraphCrunch's own contract, solve and lift (uncoloured input graph).
    Returns (vertex_orbits, edge_orbits, how, stats) exactly as GraphCrunch's ``structure``."""
    gc = _load_orbits_serial()
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    adj: dict[int, set[int]] = {v: set() for v in range(n)}
    for a, b in e.tolist():
        adj[a].add(b)
        adj[b].add(a)
    ue = sorted({(min(a, b), max(a, b)) for a, b in e.tolist()})
    return gc.structure(n, adj, ue, binary=graphcrunch_binary())


def graphcrunch_cli(n: int, edges, raw: bool, timeout: float) -> dict:
    """Run GraphCrunch's command line on an edge list, with or without --raw, and parse its own
    timing lines. A crash or timeout of the raw solve is reported, not raised."""
    with tempfile.TemporaryDirectory(prefix="gc-") as d:
        d = Path(d)
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        np.savetxt(d / "edges.tsv", e, fmt="%d", delimiter="\t")
        cmd = ["python3", str(GRAPHCRUNCH / "src" / "orbits_serial.py"), "--edges", str(d / "edges.tsv"),
               "--out", str(d / "orbits.txt"), "--binary", str(graphcrunch_binary())]
        if raw:
            cmd.append("--raw")
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            out = dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr[-2000:], timeout=False)
        except subprocess.TimeoutExpired as ex:
            out = dict(returncode=None, stdout=(ex.stdout or b"").decode() if isinstance(ex.stdout, bytes) else (ex.stdout or ""),
                       stderr="", timeout=True)
        out["wall_seconds"] = time.time() - t0
    s = out["stdout"]
    m = re.search(r"contract\+solve\+lift ([\d.]+)s", s)
    out["contract_solve_lift_seconds"] = float(m.group(1)) if m else None
    m = re.search(r"raw graph, no contraction: ([\d.]+)s, (\d+) orbits", s)
    out["raw_seconds"] = float(m.group(1)) if m else None
    m = re.search(r"IDENTICAL PARTITION: (\w+)", s)
    out["identical_partition"] = (m.group(1) == "True") if m else None
    m = re.search(r"lifted: (\d+) orbits", s)
    out["lifted_orbits"] = int(m.group(1)) if m else None
    return out
