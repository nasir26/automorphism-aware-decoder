"""Exports of the decoding graphs.

1. GraphML (regenerates the planned `ising-decoder-graphml` content, which was never published): the
   NVIDIA d = 13, T = 13 decoding graphs for bases X and Z, with per-edge probability, weight,
   flips_L0 and edge_type, and per-detector stab_type, x, y, t and NVIDIA's cnn_channel, cnn_t,
   cnn_row, cnn_col. Written to results/graphml/.
2. Automorph format (header "#nodes #edges", then zero-indexed pairs) for every swept decoding graph,
   bare and weight-subdivided, with a colours file (one colour per vertex) beside each. Automorph does
   not read colours today; automorph/README.md describes the format proposed for it.

usage: python bench/export_graphs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from autsym.circuits import memory  # noqa: E402
from autsym.coloured import subdivided, weight_colours  # noqa: E402
from qec.dem import decoding_graph  # noqa: E402


def edge_type(det, g, i) -> str:
    a, b = int(g.u[i]), int(g.v[i])
    if b == g.n_det:
        return "boundary"
    same_place = det.x[a] == det.x[b] and det.y[a] == det.y[b]
    if det.t[a] == det.t[b]:
        return "spacelike"
    return "timelike" if same_place else "spacetime"


def graphml(basis: str) -> Path:
    e = memory("nvidia", 13, 13, basis, 0.003)
    g = decoding_graph(e.circuit)
    det, cell = e.det, e.nvidia_cell
    G = nx.Graph()
    for i in range(g.n_det):
        G.add_node(i, stab_type="X" if det.stab[i] == 0 else "Z", x=int(det.x[i]), y=int(det.y[i]), t=int(det.t[i]),
                   cnn_channel=int(cell[i, 0]), cnn_t=int(det.t[i]), cnn_row=int(cell[i, 1]), cnn_col=int(cell[i, 2]),
                   is_boundary=False)
    G.add_node(g.boundary, stab_type="boundary", is_boundary=True)
    for i in range(g.n_edges):
        G.add_edge(int(g.u[i]), int(g.v[i]), probability=float(g.prob[i]), weight=float(g.weight[i]),
                   flips_L0=bool(int(g.obs[i]) & 1), edge_type=edge_type(det, g, i))
    G.graph.update(source="NVIDIA Ising-Decoding MemoryCircuit, NoiseModel.from_single_p(0.003), rotation XV",
                   distance=13, rounds=13, basis=basis)
    out = REPO / "results" / "graphml" / f"nvidia_d13_r13_{basis}_p0.003_decoding.graphml"
    out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, out)
    return out


def automorph(family: str, d: int, basis: str) -> list[Path]:
    e = memory(family, d, d, basis, 0.003)
    g = decoding_graph(e.circuit)
    base = REPO / "automorph" / e.tag
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []

    def write(path, n, edges, colours):
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        with open(path, "w") as f:
            f.write(f"{n} {len(edges)}\n")
            np.savetxt(f, edges, fmt="%d")
        np.savetxt(str(path).replace(".txt", ".colours.txt"), np.asarray(colours, dtype=np.int64), fmt="%d")
        paths.append(path)

    col = np.zeros(g.n_nodes, dtype=np.int64)
    col[g.boundary] = 1
    write(Path(f"{base}_decoding.txt"), g.n_nodes, np.stack([g.u, g.v], 1), col)
    wc, _ = weight_colours(g.weight, 9)
    n, E, c = subdivided(g.n_nodes, g.boundary, g.u, g.v, wc)
    write(Path(f"{base}_weighted_subdivided.txt"), n, E, c)
    return paths


def main():
    for b in "XZ":
        print(graphml(b))
    for fam in ("nvidia", "stim"):
        for d in (3, 5, 7, 9, 11, 13):
            for b in "XZ":
                automorph(fam, d, b)
    print("automorph exports written")


if __name__ == "__main__":
    main()
