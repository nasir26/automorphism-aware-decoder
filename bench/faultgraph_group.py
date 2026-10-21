"""H1, second half: twin contraction preserves every automorphism of the fault graph.

For each distance, the order of Aut(fault graph) computed by nauty on the uncontracted graph is
compared with log10 |K| + log10 |A|, where K is the twin group and A the automorphism group of the
coloured quotient (colour = class size and clique type), and the raw vertex orbits are compared with
the lifted ones. Also records how GraphCrunch's own solver ends on the uncontracted d = 13 graph.
Writes results/faultgraph_group.json (T4-SIM).
usage: python bench/faultgraph_group.py [--distances 3 5 7 9 11 13] [--basis X]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from autsym.circuits import memory  # noqa: E402
from autsym.coloured import coloured_orbits  # noqa: E402
from autsym.faultgraph import fault_graph  # noqa: E402
from autsym.graphcrunch import graphcrunch_binary, nauty_aut, write_graph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13])
    ap.add_argument("--families", nargs="+", default=["nvidia", "stim"])
    ap.add_argument("--timeout", type=float, default=1500)
    ap.add_argument("--raw-max-d", type=int, default=5, help="plain nauty only up to this distance")
    ap.add_argument("--raw-extra", nargs="*", default=["nvidia:7:X"], help="extra family:d:basis raw runs")
    a = ap.parse_args()
    out = {"tier": "T4-SIM", "p": 0.003, "runs": []}
    for fam in a.families:
        for d in a.distances:
            for basis in "XZ":
                e = memory(fam, d, d, basis, 0.003)
                fg = fault_graph(e.circuit)
                E = fg.edges()
                t0 = time.time()
                lifted = coloured_orbits(fg.n, E)
                t_lift = time.time() - t0
                r = {"family": fam, "distance": d, "basis": basis, "vertices": fg.n, "edges": int(len(E)),
                     "quotient_vertices": lifted.contraction.m, "log10_K": lifted.contraction.log10_K,
                     "log10_A": lifted.quotient_aut.log10_order, "log10_K_plus_A": lifted.log10_order,
                     "contract_solve_lift_seconds": round(t_lift, 2)}
                if d > a.raw_max_d and f"{fam}:{d}:{basis}" not in a.raw_extra:
                    r["raw"] = "not run (plain nauty cost grows quickly; see --raw-max-d)"
                    out["runs"].append(r)
                    print(json.dumps(r), flush=True)
                    continue
                try:
                    raw = nauty_aut(fg.n, E, timeout=a.timeout, generators=False)
                    r["raw_log10_order"] = raw.log10_order
                    r["raw_seconds"] = round(raw.seconds, 2)
                    r["raw_generators"] = raw.num_generators
                    r["log10_order_difference"] = raw.log10_order - lifted.log10_order
                    r["vertex_orbits_identical"] = ({frozenset(o) for o in raw.orbit_partition()}
                                                    == {frozenset(o) for o in lifted.vertex_orbits})
                except subprocess.TimeoutExpired:
                    r["raw"] = f"timeout after {a.timeout} s"
                out["runs"].append(r)
                print(json.dumps(r), flush=True)
    # GraphCrunch's own solver on the uncontracted d = 13 NVIDIA graph (it stores every generator)
    e = memory("nvidia", 13, 13, "X", 0.003)
    fg = fault_graph(e.circuit)
    with tempfile.TemporaryDirectory() as td:
        write_graph(Path(td) / "g.el", fg.n, fg.edges())
        t0 = time.time()
        pr = subprocess.run([str(graphcrunch_binary()), str(Path(td) / "g.el"), "", str(Path(td) / "o")],
                            capture_output=True, text=True, timeout=a.timeout)
        out["graphcrunch_solver_uncontracted_d13"] = {"returncode": pr.returncode, "seconds": round(time.time() - t0, 1),
                                                      "stderr_tail": pr.stderr[-300:]}
    print(out["graphcrunch_solver_uncontracted_d13"])
    (REPO / "results" / "faultgraph_group.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
