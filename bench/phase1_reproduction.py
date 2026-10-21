"""Phase 1: reproduce the regression table (NVIDIA MemoryCircuit, single p = 0.003, rotation XV, d = 13,
T = 13), for basis X and basis Z. Writes results/phase1_reproduction.json (tier T4-SIM).

Every row records the computed value, the expected value from the governing prompt (for basis X;
basis Z expectations are the same with X and Z exchanged, where the prompt says so) and whether
they match exactly. Nothing is adjusted to make a row match.

usage: python bench/phase1_reproduction.py [--skip-raw-nauty]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from autsym.circuits import X_STAB, Z_STAB, memory  # noqa: E402
from autsym.coloured import contract, significant, subdivided, weight_colours  # noqa: E402
from autsym.faultgraph import fault_graph, merged_probability  # noqa: E402
from autsym.graphcrunch import graphcrunch_cli, graphcrunch_structure, nauty_aut  # noqa: E402
from autsym.periodic import merge_classes, periodic_bulk_graph, periodic_coloured, round_templates, translation_classes  # noqa: E402
from autsym.symmetry import edge_permutation, is_automorphism, tensor_map  # noqa: E402
from qec.dem import decoding_graph, merged_raw_dem, raw_dem_line_count  # noqa: E402

D, T, P = 13, 13, 0.003


BASIS = {"current": "X"}


def row(name, value, expected=None, note="", both=False):
    """Expectations come from the regression table, which was measured in basis X. In basis Z they
    are attached only to rows the prompt states for both bases (``both=True``, the sigma rows)."""
    r = {"quantity": name, "value": value}
    if expected is not None and BASIS["current"] == "Z" and not both:
        r["value_in_basis_X_regression"] = expected
        expected = None
    if expected is not None:
        r["expected"] = expected
        r["match"] = value == expected
    if note:
        r["note"] = note
    return r


def decoding_rows(basis: str) -> tuple[list[dict], dict]:
    e = memory("nvidia", D, T, basis, P)
    g = decoding_graph(e.circuit)
    det = e.det
    other = Z_STAB if basis == "X" else X_STAB  # sector that sigma fixes
    moved_stab = X_STAB if basis == "X" else Z_STAB
    rows = [row("decoding graph nodes", g.n_nodes, 2101), row("decoding graph edges", g.n_edges, 10872)]

    E = np.stack([g.u, g.v], 1)
    bare = nauty_aut(g.n_nodes, E)
    parts = bare.orbit_partition()
    rows += [
        row("bare group order", bare.order, 2),
        row("bare vertex orbits", len(parts), 1555),
        row("bare vertex orbits of size 2", sum(len(o) == 2 for o in parts), 546),
        row("decoding graph twin classes (no twins means equal to node count)", contract(g.n_nodes, E).m, 2101),
    ]
    (sigma,) = bare.generators
    ep = edge_permutation(g, sigma)
    n_edge_orbits = len({frozenset((i, int(ep[i]))) for i in range(g.n_edges)})
    rows.append(row("bare edge orbits", n_edge_orbits, 8058))

    # the symmetry sigma, measured on coordinates
    mv = np.flatnonzero(sigma[: g.n_det] != np.arange(g.n_det))
    tmax = int(det.t[det.stab == moved_stab].max())
    xs, ys = det.x[det.stab == moved_stab], det.y[det.stab == moved_stab]
    cx, cy = int(xs.min() + xs.max()), int(ys.min() + ys.max())
    rule = all((int(det.x[sigma[i]]), int(det.y[sigma[i]]), int(det.t[sigma[i]]))
               == (cx - int(det.x[i]), cy - int(det.y[i]), tmax - int(det.t[i]))
               for i in np.flatnonzero(det.stab == moved_stab))
    rows += [
        row("sigma moves only detectors of the basis type", sorted(set(det.stab[mv].tolist())) == [moved_stab], True, both=True),
        row("sigma moves every detector of the basis type", len(mv) == int(np.sum(det.stab == moved_stab)), True, both=True),
        row("sigma rule (x, y, t) -> (cx - x, cy - y, tmax - t)", {"holds": rule, "cx": cx, "cy": cy, "tmax": tmax},
            {"holds": True, "cx": 26, "cy": 26, "tmax": 12}, both=True),
        row("sigma fixes the other sector and the boundary",
            bool(np.all(sigma[np.flatnonzero(det.stab == other)] == np.flatnonzero(det.stab == other))
                 and sigma[g.boundary] == g.boundary), True, both=True),
        row("sigma is an automorphism (unweighted)", is_automorphism(g, sigma), True, both=True),
    ]
    flip = np.flatnonzero(g.obs != 0)
    rows += [
        row("edges flipping L0", len(flip), 91),
        row("edges flipping L0 that are boundary edges", int(np.sum(g.v[flip] == g.boundary)), 91),
        row("images under sigma of L0 edges that flip L0", int(np.sum(g.obs[ep[flip]] != 0)), 0),
    ]

    pairs = [(i, int(ep[i])) for i in range(g.n_edges) if ep[i] > i]
    w9 = significant(g.weight, 9)
    unequal9 = [(i, j) for i, j in pairs if w9[i] != w9[j]]
    unequal_exact = [(i, j) for i, j in pairs if g.weight[i] != g.weight[j]]

    def rounds_of(i):
        return {int(det.t[a]) for a in (g.u[i], g.v[i]) if a < g.n_det}

    where = Counter()
    for i, j in unequal9:
        ts = rounds_of(i) | rounds_of(j)
        where["touches 0 or T-1" if ts & {0, tmax} else "touches 1 or T-2" if ts & {1, tmax - 1} else "only 2..T-3"] += 1
    cols9, vals9 = weight_colours(g.weight, 9)
    n2, E2, c2 = subdivided(g.n_nodes, g.boundary, g.u, g.v, cols9)
    wgrp = nauty_aut(n2, E2, c2)
    rows += [
        row("sigma-swapped edge pairs", len(pairs), 2814),
        row("swapped pairs with unequal weight (9 significant digits)", len(unequal9), 469),
        row("swapped pairs with unequal weight (exact binary64)", len(unequal_exact), note="not in the regression table"),
        row("max weight difference over unequal pairs", round(max(abs(g.weight[i] - g.weight[j]) for i, j in unequal9), 3), 0.887),
        row("unequal pairs touching rounds 0 or T-1", where["touches 0 or T-1"], 445),
        row("unequal pairs touching rounds 1 or T-2 (and not 0, T-1)", where["touches 1 or T-2"], 24),
        row("unequal pairs only in rounds 2 to T-3", where["only 2..T-3"], 0),
        row("weighted group order (weights at 9 digits as midpoint colours)", wgrp.order, 1),
        row("sigma is an automorphism with weights (9 digits)", is_automorphism(g, sigma, weighted=True, digits=9), False),
    ]

    rt = round_templates(g, det, 9, reference_round=6)
    ov = rt.overlap()
    rows += [
        row("rounds whose template equals round 6 (weights at 9 digits)", rt.bulk_rounds(), list(range(2, 12))),
        row("bulk template edges", ov[6]["edges"], 912),
        row("round 0 edges matching the bulk template", ov[0]["match_reference"], 11),
        row("round 1 edges matching the bulk template", ov[1]["match_reference"], 731),
        row("per-round template sizes and overlaps", {str(k): v for k, v in ov.items()}),
    ]

    pg = periodic_bulk_graph(rt.by_round[6], 8)
    n3, E3, c3, vals3 = periodic_coloured(pg, 9)
    pgrp = nauty_aut(n3, E3, c3)
    # sector structure: the group restricted to each stabilizer sector
    sector_orders = {}
    for s, name in ((X_STAB, "X"), (Z_STAB, "Z")):
        keep = [i for i, nd in enumerate(pg.nodes[:-1]) if nd[0] == s]
        sub = {old: new for new, old in enumerate(keep)}
        sub[pg.boundary] = len(keep)
        m = np.array([(a in sub and b in sub) for a, b in zip(pg.u, pg.v)])
        u2 = np.array([sub[a] for a in pg.u[m]])
        v2 = np.array([sub[b] for b in pg.v[m]])
        cc, _ = weight_colours(pg.weight[m], 9)
        n4, E4, c4 = subdivided(len(keep) + 1, len(keep), u2, v2, cc)
        sector_orders[name] = nauty_aut(n4, E4, c4).order
    rows += [
        row("periodic bulk graph (8 layers) weighted group order", pgrp.order, 256),
        row("periodic bulk graph sector group orders (each sector with the boundary)", sector_orders, {"X": 16, "Z": 16},
            note="D8 has order 16; 256 = 16 x 16 if the sectors are independent"),
        row("distinct bulk weights (9 digits)", len(vals3), 17),
        row("periodic bulk graph nodes and edges", {"nodes": pg.n, "edges": len(pg.u)}),
    ]

    tc = translation_classes(g, det, rt.bulk_rounds(), 9)
    tsc = merge_classes(tc, ep, w9)
    rows += [
        row("weight-table classes under translation", int(tc.max()) + 1, 2664),
        row("weight-table classes under translation and sigma", int(tsc.max()) + 1, 1994),
        row("distinct weights at 9 significant digits", len(vals9), 64),
        row("distinct weights, exact binary64", len(np.unique(g.weight)), 78,
            note="see docs/DECISIONS.md, regression differences"),
    ]

    # sigma on tensor layouts
    mvmask = det.stab == moved_stab
    nv = e.nvidia_cell
    nv_coords = np.stack([det.t, nv[:, 1], nv[:, 2]], 1)
    half = np.stack([det.t, det.x // 2, det.y // 2], 1)
    rows += [
        row("sigma on NVIDIA's CNN grid (t, row, col), moved sector",
            tensor_map(sigma, nv_coords, "nvidia (t,row,col)", mvmask).as_dict()),
        row("sigma on the halved-coordinate grid (t, x/2, y/2), moved sector",
            tensor_map(sigma, half, "halved (t,x/2,y/2)", mvmask).as_dict()),
    ]
    return rows, {"sigma": sigma.tolist()}


def fault_rows(basis: str, raw_nauty: bool) -> list[dict]:
    e = memory("nvidia", D, T, basis, P)
    fg = fault_graph(e.circuit)
    E = fg.edges()
    rows = [
        row("single faults", fg.n_fault, 121239),
        row("fault graph vertices (faults + detectors + L0)", fg.n, 123340),
        row("fault graph edges", len(E), 339920),
        row("Stim raw DEM error lines", raw_dem_line_count(e.circuit), 45392),
    ]
    ref = merged_raw_dem(e.circuit)
    rows.append(row("raw DEM lines repeating a symptom set", raw_dem_line_count(e.circuit) - len(ref), 5319))

    t0 = time.time()
    c = contract(fg.n, E)
    t_contract = time.time() - t0
    fault_classes = [cl for cl in c.classes if cl[0] < fg.n_fault]
    mixed = [cl for cl in c.classes if min(cl) < fg.n_fault <= max(cl)]
    det_twins = [cl for cl in c.classes if cl[0] >= fg.n_fault and len(cl) > 1]
    keys = {fg.footprint[cl[0]] for cl in fault_classes}
    same_footprint = all(len({fg.footprint[f] for f in cl}) == 1 for cl in fault_classes)
    rel = max(abs(merged_probability(fg, cl) - ref[fg.footprint[cl[0]]]) / ref[fg.footprint[cl[0]]] for cl in fault_classes)
    rows += [
        row("twin-contracted fault graph vertices", c.m, 42174),
        row("quotient fault vertices (mechanisms)", len(fault_classes), 40073),
        row("quotient mechanisms equal the merged DEM symptom sets (exact set equality)", keys == set(ref), True),
        row("every twin class has one footprint", same_footprint, True),
        row("classes mixing faults and detectors", len(mixed), 0),
        row("detector twin classes of size > 1", len(det_twins), 0),
        row("max relative difference, merged class probability vs Stim merged DEM", float(rel),
            note="probabilities read from instructions with Stim's conversion rules (src/autsym/faultgraph.py)"),
        row("log10 |K| (twin group)", round(c.log10_K), 46581),
        row("transposition generators of K", c.num_transpositions, 81166),
        row("python contraction time (s)", round(t_contract, 2), note="this repository's contract(), not GraphCrunch"),
    ]

    t0 = time.time()
    vo, eo, how, st = graphcrunch_structure(fg.n, E)
    rows.append(row("GraphCrunch structure() on the fault graph", {
        "wall_seconds": round(time.time() - t0, 2), "how": how,
        "quotient_vertices": st["m"], "vertex_orbits": len(vo), "edge_orbits": len(eo)}))
    cli = graphcrunch_cli(fg.n, E, raw=False, timeout=1800)
    rows.append(row("GraphCrunch CLI contract+solve+lift seconds (d = 13)", cli["contract_solve_lift_seconds"],
                    note="regression value 2.7 s was measured on another host"))
    if raw_nauty:
        cli = graphcrunch_cli(fg.n, E, raw=True, timeout=1800)
        rows.append(row("GraphCrunch --raw at d = 13 (plain nauty on the uncontracted graph)", {
            "returncode": cli["returncode"], "timeout": cli["timeout"], "wall_seconds": round(cli["wall_seconds"], 1),
            "raw_seconds": cli["raw_seconds"], "identical_partition": cli["identical_partition"],
            "stderr_tail": cli["stderr"][-300:]},
            note="regression: the plain wrapper segfaulted after 26 s storing 81,166 generators"))
    return rows


def graphcrunch_d5_rows(basis: str) -> list[dict]:
    e = memory("nvidia", 5, 5, basis, P)
    fg = fault_graph(e.circuit)
    cli = graphcrunch_cli(fg.n, fg.edges(), raw=True, timeout=1800)
    return [row("GraphCrunch on the d = 5 fault graph with --raw", {
        "contract_solve_lift_seconds": cli["contract_solve_lift_seconds"], "raw_seconds": cli["raw_seconds"],
        "identical_partition": cli["identical_partition"], "lifted_orbits": cli["lifted_orbits"],
        "returncode": cli["returncode"]},
        note="regression: 0.05 s against 67.9 s for plain nauty, identical partition (other host)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-raw-nauty", action="store_true")
    ap.add_argument("--out", default=str(REPO / "results" / "phase1_reproduction.json"))
    a = ap.parse_args()
    out = {"tier": "T4-SIM", "circuit": {"family": "nvidia", "distance": D, "rounds": T, "p": P, "rotation": "XV",
                                         "noise": "NoiseModel.from_single_p", "add_boundary_detectors": False},
           "host": platform.node() and "this host (results/env.json)", "bases": {}}
    t0 = time.time()
    for basis in ("X", "Z"):
        BASIS["current"] = basis
        drows, extra = decoding_rows(basis)
        frows = fault_rows(basis, raw_nauty=(basis == "X" and not a.skip_raw_nauty))
        grows = graphcrunch_d5_rows(basis) if basis == "X" else []
        out["bases"][basis] = {"rows": drows + frows + grows, "sigma_node_permutation": extra["sigma"]}
    out["seconds"] = round(time.time() - t0, 1)
    allrows = [r for b in out["bases"].values() for r in b["rows"] if "match" in r]
    out["summary"] = {"rows_with_expectation": len(allrows), "matching": sum(r["match"] for r in allrows),
                      "mismatches": [(r["quantity"], r["value"], r["expected"]) for r in allrows if not r["match"]]}
    Path(a.out).write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(out["summary"], indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
