"""Phase 1 sweep: symmetry structure of surface code memory decoding problems across distance, basis
and noise family. Writes results/symmetry_sweep.json (tier T4-SIM) and per-graph orbit tables under
results/orbit_tables/.

For every (family, d, basis) with T = d:
* H1: the fault graph's twin contraction against Stim's merged DEM (exact symptom-set equality, class
  probabilities against Stim's merged probabilities), twin compression ratio, log10 |K|;
* the decoding graph's bare and weighted groups (nauty), and which coordinate reflections are
  automorphisms, with and without weights;
* H2: rounds whose edge template equals the mid-circuit template (bulk), rounds where reflected
  edge pairs differ in weight, and the boundary-layer depth at each time end;
* the weighted group of the periodic bulk graph (8 layers) and its sector orders;
* weight-table compression: edges / classes under translation and under translation plus reflection;
* the reflection's action on the halved-coordinate grid (t, x/2, y/2).

SI1000 noise is not a built-in Stim generator option, so it is not swept (docs/DECISIONS.md).
usage: python bench/symmetry_sweep.py [--distances 3 5 7 9 11 13] [--p 0.003]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from autsym.circuits import (  # noqa: E402
    X_STAB,
    Z_STAB,
    memory,  # noqa: E402
)
from autsym.coloured import contract, significant, subdivided, weight_colours  # noqa: E402
from autsym.faultgraph import fault_graph, merged_probability  # noqa: E402
from autsym.graphcrunch import nauty_aut  # noqa: E402
from autsym.orbits import orbit_table, reflection_candidates, write_json  # noqa: E402
from autsym.periodic import (  # noqa: E402
    merge_classes,
    periodic_bulk_graph,
    periodic_coloured,
    periodic_is_automorphism,
    periodic_reflection,
    periodic_translation,
    round_templates,
    translation_classes,
)
from autsym.symmetry import edge_permutation, is_automorphism, tensor_map  # noqa: E402
from qec.dem import decoding_graph, merged_raw_dem, raw_dem_line_count  # noqa: E402


def h1(e) -> dict:
    fg = fault_graph(e.circuit)
    E = fg.edges()
    c = contract(fg.n, E)
    ref = merged_raw_dem(e.circuit)
    fcls = [cl for cl in c.classes if cl[0] < fg.n_fault]
    keys = {fg.footprint[cl[0]] for cl in fcls}
    rel = max(abs(merged_probability(fg, cl) - ref[fg.footprint[cl[0]]]) / ref[fg.footprint[cl[0]]] for cl in fcls)
    return {
        "faults": fg.n_fault, "fault_graph_vertices": fg.n, "fault_graph_edges": int(len(E)),
        "contracted_vertices": c.m, "mechanisms": len(fcls), "raw_dem_lines": raw_dem_line_count(e.circuit),
        "merged_dem_mechanisms": len(ref),
        "mechanisms_equal_merged_dem_exact": keys == set(ref),
        "every_class_one_footprint": all(len({fg.footprint[f] for f in cl}) == 1 for cl in fcls),
        "classes_mixing_faults_and_detectors": sum(1 for cl in c.classes if min(cl) < fg.n_fault <= max(cl)),
        "detector_or_observable_twin_classes": sum(1 for cl in c.classes if cl[0] >= fg.n_fault and len(cl) > 1),
        "max_rel_probability_difference": float(rel),
        "twin_compression_ratio": fg.n / c.m, "fault_compression_ratio": fg.n_fault / len(fcls),
        "log10_K": c.log10_K, "K_transpositions": c.num_transpositions,
    }


def decoding(e) -> tuple[dict, dict]:
    g = decoding_graph(e.circuit)
    det = e.det
    E = np.stack([g.u, g.v], 1)
    bare = nauty_aut(g.n_nodes, E)
    cols9, vals9 = weight_colours(g.weight, 9)
    n2, E2, c2 = subdivided(g.n_nodes, g.boundary, g.u, g.v, cols9)
    wgrp = nauty_aut(n2, E2, c2)
    cands = reflection_candidates(det, g.n_nodes)
    out = {
        "nodes": g.n_nodes, "edges": g.n_edges,
        "bare_group_log10_order": bare.log10_order, "bare_group_order": bare.order if bare.log10_order < 15 else None,
        "bare_vertex_orbits": bare.num_orbits, "bare_generators": len(bare.generators),
        "decoding_graph_twin_classes": contract(g.n_nodes, E).m,
        "weighted_group_order_9": wgrp.order if wgrp.log10_order < 15 else None,
        "distinct_weights_exact": int(len(np.unique(g.weight))), "distinct_weights_9": int(len(vals9)),
        "sigma": f"reflect_{e.basis}",
    }
    tables = {}
    rt = round_templates(g, det, 9)
    bulk = rt.bulk_rounds()
    tmax = int(det.t.max())
    out["template_reference_round"] = rt.reference_round
    out["template_overlap"] = {str(k): v for k, v in rt.overlap().items()}
    out["bulk_rounds"] = bulk
    out["time_layers"] = tmax + 1
    if bulk:
        out["boundary_depth_start"] = min(bulk)
        out["boundary_depth_end"] = tmax - max(bulk)
    w9 = significant(g.weight, 9)
    half = np.stack([det.t, det.x // 2, det.y // 2], 1)
    tc = translation_classes(g, det, bulk, 9) if bulk else None
    if tc is not None:
        out["weight_classes_translation"] = int(tc.max()) + 1
        tables["translation"] = orbit_table(g, tc)
    out["reflections"] = {}
    for name, p in cands.items():
        c = {"automorphism": is_automorphism(g, p),
             "automorphism_weighted_9": is_automorphism(g, p, weighted=True, digits=9),
             "preserves_observable_masks": is_automorphism(g, p, observables=True)}
        out["reflections"][name] = c
        if not c["automorphism"]:
            continue
        ep = edge_permutation(g, p)
        pairs = [(i, int(ep[i])) for i in range(g.n_edges) if ep[i] > i]
        unequal = [(i, j) for i, j in pairs if w9[i] != w9[j]]
        rounds = Counter()
        for i, j in unequal:
            for k in (i, j):
                for a in (g.u[k], g.v[k]):
                    if a < g.n_det:
                        rounds[int(det.t[a])] += 1
        broken = sorted(rounds)
        c["swapped_pairs"] = len(pairs)
        c["unequal_pairs_9"] = len(unequal)
        c["unequal_endpoint_rounds"] = {str(k): v for k, v in sorted(rounds.items())}
        c["weight_breaking_rounds"] = broken
        lo = [r for r in broken if r <= tmax // 2]
        hi = [r for r in broken if r > tmax // 2]
        c["breaking_depth_start"] = (max(lo) + 1) if lo else 0
        c["breaking_depth_end"] = (tmax - min(hi) + 1) if hi else 0
        moved = p[: g.n_det] != np.arange(g.n_det)
        tm = tensor_map(p, half, "halved", moved)
        c["on_halved_grid"] = tm.as_dict()
        c["exact_flip_on_halved_grid"] = len(tm.sums) == 1
        c["halved_grid_flip_sums"] = list(next(iter(tm.sums))) if len(tm.sums) == 1 else None
        if tc is not None:
            tsc = merge_classes(tc, ep, w9)
            c["weight_classes_translation_and_this"] = int(tsc.max()) + 1
            c["weight_compression_translation_and_this"] = g.n_edges / (int(tsc.max()) + 1)
            tables[f"translation_{name}"] = orbit_table(g, tsc)
    if bulk and len(bulk) >= 1:
        pg = periodic_bulk_graph(rt.by_round[rt.reference_round], 8)
        n3, E3, c3, vals3 = periodic_coloured(pg, 9)
        out["periodic8_weighted_group_order"] = nauty_aut(n3, E3, c3).order
        out["periodic8_distinct_weights_9"] = int(len(vals3))
        out["periodic8_translation_by_1_is_weighted_automorphism"] = periodic_is_automorphism(pg, periodic_translation(pg, 1))
        for nm, sec in (("X", X_STAB), ("Z", Z_STAB)):
            pr = periodic_reflection(pg, sec)
            out[f"periodic8_reflect_{nm}_is_weighted_automorphism"] = (pr is not None and periodic_is_automorphism(pg, pr))
    return out, tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13])
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--families", nargs="+", default=["nvidia", "stim"])
    ap.add_argument("--out", default=str(REPO / "results" / "symmetry_sweep.json"))
    a = ap.parse_args()
    res = {"tier": "T4-SIM", "p": a.p, "rounds": "T = d", "si1000": "not swept: not a built-in Stim generator",
           "weight_digits": 9, "runs": []}
    for fam in a.families:
        for d in a.distances:
            for basis in "XZ":
                t0 = time.time()
                e = memory(fam, d, d, basis, a.p)
                dec, tables = decoding(e)
                r = {"family": fam, "distance": d, "rounds": d, "basis": basis, "h1": h1(e), "decoding": dec}
                r["seconds"] = round(time.time() - t0, 2)
                res["runs"].append(r)
                for k, tab in tables.items():
                    write_json(REPO / "results" / "orbit_tables" / f"{e.tag}_{k}.json", tab)
                print(fam, d, basis, r["seconds"], "s", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
