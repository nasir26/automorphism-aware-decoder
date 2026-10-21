"""Phase 2, item 2: metamorphic invariants of minimum-weight matching under the computed symmetries.

1. σ on the NVIDIA d = 13 (and other d) decoding graphs with σ-symmetrised weights
   w'(e) = (w(e) + w(σe)) / 2: weight and logical class of decode(s) against the pull-back of
   decode(σ s); edge-set equality reported separately (ties). Syndromes are sampled from the true
   circuit; 5,000 shots per point (regression: 100% weight and logical; 75.7%, 53.2%, 37.7% edge sets
   at p = 0.003, 0.005, 0.007).
2. Time translation and both sector reflections on periodic bulk graphs (P layers wrapped in time),
   where they are exact weighted automorphisms (Phase 1). Errors are sampled edge by edge with the
   template probabilities; the periodic graph has no observable, so only weights and edge sets are
   compared.

Writes results/phase2_metamorphic.json (T4-SIM).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from autsym.circuits import X_STAB, Z_STAB, memory  # noqa: E402
from autsym.orbits import reflection_candidates  # noqa: E402
from autsym.periodic import periodic_bulk_graph, periodic_is_automorphism, periodic_reflection, periodic_translation, round_templates  # noqa: E402
from autsym.symmetry import edge_permutation, is_automorphism  # noqa: E402
from qec.dem import DecodingGraph, decoding_graph  # noqa: E402
from qec.matching import MatchingDecoder  # noqa: E402
from verify.metamorphic import check  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return (c - h, c + h)


def sigma_points(shots: int, seed: int) -> list[dict]:
    out = []
    for fam, d, basis, p in [("nvidia", 13, "X", 0.003), ("nvidia", 13, "X", 0.005), ("nvidia", 13, "X", 0.007),
                             ("nvidia", 13, "Z", 0.005), ("nvidia", 9, "X", 0.005), ("nvidia", 5, "X", 0.005),
                             ("stim", 13, "X", 0.005), ("stim", 9, "X", 0.005)]:
        e = memory(fam, d, d, basis, p)
        g = decoding_graph(e.circuit)
        sig = reflection_candidates(e.det, g.n_nodes)[f"reflect_{basis}"]
        assert is_automorphism(g, sig)
        ep = edge_permutation(g, sig)
        wsym = (g.weight + g.weight[ep]) / 2
        dec = MatchingDecoder(g, wsym)
        dets = e.circuit.compile_detector_sampler(seed=seed).sample(shots)
        t = check(dec, g, sig, ep, dets).as_dict()
        t.update(family=fam, distance=d, basis=basis, p=p, element="sigma", weights="sigma-symmetrised",
                 edges_equal_wilson95=wilson(t["edges_equal"], t["shots"]))
        out.append(t)
        print(fam, d, basis, p, t["rate_weight_equal_rel1e9"], t["rate_logical_equal"], t["rate_edges_equal"], flush=True)
    return out


def periodic_points(shots: int, seed: int) -> list[dict]:
    out = []
    rng = np.random.default_rng(seed)
    for fam in ("nvidia", "stim"):
        e = memory(fam, 13, 13, "X", 0.005)
        g = decoding_graph(e.circuit)
        rt = round_templates(g, e.det, 9)
        for P in (8, 16):
            pg = periodic_bulk_graph(rt.by_round[rt.reference_round], P)
            prob = 1 / (1 + np.exp(pg.weight))
            pgr = DecodingGraph(n_det=pg.n - 1, u=pg.u.astype(np.int64), v=pg.v.astype(np.int64), prob=prob,
                                weight=pg.weight.astype(np.float64), obs=np.zeros(len(pg.u), dtype=np.uint64))
            dec = MatchingDecoder(pgr)
            # sample independent edge errors, syndrome = parity at detectors
            err = rng.random((shots, len(pg.u))) < prob
            syn = np.zeros((shots, pg.n), dtype=np.uint8)
            for k in range(shots):
                idx = np.flatnonzero(err[k])
                np.add.at(syn[k], pg.u[idx], 1)
                np.add.at(syn[k], pg.v[idx], 1)
            syn = (syn[:, : pg.n - 1] % 2).astype(np.uint8)
            elems = {"translate_1": periodic_translation(pg, 1), "translate_3": periodic_translation(pg, 3),
                     "reflect_X": periodic_reflection(pg, X_STAB), "reflect_Z": periodic_reflection(pg, Z_STAB)}
            for name, perm in elems.items():
                assert periodic_is_automorphism(pg, perm)
                ep = edge_permutation(pgr, perm)
                t = check(dec, pgr, perm, ep, syn).as_dict()
                t.update(family=fam, distance=13, p=0.005, period=P, element=name, weights="template (exact automorphism)",
                         note="no observable in the periodic graph; logical_equal is trivially all shots")
                out.append(t)
                print(fam, "P", P, name, t["rate_weight_equal_rel1e9"], t["rate_edges_equal"], flush=True)
    return out


def main():
    res = {"tier": "T4-SIM", "decoder": "pymatching 2.4.0 via qec.matching.MatchingDecoder",
           "sigma": sigma_points(5000, 20261021), "periodic": periodic_points(2000, 20261022)}
    (REPO / "results" / "phase2_metamorphic.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
