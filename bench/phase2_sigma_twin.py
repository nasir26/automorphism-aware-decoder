"""Phase 2, items 3 and 4: the σ-twin signal under true weights, and automorphism ensemble decoding.

For each shot s (sampled from the circuit) two minimum-weight decodes with the true weights are made:
decode(s), and decode(σ s) pulled back by σ^-1. The pull-back's logical prediction is read directly
from a second matcher whose observable mask on edge e is the mask of σ^-1(e). Recorded per point:

* disagreement rate of the two logical predictions;
* P(fail | disagree), P(fail | agree) for plain MWPM, and recall P(disagree | fail);
* automorphism ensemble decoding (Geiselhart et al.; AutDEC for quantum LDPC codes): of the two
  corrections keep the one with the smaller total TRUE weight (the pulled-back correction's weight is
  computed from its edges, only on shots where the predictions disagree, since elsewhere the choice
  cannot change the prediction); ties keep plain. For an exact minimum-weight decoder the pulled-back
  correction can never be lighter, so this measures ties and PyMatching's weight discretisation;
* for reference, the rule first implemented by mistake (``lighter_own_weight``): keep the decode
  whose own reported weight is smaller, which compares the twin under σ-permuted weights;
* cost: two decodes per shot.

Each point samples until at least ``min_disagree`` disagreeing shots or ``max_shots``; points below the
target are marked inconclusive. Intervals are Wilson 95%. Writes results/phase2_sigma_twin.json (T4-SIM).
usage: python bench/phase2_sigma_twin.py [--workers 192]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_CACHE: dict = {}


def _setup(key):
    if key in _CACHE:
        return _CACHE[key]
    import pymatching

    from autsym.circuits import memory
    from autsym.orbits import reflection_candidates
    from autsym.symmetry import edge_permutation, inverse
    from qec.dem import decoding_graph

    fam, d, basis, p = key
    e = memory(fam, d, d, basis, p)
    g = decoding_graph(e.circuit)
    sig = reflection_candidates(e.det, g.n_nodes)[f"reflect_{basis}"]
    ep = edge_permutation(g, sig)
    einv = inverse(ep)

    def matcher(obs_mask):
        m = pymatching.Matching()
        for i, (a, b) in enumerate(zip(g.u.tolist(), g.v.tolist())):
            fid = {0} if int(obs_mask[i]) & 1 else set()
            if b == g.n_det:
                m.add_boundary_edge(a, weight=float(g.weight[i]), fault_ids=fid)
            else:
                m.add_edge(a, b, weight=float(g.weight[i]), fault_ids=fid)
        return m

    m1 = matcher(g.obs)
    m2 = matcher(g.obs[einv])
    sinv = inverse(sig)[: g.n_det]
    from qec.matching import MatchingDecoder

    _CACHE[key] = (e.circuit, m1, m2, sinv, MatchingDecoder(g), sig, einv)
    return _CACHE[key]


def batch(args):
    key, seed, shots = args
    circuit, m1, m2, sinv, dec, sig, einv = _setup(key)
    from verify.metamorphic import act_on_syndrome
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    obs = obs[:, 0].astype(np.uint8)
    sd = dets[:, sinv]  # (σ s)[j] = s[σ^-1 j]
    p1, w1 = m1.decode_batch(dets, return_weights=True)
    p2, w2 = m2.decode_batch(sd, return_weights=True)
    p1, p2 = p1[:, 0], p2[:, 0]
    f1 = p1 != obs
    f2 = p2 != obs
    dis = p1 != p2
    own = np.where(w2 < w1, p2, p1)
    ens = p1.copy()
    twin_lighter_true = twin_tie_true = 0
    for k in np.flatnonzero(dis).tolist():
        e1, wt1, _ = dec.decode(dets[k])
        e2, _, _ = dec.decode(act_on_syndrome(sig, dets[k]))
        back = einv[e2]
        wt2 = dec.weight_of(back)
        if wt2 < wt1:
            ens[k] = dec.logical(back)
            twin_lighter_true += 1
        elif wt2 == wt1:
            twin_tie_true += 1
    return {"shots": shots, "fail": int(f1.sum()), "fail_twin": int(f2.sum()), "disagree": int(dis.sum()),
            "fail_and_disagree": int((f1 & dis).sum()), "fail_and_agree": int((f1 & ~dis).sum()),
            "fail_ensemble": int((ens != obs).sum()), "twin_lighter_true_weight": twin_lighter_true,
            "twin_tie_true_weight_on_disagreement": twin_tie_true,
            "fail_lighter_own_weight": int((own != obs).sum()), "twin_lighter_own_weight": int((w2 < w1).sum())}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return [c - h, c + h]


def summarise(key, tot, min_dis):
    fam, d, basis, p = key
    n = tot["shots"]
    dis = tot["disagree"]
    agree = n - dis
    r = {"family": fam, "distance": d, "basis": basis, "p": p, **tot,
         "conclusive": dis >= min_dis,
         "disagree_rate": dis / n, "disagree_rate_wilson95": wilson(dis, n),
         "ler_mwpm": tot["fail"] / n, "ler_mwpm_wilson95": wilson(tot["fail"], n),
         "ler_ensemble": tot["fail_ensemble"] / n, "ler_ensemble_wilson95": wilson(tot["fail_ensemble"], n),
         "ler_lighter_own_weight": tot["fail_lighter_own_weight"] / n,
         "p_fail_given_disagree": tot["fail_and_disagree"] / dis if dis else None,
         "p_fail_given_disagree_wilson95": wilson(tot["fail_and_disagree"], dis),
         "p_fail_given_agree": tot["fail_and_agree"] / agree if agree else None,
         "p_fail_given_agree_wilson95": wilson(tot["fail_and_agree"], agree),
         "recall": tot["fail_and_disagree"] / tot["fail"] if tot["fail"] else None,
         "recall_wilson95": wilson(tot["fail_and_disagree"], tot["fail"]),
         "decodes_per_shot": 2}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(192, os.cpu_count() or 1))
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 7, 9, 11, 13])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008])
    ap.add_argument("--family", default="nvidia")
    ap.add_argument("--basis", default="X")
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--min-disagree", type=int, default=100)
    ap.add_argument("--max-shots", type=int, default=40_000_000)
    ap.add_argument("--out", default=str(REPO / "results" / "phase2_sigma_twin.json"))
    a = ap.parse_args()
    t0 = time.time()
    rows = []
    seed = 1_000_003
    with ProcessPoolExecutor(a.workers) as ex:
        for d in a.distances:
            for p in a.ps:
                key = (a.family, d, a.basis, p)
                tot = None
                while True:
                    jobs = [(key, seed + k, a.batch) for k in range(a.workers)]
                    seed += a.workers
                    for r in ex.map(batch, jobs):
                        tot = r if tot is None else {k: tot[k] + r[k] for k in tot}
                    if tot["disagree"] >= a.min_disagree or tot["shots"] >= a.max_shots:
                        break
                row = summarise(key, tot, a.min_disagree)
                rows.append(row)
                print(d, p, row["shots"], row["disagree"], f"{row['disagree_rate']:.2e}", row["ler_mwpm"], row["ler_ensemble"],
                      row["p_fail_given_disagree"], row["p_fail_given_agree"], row["recall"], flush=True)
                Path(a.out).write_text(json.dumps({"tier": "T4-SIM", "seconds": round(time.time() - t0, 1),
                                                   "min_disagree": a.min_disagree, "max_shots": a.max_shots, "rows": rows},
                                                  indent=1))


if __name__ == "__main__":
    main()
