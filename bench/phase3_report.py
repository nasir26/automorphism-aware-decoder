"""Phase 3 report: paired comparisons of every arm against the baseline of the same seed, on the same
test shots, from the per-shot failure bits of bench/phase3_eval.py; the H3 verdict by the rule fixed
in docs/DECISIONS.md (2026-10-21 "Phase 3 evaluation protocol").

Arms compared with ``baseline/plain``: ``augmented/plain``, ``equivariant/plain``, ``baseline/tts``,
``augmented/tts``. Writes results/phase3/report.json (T4-SIM).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nn.evaluate import paired, wilson  # noqa: E402

EV = REPO / "results" / "phase3" / "eval"
SEEDS = (0, 1, 2)
ARMS = [("baseline", "plain"), ("augmented", "plain"), ("equivariant", "plain"), ("baseline", "tts"), ("augmented", "tts")]
OPS = ("sdr50", "t0.5", "best")


def bits(arm, seed, key, n):
    z = np.load(EV / f"{arm}_s{seed}.npz")
    return np.unpackbits(z[key])[:n].astype(bool)


def main():
    meta = {(a, s): json.loads((EV / f"{a}_s{s}.json").read_text()) for a in ("baseline", "augmented", "equivariant") for s in SEEDS}
    ds = [r["distance"] for r in meta[("baseline", 0)]["test"]]
    rows = []
    for d in ds:
        n = next(r for r in meta[("baseline", 0)]["test"] if r["distance"] == d)["shots"]
        mw = bits("baseline", 0, f"d{d}_mwpm", n)
        for op in OPS:
            fails = {(a, v, s): bits(a, s, f"d{d}_{v}_{op}", n) for a, v in ARMS for s in SEEDS}
            for a, v in ARMS:
                per_seed = []
                for s in SEEDS:
                    f = fails[(a, v, s)]
                    per_seed.append({"seed": s, "fails": int(f.sum()), "ler": f.mean(), "ler_wilson95": wilson(int(f.sum()), n),
                                     "vs_mwpm": paired(f, mw),
                                     "vs_baseline": None if (a, v) == ("baseline", "plain")
                                     else paired(f, fails[("baseline", "plain", s)])})
                row = {"distance": d, "shots": n, "operating_point": op, "arm": a, "variant": v, "mwpm_fails": int(mw.sum()),
                       "per_seed": per_seed, "fails_total": sum(p["fails"] for p in per_seed)}
                if (a, v) != ("baseline", "plain"):
                    fa = np.concatenate([fails[(a, v, s)] for s in SEEDS])
                    fb = np.concatenate([fails[("baseline", "plain", s)] for s in SEEDS])
                    row["pooled_vs_baseline"] = paired(fa, fb)
                    row["all_seeds_lower"] = all(p["vs_baseline"]["a_only"] < p["vs_baseline"]["b_only"] for p in per_seed)
                    row["all_seeds_higher"] = all(p["vs_baseline"]["a_only"] > p["vs_baseline"]["b_only"] for p in per_seed)
                rows.append(row)

    def verdict(a, v, dset):
        rs = [r for r in rows if r["arm"] == a and r["variant"] == v and r["operating_point"] == "sdr50" and r["distance"] in dset]
        better = all(r["all_seeds_lower"] and r["pooled_vs_baseline"]["ler_diff_ci95"][1] < 0 for r in rs)
        worse = all(r["all_seeds_higher"] and r["pooled_vs_baseline"]["ler_diff_ci95"][0] > 0 for r in rs)
        return "beats baseline" if better else "baseline better" if worse else "inconclusive"

    verdicts = {f"{a}/{v}": {"d5": verdict(a, v, {5}), "d9_11_13": verdict(a, v, {9, 11, 13})} for a, v in ARMS[1:]}
    out = {"tier": "T4-SIM", "rule": "docs/DECISIONS.md 2026-10-21 Phase 3 evaluation protocol", "verdicts": verdicts, "rows": rows}
    (REPO / "results" / "phase3" / "report.json").write_text(json.dumps(out, indent=1, default=float))
    for r in rows:
        if r["operating_point"] == "sdr50":
            pv = r.get("pooled_vs_baseline")
            print(r["distance"], f"{r['arm']}/{r['variant']}", [p["fails"] for p in r["per_seed"]], "mwpm", r["mwpm_fails"],
                  "" if pv is None else f"diff {pv['ler_diff']:+.2e} ci {pv['ler_diff_ci95'][0]:+.2e} {pv['ler_diff_ci95'][1]:+.2e}")
    print(json.dumps(verdicts, indent=1))


if __name__ == "__main__":
    main()
