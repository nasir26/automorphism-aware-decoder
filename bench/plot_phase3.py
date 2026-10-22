"""Phase 3 and 4 figures and derived numbers from results/phase3/report.json, results/phase3/eval and
results/phase4/eval. Writes results/phase3/summary.json (T4-SIM) and three figures.

Derived numbers: the excess over PyMatching alone (pooled over seeds, sdr50), the ratio
LER(arm) / LER(MWPM) at every d, and the slope of log10 LER against d fitted over d = 5 to 13.
usage: python bench/plot_phase3.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "results" / "figures"
EV3 = REPO / "results" / "phase3" / "eval"
EV4 = REPO / "results" / "phase4" / "eval"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42})
ARMS = [("baseline", "plain"), ("augmented", "plain"), ("equivariant", "plain"), ("baseline", "tts"), ("augmented", "tts")]
SEEDS = (0, 1, 2)
PRECS = ("int8", "int4", "ternary")
MISMATCH = "fake_quant_fp64_vs_golden_decision_mismatches_1000_shots"


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    rep = json.loads((REPO / "results" / "phase3" / "report.json").read_text())
    rows = [r for r in rep["rows"] if r["operating_point"] == "sdr50"]
    ds = sorted({r["distance"] for r in rows})
    summ = {"tier": "T4-SIM", "operating_point": "sdr50", "arms": {}, "mwpm": {}}
    for d in ds:
        r = next(r for r in rows if r["distance"] == d)
        summ["mwpm"][d] = {"fails": r["mwpm_fails"], "shots": r["shots"], "ler": r["mwpm_fails"] / r["shots"]}
    for a, v in ARMS:
        out = {}
        for d in ds:
            r = next(r for r in rows if r["distance"] == d and r["arm"] == a and r["variant"] == v)
            n = 3 * r["shots"]
            k = r["fails_total"]
            km = 3 * r["mwpm_fails"]
            out[d] = {"fails_3_seeds": k, "shots_3_seeds": n, "ler": k / n, "ler_wilson95": wilson(k, n),
                      "ratio_to_mwpm": k / km, "excess_over_mwpm": (k - km) / n,
                      "seeds_below_mwpm": sum(p["fails"] < r["mwpm_fails"] for p in r["per_seed"])}
        x = np.array(ds, dtype=float)
        y = np.log10([out[d]["ler"] for d in ds])
        out["log10_ler_slope_per_d"] = float(np.polyfit(x, y, 1)[0])
        summ["arms"][f"{a}/{v}"] = out
    y = np.log10([summ["mwpm"][d]["ler"] for d in ds])
    summ["mwpm"]["log10_ler_slope_per_d"] = float(np.polyfit(np.array(ds, float), y, 1)[0])

    # per-edge-channel profile (sdr50) at d = 5 and 13, mean over seeds
    chans = None
    prof = {}
    for a, v in ARMS:
        acc = {}
        for s in SEEDS:
            e = json.loads((EV3 / f"{a}_s{s}.json").read_text())
            chans = e["channels"]
            for t in e["test"]:
                p = t["variants"][v]["operating_points"]["sdr50"]
                acc.setdefault(t["distance"], []).append(p["applied_per_channel"])
                acc.setdefault(f"sdr_{t['distance']}", []).append(p["syndrome_density_reduction"])
        prof[f"{a}/{v}"] = {str(k): np.mean(x, axis=0).tolist() for k, x in acc.items()}
    summ["channels"] = chans
    summ["applied_per_channel_mean_over_seeds"] = {k: {d: x[str(d)] for d in ds} for k, x in prof.items()}
    summ["sdr_test_mean_over_seeds"] = {k: {d: x[f"sdr_{d}"] for d in ds} for k, x in prof.items()}

    # Phase 4
    q = {}
    for a in ("baseline", "equivariant"):
        for pr in PRECS:
            js = [json.loads((EV4 / f"{a}_s{s}_{pr}.json").read_text()) for s in SEEDS]
            ent = {"flip_check_differing_accumulators": [j["validation"]["flip_check_differing_accumulators"] for j in js],
                   "decision_mismatches_1000_shots": [j["validation"][MISMATCH] for j in js],
                   "head_acc_max_abs": max(j["validation"]["ranges"]["head_acc_max_abs_validation"] for j in js)}
            for k, t in enumerate(js[0]["test"]):
                f = [j["test"][k]["operating_points"]["sdr50"]["fails"] for j in js]
                ent[t["distance"]] = {"fails_per_seed": f, "shots": t["shots"], "ler": sum(f) / (3 * t["shots"]),
                                      "ler_wilson95": wilson(sum(f), 3 * t["shots"]), "mwpm_fails": t["mwpm_fails"]}
            q[f"{a}/{pr}"] = ent
    summ["phase4"] = q
    (REPO / "results" / "phase3" / "summary.json").write_text(json.dumps(summ, indent=1, default=float))

    FIG.mkdir(parents=True, exist_ok=True)
    lab = {f"{a}/{v}": (f"{a}" + (" + TTS" if v == "tts" else "")) for a, v in ARMS}
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4))
    x = np.array(ds)
    axs[0].plot(x, [summ["mwpm"][d]["ler"] for d in ds], "k--", marker="x", ms=4, lw=1, label="PyMatching alone")
    for i, (key, out) in enumerate(summ["arms"].items()):
        y = np.array([out[d]["ler"] for d in ds])
        lo = np.array([out[d]["ler_wilson95"][0] for d in ds])
        hi = np.array([out[d]["ler_wilson95"][1] for d in ds])
        off = (i - 2) * 0.08
        axs[0].errorbar(x + off, y, yerr=[y - lo, hi - y], color=f"C{i}", marker="o", ms=3, lw=1, capsize=1.5, label=lab[key])
        axs[1].plot(x + off, [out[d]["ratio_to_mwpm"] for d in ds], color=f"C{i}", marker="o", ms=3, lw=1)
        axs[2].plot(x + off, [summ["sdr_test_mean_over_seeds"][key][d] for d in ds], color=f"C{i}", marker="o", ms=3, lw=1)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("logical error rate (3 seeds pooled)")
    axs[1].axhline(1, color="k", ls="--", lw=0.8)
    axs[1].set_ylabel("LER / LER(PyMatching alone)")
    axs[2].set_ylabel("syndrome density reduction")
    for a in axs:
        a.set_xlabel("code distance d (= rounds)")
        a.set_xticks(ds)
    axs[0].legend(frameon=False, loc="lower left", fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_phase3_ler.pdf")

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    names = ["fp32"] + list(PRECS)
    for i, a in enumerate(("baseline", "equivariant")):
        for j, d in enumerate((5, 9)):
            y, lo, hi = [], [], []
            o = summ["arms"][f"{a}/plain"][d]
            y.append(o["ler"])
            lo.append(o["ler_wilson95"][0])
            hi.append(o["ler_wilson95"][1])
            for pr in PRECS:
                o = q[f"{a}/{pr}"][d]
                y.append(o["ler"])
                lo.append(o["ler_wilson95"][0])
                hi.append(o["ler_wilson95"][1])
            y, lo, hi = map(np.array, (y, lo, hi))
            ax.errorbar(np.arange(4) + 0.1 * (i - 0.5), y, yerr=[y - lo, hi - y], color=f"C{2 * i}", marker="os"[j],
                        ms=3, lw=1, ls="-" if j == 0 else ":", capsize=1.5, label=f"{a}, d = {d}")
        for d, ls in ((5, "-"), (9, ":")):
            if i == 0:
                ax.axhline(summ["mwpm"][d]["ler"], color="k", ls=ls, lw=0.6)
    ax.set_xticks(range(4), names)
    ax.set_yscale("log")
    ax.set_ylabel("logical error rate (sdr50, 3 seeds)")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_phase4_qat.pdf")
    print(json.dumps({k: {d: round(v[d]["ratio_to_mwpm"], 3) for d in ds} | {"slope": round(v["log10_ler_slope_per_d"], 3)}
                      for k, v in summ["arms"].items()}, indent=1))
    print("mwpm slope", round(summ["mwpm"]["log10_ler_slope_per_d"], 3))
    print("figures in", FIG)


if __name__ == "__main__":
    main()
