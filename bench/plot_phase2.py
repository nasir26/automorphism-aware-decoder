"""Phase 2 figures from results/phase2_sigma_twin.json and results/phase2_metamorphic.json.
usage: python bench/plot_phase2.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "results" / "figures"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42})


def err(rows, key, lo_hi):
    y = np.array([r[key] if r[key] is not None else np.nan for r in rows], dtype=float)
    lo = np.array([r[lo_hi][0] if r[lo_hi][0] is not None else np.nan for r in rows], dtype=float)
    hi = np.array([r[lo_hi][1] if r[lo_hi][1] is not None else np.nan for r in rows], dtype=float)
    return y, np.vstack([y - lo, hi - y])


def main():
    tw = json.loads((REPO / "results" / "phase2_sigma_twin.json").read_text())["rows"]
    ds = sorted({r["distance"] for r in tw})
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for i, d in enumerate(ds):
        rows = sorted([r for r in tw if r["distance"] == d], key=lambda r: r["p"])
        x = np.array([r["p"] for r in rows])
        c = f"C{i}"
        y, e = err(rows, "disagree_rate", "disagree_rate_wilson95")
        axs[0].errorbar(x, y, yerr=e, color=c, marker="o", ms=3, lw=1, capsize=1.5, label=f"d = {d}")
        y, e = err(rows, "p_fail_given_disagree", "p_fail_given_disagree_wilson95")
        axs[1].errorbar(x, y, yerr=e, color=c, marker="o", ms=3, lw=1, capsize=1.5)
        y, e = err(rows, "p_fail_given_agree", "p_fail_given_agree_wilson95")
        axs[1].errorbar(x, y, yerr=e, color=c, marker="s", ms=3, lw=1, ls="--", capsize=1.5)
        y, e = err(rows, "recall", "recall_wilson95")
        axs[2].errorbar(x, y, yerr=e, color=c, marker="o", ms=3, lw=1, capsize=1.5)
        for k, r in enumerate(rows):
            if not r["conclusive"]:
                axs[0].plot(x[k], r["disagree_rate"] or np.nan, "x", color="k", ms=5)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("σ-twin disagreement rate")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("P(MWPM fails | disagree), solid\nP(MWPM fails | agree), dashed")
    axs[2].set_ylabel("recall: P(disagree | MWPM fails)")
    for a in axs:
        a.set_xlabel("physical error rate p")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_sigma_twin.pdf")

    print("figures in", FIG)


if __name__ == "__main__":
    main()
