"""Phase 1 figures from results/symmetry_sweep.json and results/faultgraph_group.json (vector PDF,
text at least 8 pt at the stated size). usage: python bench/plot_phase1.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "results" / "figures"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42})
STYLE = {("nvidia", "X"): ("C0", "o", "-"), ("nvidia", "Z"): ("C0", "s", "--"),
         ("stim", "X"): ("C1", "o", "-"), ("stim", "Z"): ("C1", "s", "--")}


def series(runs, f):
    out = {}
    for r in runs:
        k = (r["family"], r["basis"])
        v = f(r)
        if v is not None:
            out.setdefault(k, ([], []))
            out[k][0].append(r["distance"])
            out[k][1].append(v)
    return out


def lines(ax, s, label=lambda k: f"{k[0]}, basis {k[1]}"):
    for k, (x, y) in s.items():
        c, m, ls = STYLE[k]
        ax.plot(x, y, color=c, marker=m, ls=ls, ms=4, lw=1, label=label(k))


def main():
    sw = json.loads((REPO / "results" / "symmetry_sweep.json").read_text())["runs"]
    FIG.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    lines(ax, series(sw, lambda r: r["h1"]["twin_compression_ratio"]))
    ax.set_xlabel("distance d (T = d)")
    ax.set_ylabel("fault graph vertices / quotient vertices")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_twin_compression.pdf")

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4))
    lines(axs[0], series(sw, lambda r: r["decoding"]["bare_group_order"]))
    axs[0].set_yscale("log", base=2)
    axs[0].set_title("decoding graph, bare")
    lines(axs[1], series(sw, lambda r: r["decoding"]["weighted_group_order_9"]))
    axs[1].set_yscale("log", base=2)
    axs[1].set_title("decoding graph, weights at 9 digits")
    for a in axs:
        a.set_xlabel("distance d (T = d)")
        a.set_ylabel("|Aut|")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_group_orders.pdf")

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for k, off in ((("nvidia", "X"), -0.15), (("stim", "X"), 0.15)):
        c, m, _ = STYLE[k]
        first = True
        for r in sw:
            if (r["family"], r["basis"]) != k:
                continue
            refl = r["decoding"]["reflections"][r["decoding"]["sigma"]]
            tmax = r["decoding"]["time_layers"] - 1
            for t in refl["weight_breaking_rounds"]:
                y = t if t <= tmax // 2 else t - tmax - 1
                ax.plot(r["distance"] + off, y, marker=m, color=c, ls="", ms=4,
                        label=f"{k[0]}, basis X" if first else None)
                first = False
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set_xlabel("distance d (T = d)")
    ax.set_ylabel("layer breaking σ\n(t from the start, t - T from the end)")
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_sigma_breaking_rounds.pdf")

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    s1 = series(sw, lambda r: r["decoding"]["edges"] / r["decoding"]["weight_classes_translation"]
                if r["decoding"].get("weight_classes_translation") else None)
    s2 = series(sw, lambda r: r["decoding"]["reflections"][r["decoding"]["sigma"]].get("weight_compression_translation_and_this"))
    for k in (("nvidia", "X"), ("stim", "X")):
        c, m, _ = STYLE[k]
        ax.plot(*s1[k], color=c, marker=m, ls=":", ms=4, lw=1, label=f"{k[0]}, translation")
        ax.plot(*s2[k], color=c, marker=m, ls="-", ms=4, lw=1, label=f"{k[0]}, translation and σ")
    ax.set_xlabel("distance d (T = d), basis X")
    ax.set_ylabel("edges / weight classes")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_weight_compression.pdf")

    fgp = REPO / "results" / "faultgraph_group.json"
    if fgp.exists():
        fg = json.loads(fgp.read_text())["runs"]
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        for fam, c in (("nvidia", "C0"), ("stim", "C1")):
            pts = [r for r in fg if r["family"] == fam and r["basis"] == "X"]
            ax.plot([r["distance"] for r in pts], [r["contract_solve_lift_seconds"] for r in pts], color=c,
                    marker="o", ms=4, lw=1, label=f"{fam}: contract, solve, lift")
            raw = [r for r in pts if "raw_seconds" in r]
            ax.plot([r["distance"] for r in raw], [r["raw_seconds"] for r in raw], color=c, marker="x", ms=5,
                    ls="--", lw=1, label=f"{fam}: nauty on the raw graph")
        ax.set_yscale("log")
        ax.set_xlabel("distance d (T = d), basis X")
        ax.set_ylabel("seconds (this host)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "fig_graphcrunch_timing.pdf")
    print("figures in", FIG)


if __name__ == "__main__":
    main()
