"""
Schematic of the four ordering variants of the pseudotime CardamomOT (``PLAN.md`` §5).

Draws, side by side, the *calculation steps* of the base (temporal) model and of the three
pseudotime options, and — under them — the machinery they all share and feed.  Two arrows close the
loop:

* each option turns its own quantity into ``τ``, which becomes the **bins**;
* the bins give the **E-step** (Θ, and the protein state ``P`` of each cell), which are fed **back**
  into the ordering of the next iteration.

Usage::

    conda run -n cardamom_env python CardamomOT/pseudotime/figure_methods.py
    # -> data/_figures/methods_schematic.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "data" / "_figures"

BASE = "#5c5c5c"
OPT3 = "#1f77b4"
OPT1 = "#2ca02c"
OPT2 = "#c0392b"
SHARED = "#e08214"

COLUMNS = [
    dict(x=2.30, color=BASE, title="Base — temporal CardamomOT",
         sub="the sampling time is known",
         steps=["obs['time']\nknown time of every cell",
                "bin the cells at the K time points\n(balanced: N cells per time)",
                "no ordering step\nτ = the given time"]),
    dict(x=6.07, color=OPT3, title="Option 3 — mechanistic Palantir",
         sub="τ = shortest path in a Θ-weighted kNN graph",
         steps=["1. features = expected burst modes\nΣm π·k  (gene-weighted, standardised)",
                "2. kNN graph (k = 60)\nedge cost: mechanistic / plausibility / geom.",
                "3. Dijkstra from the roots → d(c₀, c)",
                "4. waypoints (max-min) + consensus\nτ ∝ min_w [d(c₀,w) + d(w,c)] / 2"]),
    dict(x=9.84, color=OPT1, title="Option 1 — partial OT",
         sub="τ = waves of transport, one mass budget at a time",
         steps=["1. frontier = reached cells (leading edge)\ncandidates = kNN of the frontier, unexplained",
                "2. cost C(c,c') = CE(kon(Pc→c'), k_mix(c'))\n+ α‖·‖   — same PDMP bridge as the E-step",
                "3. partial OT: γ minimises ⟨C,γ⟩ + ε·KL\nsubject to Σγ = β   (ot.partial)",
                "4. τ(c') = Σc γ·(τ(c) + Δτ) / Σc γ\nnew cells join the frontier → next wave"]),
    dict(x=13.61, color=OPT2, title="Option 2 — path-dependent TSP",
         sub="τ = cumulative length of a tour through the cells",
         steps=["1. initial tour = cells sorted by the previous τ\n(at the first iteration: Option 1 or 3)",
                "2. propagate P along the tour\nstep cost depends on the state reached",
                "3. windowed 2-opt: reverse ≤ window cells,\nre-propagating only that window",
                "4. τ = cumulative path length / total"]),
]

SHARED_STEPS = [
    "NB mixture on the bins\nper-bin mode masses ν\nforce_basins",
    "E-step\nOT between consecutive bins\ncost = network CE + mRNA + geom\nPDMP bridge find_next_prot",
    "M-step\nΘ by cross-entropy\nσ(ΘP) vs sigmoid_target(kon)",
    "α\nswitching fraction\ninside a step",
    "basins / modes\nper-bin EMD\nposteriors ↔ modes",
    "growth pass\nunbalanced OT on the\nsame couplings → R_opt",
    "refine\nd₁/d₀ + ProliferationMLP\nthen simulate_network",
]


def box(ax, x, y, w, h, text, fc, ec, fs=7.4, weight="normal", ls="-", fill=True, z=2, tc=None):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.07",
        linewidth=1.2, edgecolor=ec, facecolor=fc if fill else "none",
        linestyle=ls, alpha=0.92 if fill else 1.0, zorder=z)
    ax.add_patch(patch)
    if tc is None:                      # dark text on a light face, white on a coloured one
        tc = "white" if fc not in ("none", "white") else ec
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight,
            zorder=z + 1, linespacing=1.35, color=tc)
    return patch


def arrow(ax, p1, p2, color="0.35", lw=1.2, ls="-", rad=0.0, z=1):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=11, linewidth=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=z))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16.2, 10.4))
    ax.set_xlim(-0.55, 16.55)
    ax.set_ylim(0.2, 10.9)
    ax.axis("off")

    # ---------------- title -------------------------------------------------
    ax.text(8.0, 10.62, "Pseudotime CardamomOT — the four ordering variants",
            ha="center", va="center", fontsize=15, weight="bold")
    ax.text(8.0, 10.22,
            "same bursty model, same mixture, same E/M/basin/growth code — only the step that turns "
            "the cells into a time axis changes",
            ha="center", va="center", fontsize=9.6, style="italic", color="0.3")

    # ---------------- the ordering columns ---------------------------------
    step_h, step_gap = 0.60, 0.15
    x_left, x_right = -0.30, 16.30
    for col in COLUMNS:
        x, color = col["x"], col["color"]
        box(ax, x, 9.10, 3.46, 0.46, col["title"], color, color, fs=9.4, weight="bold")
        ax.text(x, 8.62, col["sub"], ha="center", va="center", fontsize=7.6,
                style="italic", color=color)
        y = 8.16
        for step in col["steps"]:
            h = step_h + (0.16 if step.count("\n") == 2 else 0.0)
            box(ax, x, y, 3.40, h, step, "white", color, fs=7.3)
            ax.text(x - 1.62, y + h / 2 + 0.10, "", fontsize=1)   # keeps the bbox tight
            y -= h + step_gap
        # τ leaves the column and enters the shared band
        ax.text(x + 0.20, 4.86, "τ", ha="left", va="center", fontsize=9.4,
                weight="bold", color=color)
        arrow(ax, (x, 4.66), (x, 4.02), color=color, lw=1.7)

    # ---------------- the shared band --------------------------------------
    n, w, gap = len(SHARED_STEPS), 2.06, 0.20
    total = n * w + (n - 1) * gap
    x0 = 8.0 - total / 2
    band_top, band_bot = 4.02, 1.62
    ax.add_patch(FancyBboxPatch((x_left + 0.10, band_bot - 0.16),
                                (x_right - 0.10) - (x_left + 0.10),
                                band_top - band_bot + 0.30,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.6, edgecolor=SHARED, facecolor=SHARED,
                                alpha=0.07, linestyle="--", zorder=0))
    ax.text(x_left + 0.18, 1.30,
            "the dashed band is shared by all four — unchanged base code (τ only changes which bin a cell falls in)",
            ha="left", va="center", fontsize=8.0, weight="bold", color=SHARED)
    centres = []
    for i, text in enumerate(SHARED_STEPS):
        cx = x0 + w / 2 + i * (w + gap)
        centres.append(cx)
        box(ax, cx, 3.20, w, 1.28, text, "white", SHARED, fs=6.8)
        if i:
            arrow(ax, (cx - w / 2 - gap + 0.02, 3.20), (cx - w / 2 - 0.02, 3.20),
                  color=SHARED, lw=1.3)

    # ---------------- the two feedback arrows ------------------------------
    # (i) the network and the protein state go back into the ordering of all four columns
    y_bus, x_bus = 9.70, -0.42
    ax.plot([x_bus, x_bus], [2.90, y_bus], color=SHARED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.plot([x_bus, COLUMNS[-1]["x"]], [y_bus, y_bus], color=SHARED, lw=1.5,
            ls=(0, (4, 3)), zorder=1)
    for col in COLUMNS:
        ax.add_patch(FancyArrowPatch((col["x"], y_bus), (col["x"], 9.34), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1.4, color=SHARED,
                                     linestyle=(0, (4, 3)), shrinkA=1, shrinkB=1, zorder=1))
    ax.text(-0.30, 6.10, "Θ  and  P\n(network, per-cell proteins)\n→ next iteration's ordering",
            ha="center", va="center", fontsize=7.6, color=SHARED, weight="bold", rotation=90)

    # (ii) the mixture is re-fitted on the new bins: a self-loop on its own box
    ax.add_patch(FancyArrowPatch(
        (centres[0] - 0.62, 2.56), (centres[0] + 0.62, 2.56), arrowstyle="-|>",
        mutation_scale=10, linewidth=1.2, color="0.45", linestyle=(0, (3, 3)),
        connectionstyle="arc3,rad=-0.85", shrinkA=1, shrinkB=1, zorder=1))
    ax.text(centres[0], 1.86, "re-fit on the new bins\n(every 5 iterations)",
            ha="center", va="center", fontsize=6.8, color="0.4", style="italic")

    # ---------------- the outputs -----------------------------------------
    ax.text(8.0, 0.86,
            "outputs:  τ (the cell axis)   ·   Θ (the network)   ·   R_opt (net growth)   ·   "
            "d₁, d₀ (degradations)   ·   simulated cells",
            ha="center", va="center", fontsize=8.6, weight="bold", color="0.25")
    ax.text(8.0, 0.50,
            "evaluated against the truth by AUPRC / AUROC (network), ρ and Kendall-τb (τ), "
            "order accuracy, separation and recovered gap ratios (irregular snapshot design)",
            ha="center", va="center", fontsize=7.8, color="0.4")

    fig.savefig(OUT / "methods_schematic.pdf", bbox_inches="tight")
    fig.savefig(OUT / "methods_schematic.png", dpi=190, bbox_inches="tight")
    print(f"figure -> {OUT / 'methods_schematic.pdf'} (+ .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
