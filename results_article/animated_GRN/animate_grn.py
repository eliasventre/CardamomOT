#!/usr/bin/env python3
"""
Animated Gene Regulatory Network — Multi-topology GRN illustration.

Simulates a GRN using the harissa bursty PDMP model (stochastic transcriptional
bursting) and produces an animated network graph where node sizes reflect
protein (outer disc) and mRNA (inner dot) levels, and edge opacity reflects
instantaneous regulatory signal strength.

Supported topologies
--------------------
  cell-cycle    Circular ring with feed-forward activation + delayed inhibition
                (default 15 genes, configurable via --n-genes).  Produces a
                travelling wave evocative of the cell cycle, with G1/S/G2/M
                phase quadrants and a phase arc.

  BN8           8-gene benchmark network from simulate_data_BN8.py.
  FN8           8-gene feed-forward network from simulate_data_FN8.py.
  CN5           5-gene cycle network from simulate_data_CN5.py.
  FN4           4-gene feed-forward network from simulate_data_FN4.py.

The non-cell-cycle topologies use a force-directed (spring) layout when
networkx is available, falling back to a circular layout.

Output formats
--------------
  .mp4  — best for PowerPoint / Beamer (requires ffmpeg)
  .gif  — universal, smaller filesize, good for quick previews

Usage
-----
  python animate_grn.py                                          # cell-cycle, 15 genes
  python animate_grn.py --topology BN8                           # BN8 benchmark
  python animate_grn.py --topology cell-cycle --n-genes 20       # larger ring
  python animate_grn.py --topology cell-cycle --stimulus-duration 48   # 48h stimulus pulse
  python animate_grn.py --topology FN4 --duration 120 --format gif
  python animate_grn.py --topology CN5 --no-save                  # preview only
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG_COLOR = "#0a0a14"
ACTIVATION_COLOR = "#4caf50"
INHIBITION_COLOR = "#f44336"
NODE_GREY = "#888899"
NODE_EDGE = "#ccccdd"
BLACK = "#111111"
QUADRANT_COLORS = {
    "G1": "#bbdefb",
    "S":  "#e1bee7",
    "G2": "#ffe0b2",
    "M":  "#ffcdd2",
}

# ===================================================================
# 1.  Topology registry
# ===================================================================
# Each entry is a dict with keys:
#   name          : str
#   n_genes       : int  (biological genes only)
#   d0            : float  mRNA degradation rate
#   d1            : float  protein degradation rate
#   basal         : (G+1,) array  (index 0 = stimulus)
#   inter         : (G+1, G+1) array  full harissa interaction matrix
#   gene_names    : list[str]
#   has_phase_cycle : bool  show G1/S/G2/M quadrants + phase arc
#
# The stimulus is at index 0; biological genes are at 1..G.


def _circular_positions(n_genes: int, seed: int = 42) -> NDArray:
    """Place *n_genes* nodes evenly on a slightly jittered unit circle."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n_genes, endpoint=False)
    r = 1.0 + rng.uniform(-0.12, 0.22, size=n_genes)
    theta += rng.uniform(-0.05, 0.05, size=n_genes)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _spring_positions(
    inter_bio: NDArray, seed: int = 42,
) -> NDArray:
    """
    Force-directed layout for biological genes.

    Uses networkx if available; falls back to a circular layout otherwise.
    """
    G = inter_bio.shape[0]
    try:
        import networkx as nx
    except ImportError:
        return _circular_positions(G, seed)

    graph = nx.DiGraph()
    for i in range(G):
        graph.add_node(i)
    for i in range(G):
        for j in range(G):
            w = inter_bio[i, j]
            if w != 0:
                graph.add_edge(i, j, weight=abs(w))

    pos_dict = nx.spring_layout(
        graph, seed=seed, k=2.0 / np.sqrt(G), iterations=100,
    )
    positions = np.array([pos_dict[i] for i in range(G)])
    # Normalise to unit-circle scale
    r_max = np.max(np.hypot(positions[:, 0], positions[:, 1])) or 1.0
    return positions / r_max


# ------------------------------------------------------------------
# 1a.  cell-cycle  (configurable n_genes)
# ------------------------------------------------------------------

def _make_cell_cycle(
    n_genes: int = 15,
    weight_scale: float = 10.0,
    stim_weight: float = 10.0,
    d0: float = 0.25,
    d1: float = 0.05,
    basal_val: float = -5.0,
    seed: int = 42,
) -> dict:
    """
    Circular GRN: gene *i* activates *i*+1, *i*+2 and inhibits *i*−2.

    A stimulus → gene-1 edge kicks off a travelling wave.
    """
    G = n_genes
    s = weight_scale

    # --- interaction matrix (harissa-full: includes stimulus) ---
    inter = np.zeros((G + 1, G + 1))
    inter[0, 1] = stim_weight                # stimulus → first gene
    for i in range(G):
        inter[i + 1, (i + 1) % G + 1] = s       # fwd
        inter[i + 1, (i + 2) % G + 1] = s       # fwd-skip
        inter[i + 1, (i - 2) % G + 1] = -s      # delayed inhibition

    # --- basal ---
    basal = np.full(G + 1, basal_val)
    basal[0] = 0.0  # stimulus

    return {
        "name": "cell-cycle",
        "n_genes": G,
        "d0": d0,
        "d1": d1,
        "basal": basal,
        "inter": inter,
        "gene_names": [f"G{i+1:02d}" for i in range(G)],
        "positions": _circular_positions(G, seed),
        "has_phase_cycle": True,
    }


# ------------------------------------------------------------------
# 1b.  BN8  — from simulate_data_BN8.py
# ------------------------------------------------------------------

def _make_bn8(seed: int = 42) -> dict:
    G = 8
    inter = np.zeros((G + 1, G + 1))
    inter[0, 1] = 10
    inter[1, 2] = 10
    inter[1, 3] = 10
    inter[3, 2] = -10
    inter[2, 3] = -10
    inter[2, 2] = 5
    inter[3, 3] = 5
    inter[2, 4] = 10
    inter[3, 5] = 10
    inter[2, 5] = -10
    inter[3, 4] = -10
    inter[4, 7] = -10
    inter[5, 6] = -10
    inter[4, 6] = 10
    inter[5, 7] = 10
    inter[7, 8] = 10
    inter[6, 8] = -10

    basal = np.zeros(G + 1)
    basal[1:] = -4.0
    basal[0] = 0.0

    return {
        "name": "BN8",
        "n_genes": G,
        "d0": 0.25,
        "d1": 0.05,
        "basal": basal,
        "inter": inter,
        "gene_names": [f"G{i+1}" for i in range(G)],
        "positions": _spring_positions(inter[1:, 1:], seed),
        "has_phase_cycle": False,
    }


# ------------------------------------------------------------------
# 1c.  CN5  — from simulate_data_CN5.py
# ------------------------------------------------------------------

def _make_cn5(seed: int = 42) -> dict:
    G = 5
    inter = np.zeros((G + 1, G + 1))
    inter[0, 1] = 10
    inter[1, 2] = -10
    inter[2, 3] = -10
    inter[3, 4] = 10
    inter[4, 5] = 10
    inter[5, 1] = -10

    basal = np.zeros(G + 1)
    basal[1:] = [-5.0, 4.0, 4.0, -5.0, -5.0]

    return {
        "name": "CN5",
        "n_genes": G,
        "d0": 0.5,
        "d1": 0.1,
        "basal": basal,
        "inter": inter,
        "gene_names": [f"G{i+1}" for i in range(G)],
        "positions": _spring_positions(inter[1:, 1:], seed),
        "has_phase_cycle": False,
    }


# ------------------------------------------------------------------
# 1d.  FN4  — from simulate_data_FN4.py
# ------------------------------------------------------------------

def _make_fn4(seed: int = 42) -> dict:
    G = 4
    inter = np.zeros((G + 1, G + 1))
    inter[0, 1] = 10
    inter[1, 2] = 10
    inter[1, 3] = 10
    inter[3, 4] = 10
    inter[4, 1] = -10
    inter[2, 2] = 10
    inter[3, 3] = 10

    basal = np.full(G + 1, -5.0)
    basal[0] = 0.0

    return {
        "name": "FN4",
        "n_genes": G,
        "d0": 0.2,   # original 1.0 then /= 5
        "d1": 0.04,  # original 0.2 then /= 5
        "basal": basal,
        "inter": inter,
        "gene_names": [f"G{i+1}" for i in range(G)],
        "positions": _spring_positions(inter[1:, 1:], seed),
        "has_phase_cycle": False,
    }


# ------------------------------------------------------------------
# 1e.  FN8  — from simulate_data_FN8.py
# ------------------------------------------------------------------

def _make_fn8(seed: int = 42) -> dict:
    G = 8
    inter = np.zeros((G + 1, G + 1))
    inter[0, 1] = 10
    inter[1, 2] = 10
    inter[2, 3] = 10
    inter[3, 4] = 10
    inter[3, 5] = 10
    inter[3, 6] = 10
    inter[4, 1] = -10
    inter[5, 1] = -10
    inter[6, 1] = -10
    inter[4, 4] = 10
    inter[5, 5] = 10
    inter[6, 6] = 10
    inter[4, 8] = -10
    inter[4, 7] = -10
    inter[6, 7] = 10
    inter[7, 6] = 10
    inter[8, 8] = 10

    basal = np.full(G + 1, -5.0)
    basal[0] = 0.0

    return {
        "name": "FN8",
        "n_genes": G,
        "d0": 0.4,
        "d1": 0.08,
        "basal": basal,
        "inter": inter,
        "gene_names": [f"G{i+1}" for i in range(G)],
        "positions": _spring_positions(inter[1:, 1:], seed),
        "has_phase_cycle": False,
    }


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

TOPOLOGY_BUILDERS = {
    "cell-cycle": _make_cell_cycle,
    "BN8":        _make_bn8,
    "CN5":        _make_cn5,
    "FN4":        _make_fn4,
    "FN8":        _make_fn8,
}

# ===================================================================
# 2.  Harissa simulation  (topology-agnostic)
# ===================================================================

def simulate_harissa(
    topology: dict,
    *,
    duration: float = 60.0,
    n_frames: int = 1801,
    seed: int = 123,
    use_numba: bool = False,
    stimulus_duration: Optional[float] = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Simulate the GRN using harissa's bursty PDMP model.

    Parameters
    ----------
    topology : dict
        Topology descriptor (as returned by the registry builders).
    duration : float
        Total simulation time in hours (after burnin).
    n_frames : int
        Number of equally-spaced time points to record.
    seed : int
        Random seed.
    use_numba : bool
        Use numba-accelerated PDMP simulation.
    stimulus_duration : float or None
        If set (in hours), the stimulus gene is active only for the first
        ``stimulus_duration`` hours and then turned off (basal[0] → −9).
        When None (default), the stimulus stays on permanently.

    Returns
    -------
    t_snap : (n_frames,) ndarray
        Time points (hours).
    m_traj : (n_frames, G) ndarray
        mRNA levels (biological genes only).
    p_traj : (n_frames, G) ndarray
        Protein levels (biological genes only).
    """
    from harissa import NetworkModel

    G_bio = topology["n_genes"]

    model = NetworkModel(G_bio)
    model.d[0, :] = topology["d0"]
    model.d[1, :] = topology["d1"]
    model.basal[:] = topology["basal"]
    model.inter[:, :] = topology["inter"]

    # ── Two-phase simulation when stimulus has a finite duration ──
    if stimulus_duration is not None and 0 < stimulus_duration < duration:
        # Phase 1: stimulus ON  (0 → stimulus_duration)
        n_frames_1 = max(2, int(n_frames * stimulus_duration / duration))
        t_eval_1 = np.linspace(0, stimulus_duration, n_frames_1)
        np.random.seed(seed)
        sim1 = model.simulate(t_eval_1, burnin=5, use_numba=use_numba)

        # Phase 2: stimulus OFF  (stimulus_duration → duration)
        model.basal[0] = -9.0  # silence the stimulus gene
        n_frames_2 = n_frames - n_frames_1 + 1
        t_eval_2 = np.linspace(stimulus_duration, duration, n_frames_2)
        np.random.seed(seed + 10_000)
        # harissa's simulate() expects M0/P0 of shape (G_bio+1,) with the
        # stimulus at index 0 (M0[0] is ignored — only M0[1:] is used).
        M0_pad = np.concatenate([[0.0], sim1.m[-1]])
        P0_pad = np.concatenate([[0.0], sim1.p[-1]])
        sim2 = model.simulate(
            t_eval_2,
            M0=M0_pad,
            P0=P0_pad,
            burnin=None,
            use_numba=use_numba,
        )

        # Concatenate, dropping the duplicate frame at the seam
        t_snap = np.concatenate([sim1.t, sim2.t[1:]])
        m_traj = np.concatenate([sim1.m, sim2.m[1:]], axis=0)
        p_traj = np.concatenate([sim1.p, sim2.p[1:]], axis=0)
        return t_snap, m_traj, p_traj

    # ── Single-phase (permanent stimulus) ──
    t_eval = np.linspace(0, duration, n_frames)
    np.random.seed(seed)
    sim = model.simulate(t_eval, burnin=5, use_numba=use_numba)

    # harissa excludes the stimulus gene from sim.m / sim.p
    return sim.t, sim.m, sim.p


# ===================================================================
# 3.  Animation  (network graph)
# ===================================================================

def make_animation(
    positions: NDArray,
    W_bio: NDArray,
    t_snap: NDArray,
    m_traj: NDArray,
    p_traj: NDArray,
    gene_names: list[str],
    *,
    has_phase_cycle: bool = False,
    fps: int = 30,
    save_path: Optional[str] = None,
    dpi: int = 150,
    title: str = "Gene Regulatory Network",
) -> animation.FuncAnimation:
    """
    Single-panel animation: GRN graph with pulsing nodes and dynamic edges.

    Node sizes reflect protein (outer disc) and mRNA (inner dot) levels.
    Edge opacity reflects the instantaneous regulatory signal strength.
    When *has_phase_cycle* is True, G1/S/G2/M quadrant backgrounds and a
    phase-tracking arc are shown (suited for the cell-cycle topology).
    """
    G = positions.shape[0]
    n_frames = len(t_snap)

    # ---- figure ----
    fig, ax_net = plt.subplots(figsize=(10, 10), facecolor=BG_COLOR)
    ax_net.set_facecolor(BG_COLOR)
    ax_net.set_xlim(-1.65, 1.65)
    ax_net.set_ylim(-1.65, 1.65)
    ax_net.set_aspect("equal")
    ax_net.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04)

    # ---- background: phase quadrants (cell-cycle only) ----
    phase_patches: list = []
    if has_phase_cycle:
        quadrant_radius = 1.65
        for phase_name, t1, t2 in [
            ("G1", 0, 90), ("S", 90, 180),
            ("G2", 180, 270), ("M", 270, 360),
        ]:
            ax_net.add_patch(Wedge(
                (0, 0), quadrant_radius, t1, t2,
                facecolor=QUADRANT_COLORS[phase_name],
                edgecolor="none", alpha=0.22, zorder=0,
            ))
            mid_angle = np.radians((t1 + t2) / 2)
            lbl_r = quadrant_radius * 0.55
            ax_net.text(
                lbl_r * np.cos(mid_angle), lbl_r * np.sin(mid_angle),
                phase_name, color=BLACK,
                fontsize=22, ha="center", va="center",
                fontweight="bold", alpha=0.15, zorder=0,
            )

    # ---- edge arrows ----
    arrow_patches: list[FancyArrowPatch] = []
    arrow_indices: list[tuple[int, int]] = []
    node_r = 0.12

    bidir_pairs: set[tuple[int, int]] = set()
    for i in range(G):
        for j in range(G):
            if W_bio[i, j] != 0 and W_bio[j, i] != 0 and i < j:
                bidir_pairs.add((i, j))
                bidir_pairs.add((j, i))

    for i in range(G):
        for j in range(G):
            w = W_bio[i, j]
            if w == 0:
                continue
            xi, yi = positions[i]
            xj, yj = positions[j]
            d = np.hypot(xj - xi, yj - yi)
            if d < 1e-9:
                continue
            shrink = node_r / d
            xs = xi + (xj - xi) * shrink
            ys = yi + (yj - yi) * shrink
            xe = xj - (xj - xi) * shrink
            ye = yj - (yj - yi) * shrink
            color = ACTIVATION_COLOR if w > 0 else INHIBITION_COLOR

            if (i, j) in bidir_pairs:
                side = 1 if w > 0 else -1
                arrow = FancyArrowPatch(
                    (xs, ys), (xe, ye),
                    arrowstyle="-|>", mutation_scale=12,
                    connectionstyle=f"arc3,rad={side * 0.35:.3f}",
                    facecolor=color, edgecolor=color,
                    linewidth=1.0, alpha=0.10, zorder=1,
                    shrinkA=0, shrinkB=0,
                )
            else:
                rad = 0.22 if w > 0 else -0.22
                arrow = FancyArrowPatch(
                    (xs, ys), (xe, ye),
                    arrowstyle="-|>", mutation_scale=12,
                    connectionstyle=f"arc3,rad={rad:.3f}",
                    facecolor=color, edgecolor=color,
                    linewidth=0.8, alpha=0.10, zorder=1,
                    shrinkA=0, shrinkB=0,
                )
            ax_net.add_patch(arrow)
            arrow_patches.append(arrow)
            arrow_indices.append((i, j))

    # ---- node scatters ----
    node_size_base = 800
    scatter_prot = ax_net.scatter(
        positions[:, 0], positions[:, 1],
        s=[node_size_base] * G,
        c=NODE_GREY,
        edgecolors=NODE_EDGE, linewidths=0.7, zorder=3, alpha=0.9,
    )
    scatter_mrna = ax_net.scatter(
        positions[:, 0], positions[:, 1],
        s=[node_size_base * 0.18] * G,
        c="white", edgecolors="none", zorder=4, alpha=0.75,
    )

    # ---- gene labels ----
    for i in range(G):
        x, y = positions[i]
        r_lbl = np.hypot(x, y) + 0.17
        th = np.arctan2(y, x)
        ax_net.text(
            r_lbl * np.cos(th), r_lbl * np.sin(th), gene_names[i],
            color="white", fontsize=6.5, ha="center", va="center",
            zorder=5, alpha=0.65,
        )

    # ---- phase arc (cell-cycle only) ----
    cycle_arc: Optional[Wedge] = None
    phase_text: Optional[plt.Text] = None
    if has_phase_cycle:
        cycle_arc = Wedge(
            (0, 0), 1.55, 0, 10, width=0.04,
            facecolor=BLACK, edgecolor="none", alpha=0.45, zorder=0,
        )
        ax_net.add_patch(cycle_arc)
        phase_text = ax_net.text(
            0, -1.63, "Phase: G1", color=BLACK, fontsize=9,
            ha="center", va="top", fontfamily="monospace", alpha=0.85,
        )

    # ---- legend ----
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NODE_GREY,
               markersize=9, label="Protein", linestyle="none"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
               markersize=5, label="mRNA", linestyle="none"),
        Line2D([0], [0], color=ACTIVATION_COLOR, lw=1.5, label="Activation (+)"),
        Line2D([0], [0], color=INHIBITION_COLOR, lw=1.5, label="Inhibition (−)"),
    ]
    leg = ax_net.legend(
        handles=legend_elements, loc="upper right",
        fontsize=7.5, facecolor="#1a1a2e", edgecolor="#333355",
        labelcolor="white",
    )
    leg.set_zorder(10)

    # ---- time counter ----
    time_text = ax_net.text(
        0.02, 0.98, "t = 0.0 h", transform=ax_net.transAxes,
        color="white", fontsize=10, fontfamily="monospace",
        ha="left", va="top", alpha=0.8,
    )

    # ---- title ----
    fig.suptitle(title, color="white", fontsize=15, y=0.97, alpha=0.9)

    # ---- per-frame update ----
    def update(frame_idx: int) -> list:
        m = m_traj[frame_idx]
        p = p_traj[frame_idx]
        t_now = t_snap[frame_idx]

        # --- Protein node sizes ---
        p_ref = np.mean(p) + 1e-8
        p_scale = np.clip(p / p_ref, 0.15, 4.0)
        scatter_prot.set_sizes(node_size_base * p_scale)

        # --- mRNA node sizes ---
        m_ref = np.mean(m) + 1e-8
        m_scale = np.clip(m / m_ref, 0.15, 4.0)
        scatter_mrna.set_sizes(node_size_base * 0.22 * m_scale)

        # --- Edge alphas ---
        reg_input = p @ W_bio
        max_abs = max(np.max(np.abs(reg_input)), 1e-8)
        for k, (i_src, i_dst) in enumerate(arrow_indices):
            signal = p[i_src] * W_bio[i_src, i_dst]
            alpha = float(np.clip(abs(signal) / (max_abs + 1e-8), 0.02, 0.85))
            arrow_patches[k].set_alpha(alpha)

        # --- Node edge glow ---
        node_edge_colors = []
        for i in range(G):
            glow = 0.5 + 0.5 * float(np.clip(p_scale[i] / 4.0, 0, 1))
            g = 0.8 + 0.2 * glow
            node_edge_colors.append((g, g, g, 0.9))
        scatter_prot.set_edgecolor(node_edge_colors)

        # --- Phase arc (cell-cycle only) ---
        if has_phase_cycle and cycle_arc is not None and phase_text is not None:
            phase_vec_x = float(np.sum(p * positions[:, 0]))
            phase_vec_y = float(np.sum(p * positions[:, 1]))
            phase_angle = np.arctan2(phase_vec_y, phase_vec_x)
            phase_deg = np.degrees(phase_angle) % 360

            if phase_deg < 90:
                stage = "G1"
            elif phase_deg < 180:
                stage = "S"
            elif phase_deg < 270:
                stage = "G2"
            else:
                stage = "M"

            arc_hw = 22
            cycle_arc.set(
                theta1=np.degrees(phase_angle - np.radians(arc_hw)),
                theta2=np.degrees(phase_angle + np.radians(arc_hw)),
            )
            phase_text.set_text(f"Phase: {stage}")

        # --- Time ----
        time_text.set_text(f"t = {t_now:.1f} h")

        result = [scatter_prot, scatter_mrna, *arrow_patches, time_text]
        if has_phase_cycle and cycle_arc is not None and phase_text is not None:
            result.extend([cycle_arc, phase_text])
        return result

    # ---- build animation ----
    interval = 1000 / fps
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=interval, blit=False, repeat=True,
    )

    # ---- save ----
    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        print(f"Saving animation to {save_path} ({ext}) ...")
        if ext == ".gif":
            writer = animation.PillowWriter(fps=fps)
            ani.save(save_path, writer=writer, dpi=dpi)
        elif ext in (".mp4", ".mov", ".mkv"):
            writer = animation.FFMpegWriter(
                fps=fps, metadata={"artist": "CARDAMOM"},
                bitrate=2500, codec="h264",
            )
            ani.save(save_path, writer=writer, dpi=dpi)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2500, codec="h264")
            ani.save(save_path, writer=writer, dpi=dpi)
        print("Done.")

    return ani


# ===================================================================
# 4.  CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animated GRN — multi-topology (harissa PDMP backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available topologies: " + ", ".join(TOPOLOGY_BUILDERS),
    )
    parser.add_argument(
        "--topology", type=str, default="cell-cycle",
        choices=list(TOPOLOGY_BUILDERS),
        help="Network topology to animate (default: cell-cycle).",
    )
    # cell-cycle specific
    parser.add_argument("--n-genes", type=int, default=15,
                        help="Number of genes (cell-cycle only, default: 15).")
    parser.add_argument("--weight-scale", type=float, default=10.0,
                        help="Interaction weight magnitude (cell-cycle only, default: 10).")
    parser.add_argument("--stim-weight", type=float, default=10.0,
                        help="Stimulus → first gene weight (cell-cycle only, default: 10).")
    parser.add_argument("--d0", type=float, default=None,
                        help="Override mRNA degradation rate.")
    parser.add_argument("--d1", type=float, default=None,
                        help="Override protein degradation rate.")
    parser.add_argument("--basal", type=float, default=None,
                        help="Override basal activity (cell-cycle only).")
    # general
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Animation wall-clock duration in seconds (default: 60).")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Time multiplier: each second of animation covers "
                             "this many hours of simulation (default: 10).")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stimulus-duration", type=float, default=None,
                        help="Hours the stimulus stays active (default: None = "
                             "permanent).  E.g. --stimulus-duration 48 cuts the "
                             "stimulus after 48 h of simulated time.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--use-numba", action="store_true",
                        help="Use numba-accelerated PDMP simulation.")
    args = parser.parse_args()

    # ---- Build topology ----
    build_kwargs: dict = {"seed": args.seed}
    if args.topology == "cell-cycle":
        build_kwargs["n_genes"] = args.n_genes
        build_kwargs["weight_scale"] = args.weight_scale
        build_kwargs["stim_weight"] = args.stim_weight
        if args.d0 is not None:
            build_kwargs["d0"] = args.d0
        if args.d1 is not None:
            build_kwargs["d1"] = args.d1
        if args.basal is not None:
            build_kwargs["basal_val"] = args.basal
    elif args.n_genes != 15:
        print(f"Note: --n-genes is ignored for topology '{args.topology}'")

    topo = TOPOLOGY_BUILDERS[args.topology](**build_kwargs)

    # Allow overriding d0/d1 for any topology
    if args.d0 is not None and args.topology != "cell-cycle":
        topo["d0"] = args.d0
    if args.d1 is not None and args.topology != "cell-cycle":
        topo["d1"] = args.d1

    # Extract biological interaction submatrix for visualisation
    W_bio = topo["inter"][1:, 1:]

    n_edges = int(np.sum(np.abs(W_bio) > 0))
    n_act = int(np.sum(W_bio > 0))
    n_inh = int(np.sum(W_bio < 0))
    print(f"Topology: {topo['name']}  |  "
          f"{topo['n_genes']} genes, {n_edges} edges "
          f"({n_act} act, {n_inh} inh)  |  "
          f"d0={topo['d0']}, d1={topo['d1']}")

    # ---- Simulate ----
    sim_duration = args.duration * args.speed
    n_frames = int(args.duration * args.fps) + 1
    print(f"Simulating {sim_duration:.0f} h with harissa PDMP "
          f"({n_frames} frames, {args.speed:.0f}× speed) ...")
    t_snap, m_traj, p_traj = simulate_harissa(
        topo,
        duration=sim_duration,
        n_frames=n_frames,
        seed=args.seed + 1,
        use_numba=args.use_numba,
        stimulus_duration=args.stimulus_duration,
    )
    print(f"  {len(t_snap)} frames, "
          f"mRNA ∈ [{m_traj.min():.3f}, {m_traj.max():.3f}], "
          f"protein ∈ [{p_traj.min():.3f}, {p_traj.max():.3f}]")

    # ---- Output path ----
    if args.output:
        save_path = args.output
    elif not args.no_save:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Build a descriptive default name: e.g. grn_cell-cycle_G15.mp4
        if topo["name"] == "cell-cycle":
            stem = f"grn_{topo['name']}_G{topo['n_genes']}"
        else:
            stem = f"grn_{topo['name']}"
        save_path = os.path.join(script_dir, f"{stem}.{args.format}")
    else:
        save_path = None

    # ---- Title ----
    title = args.title or f"Gene Regulatory Network — {topo['name']}"

    # ---- Animate ----
    print("Rendering ...")
    ani = make_animation(
        topo["positions"], W_bio, t_snap, m_traj, p_traj,
        topo["gene_names"],
        has_phase_cycle=topo["has_phase_cycle"],
        fps=args.fps, save_path=save_path, dpi=args.dpi, title=title,
    )

    if args.no_save:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
