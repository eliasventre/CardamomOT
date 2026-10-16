#!/usr/bin/env python3
"""Educational GRN animation with explicit causal flow and molecular levels.

This renderer deliberately keeps the graph topology fixed: regulation is shown
as animated messages travelling along the directed edges, so a viewer can
follow source -> target ordering.  The right-hand panels show the same
trajectory as current mRNA/protein bars and short time-series traces.

The topology builders and Harissa simulator are reused from ``animate_grn.py``
in this directory.  Example:

    python animate_grn_explainer.py --topology FN8 --duration 25 --speed 8
    python animate_grn_explainer.py --topology cell-cycle --n-genes 12 \
        --stimulus-duration 36 --format gif
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
from numpy.typing import NDArray

try:
    from animate_grn import TOPOLOGY_BUILDERS, simulate_harissa
except ModuleNotFoundError:
    from results_article.animated_GRN.animate_grn import (
        TOPOLOGY_BUILDERS,
        simulate_harissa,
    )


BACKGROUND = "#101522"
PANEL = "#182235"
GRID = "#34445e"
TEXT = "#f3f6fb"
MUTED = "#aebbd0"
MRNA = "#ffd166"
PROTEIN = "#56b4e9"
ACTIVATION = "#55d68a"
INHIBITION = "#ff6b6b"
STIMULUS = "#c084fc"


def _normalise_trajectory(values: NDArray) -> NDArray:
    """Return one fixed scale for the entire movie, never one scale per frame."""
    reference = float(np.percentile(values, 98))
    return np.clip(values / max(reference, 1e-8), 0.0, 1.0)


def _explanatory_positions(weights: NDArray, seed: int = 42) -> NDArray:
    """Lay out connected genes with generous spacing and short edge paths."""
    n_genes = weights.shape[0]
    try:
        import networkx as nx
    except ImportError:
        angles = np.linspace(0, 2 * np.pi, n_genes, endpoint=False)
        return np.column_stack((1.05 * np.cos(angles), 0.88 * np.sin(angles)))

    graph = nx.Graph()
    graph.add_nodes_from(range(n_genes))
    for source in range(n_genes):
        for target in range(source + 1, n_genes):
            if weights[source, target] != 0 or weights[target, source] != 0:
                graph.add_edge(source, target)

    # A larger k keeps labels and arrowheads apart; high iterations make the
    # result stable enough for a talk where the same topology is reused.
    layout = nx.spring_layout(
        graph, seed=seed, k=1.35 / np.sqrt(max(n_genes, 1)),
        iterations=300, scale=1.0, weight=None,
    )
    positions = np.array([layout[index] for index in range(n_genes)])
    positions -= positions.mean(axis=0, keepdims=True)
    span = np.ptp(positions, axis=0)
    positions *= np.array([2.25 / max(span[0], 1e-8), 1.85 / max(span[1], 1e-8)])
    return positions


def make_animation(
    positions: NDArray,
    weights: NDArray,
    t_snap: NDArray,
    m_traj: NDArray,
    p_traj: NDArray,
    gene_names: list[str],
    *,
    fps: int = 30,
    save_path: Optional[str] = None,
    dpi: int = 150,
    history_genes: int = 4,
    title: str = "How a gene-regulatory network changes over time",
) -> animation.FuncAnimation:
    """Build a synchronized network, current-level, and history animation."""
    n_genes = len(gene_names)
    n_frames = len(t_snap)
    m_norm = _normalise_trajectory(m_traj)
    p_norm = _normalise_trajectory(p_traj)
    # Show the most dynamic genes rather than eight noisy traces on top of
    # one another. The full gene-by-gene levels remain visible in the bars.
    dynamicity = np.std(m_norm, axis=0) + np.std(p_norm, axis=0)
    history_indices = np.argsort(dynamicity)[-min(history_genes, n_genes):]
    history_indices = np.sort(history_indices)

    fig = plt.figure(figsize=(14, 8), facecolor=BACKGROUND)
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1.65, 1), height_ratios=(1, 1.25),
        wspace=0.16, hspace=0.22,
    )
    ax_net = fig.add_subplot(grid[:, 0])
    ax_levels = fig.add_subplot(grid[0, 1])
    ax_history = fig.add_subplot(grid[1, 1])
    for ax in (ax_net, ax_levels, ax_history):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    # Network panel: arrows are fixed, while line brightness and particles move.
    ax_net.set_xlim(-1.4, 1.4)
    ax_net.set_ylim(-1.2, 1.2)
    ax_net.set_aspect("equal")
    ax_net.axis("off")
    edge_patches: list[FancyArrowPatch] = []
    edge_pairs: list[tuple[int, int]] = []
    edge_colors: list[str] = []
    edge_particle_offsets: list[float] = []
    node_radius = 0.105
    bidirectional = {
        (i, j) for i in range(n_genes) for j in range(i + 1, n_genes)
        if weights[i, j] != 0 and weights[j, i] != 0
    }
    for source in range(n_genes):
        for target in range(n_genes):
            weight = weights[source, target]
            if weight == 0 or source == target:
                continue
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            distance = np.hypot(x2 - x1, y2 - y1)
            if distance < 1e-9:
                continue
            ux, uy = (x2 - x1) / distance, (y2 - y1) / distance
            start = (x1 + ux * node_radius, y1 + uy * node_radius)
            end = (x2 - ux * node_radius, y2 - uy * node_radius)
            pair = tuple(sorted((source, target)))
            if pair in bidirectional:
                bend = 0.25 if source < target else -0.25
            else:
                bend = 0.14 if weight > 0 else -0.14
            color = ACTIVATION if weight > 0 else INHIBITION
            edge = FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=14,
                connectionstyle=f"arc3,rad={bend}",
                linewidth=1.4, edgecolor=color, facecolor=color,
                alpha=0.18, zorder=1,
            )
            ax_net.add_patch(edge)
            edge_patches.append(edge)
            edge_pairs.append((source, target))
            edge_colors.append(color)
            edge_particle_offsets.append((source * 0.37 + target * 0.19) % 1.0)

    protein_nodes = ax_net.scatter(
        positions[:, 0], positions[:, 1], s=np.full(n_genes, 850.0),
        c=PROTEIN, edgecolors="white", linewidths=1.1, alpha=0.9, zorder=3,
    )
    mrna_nodes = ax_net.scatter(
        positions[:, 0], positions[:, 1], s=np.full(n_genes, 130.0),
        c=MRNA, edgecolors=BACKGROUND, linewidths=0.8, zorder=4,
    )
    particles = ax_net.scatter(
        [], [], s=[], c=[], edgecolors="white", linewidths=0.35, zorder=5,
    )
    label_artists = []
    for index, name in enumerate(gene_names):
        x, y = positions[index]
        angle = np.arctan2(y, x)
        label_radius = np.hypot(x, y) + 0.17
        label_artists.append(ax_net.text(
            label_radius * np.cos(angle), label_radius * np.sin(angle), name,
            color=TEXT, fontsize=8, ha="center", va="center", zorder=6,
        ))
    # Current levels: the common y-axis makes gene-to-gene comparison honest.
    x = np.arange(n_genes)
    width = 0.38
    ax_levels.bar(x - width / 2, np.zeros(n_genes), width, color=MRNA,
                  label="mRNA", zorder=3)
    ax_levels.bar(x + width / 2, np.zeros(n_genes), width, color=PROTEIN,
                  label="Protein", zorder=3)
    ax_levels.set_ylim(0, 1.08)
    ax_levels.set_xticks(x, gene_names, fontsize=8, color=TEXT)
    ax_levels.set_yticks([0, 0.5, 1.0], ["low", "half", "high"], color=MUTED)
    ax_levels.tick_params(axis="x", length=0)
    ax_levels.grid(axis="y", color=GRID, alpha=0.7, linewidth=0.7)
    ax_levels.set_title("What is present right now?", color=TEXT, fontsize=12,
                        loc="left", pad=10, fontweight="bold")
    ax_levels.legend(frameon=False, labelcolor=TEXT, fontsize=8, ncol=2,
                     loc="upper right")
    for tick in ax_levels.get_yticklabels():
        tick.set_fontsize(8)
    level_bars = ax_levels.containers

    # History: protein traces remain uncluttered so their relative timing is
    # readable at presentation speed. RNA is already visible in the bars.
    for index in history_indices:
        ax_history.plot(t_snap, p_norm[:, index], color=PROTEIN, alpha=0.12,
                        linewidth=0.8)
    curve_colors = plt.get_cmap("tab10")(np.linspace(0.05, 0.85, len(history_indices)))
    history_protein = [
        ax_history.plot([], [], color=curve_colors[offset], linewidth=2.0)[0]
        for offset, _ in enumerate(history_indices)
    ]
    history_labels = [
        ax_history.text(
            t_snap[0], 0, gene_names[index], color=curve_colors[offset],
            fontsize=8, va="center", ha="left",
        )
        for offset, index in enumerate(history_indices)
    ]
    cursor = ax_history.axvline(t_snap[0], color=TEXT, alpha=0.8, linewidth=1)
    ax_history.set_xlim(t_snap[0], t_snap[-1])
    ax_history.set_ylim(0, 1.08)
    ax_history.set_xlabel("simulation time (hours)", color=MUTED, fontsize=9)
    ax_history.set_ylabel("relative level", color=MUTED, fontsize=9)
    ax_history.set_title(
        "Protein levels over time (selected genes)",
        color=TEXT, fontsize=12, loc="left", pad=10, fontweight="bold")
    ax_history.grid(color=GRID, alpha=0.65, linewidth=0.7)
    ax_history.tick_params(colors=MUTED, labelsize=8)
    time_text = ax_net.text(
        0.02, 0.98, "", transform=ax_net.transAxes, color=TEXT, fontsize=14,
        family="monospace", va="top", ha="left",
    )
    fig.suptitle(title, color=TEXT, fontsize=17, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.91, bottom=0.17)
    fig.legend(
        handles=[
            Line2D([0], [0], color=ACTIVATION, lw=2.5, label="activation"),
            Line2D([0], [0], color=INHIBITION, lw=2.5, label="inhibition"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=PROTEIN,
                   markersize=9, label="protein (blue outer circle)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=MRNA,
                   markersize=6, label="mRNA (yellow centre)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=TEXT,
                   markersize=5, label="moving particle = regulatory message"),
        ],
        loc="lower center", bbox_to_anchor=(0.30, 0.075), ncol=2,
        frameon=False, labelcolor=TEXT, fontsize=8.5,
    )

    def update(frame: int) -> list:
        current_mrna = m_norm[frame]
        current_protein = p_norm[frame]
        protein_nodes.set_sizes(500 + 1050 * current_protein)
        mrna_nodes.set_sizes(35 + 360 * current_mrna)

        signals = np.array([
            current_protein[source] * abs(weights[source, target])
            for source, target in edge_pairs
        ])
        signal_scale = max(float(np.max(signals, initial=0.0)), 1e-8)
        particle_xy = []
        particle_sizes = []
        particle_colors = []
        for edge_index, (source, target) in enumerate(edge_pairs):
            strength = signals[edge_index] / signal_scale
            edge_patches[edge_index].set_alpha(0.10 + 0.75 * strength)
            edge_patches[edge_index].set_linewidth(1.0 + 2.0 * strength)
            if strength < 0.06:
                continue
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            # A train of particles makes direction visible without moving nodes.
            phase = (frame / max(fps * 0.55, 1) + edge_particle_offsets[edge_index]) % 1
            for particle_phase in (phase, (phase + 0.5) % 1):
                eased = particle_phase * particle_phase * (3 - 2 * particle_phase)
                particle_xy.append((x1 + eased * (x2 - x1),
                                    y1 + eased * (y2 - y1)))
                particle_sizes.append(18 + 42 * strength)
                particle_colors.append(edge_colors[edge_index])
        particles.set_offsets(np.asarray(particle_xy) if particle_xy else np.empty((0, 2)))
        particles.set_sizes(particle_sizes)
        particles.set_color(particle_colors)

        for bar, value in zip(level_bars[0], current_mrna):
            bar.set_height(value)
        for bar, value in zip(level_bars[1], current_protein):
            bar.set_height(value)
        for line, label, index in zip(history_protein, history_labels, history_indices):
            line.set_data(t_snap[:frame + 1], p_norm[:frame + 1, index])
        label_levels = p_norm[frame, history_indices].copy()
        order = np.argsort(label_levels)
        for offset in range(1, len(order)):
            previous = order[offset - 1]
            current = order[offset]
            label_levels[current] = max(
                label_levels[current], label_levels[previous] + 0.035
            )
        if len(label_levels) and label_levels.max() > 1.02:
            label_levels -= label_levels.max() - 1.02
        for label, level in zip(history_labels, label_levels):
            label.set_position((t_snap[frame], np.clip(level, 0.02, 1.02)))
        cursor.set_xdata([t_snap[frame], t_snap[frame]])
        time_text.set_text(f"t = {t_snap[frame]:6.1f} h")
        return [protein_nodes, mrna_nodes, particles, cursor, time_text,
                *edge_patches, *history_protein, *history_labels,
                *level_bars[0], *level_bars[1]]

    return animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
        repeat=True, init_func=lambda: update(0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=list(TOPOLOGY_BUILDERS),
                        default="cell-cycle")
    parser.add_argument("--n-genes", type=int, default=15)
    parser.add_argument("--duration", type=float, default=45,
                        help="Wall-clock duration of the movie in seconds.")
    parser.add_argument("--speed", type=float, default=10,
                        help="Simulated hours per wall-clock second.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--stimulus-duration", type=float, default=None)
    parser.add_argument("--history-genes", type=int, default=4,
                        help="Number of dynamic genes shown in the history panel.")
    parser.add_argument("--use-numba", action="store_true")
    args = parser.parse_args()

    if args.topology == "cell-cycle":
        topology = TOPOLOGY_BUILDERS[args.topology](
            n_genes=args.n_genes, seed=args.seed,
        )
    else:
        topology = TOPOLOGY_BUILDERS[args.topology](seed=args.seed)
    n_frames = max(2, int(args.duration * args.fps) + 1)
    t_snap, m_traj, p_traj = simulate_harissa(
        topology, duration=args.duration * args.speed, n_frames=n_frames,
        seed=args.seed + 1, use_numba=args.use_numba,
        stimulus_duration=args.stimulus_duration,
    )
    output = args.output
    if output is None:
        output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"grn_explainer_{topology['name']}.{args.format}",
        )
    animation_object = make_animation(
        _explanatory_positions(topology["inter"][1:, 1:], seed=args.seed),
        topology["inter"][1:, 1:], t_snap, m_traj, p_traj,
        topology["gene_names"], fps=args.fps, save_path=None, dpi=args.dpi,
        history_genes=args.history_genes,
        title=f"Gene regulation: {topology['name']}",
    )
    print(f"Rendering {output} ...")
    if args.format == "gif":
        animation_object.save(output, writer=animation.PillowWriter(fps=args.fps),
                              dpi=args.dpi)
    else:
        animation_object.save(
            output,
            writer=animation.FFMpegWriter(
                fps=args.fps, codec="h264", bitrate=3200,
                metadata={"artist": "CARDAMOM"},
            ),
            dpi=args.dpi,
        )
    plt.close(animation_object._fig)
    print("Done.")


if __name__ == "__main__":
    main()
