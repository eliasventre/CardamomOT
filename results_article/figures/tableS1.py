#!/usr/bin/env python
"""Generate LaTeX Table S1: Comparison of parameters for different network models.

This script simulates four gene regulatory networks (FN4, CN5, BN8, FN8) using
Harissa, runs CardamomOT inference on each, and produces a LaTeX table comparing
reference and inferred parameters (k0/d0, k1/d0, k_off/s0, and attribution accuracy).
"""

import sys
import os
import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from CardamomOT import NetworkModel as CardamomNetworkModel
from harissa import NetworkModel as HarissaNetworkModel


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def run_inference(rna_data, time, deg_rates):
    """Fit a CardamomOT NetworkModel on simulated Harissa data."""
    x = rna_data[1:, 1:].copy()
    x[:, 0] = time
    G = x.shape[1]
    model = CardamomNetworkModel(G - 1)
    model.d = deg_rates
    model.fit(x)
    return model


def kon(model_harissa, p):
    """Interaction function kon (off->on rate), given protein levels p."""
    sigma = expit(model_harissa.basal + p @ model_harissa.inter)
    Kon = sigma
    Kon[0] = 0  # Ignore stimulus
    return Kon


def correctly_attributed(model_harissa, prot_traj_harissa, model_cardamom):
    """Fraction of correctly attributed modes between Harissa and CardamomOT."""
    kon_harissa = kon(model_harissa, prot_traj_harissa)[:, 1:]
    modes_cardamom = model_cardamom.modes[:, 1:]  # Ignore stimulus
    kon_bin = (kon_harissa > 0.5).astype("int")
    modes_bin = (modes_cardamom > 0.5).astype("int")
    return (kon_bin == modes_bin).mean()


def _format_ref(values: np.ndarray) -> str:
    """Format reference (true) value as a clean decimal."""
    mean = float(np.mean(values))
    return f"{mean:.3f}".rstrip("0").rstrip(".")


def _format_inf(values: np.ndarray) -> str:
    """Format inferred value as mean ± std."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    return f"{mean:.3f}$\\pm${std:.3f}"


def _ratio_values(net_model, a_idx: int) -> np.ndarray:
    """Compute a[a_idx] / d[0] ratio for a network model."""
    a = np.asarray(net_model.a)
    d = np.asarray(net_model.d)

    if a.ndim == 1:
        num = np.atleast_1d(a[a_idx]).astype(float)
    else:
        num = np.asarray(a[a_idx, 1:], dtype=float)

    if d.ndim == 1:
        den = np.full(num.shape, float(d[0]))
    else:
        den = np.asarray(d[0, 1:], dtype=float)

    return np.asarray(num / den, dtype=float).ravel()


def _a_off_values(net_model) -> np.ndarray:
    """Extract a_off (last row of a) values from a network model."""
    a = np.asarray(net_model.a)

    if a.ndim == 1:
        values = np.atleast_1d(a[-1]).astype(float)
    else:
        values = np.asarray(a[-1, 1:], dtype=float)

    return np.asarray(values, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Common simulation parameters
# ---------------------------------------------------------------------------

C = 1000  # Number of cells
T = [0, 6, 12, 24, 36, 48, 60, 72, 84, 96]  # Time points
N = int(C / len(T))
k = np.linspace(0, C, len(T) + 1, dtype="int")


def _build_time_array():
    """Build the time-point array for C cells across T time points."""
    time = np.zeros(C, dtype="int")
    for i in range(len(T)):
        time[k[i] : k[i + 1]] = T[i]
    return time


def _init_data(time, G):
    """Initialise the Harissa-format data array (C+1 × G+2)."""
    data = np.zeros((C + 1, G + 2), dtype="int")
    data[0, 1:] = np.arange(G + 1)
    data[1:, 0] = time
    data[1:, 1] = 100 * (time > 0)  # Stimulus
    return data


# ---------------------------------------------------------------------------
# Network: FN4
# ---------------------------------------------------------------------------

def simulate_fn4():
    np.random.seed(0)
    G = 4
    time = _build_time_array()
    data = _init_data(time, G)

    model_harissa = HarissaNetworkModel(G)
    model_harissa.d[0] = 1
    model_harissa.d[1] = 0.2
    model_harissa.d /= 5
    model_harissa.basal[1:] = -5
    model_harissa.inter[0, 1] = 10
    model_harissa.inter[1, 2] = 10
    model_harissa.inter[1, 3] = 10
    model_harissa.inter[3, 4] = 10
    model_harissa.inter[4, 1] = -10
    model_harissa.inter[2, 2] = 10
    model_harissa.inter[3, 3] = 10

    prot_traj = np.ones((C, G + 1), dtype="float32")
    prot_traj[:, 0] = (time > 0).astype("float32")
    for c in range(C):
        sim = model_harissa.simulate(time[c], burnin=5)
        prot_traj[c, 1:] = sim.p[-1]
        data[c + 1, 2:] = np.random.poisson(sim.m[-1])

    model_cardamom = run_inference(data, time, model_harissa.d.copy())
    return model_harissa, model_cardamom, prot_traj


# ---------------------------------------------------------------------------
# Network: CN5
# ---------------------------------------------------------------------------

def simulate_cn5():
    np.random.seed(0)
    G = 5
    time = _build_time_array()
    data = _init_data(time, G)

    model_harissa = HarissaNetworkModel(G)
    model_harissa.d[0] = 0.5
    model_harissa.d[1] = 0.1
    model_harissa.basal[1:] = [-5, 4, 4, -5, -5]
    model_harissa.inter[0, 1] = 10
    model_harissa.inter[1, 2] = -10
    model_harissa.inter[2, 3] = -10
    model_harissa.inter[3, 4] = 10
    model_harissa.inter[4, 5] = 10
    model_harissa.inter[5, 1] = -10

    prot_traj = np.ones((C, G + 1), dtype="float32")
    prot_traj[:, 0] = (time > 0).astype("float32")
    for c in range(C):
        sim = model_harissa.simulate(time[c], burnin=5)
        prot_traj[c, 1:] = sim.p[-1]
        data[c + 1, 2:] = np.random.poisson(sim.m[-1])

    model_cardamom = run_inference(data, time, model_harissa.d.copy())
    return model_harissa, model_cardamom, prot_traj


# ---------------------------------------------------------------------------
# Network: BN8
# ---------------------------------------------------------------------------

def simulate_bn8():
    np.random.seed(0)
    G = 8
    time = _build_time_array()
    data = _init_data(time, G)

    model_harissa = HarissaNetworkModel(G)
    model_harissa.d[0] = 0.25
    model_harissa.d[1] = 0.05
    model_harissa.basal[1:] = [-4] * 8
    model_harissa.inter[0, 1] = 10
    model_harissa.inter[1, 2] = 10
    model_harissa.inter[1, 3] = 10
    model_harissa.inter[3, 2] = -10
    model_harissa.inter[2, 3] = -10
    model_harissa.inter[2, 2] = 5
    model_harissa.inter[3, 3] = 5
    model_harissa.inter[2, 4] = 10
    model_harissa.inter[3, 5] = 10
    model_harissa.inter[2, 5] = -10
    model_harissa.inter[3, 4] = -10
    model_harissa.inter[4, 7] = -10
    model_harissa.inter[5, 6] = -10
    model_harissa.inter[4, 6] = 10
    model_harissa.inter[5, 7] = 10
    model_harissa.inter[7, 8] = 10
    model_harissa.inter[6, 8] = -10

    prot_traj = np.ones((C, G + 1), dtype="float32")
    prot_traj[:, 0] = (time > 0).astype("float32")
    for c in range(C):
        sim = model_harissa.simulate(time[c], burnin=5)
        prot_traj[c, 1:] = sim.p[-1]
        data[c + 1, 2:] = np.random.poisson(sim.m[-1])

    model_cardamom = run_inference(data, time, model_harissa.d.copy())
    return model_harissa, model_cardamom, prot_traj


# ---------------------------------------------------------------------------
# Network: FN8
# ---------------------------------------------------------------------------

def simulate_fn8():
    np.random.seed(0)
    G = 8
    time = _build_time_array()
    data = _init_data(time, G)

    model_harissa = HarissaNetworkModel(G)
    model_harissa.d[0] = 0.4
    model_harissa.d[1] = 0.08
    model_harissa.basal[1:] = [-5] * 8
    model_harissa.inter[0, 1] = 10
    model_harissa.inter[1, 2] = 10
    model_harissa.inter[2, 3] = 10
    model_harissa.inter[3, 4] = 10
    model_harissa.inter[3, 5] = 10
    model_harissa.inter[3, 6] = 10
    model_harissa.inter[4, 1] = -10
    model_harissa.inter[5, 1] = -10
    model_harissa.inter[6, 1] = -10
    model_harissa.inter[4, 4] = 10
    model_harissa.inter[5, 5] = 10
    model_harissa.inter[6, 6] = 10
    model_harissa.inter[4, 8] = -10
    model_harissa.inter[4, 7] = -10
    model_harissa.inter[6, 7] = 10
    model_harissa.inter[7, 6] = 10
    model_harissa.inter[8, 8] = 10

    prot_traj = np.ones((C, G + 1), dtype="float32")
    prot_traj[:, 0] = (time > 0).astype("float32")
    for c in range(C):
        sim = model_harissa.simulate(time[c], burnin=5)
        prot_traj[c, 1:] = sim.p[-1]
        data[c + 1, 2:] = np.random.poisson(sim.m[-1])

    model_cardamom = run_inference(data, time, model_harissa.d.copy())
    return model_harissa, model_cardamom, prot_traj


# ---------------------------------------------------------------------------
# Build the LaTeX table
# ---------------------------------------------------------------------------

def build_latex_table(
    k0_ref_inf, k1_ref_inf, koff_ref_inf, attrib_inf, label="tableS2"
):
    """Assemble the full LaTeX table from pre-computed row dictionaries."""
    line_break = r"\\"

    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Comparison of reference parameters and parameters inferred by "
        r"CardamomOT for different simulated datasets. "
        r"Results show mean $\pm$ standard deviation across all genes.}",
        r"\small",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}",
        r"\hline",
        r"\multirow{2}{*}{Parameter} & \multicolumn{2}{c|}{FN4} & "
        r"\multicolumn{2}{c|}{CN5} & \multicolumn{2}{c|}{BN8} & "
        r"\multicolumn{2}{c|}{FN8} " + line_break,
        r"\cline{2-9}",
        r"& Ref & Inf & Ref & Inf & Ref & Inf & Ref & Inf " + line_break,
        r"\hline",
    ]

    row_k0 = (
        "$k_0 /d_0$"
        f" & {k0_ref_inf['FN4'][0]} & {k0_ref_inf['FN4'][1]}"
        f" & {k0_ref_inf['CN5'][0]} & {k0_ref_inf['CN5'][1]}"
        f" & {k0_ref_inf['BN8'][0]} & {k0_ref_inf['BN8'][1]}"
        f" & {k0_ref_inf['FN8'][0]} & {k0_ref_inf['FN8'][1]} " + line_break
    )

    row_k1 = (
        "$k_1 /d_0$"
        f" & {k1_ref_inf['FN4'][0]} & {k1_ref_inf['FN4'][1]}"
        f" & {k1_ref_inf['CN5'][0]} & {k1_ref_inf['CN5'][1]}"
        f" & {k1_ref_inf['BN8'][0]} & {k1_ref_inf['BN8'][1]}"
        f" & {k1_ref_inf['FN8'][0]} & {k1_ref_inf['FN8'][1]} " + line_break
    )

    row_koff = (
        "$k_{off}/s_0$"
        f" & {koff_ref_inf['FN4'][0]} & {koff_ref_inf['FN4'][1]}"
        f" & {koff_ref_inf['CN5'][0]} & {koff_ref_inf['CN5'][1]}"
        f" & {koff_ref_inf['BN8'][0]} & {koff_ref_inf['BN8'][1]}"
        f" & {koff_ref_inf['FN8'][0]} & {koff_ref_inf['FN8'][1]} " + line_break
    )

    row_attrib = (
        r"\shortstack[l]{\% of correctly\\attributed modes}"
        f" &  & {attrib_inf['FN4']}"
        f" &  & {attrib_inf['CN5']}"
        f" &  & {attrib_inf['BN8']}"
        f" &  & {attrib_inf['FN8']} " + line_break
    )

    footer = [
        row_koff,
        row_attrib,
        r"\hline",
        r"\end{tabular}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]

    return "\n".join(header + [row_k0, row_k1] + footer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Simulating FN4 …")
    harissa_fn4, cardamom_fn4, prot_fn4 = simulate_fn4()

    print("Simulating CN5 …")
    harissa_cn5, cardamom_cn5, prot_cn5 = simulate_cn5()

    print("Simulating BN8 …")
    harissa_bn8, cardamom_bn8, prot_bn8 = simulate_bn8()

    print("Simulating FN8 …")
    harissa_fn8, cardamom_fn8, prot_fn8 = simulate_fn8()

    networks = {
        "FN4": (harissa_fn4, cardamom_fn4),
        "CN5": (harissa_cn5, cardamom_cn5),
        "BN8": (harissa_bn8, cardamom_bn8),
        "FN8": (harissa_fn8, cardamom_fn8),
    }

    prot_trajectories = {
        "FN4": prot_fn4,
        "CN5": prot_cn5,
        "BN8": prot_bn8,
        "FN8": prot_fn8,
    }

    k0_ref_inf = {}
    k1_ref_inf = {}
    koff_ref_inf = {}
    attrib_inf = {}

    for name, (harissa_model, cardamom_model) in networks.items():
        k0_ref_inf[name] = (
            _format_ref(_ratio_values(harissa_model, 0)),
            _format_inf(_ratio_values(cardamom_model, 0)),
        )
        k1_ref_inf[name] = (
            _format_ref(_ratio_values(harissa_model, 1)),
            _format_inf(_ratio_values(cardamom_model, 1)),
        )
        koff_ref_inf[name] = (
            _format_ref(_a_off_values(harissa_model)),
            _format_inf(_a_off_values(cardamom_model)),
        )
        attrib = correctly_attributed(
            harissa_model, prot_trajectories[name], cardamom_model
        )
        attrib_inf[name] = f"{float(attrib):.3f}"

    latex = build_latex_table(k0_ref_inf, k1_ref_inf, koff_ref_inf, attrib_inf)
    print("\n" + latex)

    # Optionally write to file
    out_path = os.path.join(os.path.dirname(__file__), "tableS1.tex")
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table written to {out_path}")


if __name__ == "__main__":
    main()
