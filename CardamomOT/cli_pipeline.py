"""
Interactive pipeline runner for CardamomOT.

Runs the pipeline steps chosen interactively (preselected from the parameters of the project:
select_genes, build_prior_network, estimate_proliferation_rates, run_test, simulate_perturbations),
asking once for the hard-to-calibrate options (stimulus, prior, mean_forcing, force_basins,
temporal_basins; empty = workbook value or default). Each step receives -i and the options it uses.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List
import subprocess

from CardamomOT.config import STATIONARY_EXIT_CODE, STATIONARY_MESSAGE


# Pipeline steps in order ("default" is replaced by the steps given by the project parameters)
PIPELINE_STEPS = [
    {
        "id": "estimate_cell_depth",
        "name": "Cell depth",
        "script": "estimate_cell_depth.py",
        "description": "Diagnose per-cell sequencing depth effects on the whole transcriptome and, if "
                        "needed and allowed, store a depth factor per cell (obs['depth_factor'])",
    },
    {
        "id": "fit_population_anchors",
        "name": "Population anchors",
        "script": "fit_population_anchors.py",
        "description": "Correct the prescribed proliferation / transition rates of each sample on the evolution of "
                        "its cell types and population sizes (small growth and transition model), borrow them for "
                        "the samples without constraint",
    },
    {
        "id": "get_proliferation_rates",
        "name": "Proliferation rates",
        "script": "get_proliferation_rates.py",
        "description": "Estimate net proliferation rate per cell from literature gene "
                        "signatures (runs on the full gene set, before gene selection)",
    },
    {
        "id": "split_dataset",
        "name": "Train/test split",
        "script": "split_dataset.py",
        "description": "Split the cells into train/test (obs['split'] of Data/data.h5ad)",
    },
    {
        "id": "run_classical_OT",
        "name": "Classical OT",
        "script": "run_classical_OT.py",
        "description": "Waddington-OT-style couplings per sample (every gene, train cells): cell-type "
                        "transitions, fate genes, velocities",
    },
    {
        "id": "select_genes",
        "name": "Gene selection",
        "script": "select_genes.py",
        "description": "Select the genes on the train cells, write data_full/train/test",
    },
    {
        "id": "build_reference_network",
        "name": "Network constraint (optional)",
        "script": "build_reference_network.py",
        "description": "Build prior knowledge network from biological databases",
    },
    {
        "id": "get_degradation_rates",
        "name": "Kinetics",
        "script": "get_degradation_rates.py",
        "description": "Assign literature mRNA and protein degradation rates (h^-1)",
    },
    {
        "id": "infer_mixture",
        "name": "Mixture model",
        "script": "infer_mixture.py",
        "description": "Fit negative-binomial burst parameters per gene",
    },
    {
        "id": "check_mixture_to_data",
        "name": "Check mixture",
        "script": "check_mixture_to_data.py",
        "description": "Validate mixture model parameters against data",
    },
    {
        "id": "infer_network_structure",
        "name": "Network inference",
        "script": "infer_network_structure.py",
        "description": "Learn gene regulatory interactions via optimal transport",
    },
    {
        "id": "infer_network_simul",
        "name": "Network adaptation",
        "script": "infer_network_simul.py",
        "description": "Prepare network parameters for forward simulation "
                        "(optionally learns a proliferation-rate MLP)",
    },
    {
        "id": "simulate_network",
        "name": "Simulation",
        "script": "simulate_network.py",
        "description": "Generate synthetic single-cell trajectories from the learned model",
    },
    {
        "id": "check_sim_to_data",
        "name": "Check simulation",
        "script": "check_sim_to_data.py",
        "description": "Validate simulations against experimental data distribution",
    },
    {
        "id": "infer_test",
        "name": "Test — inference (optional)",
        "script": "infer_test.py",
        "description": "Infer network and simulate on held-out test set",
    },
    {
        "id": "check_test_to_train",
        "name": "Test — check (optional)",
        "script": "check_test_to_train.py",
        "description": "Compare test predictions to training observations",
    },
    {
        "id": "simulate_network_KOV",
        "name": "Perturb — KO/OV simulation",
        "script": "simulate_network_KOV.py",
        "description": "Simulate gene expression under in-silico knock-out/over-expression",
    },
    {
        "id": "check_KOV_to_sim",
        "name": "Perturb — check KO/OV",
        "script": "check_KOV_to_sim.py",
        "description": "Compare perturbation simulations to wild-type",
    },
    {
        "id": "report_results",
        "name": "Report — final PDF",
        "script": "report_results.py",
        "description": "Write the PDF report (generative model, GRN top regulators, KO/OV predictions)",
    },
]

def validate_project_structure(project_path: str) -> bool:
    """
    Check if the project has the expected structure.

    Expected structure:
        project/
        ├── Data/
        │   └── *.h5ad
        └── (will create cardamom/ if missing)
    """
    p = Path(project_path)
    if not p.exists() or not p.is_dir():
        print(f"❌ Project directory does not exist: {project_path}")
        return False

    data_dir = p / "Data"
    if not data_dir.exists():
        print(f"⚠️  Data/ directory not found in {project_path}")
        print("   Please create a Data/ subdirectory with your dataset (*.h5ad)")
        return False

    h5ad_files = list(data_dir.glob("*.h5ad"))
    if not h5ad_files:
        print(f"⚠️  No .h5ad files found in {project_path}/Data/")
        return False

    return True


def interactive_step_selection(default_ids) -> List[str]:
    """Y/n for each step, preselected by the project parameters; returns the script names."""
    print("\n" + "=" * 60)
    print("SELECT PIPELINE STEPS (Enter = preselected answer from the project parameters)")
    print("=" * 60 + "\n")
    selected = []
    for step in PIPELINE_STEPS:
        default = step["id"] in default_ids
        print(f"\n{step['name']}")
        print(f"  -> {step['description']}")
        response = input(f"  Include this step? [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if (response == "" and default) or response == "y":
            selected.append(step["script"])
    if not selected:
        print("No steps selected.")
        sys.exit(1)
    return selected


def prompt_hard_options() -> Dict[str, str]:
    """Ask once for the hard-to-calibrate options; empty = workbook value or default."""
    from CardamomOT.run_options import HARD_OPTIONS
    print("\n" + "=" * 60)
    print("HARD-TO-CALIBRATE PARAMETERS (empty = Data/CardamomOT_inputs.xlsx value, else default)")
    print("=" * 60)
    values = {}
    for h, attr in HARD_OPTIONS.items():
        v = input(f"  {h} (model.{attr}): ").strip()
        if v:
            values[h] = v
    return values


def run_step(script_name: str, args: List[str], repo_root: str) -> bool:
    """Execute a single pipeline step; returns False to stop."""
    script_path = Path(repo_root) / script_name
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return False
    cmd = [sys.executable, str(script_path)] + args
    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print(f"    Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print(f"{script_name} completed successfully")
        return True
    if result.returncode == STATIONARY_EXIT_CODE:
        print(f"{STATIONARY_MESSAGE} Stopping pipeline.")
        sys.exit(0)
    print(f"{script_name} exited with code {result.returncode}")
    return input("Continue to next step? [Y/n]: ").strip().lower() != "n"


def run_pipeline_interactive(project_path: str, use_defaults: bool = False):
    """Interactive (or, with use_defaults, parameter-driven) pipeline runner."""
    from CardamomOT.cli import pipeline_steps
    from CardamomOT.run_options import StepOptions, settings, step_arguments, HARD_OPTIONS

    if not validate_project_structure(project_path):
        sys.exit(1)
    print(f"Project validated: {project_path}\n")
    repo_root = Path(__file__).parent.parent

    values = {} if use_defaults else prompt_hard_options()
    opts = StepOptions(p=os.path.join(project_path, ''),
                       values={HARD_OPTIONS[h]: float(v) for h, v in values.items() if float(v) >= 0})
    default_ids = set(pipeline_steps(settings(opts), opts.p))
    if use_defaults:
        selected = [s["script"] for s in PIPELINE_STEPS if s["id"] in default_ids]
    else:
        selected = interactive_step_selection(default_ids)

    print(f"\nSelected {len(selected)} steps:")
    for script in selected:
        print(f"   - {next(s['name'] for s in PIPELINE_STEPS if s['script'] == script)}")
    if not use_defaults and input("\nProceed with these steps? [Y/n]: ").strip().lower() == "n":
        print("Pipeline cancelled.")
        sys.exit(0)

    failed = []
    for i, script in enumerate(selected, 1):
        step = next(s for s in PIPELINE_STEPS if s["script"] == script)
        print(f"\n[{i}/{len(selected)}] {step['name']}")
        if not run_step(script, step_arguments(step["id"], project_path, values), repo_root):
            failed.append(script)
            break

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    if failed:
        print(f"Stopped at: {', '.join(failed)}")
    else:
        print(f"All {len(selected)} steps completed successfully!")
    print(f"\nResults saved to: {project_path}/cardamomOT/")


def main():
    """Entry point for CLI."""
    parser = argparse.ArgumentParser(prog="cardamomot run",
                                     description="Run the CardamomOT analysis pipeline interactively")
    parser.add_argument("project_path", type=str, help="Path to the project directory containing Data/")
    parser.add_argument("--default", action="store_true",
                        help="Run the steps given by the project parameters without questions")
    args = parser.parse_args()
    run_pipeline_interactive(args.project_path, use_defaults=args.default)


if __name__ == "__main__":
    main()
