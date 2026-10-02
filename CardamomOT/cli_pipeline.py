"""
Interactive pipeline runner for CardamomOT.

Provides command-line interface for users to run the full analysis pipeline
on their own datasets with customizable step selection and hyperparameters.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
import subprocess

from CardamomOT.config import STATIONARY_EXIT_CODE, STATIONARY_MESSAGE

try:
    import questionary
    # Force disable questionary due to macOS terminal compatibility issues
    # The simple fallback prompts work more reliably
    HAS_QUESTIONARY = False  # Set to False to use simple Y/n prompts
except ImportError:
    HAS_QUESTIONARY = False


# Define pipeline steps in order.
# "default": True  → checked by default in interactive mode
# "default": False → unchecked by default (optional step)
PIPELINE_STEPS = [
    {
        "id": "estimate_cell_depth",
        "name": "Cell depth",
        "script": "estimate_cell_depth.py",
        "description": "Diagnose per-cell sequencing depth effects on the whole transcriptome and, if "
                        "needed and allowed, store a depth factor per cell (obs['depth_factor'])",
        "default": True,
    },
    {
        "id": "get_proliferation_rates",
        "name": "Proliferation rates",
        "script": "get_proliferation_rates.py",
        "description": "Estimate net proliferation rate per cell from literature gene "
                        "signatures (runs on the full gene set, before gene selection)",
        "default": True,
    },
    {
        "id": "select_genes_and_split",
        "name": "Gene selection",
        "script": "select_genes_and_split.py",
        "description": "Filter differentially expressed genes and split cells into train/test",
        "default": True,
    },
    {
        "id": "build_reference_network",
        "name": "Network constraint (optional)",
        "script": "build_reference_network.py",
        "description": "Build prior knowledge network from biological databases",
        "default": False,
    },
    {
        "id": "get_degradation_rates",
        "name": "Kinetics",
        "script": "get_degradation_rates.py",
        "description": "Assign literature mRNA and protein degradation rates (h^-1)",
        "default": True,
    },
    {
        "id": "infer_mixture",
        "name": "Mixture model",
        "script": "infer_mixture.py",
        "description": "Fit negative-binomial burst parameters per gene",
        "default": True,
    },
    {
        "id": "check_mixture_to_data",
        "name": "Check mixture",
        "script": "check_mixture_to_data.py",
        "description": "Validate mixture model parameters against data",
        "default": True,
    },
    {
        "id": "infer_network_structure",
        "name": "Network inference",
        "script": "infer_network_structure.py",
        "description": "Learn gene regulatory interactions via optimal transport",
        "default": True,
    },
    {
        "id": "infer_network_simul",
        "name": "Network adaptation",
        "script": "infer_network_simul.py",
        "description": "Prepare network parameters for forward simulation "
                        "(optionally learns a proliferation-rate MLP)",
        "default": True,
    },
    {
        "id": "simulate_network",
        "name": "Simulation",
        "script": "simulate_network.py",
        "description": "Generate synthetic single-cell trajectories from the learned model",
        "default": True,
    },
    {
        "id": "check_sim_to_data",
        "name": "Check simulation",
        "script": "check_sim_to_data.py",
        "description": "Validate simulations against experimental data distribution",
        "default": True,
    },
    {
        "id": "infer_test",
        "name": "Test — inference (optional)",
        "script": "infer_test.py",
        "description": "Infer network and simulate on held-out test set",
        "default": False,
    },
    {
        "id": "check_test_to_train",
        "name": "Test — check (optional)",
        "script": "check_test_to_train.py",
        "description": "Compare test predictions to training observations",
        "default": False,
    },
    {
        "id": "simulate_network_KOV",
        "name": "Perturb — KO/OV simulation",
        "script": "simulate_network_KOV.py",
        "description": "Simulate gene expression under in-silico knock-out/over-expression",
        "default": True,
    },
    {
        "id": "check_KOV_to_sim",
        "name": "Perturb — check KO/OV",
        "script": "check_KOV_to_sim.py",
        "description": "Compare perturbation simulations to wild-type",
        "default": True,
    },
    {
        "id": "report_results",
        "name": "Report — final PDF",
        "script": "report_results.py",
        "description": "Write the PDF report (generative model, GRN top regulators, KO/OV predictions)",
        "default": True,
    },
]

# Default hyperparameters
DEFAULT_PARAMS = {
    "estimate_cell_depth": {
        "-i": "input project path",
        "--allow": "1/0: apply the depth factor if recommended (default: model.allow_depth_correction)",
        "--method": "group_median (default), poissonian, or <project>/depth_methods/<name>.py",
    },
    "get_proliferation_rates": {
        "-i": "input project path",
        "--species": "organism for proliferation/death gene signatures: auto (detected from gene names, default), human or mouse",
    },
    "select_genes_and_split": {
        "-i": "input project path",
        "-c": "change flag (default: 0)",
        "-r": "rate parameter (default: 1.0)",
        "-s": "split name (default: 'train')",
        "-m": "mean constraint (default: 1.0)",
        "--prior": "prior weight of the run (hard prior 0 + --ref 1: gene budget from model.max_free_params)",
        "--ref": "1 if the literature prior is built (build_reference_network step selected)",
    },
    "build_reference_network": {
        "-i": "input project path",
        "-d": "max literature path length (default: model.literature_depth = 3)",
        "--species": "auto (detected from gene names, default), human or mouse",
        "--resources": "extended (default: OmniPath, CollecTRI + less curated resources) or core (OmniPath + CollecTRI)",
    },
    "get_degradation_rates": {
        "-i": "input project path",
        "-s": "split name (default: 'train')",
        "--species": "auto (detected from gene names, default), human or mouse",
        "--overwrite": "replace d0/d1 already stored in the AnnData files",
    },
    "infer_mixture": {
        "-i": "input project path",
        "-s": "split name (default: 'train')",
        "-m": "mean constraint (default: 1.0)",
    },
    "check_mixture_to_data": {
        "-i": "input project path",
        "-s": "split name (default: 'train')",
    },
    "infer_network_structure": {
        "-i": "input project path",
        "-s": "split name (default: 'train')",
    },
    "infer_network_simul": {
        "-i": "input project path",
        "-s": "split name (default: 'train')",
    },
    "simulate_network": {
        "-i": "input project path",
        "-s": "split name (default: 'train')",
    },
    "check_sim_to_data": {
        "-i": "input project path",
        "-s": "split name (default: 'train')",
    },
    "simulate_network_KOV": {
        "-i": "input project path",
        "-s": "split name (default: 'train')",
    },
    "check_KOV_to_sim": {
        "-i": "input project path",
        "-s": "split name (default: 'train')",
    },
    "report_results": {
        "-i": "input project path",
        "-s": "split name (default: 'train')",
    },
    "infer_test": {
        "-i": "input project path",
    },
    "check_test_to_train": {
        "-i": "input project path",
        "-s": "split name (default: 'train')",
    },
}


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


def interactive_step_selection() -> List[str]:
    """
    Present checkboxes to user to select which steps to run.
    Returns list of script names to execute.
    """
    if not HAS_QUESTIONARY:
        return simple_step_selection()

    print("\n" + "=" * 60)
    print("📋 SELECT PIPELINE STEPS")
    print("=" * 60 + "\n")

    # Display options with descriptions
    choices = []
    for step in PIPELINE_STEPS:
        choices.append({
            "name": f"{step['name']:<45} {step['description']}",
            "value": step["script"],
            "checked": step.get("default", True),
        })

    selected = questionary.checkbox(
        "Which steps do you want to run?",
        choices=choices,
    ).ask()

    if selected is None:
        print("❌ Step selection cancelled.")
        sys.exit(1)

    return selected


def simple_step_selection() -> List[str]:
    """
    Fallback step selection without questionary.
    Uses Y/n prompts for each step.
    """
    print("\n" + "=" * 60)
    print("SELECT PIPELINE STEPS (Y/n for each)")
    print("=" * 60 + "\n")

    selected = []
    for step in PIPELINE_STEPS:
        print(f"\n{step['name']}")
        print(f"  → {step['description']}")
        response = input("  Include this step? [Y/n]: ").strip().lower()
        if response != "n":
            selected.append(step["script"])

    if not selected:
        print("❌ No steps selected.")
        sys.exit(1)

    return selected



# Steps that consume the branching-simulation MLP (train it, or apply it).
# --simulate-proliferation must be passed consistently to all three, or not
# at all -- see docs/advanced.md#proliferation-aware-simulation---simulate-proliferation.
STEPS_WITH_SIMULATE_PROLIFERATION = ["infer_network_simul", "simulate_network", "simulate_network_KOV"]

# Steps taking --prior (weight of the edges absent from cardamomOT/ref_network.csv); the same
# value must reach all of them (output file names embed it).
STEPS_WITH_PRIOR = ["infer_network_structure", "infer_network_simul", "check_sim_to_data", "infer_test",
                    "check_test_to_train", "check_KOV_to_sim", "report_results"]


def prompt_prior() -> str:
    """Ask once for the prior weight; '' keeps the model default."""
    print("\n" + "=" * 60)
    print("PRIOR NETWORK (optional)")
    print("=" * 60)
    print("  Weight of the edges absent from the literature prior (cardamomOT/ref_network.csv,")
    print("  written by the gene selection or build_reference_network): 0 = hard constraint")
    print("  (sparse network), 1 = prior ignored, in between = soft penalty.")
    if HAS_QUESTIONARY:
        value = questionary.text("Prior weight in [0, 1] [default: model default]:", default="").ask() or ""
    else:
        value = input("  Prior weight in [0, 1] [model default]: ").strip()
    return value


def prompt_simulate_proliferation() -> bool:
    """
    Ask once, up front, whether to enable proliferation-aware simulation
    (--simulate-proliferation) for this run. Off by default, matching
    run.sh / cardamomot pipeline.
    """
    print("\n" + "=" * 60)
    print("PROLIFERATION-AWARE SIMULATION (optional)")
    print("=" * 60)
    print("  Learns a small MLP mapping protein levels -> net proliferation rate")
    print("  from the inferred optimal-transport couplings, and simulates with")
    print("  branching (birth/death) resampling instead of a fixed cell number.")
    print("  See Advanced Features -> Proliferation-aware simulation for details.")
    response = input("  Enable proliferation-aware simulation (--simulate-proliferation)? [y/N]: ").strip().lower()
    return response == "y"


def interactive_parameter_input(step_id: str, project_path: str,
                                 simulate_proliferation: bool = False, prior: str = "",
                                 build_prior: bool = False) -> Dict[str, str]:
    """
    Prompt user for parameter values for a given step.
    Returns dictionary of parameters to pass to the script.
    """
    params = {}

    # Add -i to all steps
    params["-i"] = project_path

    # Add -s only to steps that use it (not get_proliferation_rates,
    # build_reference_network, or infer_test)
    steps_with_split = [
        "select_genes_and_split", "get_degradation_rates", "infer_mixture",
        "check_mixture_to_data", "infer_network_structure",
        "infer_network_simul", "simulate_network", "check_sim_to_data",
        "simulate_network_KOV", "check_KOV_to_sim", "check_test_to_train",
        "report_results"
    ]
    if step_id in steps_with_split:
        params["-s"] = "train"  # Default split

    # Flag-only parameter (no value) -- forwarded consistently to all three
    # steps that need it, decided once via prompt_simulate_proliferation().
    if step_id in STEPS_WITH_SIMULATE_PROLIFERATION and simulate_proliferation:
        params["--simulate-proliferation"] = ""
    if step_id in STEPS_WITH_PRIOR and prior:
        params["--prior"] = prior
    # The selection builds the literature prior; with a hard prior its budget is in parameters
    if step_id == "select_genes_and_split":
        if prior:
            params["--prior"] = prior
        params["--ref"] = "1" if build_prior else "0"

    # Step-specific parameters
    if step_id == "select_genes_and_split":
        print("\n" + "=" * 60)
        print("SELECT DE GENES & SPLIT - Parameters")
        print("=" * 60)
        
        if HAS_QUESTIONARY:
            # Change flag
            change = questionary.text(
                "Change detection flag (0=off, 1=on) [default: 0]:",
                default="0",
            ).ask()
            if change:
                params["-c"] = change
            else:
                params["-c"] = "0"
            
            # Rate parameter
            rate = questionary.text(
                "Rate parameter [default: 1.0]:",
                default="1.0",
            ).ask()
            if rate:
                params["-r"] = rate
            else:
                params["-r"] = "1.0"
            
            # Mean constraint
            mean = questionary.text(
                "Mean constraint [default: 1.0]:",
                default="1.0",
            ).ask()
            if mean:
                params["-m"] = mean
        else:
            # Fallback to simple text input
            change = input("Change detection flag (0=off, 1=on) [0]: ").strip() or "0"
            params["-c"] = change
            
            rate = input("Rate parameter [1.0]: ").strip() or "1.0"
            params["-r"] = rate
            
            mean = input("Mean constraint [1.0]: ").strip() or "1.0"
            params["-m"] = mean

    elif step_id == "get_proliferation_rates":
        print("\n" + "=" * 60)
        print("PROLIFERATION RATES - Parameters")
        print("=" * 60)

        if HAS_QUESTIONARY:
            species = questionary.select(
                "Organism for proliferation/death gene signatures:",
                choices=["auto", "human", "mouse"],
                default="auto",
            ).ask()
            params["--species"] = species or "auto"
        else:
            species = input("Organism for proliferation/death gene signatures (auto/human/mouse) [auto]: ").strip().lower() or "auto"
            params["--species"] = species

    elif step_id == "build_reference_network":
        print("\n" + "=" * 60)
        print("BUILD REFERENCE NETWORK - Parameters")
        print("=" * 60)
        
        if HAS_QUESTIONARY:
            depth = questionary.text(
                "Network depth to query [default: 3]:",
                default="3",
            ).ask()
            if depth:
                params["-d"] = depth
        else:
            depth = input("Network depth to query [3]: ").strip() or "3"
            params["-d"] = depth

    elif step_id == "infer_mixture":
        print("\n" + "=" * 60)
        print("INFER MIXTURE MODEL - Parameters")
        print("=" * 60)
        
        if HAS_QUESTIONARY:
            mean = questionary.text(
                "Mean constraint [default: 1.0]:",
                default="1.0",
            ).ask()
            if mean:
                params["-m"] = mean
        else:
            mean = input("Mean constraint [1.0]: ").strip() or "1.0"
            params["-m"] = mean

    elif step_id == "infer_network_structure":
        print("\n" + "=" * 60)
        print("INFER NETWORK STRUCTURE - Parameters")
        print("=" * 60)
        
        print(f"  → Prior weight: {prior or 'model default'} (cardamomOT/ref_network.csv if present)")

    elif step_id == "simulate_network_KOV":
        print("\n" + "=" * 60)
        print("SIMULATE KOCKOUTS/OVEREXPRESSIONS - Parameters")
        print("=" * 60)
        print("  → Uses -i and -s parameters (no additional configuration needed)")

    elif step_id == "check_KOV_to_sim":
        print("\n" + "=" * 60)
        print("CHECK KO/OV SIMULATIONS - Parameters")
        print("=" * 60)
        print("  → Uses -i and -s parameters (no additional configuration needed)")

    elif step_id == "infer_test":
        print("\n" + "=" * 60)
        print("INFER ON TEST SET - Parameters")
        print("=" * 60)
        print("  → Uses only -i parameter (no additional configuration needed)")

    elif step_id == "check_test_to_train":
        print("\n" + "=" * 60)
        print("CHECK TEST VS TRAIN PREDICTIONS - Parameters")
        print("=" * 60)
        print("  → Uses -i and -s parameters (no additional configuration needed)")

    return params


def run_step(script_name: str, params: Dict[str, str], repo_root: str) -> bool:
    """
    Execute a single pipeline step.
    """
    script_path = Path(repo_root) / script_name

    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False

    cmd = ["python", str(script_path)]
    for key, value in params.items():
        if value == "":
            cmd.append(key)  # flag-only parameter, e.g. --simulate-proliferation
        else:
            cmd.extend([key, value])

    print(f"\n{'=' * 60}")
    print(f"▶️  Running: {script_name}")
    print(f"    Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return True
        elif result.returncode == STATIONARY_EXIT_CODE:
            print(f"⏹️  {STATIONARY_MESSAGE} Stopping pipeline.")
            sys.exit(0)
        else:
            print(f"⚠️  {script_name} exited with code {result.returncode}")
            response = input("Continue to next step? [Y/n]: ").strip().lower()
            return response != "n"
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False


def run_pipeline_interactive(project_path: str, use_defaults: bool = False):
    """
    Main pipeline runner with interactive or default mode.
    """
    # 1. Validate project structure
    if not validate_project_structure(project_path):
        sys.exit(1)

    print(f"✅ Project validated: {project_path}\n")

    # Get repo root (parent of CardamomOT/)
    repo_root = Path(__file__).parent.parent

    # 2. Select steps
    if use_defaults:
        print("🚀 Running pipeline with DEFAULT settings...")
        selected_scripts = [step["script"] for step in PIPELINE_STEPS if step.get("default", True)]
    else:
        selected_scripts = interactive_step_selection()

    print(f"\n📌 Selected {len(selected_scripts)} steps:\n")
    for script in selected_scripts:
        step = next((s for s in PIPELINE_STEPS if s["script"] == script), None)
        print(f"   • {step['name']}")

    # 3. Confirm and run
    if not use_defaults:
        response = input("\n✓ Proceed with these steps? [Y/n]: ").strip().lower()
        if response == "n":
            print("❌ Pipeline cancelled.")
            sys.exit(0)

    # 3.5. Proliferation-aware simulation is opt-in and shared across three steps
    # (infer_network_simul, simulate_network, simulate_network_KOV) -- ask once,
    # up front, rather than per-step, so the same choice is applied consistently.
    selected_ids = {step["id"] for step in PIPELINE_STEPS if step["script"] in selected_scripts}
    simulate_proliferation = False
    from CardamomOT.inputs import input_dir, project_parameters
    input_dir(project_path)  # sync Data/CardamomOT_inputs.xlsx
    fixed = project_parameters(project_path)
    if 'simulate_with_proliferation' in fixed:
        # Fixed in the workbook: the scripts read it themselves, no prompt
        print(f"✓ simulate_with_proliferation = {fixed['simulate_with_proliferation']} (Data/CardamomOT_inputs.xlsx)")
    elif selected_ids & set(STEPS_WITH_SIMULATE_PROLIFERATION):
        simulate_proliferation = prompt_simulate_proliferation() if not use_defaults else False
        if simulate_proliferation:
            print("✓ Proliferation-aware simulation enabled (--simulate-proliferation)")
    prior = ""
    if selected_ids & set(STEPS_WITH_PRIOR) and not use_defaults:
        prior = prompt_prior()
    build_prior = "build_reference_network" in selected_ids

    # 4. Execute each step
    failed_steps = []
    for i, script in enumerate(selected_scripts, 1):
        step = next((s for s in PIPELINE_STEPS if s["script"] == script), None)

        print(f"\n[{i}/{len(selected_scripts)}] {step['name']}")

        params = interactive_parameter_input(step["id"], project_path,
                                              simulate_proliferation=simulate_proliferation, prior=prior,
                                              build_prior=build_prior)

        if not run_step(script, params, repo_root):
            failed_steps.append(script)

    # 5. Summary
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    if failed_steps:
        print(f"⚠️  {len(selected_scripts) - len(failed_steps)}/{len(selected_scripts)} steps completed")
        print(f"❌ Failed steps: {', '.join(failed_steps)}")
    else:
        print(f"✅ All {len(selected_scripts)} steps completed successfully!")

    print(f"\n📁 Results saved to: {project_path}/cardamom/")


def main():
    """Entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="cardamomot run",
        description="Run the CardamomOT analysis pipeline interactively",
    )
    parser.add_argument(
        "project_path",
        type=str,
        help="Path to the project directory containing Data/ subdirectory",
    )
    parser.add_argument(
        "--default",
        action="store_true",
        help="Use default parameters without interaction",
    )

    args = parser.parse_args()
    run_pipeline_interactive(args.project_path, use_defaults=args.default)


if __name__ == "__main__":
    main()
