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

try:
    import questionary
    # Force disable questionary due to macOS terminal compatibility issues
    # The simple fallback prompts work more reliably
    HAS_QUESTIONARY = False  # Set to False to use simple Y/n prompts
except ImportError:
    HAS_QUESTIONARY = False


# Define pipeline steps in order
PIPELINE_STEPS = [
    {
        "id": "infer_rd",
        "name": "Estimate read depth correction",
        "script": "infer_rd.py",
        "description": "Compute per-cell read depth factors (optional)",
    },
    {
        "id": "select_DEgenes",
        "name": "Select DE genes & split cells",
        "script": "select_DEgenes_and_split.py",
        "description": "Filter genes by variability and split cells into train/test",
    },
    {
        "id": "prepare_reference_network",
        "name": "Prepare reference network (optional)",
        "script": "prepare_reference_network.py",
        "description": "Build prior knowledge network from biological databases",
    },
    {
        "id": "get_kinetic_rates",
        "name": "Get kinetic rates",
        "script": "get_kinetic_rates.py",
        "description": "Estimate mRNA degradation and synthesis rates",
    },
    {
        "id": "infer_mixture",
        "name": "Infer mixture model",
        "script": "infer_mixture.py",
        "description": "Learn kinetic parameters and cell state assignments",
    },
    {
        "id": "check_mixture_to_data",
        "name": "Check mixture model fit",
        "script": "check_mixture_to_data.py",
        "description": "Validate mixture model parameters against data",
    },
    {
        "id": "infer_network_structure",
        "name": "Infer network structure",
        "script": "infer_network_structure.py",
        "description": "Infer gene regulatory network from data and priors",
    },
    {
        "id": "infer_network_simul",
        "name": "Adapt network for simulation",
        "script": "infer_network_simul.py",
        "description": "Prepare network parameters for forward simulation",
    },
    {
        "id": "simulate_network",
        "name": "Simulate network dynamics",
        "script": "simulate_network.py",
        "description": "Generate synthetic trajectories from learned model",
    },
    {
        "id": "check_sim_to_data",
        "name": "Check simulated vs observed data",
        "script": "check_sim_to_data.py",
        "description": "Validate simulations match experimental data distribution",
    },
    {
        "id": "simulate_network_KOV",
        "name": "Simulate knockouts/overexpressions",
        "script": "simulate_network_KOV.py",
        "description": "Simulate gene expression under genetic perturbations (KO/OV)",
    },
    {
        "id": "check_KOV_to_sim",
        "name": "Check KO/OV simulations",
        "script": "check_KOV_to_sim.py",
        "description": "Validate perturbation simulations against wildtype data",
    },
    {
        "id": "infer_test",
        "name": "Infer on test set",
        "script": "infer_test.py",
        "description": "Infer regulatory network and simulate on test dataset",
    },
    {
        "id": "check_test_to_train",
        "name": "Check test vs train predictions",
        "script": "check_test_to_train.py",
        "description": "Validate network inference by comparing test predictions to observations",
    },
]

# Default hyperparameters
DEFAULT_PARAMS = {
    "infer_rd": {
        "-i": "input project path",
    },
    "select_DEgenes": {
        "-i": "input project path",
        "-c": "change flag (default: 0)",
        "-r": "rate parameter (default: 1.0)",
        "-s": "split name (default: 'train')",
        "-m": "mean constraint (default: 1.0)",
    },
    "prepare_reference_network": {
        "-i": "input project path",
        "-d": "network depth to query (default: 3)",
    },
    "get_kinetic_rates": {
        "-i": "input project path",
        "-s": "split name (default: 'train')",
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
            "name": f"{step['name']:<35} {step['description']}",
            "value": step["script"],
            "checked": True,  # All checked by default
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


def interactive_parameter_input(step_id: str, project_path: str) -> Dict[str, str]:
    """
    Prompt user for parameter values for a given step.
    Returns dictionary of parameters to pass to the script.
    """
    params = {}

    # Add -i to all steps
    params["-i"] = project_path
    
    # Add -s only to steps that use it (not infer_rd or prepare_reference_network or infer_test)
    steps_with_split = [
        "select_DEgenes", "get_kinetic_rates", "infer_mixture", 
        "check_mixture_to_data", "infer_network_structure", 
        "infer_network_simul", "simulate_network", "check_sim_to_data",
        "simulate_network_KOV", "check_KOV_to_sim", "check_test_to_train"
    ]
    if step_id in steps_with_split:
        params["-s"] = "train"  # Default split

    # Step-specific parameters
    if step_id == "select_DEgenes":
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

    elif step_id == "prepare_reference_network":
        print("\n" + "=" * 60)
        print("PREPARE REFERENCE NETWORK - Parameters")
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
        
        if HAS_QUESTIONARY:
            has_prior = questionary.confirm(
                "Use prior network from previous step?",
                default=True,
            ).ask()
            if has_prior:
                print("  ✓ Will use prepared prior network")
        else:
            response = input("Use prior network? [Y/n]: ").strip().lower()
            # If they say no, don't add any special parameters

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
        selected_scripts = [step["script"] for step in PIPELINE_STEPS]
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

    # 4. Execute each step
    failed_steps = []
    for i, script in enumerate(selected_scripts, 1):
        step = next((s for s in PIPELINE_STEPS if s["script"] == script), None)

        print(f"\n[{i}/{len(selected_scripts)}] {step['name']}")

        params = interactive_parameter_input(step["id"], project_path)

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
