"""Command-line interface of CardamomOT (console command ``cardamomot``).

The pipeline takes the project directory and, optionally, the hard-to-calibrate parameters (in this
order) --stimulus, --prior, --mean-forcing, --force-basins, --temporal-basins. Everything else (split,
gene selection, literature prior, test, perturbations, proliferation, species...) is a parameter of
NetworkModel (CardamomOT/model/base.py), fixed per project in the model_parameters sheet of
Data/CardamomOT_inputs.xlsx. Precedence: default < workbook < command-line option.

Usage examples
--------------

  # full analysis pipeline (same as run.sh)
  cardamomot pipeline -i data/myproject --stimulus 1 --prior 0

  # interactive step selection
  cardamomot run data/myproject

  # a single step
  cardamomot step infer_mixture -i data/myproject --mean-forcing 0.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from . import cli_pipeline
from .run_options import HARD_OPTIONS, StepOptions, settings, step_arguments


def _run_script(script: str, args: List[str]) -> int:
    repo = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(repo / script)] + args
    print(">>>", " ".join(cmd))
    return subprocess.call(cmd)


def pipeline_steps(cfg, project=None) -> List[str]:
    """Steps of a run given the configured NetworkModel `cfg` (pipeline parameters) of `project`."""
    steps = ['estimate_cell_depth']
    if cfg.estimate_proliferation_rates:
        steps.append('get_proliferation_rates')
    steps.append('select_genes_and_split')
    # The selection builds the literature prior only with literature_selection and a hard prior
    lit_selection = cfg.select_genes and cfg.literature_selection and cfg.prior_network_pen == 0
    if cfg.build_prior_network and not lit_selection:
        steps.append('build_reference_network')
    steps += ['get_degradation_rates', 'infer_mixture', 'check_mixture_to_data', 'infer_network_structure',
              'infer_network_simul', 'simulate_network', 'check_sim_to_data']
    if cfg.run_test:
        from .inputs import removed_samples
        removed = removed_samples(project)[0] if project is not None else []
        if cfg.split == 'train' or removed:
            steps += ['infer_test', 'check_test_to_train']
        else:
            print("Warning: run_test = True needs split = 'train' or samples with remove_from_inference "
                  "(held-out cells); test steps skipped")
    if cfg.simulate_perturbations:
        steps += ['simulate_network_KOV', 'check_KOV_to_sim']
    steps.append('report_results')
    return steps


def hard_values(args: argparse.Namespace) -> dict:
    """{option: value} of the hard-to-calibrate options given on the command line."""
    return {h: getattr(args, h.replace('-', '_')) for h in HARD_OPTIONS
            if getattr(args, h.replace('-', '_')) is not None}


def _pipeline(args: argparse.Namespace) -> None:
    from .config import STATIONARY_EXIT_CODE, STATIONARY_MESSAGE
    values = hard_values(args)
    opts = StepOptions(p=str(Path(args.input)) + '/',
                       values={HARD_OPTIONS[h]: float(v) for h, v in values.items() if float(v) >= 0})
    cfg = settings(opts)
    steps = pipeline_steps(cfg, opts.p)
    print(f"Pipeline on {args.input}: split={cfg.split}, select_genes={cfg.select_genes}, "
          f"build_prior_network={cfg.build_prior_network}, estimate_proliferation_rates="
          f"{cfg.estimate_proliferation_rates}, run_test={cfg.run_test}, simulate_perturbations="
          f"{cfg.simulate_perturbations}, simulate_with_proliferation={cfg.simulate_with_proliferation}")
    for step in steps:
        code = _run_script(f'{step}.py', step_arguments(step, args.input, values))
        if code == STATIONARY_EXIT_CODE and step == 'infer_network_structure':
            print(f"{STATIONARY_MESSAGE} Stopping pipeline.")
            return
        if code != 0:
            sys.exit(f"Step {step} failed (exit code {code})")
    print("\nPipeline complete.")


def _run_pipeline_interactive(args: argparse.Namespace) -> None:
    cli_pipeline.run_pipeline_interactive(project_path=args.project, use_defaults=args.default)


def add_hard_options(parser: argparse.ArgumentParser) -> None:
    """--stimulus, --prior, --mean-forcing, --force-basins, --temporal-basins (absent = workbook/default)."""
    helps = {
        'stimulus': 'stimulus-edge penalisation in [0, 1] (model.stimulus)',
        'prior': 'weight of the edges absent from the literature prior cardamomOT/ref_network.csv '
                 '(0 = hard constraint, sparse; 1 = prior ignored) (model.prior_network_pen)',
        'mean-forcing': 'mean-forcing intensity of the NB mixture (model.mean_forcing_em)',
        'force-basins': 'preservation of the NB basin weights in the network inference, in [0, 1] (model.force_basins)',
        'temporal-basins': 'temporal consistency of the basins, 0 or 1 (model.temporal_basins)',
    }
    for h in HARD_OPTIONS:
        parser.add_argument(f'--{h}', default=None, help=helps[h] + '; absent or negative = workbook value or default')


def main() -> None:
    parser = argparse.ArgumentParser(prog='cardamomot', description='CardamomOT command-line interface')
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_run = subparsers.add_parser('run', help='run the pipeline with interactive step selection')
    p_run.add_argument('project', help='path to project directory')
    p_run.add_argument('--default', action='store_true',
                       help='run the steps given by the parameters of the project, without questions')
    p_run.set_defaults(func=_run_pipeline_interactive)

    p_pipe = subparsers.add_parser('pipeline', help='run the full analysis pipeline')
    p_pipe.add_argument('-i', '--input', required=True, help='project directory')
    add_hard_options(p_pipe)
    p_pipe.set_defaults(func=_pipeline)

    p_step = subparsers.add_parser('step', help='run an individual step')
    p_step.add_argument('name', help='script name without .py')
    p_step.add_argument('extra', nargs=argparse.REMAINDER, help='arguments forwarded to the script')
    p_step.set_defaults(func=lambda a: sys.exit(_run_script(a.name + '.py', a.extra)))

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
