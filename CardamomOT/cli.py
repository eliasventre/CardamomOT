"""Command-line helpers and high-level CLI for CARDAMOM.

The repository previously contained a collection of standalone scripts
(e.g. ``infer_rd.py``, ``infer_mixture.py``) invoked via shell wrappers.  In
Tier‑3 these have been consolidated behind a single console command named
``cardamom``.  Each original script still exists for backwards compatibility
but they now import common argument parsing utilities from this module.

Utilities such as ``create_pipeline_parser`` remain here so that the separate
scripts can import them, keeping interfaces consistent.

Usage examples
--------------

  # run the full analysis pipeline (identical to the old run.sh)
  cardamom pipeline -i data/myproject -s train -c 1 -r 0.6 -m 0.5

  # execute a single step with arbitrary options
  cardamom step infer_mixture -i data/myproject -s train -m 1.0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, List

# ---------------------------------------------------------------------------
# argument parser helpers for individual scripts
# ---------------------------------------------------------------------------

def create_pipeline_parser(
    description: str,
    epilog: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Build a standardized parser used by all pipeline scripts.

    Args:
        description: short text describing the script's purpose.
        epilog: optional text appended to the help message.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        metavar="PATH",
        help="Project directory containing Data/ and cardamom/",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only log warnings and errors",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        metavar="PATH",
        default=None,
        help="Optional log file",
    )

    return parser


def add_split_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-s", "--split",
        type=str,
        default="train",
        metavar="SPLIT",
        help="Data split to use (train/test/full)",
    )


def add_means_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m", "--means",
        type=float,
        default=-1,
        metavar="VALUE",
        help="Mean expression threshold (default: auto)",
    )


def add_change_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--change",
        type=str,
        default="default",
        metavar="TYPE",
        help="Change condition for inference",
    )


def add_rate_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-r", "--rate",
        type=str,
        default="default",
        metavar="TYPE",
        help="Rate parameter for kinetics",
    )


def validate_input_path(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"Input path does not exist: {path}")
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"Input path is not a directory: {path}")
    return p


def validate_output_path(path: str) -> Path:
    p = Path(path)
    if not p.parent.exists():
        raise argparse.ArgumentTypeError(f"Output directory does not exist: {p.parent}")
    return p


def handle_common_args(
    args: argparse.Namespace,
    module_name: str = "cardamom",
) -> None:
    """Configure logging based on shared options."""
    from .logging import configure_logging
    import logging

    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    log_file = Path(args.log_file) if args.log_file else None
    configure_logging(level=level, log_file=log_file)

# ---------------------------------------------------------------------------
# high-level pipeline CLI
# ---------------------------------------------------------------------------

def _run_script(script: str, args: List[str]) -> None:
    repo = Path(__file__).resolve().parent.parent
    path = repo / script
    cmd = [sys.executable, str(path)] + args
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)


def _pipeline(args: argparse.Namespace) -> None:
    kw = ['-i', args.input, '-s', args.split, '-c', args.change,
          '-r', args.rate, '-m', args.mean]
    _run_script('infer_rd.py', ['-i', args.input])
    _run_script('select_DEgenes_and_split.py', kw)
    _run_script('get_kinetic_rates.py', ['-i', args.input, '-s', args.split])
    _run_script('infer_mixture.py', ['-i', args.input, '-s', args.split, '-m', args.mean])
    _run_script('check_mixture_to_data.py', ['-i', args.input, '-s', args.split])
    _run_script('infer_network_structure.py', ['-i', args.input, '-s', args.split])
    _run_script('infer_network_simul.py', ['-i', args.input, '-s', args.split])
    _run_script('simulate_network.py', ['-i', args.input, '-s', args.split])
    _run_script('check_sim_to_data.py', ['-i', args.input, '-s', args.split])
    _run_script('simulate_network_KOV.py', ['-i', args.input, '-s', args.split])
    _run_script('check_KOV_to_sim.py', ['-i', args.input, '-s', args.split])
    print("\nPipeline complete.")


def main() -> None:
    parser = argparse.ArgumentParser(prog='cardamom',
                                     description='CARDAMOM command-line interface')
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_pipe = subparsers.add_parser('pipeline', help='run the full analysis pipeline')
    p_pipe.add_argument('-i', '--input', required=True, help='project directory')
    p_pipe.add_argument('-s', '--split', default='full', help='data split')
    p_pipe.add_argument('-c', '--change', default='0', help='change flag')
    p_pipe.add_argument('-r', '--rate', default='1.0', help='rate parameter')
    p_pipe.add_argument('-m', '--mean', default='1.0', help='mean parameter')
    p_pipe.set_defaults(func=_pipeline)

    p_step = subparsers.add_parser('step', help='run individual step')
    p_step.add_argument('name', help='script name without .py')
    p_step.add_argument('extra', nargs=argparse.REMAINDER,
                        help='additional arguments forwarded to script')
    p_step.set_defaults(func=lambda a: _run_script(a.name + '.py', a.extra))

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
