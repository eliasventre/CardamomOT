"""
Command-line options of the pipeline steps.

A step takes only the project path (-i) and, among the hard-to-calibrate parameters (in this
order: --stimulus, --prior, --mean-forcing, --force-basins, --temporal-basins), those it uses
(STEP_OPTIONS). Everything else (split, gene selection, literature prior, test, perturbations,
proliferation, species...) is a NetworkModel parameter (CardamomOT/model/base.py), fixed per project
in the Model_parameters sheet of Data/CardamomOT_inputs.xlsx.

Precedence: default of base.py < workbook < command-line option; an absent or negative option
keeps the workbook value (or the default).
"""
import getopt
import os
import sys
from dataclasses import dataclass, field

# Option -> NetworkModel attribute, in the order of the pipeline calls
HARD_OPTIONS = {
    'stimulus': 'stimulus',
    'prior': 'prior_network_pen',
    'mean-forcing': 'mean_forcing_em',
    'force-basins': 'force_basins',
    'temporal-basins': 'temporal_basins',
}

# Hard-to-calibrate options used by each step (the others are not accepted)
STEP_OPTIONS = {
    'estimate_cell_depth': (),
    'get_proliferation_rates': (),
    'select_genes_and_split': ('prior',),
    'build_reference_network': (),
    'get_degradation_rates': (),
    'infer_mixture': ('mean-forcing',),
    'check_mixture_to_data': (),
    'infer_network_structure': ('stimulus', 'prior', 'force-basins', 'temporal-basins'),
    'infer_network_simul': ('stimulus', 'prior'),
    'simulate_network': (),
    'check_sim_to_data': ('stimulus', 'prior'),
    'infer_test': ('stimulus', 'prior', 'force-basins', 'temporal-basins'),
    'check_test_to_train': ('stimulus', 'prior'),
    'simulate_network_KOV': (),
    'check_KOV_to_sim': ('stimulus', 'prior'),
    'report_results': ('stimulus', 'prior'),
}

# Former options, now parameters of the Model_parameters sheet (error message only)
REMOVED_OPTIONS = {
    '-s': 'split', '--split': 'split', '-r': 'train_rate', '--rate': 'train_rate', '-c': 'select_genes',
    '--change': 'select_genes', '--ref': 'build_prior_network', '--simulate-proliferation': 'simulate_with_proliferation',
    '--species': 'species', '--allow': 'allow_depth_correction', '--method': 'depth_method',
    '-d': 'literature_depth', '--depth': 'literature_depth', '--resources': 'literature_resources',
    '--overwrite': 'overwrite_degradation_rates', '--no-senescence-gating': 'senescence_gating',
    '--integrate-samples': 'integrate_samples', '--no-integrate-samples': 'integrate_samples',
    '--ref-sample': 'ref_sample_integration',
    '--soft-em-refinement': 'soft_em_refinement', '--net-index': 'report_net_index', '--norm': 'report_normalize',
    '--no-log': 'report_log1p', '--n-umap': 'report_n_umap',
}


@dataclass
class StepOptions:
    p: str                                        # project path, with a trailing separator
    values: dict = field(default_factory=dict)    # NetworkModel attribute -> value given on the command line
    output: str = None                            # -o (report_results only)


def parse_step_options(argv, step, doc=None, output=False):
    """-i <project> and the hard options of `step` (STEP_OPTIONS); exits with a message otherwise."""
    hard = STEP_OPTIONS[step]
    longs = ['input='] + [h + '=' for h in hard] + (['output='] if output else [])
    tag = f'[{step}]'
    try:
        opts, rest = getopt.getopt(argv, 'hi:' + ('o:' if output else ''), longs)
    except getopt.GetoptError as e:
        bad = next((a.split('=')[0] for a in argv if a.split('=')[0] in REMOVED_OPTIONS), None)
        if bad:
            print(f"{tag} Error: option {bad} was removed: set the parameter '{REMOVED_OPTIONS[bad]}' in the "
                  f"Model_parameters sheet of Data/CardamomOT_inputs.xlsx (or its default in CardamomOT/model/base.py)")
        else:
            print(f"{tag} Error: {e}. Usage: python {step}.py -i <project>"
                  + ''.join(f' [--{h} <value>]' for h in hard) + (' [-o <out.pdf>]' if output else ''))
        sys.exit(2)
    if rest:
        print(f"{tag} Error: unexpected arguments {rest} (positional arguments are not accepted)")
        sys.exit(2)
    inputfile, values, out = '', {}, None
    for opt, arg in opts:
        if opt == '-h':
            print(doc or __doc__)
            sys.exit(0)
        elif opt in ('-i', '--input'):
            inputfile = arg
        elif opt in ('-o', '--output'):
            out = arg
        else:
            v = float(arg)
            if v >= 0:  # negative = default of the model
                values[HARD_OPTIONS[opt[2:]]] = v
    if not inputfile:
        print(f"{tag} Error: missing required argument -i <project>")
        sys.exit(1)
    return StepOptions(p=os.path.join(inputfile, ''), values=values, output=out)


def configure(model, opts, verb=True):
    """Workbook values, then the command-line options (which dominate), on a NetworkModel; returns it."""
    model.apply_project_parameters(opts.p, verb=verb)
    for attr, v in opts.values.items():
        cur = getattr(model, attr)
        setattr(model, attr, type(cur)(v) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else v)
        model._project_overrides = getattr(model, '_project_overrides', set()) | {attr}
    if opts.values and verb:
        print(f"[CardamomOT] Command-line options (override the workbook): {opts.values}")
    return model


def settings(opts):
    """NetworkModel(1) configured as the run (to read the pipeline parameters before the model of a step)."""
    from .model.base import NetworkModel
    return configure(NetworkModel(1), opts, verb=False)


def step_arguments(step, p, values):
    """Command line of `step` for the pipeline runners: -i and the hard options it uses (given ones only)."""
    args = ['-i', p]
    for h in STEP_OPTIONS[step]:
        v = values.get(h)
        if v is not None and str(v) != '' and float(v) >= 0:
            args += [f'--{h}', str(v)]
    return args
