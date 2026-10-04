"""
Optional run inputs of a project: one Excel workbook, Data/CardamomOT_inputs.xlsx.

Each sheet holds one kind of information (gene lists, stimulus schedules and targets,
perturbations, timepoints, proliferation and transition rates...); empty cells mean "not given".
The first call of input_dir(project) in a process:
1. creates the workbook with its documented structure if it does not exist;
2. imports the legacy text files still present in Data/ (genes_queries.txt, KO_OV_simulate.txt,
   stimulus_schedule.txt...) into their sheet, with a warning: a text file overwrites the
   corresponding cells of the workbook (old projects run unchanged and get a filled workbook);
3. exports the workbook to cardamomOT/inputs/ as the text files the pipeline reads.
The scripts read their inputs from input_dir(project). Large numeric arrays (reference_network.csv,
basal_init / basal_ref, inter_init / inter_ref, inter_simul_ref) stay files in Data/.
"""
import glob
import os
import re
import shutil

import numpy as np
import pandas as pd

WORKBOOK = 'CardamomOT_inputs.xlsx'
CACHE = os.path.join('cardamomOT', 'inputs')

# Parameters listed in the model_parameters sheet, grouped by use (empty value = default of base.py;
# other attributes of NetworkModel can be added as rows)
PARAMETER_GROUPS = [
    ('Pipeline: steps and data', [
        'estimate_proliferation_rates', 'simulate_with_proliferation', 'select_genes', 'build_prior_network',
        'simulate_perturbations', 'run_test', 'split', 'train_rate', 'species']),
    ('Gene selection', [
        'num_max_genes', 'n_query_genes', 'n_entropy_genes', 'network_method', 'k_in_steiner', 'closure_min',
        'min_entropy_change', 'min_nb_separation']),
    ('Literature prior network', [
        'prior_network_pen', 'max_free_params', 'literature_selection', 'literature_depth', 'literature_resources']),
    ('Stimulus', ['stimulus']),
    ('Per-cell sequencing depth', [
        'use_depth_factor', 'compute_depth_factor', 'allow_depth_correction', 'depth_method', 'depth_by_cell_type']),
    ('NB mixture and samples', [
        'mean_forcing_em', 'batch_size_mixture', 'soft_em_refinement', 'integrate_samples', 'ref_sample_integration']),
    ('Network inference', [
        'force_basins', 'temporal_basins', 'min_n_loops', 'max_iter', 'batch_size_traj', 'n_network_fits']),
    ('Proliferation and simulation', [
        'growth_reg_source', 'prolif_uses_stimulus', 'simulation_stochastic']),
    ('Reproducibility', ['seed']),
]
MODEL_PARAMETERS = [name for _, names in PARAMETER_GROUPS for name in names]
GROUP_TITLES = {title for title, _ in PARAMETER_GROUPS} | {'Other parameters'}

# Sheet -> (columns with their description, purpose, legacy files)
SHEETS = {
    'model_parameters': (
        [('parameter', 'Attribute of NetworkModel (CardamomOT/model/base.py); rows can be added for other attributes'),
         ('value', 'Value for this project (empty = default of base.py); the command-line options of the pipeline '
                   '(--stimulus, --prior, --mean-forcing, --force-basins, --temporal-basins) override it'),
         ('description', 'What the parameter does (comment of base.py)')],
        'Parameters of the model for this project, grouped by use: empty value = default of CardamomOT/model/base.py; '
        'a filled value overrides it, and the command-line options override the workbook.',
        []),
    'gene_lists': (
        [('genes_queries', 'Genes of interest for the gene selection (select_genes_and_split with select_genes = True), one per row'),
         ('proliferation_signatures', 'Proliferation marker genes (get_proliferation_rates), replace the built-in list'),
         ('death_signatures', 'Death marker genes (get_proliferation_rates), replace the built-in list'),
         ('senescence_signatures', 'Senescence / arrest marker genes (get_proliferation_rates), replace the built-in list')],
        'Gene lists, one gene per row in each column. Any other column is a named gene list, usable as the '
        'targets (STIMk) or a RATE target of perturbation_inference and perturbation_simulation '
        '(exported as gene_list_<name>.txt; legacy Data/gene_list_<name>.txt).',
        ['genes_queries.txt', 'proliferation_signatures', 'death_signatures', 'senescence_signatures']),
    'stimulus_inference_schedule': (
        [('time', 'Timepoint (optional: without times, rows follow the sorted timepoints of the data)'),
         ('stimulus_1', 'Value of stimulus 1 at this timepoint (default: 0 at the first timepoint, 1 after)')],
        'Schedule of the stimuli of the measured data (inference); one row per timepoint, one column per stimulus '
        '(add stimulus_2, stimulus_3... for several stimuli; without this sheet: one stimulus).',
        ['stimulus_schedule_inference.txt', 'stimulus_schedule.txt']),
    'stimulus_simulation_schedule': (
        [('time', 'Simulated timepoint (optional: without times, rows follow the sorted simulated timepoints)'),
         ('stimulus_1', 'Inference stimulus 1 during the simulations (default: its inference schedule)'),
         ('STIM1', 'Perturbation stimulus STIM1 of perturbation_simulation (default: 0 at the first time, 1 after)')],
        'Schedules of the simulations, one row per simulated timepoint: inference stimuli (stimulus_k, acting '
        'on the genes and, through the RATEk of perturbation_inference, on proliferation and death), then '
        'perturbation stimuli (STIMk, as in perturbation_simulation).',
        ['stimulus_schedule_simulate.txt', 'stimulus_schedule_simul.txt']),
    'perturbation_inference': (
        [('sample_id', 'dataset_id of a measured sample carrying genetic perturbations (KO / OV), or "all" '
                       'for the row describing the inference stimuli (STIMk / RATEk)'),
         ('KO', 'Genes knocked-out in this sample, comma-separated'),
         ('OV', 'Genes over-expressed in this sample, comma-separated'),
         ('STIM1', 'Row "all": possible direct targets of inference stimulus 1: a gene list of gene_lists or '
                   'comma-separated genes (empty: every gene)'),
         ('RATE1', 'Row "all": effect of inference stimulus 1 on the net proliferation rate (per hour), '
                   'TARGET:delta comma-separated, scaled by its schedule: TARGET = a cell type of '
                   'cell_type_proliferation (delta added to its rate of proliferation_rates, which is then the '
                   'rate without stimulus), or a gene list of gene_lists / GENE1+GENE2 / a gene (delta x mRNA '
                   'signature score in [0, 1])'),
         ('comment', 'Free comment (ignored)')],
        'Perturbations of the measured data, used by the inference: genetic perturbations of samples (KO / OV), '
        'and the inference stimuli (row sample_id = all: targets STIMk and effects on proliferation RATEk, '
        'k = column of stimulus_inference_schedule). Empty: one stimulus, every gene a possible target, no '
        'effect on the rates.',
        ['KO_OV_inference.txt', 'stimulus_targets']),
    'perturbation_simulation': (
        [('KO', 'Knocked-out genes, comma-separated; GENE-X for a partial KO of X%'),
         ('OV', 'Over-expressed genes, comma-separated; GENE-X for a partial OV of X%'),
         ('STIM1', "Perturbation stimulus 1: targets followed by + (activated) or - (inhibited), e.g. CHGA+STMN2-"),
         ('RATE1', "Effect of stimulus 1 (its schedule, STIM1 may be empty) on the net proliferation rate: "
                   "TARGET:delta, comma-separated; TARGET = a gene list of gene_lists, GENE1+GENE2..., a gene or "
                   "'all'; delta (per time unit) added to R of a cell at the maximal score, e.g. "
                   "ferroptosis_sensitive:-0.01 (needs the proliferation MLP)"),
         ('comment', 'Free comment (ignored)')],
        'In-silico perturbations simulated by simulate_network_KOV, one condition per row '
        '(add STIM2, STIM3... for several perturbation stimuli; their effects add up).',
        ['KO_OV_Stim_simulate.txt', 'KO_OV_simulate.txt']),
    'times': (
        [('times_to_inference', 'Inference restricted to the timepoints <= the largest value of this column'),
         ('times_to_simulate', 'Timepoints of the simulations (0 added if absent)')],
        'Timepoints, one per row.',
        ['times_to_inference.txt', 'times_to_simulate.txt']),
    'proliferation_rates': (
        [('cell_type', 'Cell type (as in obs cell_type_proliferation, else cell_type_transition, else cell_type)'),
         ('net_rate_per_hour', 'Reference net proliferation rate (birth - death), in h^-1; without the inference '
                               'stimulus if perturbation_inference gives its effect on this cell type (RATEk)')],
        'Anchors of the net proliferation rates per cell type (get_proliferation_rates); every cell type '
        'needs a value.',
        ['proliferation_rates']),
    'population_sizes': (
        [('time', 'Timepoint'), ('population_size', 'Total population size at this timepoint')],
        'Absolute population sizes, anchoring the growth estimated by optimal transport.',
        ['population_sizes']),
    'transition_rates': (
        [('from \\ to', 'Source cell type (rows) and target cell types (header): allowed transition rates')],
        'Cell-type transition rate matrix constraining optimal transport: first column = source cell types, '
        'header = target cell types (as in obs cell_type_transition, else cell_type).',
        ['transition_rates']),
}

# Sheet names of earlier versions -> current names (workbooks are migrated at the first use)
RENAMED = {'README': 'readme', 'Model_parameters': 'model_parameters', 'Gene_lists': 'gene_lists',
           'Stimulus_inference': 'stimulus_inference_schedule', 'Simulation_schedule': 'stimulus_simulation_schedule',
           'Perturbations': 'perturbation_simulation', 'KO_OV_inference': 'perturbation_inference',
           'Times': 'times', 'Proliferation_rates': 'proliferation_rates', 'Population_sizes': 'population_sizes',
           'Transition_rates': 'transition_rates'}


def _is_empty(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip())


def _clean(v):
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


# ---------------------------------------------------------------------------
# Workbook creation and access
# ---------------------------------------------------------------------------

def _style_header(ws, columns):
    from openpyxl.comments import Comment
    from openpyxl.styles import Font, PatternFill, Alignment
    for j, (name, desc) in enumerate(columns, start=1):
        c = ws.cell(row=1, column=j, value=name)
        c.font = Font(bold=True, color='FFFFFF')
        c.fill = PatternFill('solid', fgColor='1F3864')
        c.alignment = Alignment(horizontal='center')
        if desc:
            c.comment = Comment(desc, 'CardamomOT', width=320, height=90)
        ws.column_dimensions[c.column_letter].width = max(16, len(name) + 4)
    ws.freeze_panes = 'A2'


def _fill_readme(ws):
    """Content of the readme sheet: purpose of the workbook and of each sheet."""
    from openpyxl.styles import Font, Alignment
    lines = [
        ('CardamomOT — optional inputs of the run', 'title'),
        ('Fill only what you need: empty cells mean "not given" and the defaults of CardamomOT apply. '
         'Each sheet holds one kind of information; the header cells carry a comment (hover them) '
         'explaining their content.', ''),
        ('Text files of older projects (genes_queries.txt, KO_OV_simulate.txt, stimulus_schedule.txt...) '
         'still present in Data/ are imported into this workbook at each run and OVERWRITE the '
         'corresponding cells (a warning is printed): delete them to edit the workbook instead.', ''),
        ('Large numeric arrays stay files in Data/: reference_network.csv (structural prior), '
         'basal_init / basal_ref, inter_init / inter_ref (.npy or .csv), inter_simul_ref (.npy or .csv).', ''),
        ('', ''),
        ('Sheet', 'header'),
    ]
    for i, (text, kind) in enumerate(lines, start=1):
        c = ws.cell(row=i, column=1, value=text)
        if kind == 'title':
            c.font = Font(bold=True, size=14)
        elif kind == 'header':
            for j, h in enumerate(['Sheet', 'Content', 'Replaces (older text files)'], start=1):
                ws.cell(row=i, column=j, value=h).font = Font(bold=True)
        else:
            c.alignment = Alignment(wrap_text=True, vertical='top')
            ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=3)
            ws.row_dimensions[i].height = 45
    r = len(lines) + 1
    for name, (columns, purpose, legacy) in SHEETS.items():
        ws.cell(row=r, column=1, value=name).font = Font(bold=True)
        ws.cell(row=r, column=2, value=purpose).alignment = Alignment(wrap_text=True, vertical='top')
        ws.cell(row=r, column=3, value=', '.join(f + ('' if f.endswith(('.txt', '.csv')) else '.txt/.csv')
                                                 for f in legacy)).alignment = Alignment(wrap_text=True, vertical='top')
        ws.row_dimensions[r].height = 48
        r += 1
    ws.column_dimensions['A'].width = 28
    ws.column_dimensions['B'].width = 95
    ws.column_dimensions['C'].width = 45


def create_workbook(path):
    """Empty workbook with the documented structure (readme sheet + one sheet per input)."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = 'readme'
    _fill_readme(ws)
    for name, (columns, purpose, legacy) in SHEETS.items():
        _style_header(wb.create_sheet(name), columns)
    _fill_parameter_rows(wb['model_parameters'])
    wb.save(path)


def _all_row(df):
    """Index of the row sample_id = all of a perturbation_inference table (created if absent), and the table."""
    if 'sample_id' not in df:
        df['sample_id'] = None
    mask = df['sample_id'].astype(str).str.strip().str.lower() == 'all'
    if not mask.any():
        df = pd.concat([pd.DataFrame([{'sample_id': 'all'}]), df], ignore_index=True)
        return 0, df
    return df.index[mask][0], df


def _migrate(wb):
    """Sheets of earlier versions renamed (lowercase names), Stimulus_targets merged into
    perturbation_inference (row all, STIMk = gene list stimulus_k_targets); returns True if changed."""
    changed = False
    for old, new in RENAMED.items():
        if old in wb.sheetnames and new not in wb.sheetnames:
            wb[old].title = '_renaming_'  # openpyxl compares titles case-insensitively
            wb['_renaming_'].title = new
            changed = True
            if new not in ('readme', 'model_parameters'):
                df = read_sheet(wb, new)
                if len(df.columns):
                    write_sheet(wb, new, df)  # headers and comments of the current version
                else:
                    wb[new].delete_rows(1, wb[new].max_row)
                    _style_header(wb[new], SHEETS[new][0])
    if changed and 'readme' in wb.sheetnames:
        idx = wb.sheetnames.index('readme')
        del wb['readme']
        _fill_readme(wb.create_sheet('readme', idx))
    if 'Stimulus_targets' in wb.sheetnames:
        tg = read_sheet(wb, 'Stimulus_targets')
        gl, pi = read_sheet(wb, 'gene_lists'), read_sheet(wb, 'perturbation_inference')
        for c in [c for c in tg.columns if str(c).startswith('stimulus_')]:
            genes = _values(tg, c)
            if genes:
                k = int(str(c).split('_')[1])
                gl = _set_column(gl, f'stimulus_{k}_targets', genes)
                r, pi = _all_row(pi)
                pi.loc[r, f'STIM{k}'] = f'stimulus_{k}_targets'
        write_sheet(wb, 'gene_lists', gl)
        write_sheet(wb, 'perturbation_inference', _pi_columns(pi))
        del wb['Stimulus_targets']
        changed = True
    # Missing sheets of the current version (empty, with their header)
    for name, (columns, _, _) in SHEETS.items():
        if name not in wb.sheetnames:
            _style_header(wb.create_sheet(name), columns)
            if name == 'model_parameters':
                _fill_parameter_rows(wb[name])
            changed = True
    return changed


def _pi_columns(df):
    """perturbation_inference columns in their usual order (sample_id, KO, OV, STIMk / RATEk, comment, others)."""
    ks = sorted({int(c[4:]) for c in df.columns if re.fullmatch(r'(STIM|RATE)\d+', str(c))} | {1})
    order = ['sample_id', 'KO', 'OV'] + [x for k in ks for x in (f'STIM{k}', f'RATE{k}')] + ['comment']
    return df.reindex(columns=order + [c for c in df.columns if c not in order])


def parameter_docs():
    """{attribute: (default, comment)} of NetworkModel, comments read from base.py."""
    from .model.base import NetworkModel
    import inspect
    model = NetworkModel(1)
    src = inspect.getsource(NetworkModel.__init__)
    docs = {}
    for name, comment in re.findall(r'self\.(\w+)\s*=\s*[^#\n]*#\s*(.*)', src):
        docs.setdefault(name, comment.strip())
    return {name: (getattr(model, name, None), docs.get(name, '')) for name in vars(model)}


def _fill_parameter_rows(ws, values=None, extra=()):
    """Group title rows, then parameter / value / description of each listed parameter (+ extra rows)."""
    from openpyxl.styles import Alignment, Font, PatternFill
    docs, values = parameter_docs(), values or {}
    groups = PARAMETER_GROUPS + ([('Other parameters', list(extra))] if extra else [])
    r = 2
    for title, names in groups:
        for j in (1, 2, 3):
            ws.cell(row=r, column=j).fill = PatternFill('solid', fgColor='D9E1F2')
        ws.cell(row=r, column=1, value=title).font = Font(bold=True)
        r += 1
        for name in names:
            ws.cell(row=r, column=1, value=name)
            if name in values:
                ws.cell(row=r, column=2, value=values[name])
            ws.cell(row=r, column=3, value=docs.get(name, (None, ''))[1]).alignment = Alignment(wrap_text=True,
                                                                                               vertical='top')
            r += 1
    ws.column_dimensions['A'].width = 30
    ws.column_dimensions['B'].width = 14
    ws.column_dimensions['C'].width = 120


def _sync_parameter_sheet(ws):
    """Bring an existing model_parameters sheet to the current layout (groups, descriptions, no default column),
    keeping its values and its added rows; returns True if it was rewritten."""
    rows = [r for r in ws.iter_rows(min_row=1, values_only=True)]
    header = [str(c).strip() if c is not None else '' for c in (rows[0] if rows else ())]
    j_val = header.index('value') if 'value' in header else 1
    values, order = {}, []
    for r in rows[1:]:
        name = str(r[0]).strip() if r and r[0] is not None else ''
        if not name or name in GROUP_TITLES:
            continue
        order.append(name)
        v = r[j_val] if len(r) > j_val else None
        if v is not None and str(v).strip() != '':
            values[name] = v
    extra = [n for n in order if n not in MODEL_PARAMETERS]
    docs = parameter_docs()
    expected = [('parameter', 'value', 'description')]
    for title, names in PARAMETER_GROUPS + ([('Other parameters', extra)] if extra else []):
        expected.append((title, None, None))
        expected += [(n, values.get(n), docs.get(n, (None, ''))[1]) for n in names]
    current = [tuple((list(r) + [None] * 3)[:3]) for r in rows]
    if [tuple(e) for e in expected] == current and len(header) == 3:
        return False
    ws.delete_rows(1, ws.max_row)
    _style_header(ws, SHEETS['model_parameters'][0])
    _fill_parameter_rows(ws, values, extra)
    return True


def read_sheet(wb, name):
    """DataFrame of a sheet (header row 1), empty cells as NaN, empty rows dropped."""
    if name not in wb.sheetnames:
        return pd.DataFrame()
    rows = list(wb[name].iter_rows(values_only=True))
    if not rows:
        return pd.DataFrame()
    header = [str(h).strip() if h is not None else f'_col{j}' for j, h in enumerate(rows[0])]
    df = pd.DataFrame([list(r) + [None] * (len(header) - len(r)) for r in rows[1:]], columns=header)
    df = df.loc[:, [not h.startswith('_col') or df[h].notna().any() for h in header]]
    return df.dropna(how='all').reset_index(drop=True)


def write_sheet(wb, name, df):
    """Replace the content of a sheet by df (header kept and styled for the known columns)."""
    ws = wb[name] if name in wb.sheetnames else wb.create_sheet(name)
    ws.delete_rows(1, ws.max_row)
    known = dict(SHEETS.get(name, ([], '', []))[0])
    _style_header(ws, [(str(c), known.get(str(c), '')) for c in df.columns])
    for i, row in enumerate(df.itertuples(index=False), start=2):
        for j, v in enumerate(row, start=1):
            if not _is_empty(v):
                ws.cell(row=i, column=j, value=_clean(v))


def _set_column(df, col, values):
    """df with column col replaced by values (other columns kept, rows extended as needed)."""
    n = max(len(df), len(values))
    df = df.reindex(range(n))
    df[col] = list(values) + [None] * (n - len(values))
    return df


# ---------------------------------------------------------------------------
# Legacy text files -> workbook
# ---------------------------------------------------------------------------

def _legacy_path(data_dir, name):
    if name.endswith(('.txt', '.csv')):
        path = os.path.join(data_dir, name)
        return path if os.path.exists(path) else None
    for ext in ('.csv', '.txt'):
        path = os.path.join(data_dir, name + ext)
        if os.path.exists(path):
            return path
    return None


def _gene_list(path):
    text = '\n'.join(line.split('#', 1)[0] for line in open(path).read().splitlines())
    return [g for g in re.split(r'[,\s]+', text) if g]


def _raw_table(path):
    """Tab-separated table with a header, as strings (comment lines dropped)."""
    lines = [l for l in open(path).read().splitlines() if l.strip() and not l.lstrip().startswith('#')]
    header = [h.strip() for h in lines[0].split('\t')]
    rows = [[c.strip() for c in l.split('\t')] for l in lines[1:]]
    rows = [r + [''] * (len(header) - len(r)) for r in rows]
    return pd.DataFrame([r[:len(header)] for r in rows], columns=header)


def import_legacy(wb, data_dir):
    """Import the legacy text files of data_dir into the workbook; returns the names imported."""
    imported = []

    def note(path, sheet):
        imported.append(os.path.basename(path))
        print(f"[CardamomOT] Warning: Data/{os.path.basename(path)} imported into Data/{WORKBOOK} "
              f"(sheet {sheet}); text files overwrite the workbook: delete it to edit the workbook instead")

    # Gene lists
    gl = read_sheet(wb, 'gene_lists')
    for col, name in [('genes_queries', 'genes_queries.txt'), ('proliferation_signatures', 'proliferation_signatures'),
                      ('death_signatures', 'death_signatures'), ('senescence_signatures', 'senescence_signatures')]:
        path = _legacy_path(data_dir, name)
        if path:
            gl = _set_column(gl, col, _gene_list(path))
            note(path, 'gene_lists')
    # Named gene lists (RATE targets)
    for path in sorted(glob.glob(os.path.join(data_dir, 'gene_list_*.txt'))):
        gl = _set_column(gl, os.path.basename(path)[len('gene_list_'):-4], _gene_list(path))
        note(path, 'gene_lists')
    if imported:
        write_sheet(wb, 'gene_lists', gl)

    # Inference schedule (rows in timepoint order, no time column)
    n_inf = None
    for name in ('stimulus_schedule_inference.txt', 'stimulus_schedule.txt'):
        path = _legacy_path(data_dir, name)
        if path:
            arr = np.loadtxt(path, ndmin=2)
            n_inf = arr.shape[1]
            df = pd.DataFrame(arr, columns=[f'stimulus_{k + 1}' for k in range(n_inf)])
            df.insert(0, 'time', None)
            write_sheet(wb, 'stimulus_inference_schedule', df)
            note(path, 'stimulus_inference_schedule')
            break
    if n_inf is None:
        n_inf = max(1, sum(c.startswith('stimulus_') for c in read_sheet(wb, 'stimulus_inference_schedule').columns))

    # Stimulus targets (tab columns, or one list) -> gene lists stimulus_k_targets, row all of perturbation_inference
    path = _legacy_path(data_dir, 'stimulus_targets')
    if path:
        from .config import read_stimulus_targets
        cols = read_stimulus_targets(data_dir) or []
        gl, pi = read_sheet(wb, 'gene_lists'), read_sheet(wb, 'perturbation_inference')
        for k, genes in enumerate(cols, start=1):
            if genes:
                gl = _set_column(gl, f'stimulus_{k}_targets', genes)
                r, pi = _all_row(pi)
                pi.loc[r, f'STIM{k}'] = f'stimulus_{k}_targets'
        write_sheet(wb, 'gene_lists', gl)
        write_sheet(wb, 'perturbation_inference', _pi_columns(pi))
        note(path, 'gene_lists / perturbation_inference')

    # Simulation schedule: inference stimuli, then perturbation stimuli
    for name in ('stimulus_schedule_simulate.txt', 'stimulus_schedule_simul.txt'):
        path = _legacy_path(data_dir, name)
        if path:
            arr = np.loadtxt(path, ndmin=2)
            cols = [f'stimulus_{k + 1}' if k < n_inf else f'STIM{k - n_inf + 1}' for k in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=cols)
            df.insert(0, 'time', None)
            write_sheet(wb, 'stimulus_simulation_schedule', df)
            note(path, 'stimulus_simulation_schedule')
            break

    # Perturbations
    for name in ('KO_OV_Stim_simulate.txt', 'KO_OV_simulate.txt'):
        path = _legacy_path(data_dir, name)
        if path:
            df = _raw_table(path)
            df.columns = [{'STIM': 'STIM1', 'RATE': 'RATE1'}.get(c.upper(), c.upper() if c.upper() in ('KO', 'OV') or
                          re.fullmatch(r'(STIM|RATE)\d+', c.upper()) else c) for c in df.columns]
            df = df.replace({'0': None, '': None})
            write_sheet(wb, 'perturbation_simulation', df)
            note(path, 'perturbation_simulation')
            break

    # Perturbations of the measured samples
    path = _legacy_path(data_dir, 'KO_OV_inference.txt')
    if path:
        df = _raw_table(path)
        df.columns = ['sample_id' if c.upper() in ('SAMPLE_ID', 'DATASET_ID') else c.upper() for c in df.columns]
        # Sample rows replaced, row all (inference stimuli) kept
        pi = read_sheet(wb, 'perturbation_inference')
        keep = pi[pi['sample_id'].astype(str).str.strip().str.lower() == 'all'] if 'sample_id' in pi else pi.iloc[:0]
        write_sheet(wb, 'perturbation_inference',
                    _pi_columns(pd.concat([keep, df.replace({'0': None, '': None})], ignore_index=True)))
        note(path, 'perturbation_inference')

    # Timepoints
    tm = read_sheet(wb, 'times')
    changed = False
    for col in ('times_to_inference', 'times_to_simulate'):
        path = _legacy_path(data_dir, f'{col}.txt')
        if path:
            vals = [float(l.strip()) for l in open(path) if l.strip()]
            tm = _set_column(tm, col, vals)
            note(path, 'times')
            changed = True
    if changed:
        write_sheet(wb, 'times', tm)

    # Two-column tables
    for sheet, name, cols in [('proliferation_rates', 'proliferation_rates', ['cell_type', 'net_rate_per_hour']),
                              ('population_sizes', 'population_sizes', ['time', 'population_size'])]:
        path = _legacy_path(data_dir, name)
        if path:
            df = pd.read_csv(path, sep=None, engine='python', header=None).iloc[:, :2]
            df.columns = cols
            write_sheet(wb, sheet, df)
            note(path, sheet)

    # Transition rate matrix
    path = _legacy_path(data_dir, 'transition_rates')
    if path:
        df = pd.read_csv(path, sep=None, engine='python', index_col=0)
        df.insert(0, 'from \\ to', df.index.astype(str))
        write_sheet(wb, 'transition_rates', df.reset_index(drop=True))
        note(path, 'transition_rates')
    return imported


# ---------------------------------------------------------------------------
# Workbook -> text files read by the pipeline
# ---------------------------------------------------------------------------

def _values(df, col):
    return [_clean(v) for v in df[col] if not _is_empty(v)] if col in df else []


def _schedule(df, cols):
    """Matrix of the schedule columns, rows sorted by time when every row has one."""
    df = df.dropna(subset=cols, how='all')
    if 'time' in df and df['time'].notna().all() and len(df):
        df = df.sort_values('time')
    return df[cols].astype(float).to_numpy()


def export(wb, out_dir):
    """Write the content of the workbook as the text files read by the pipeline."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    write = lambda name, text: open(os.path.join(out_dir, name), 'w').write(text)

    gl = read_sheet(wb, 'gene_lists')
    for col, name in [('genes_queries', 'genes_queries.txt'), ('proliferation_signatures', 'proliferation_signatures.txt'),
                      ('death_signatures', 'death_signatures.txt'), ('senescence_signatures', 'senescence_signatures.txt')]:
        genes = _values(gl, col)
        if genes:
            write(name, '\n'.join(map(str, genes)) + '\n')
    known = {'genes_queries', 'proliferation_signatures', 'death_signatures', 'senescence_signatures'}
    for col in gl.columns:
        genes = _values(gl, col) if col not in known and not str(col).startswith('_col') else []
        if genes:
            write(f'gene_list_{col}.txt', '\n'.join(map(str, genes)) + '\n')

    st = read_sheet(wb, 'stimulus_inference_schedule')
    inf_cols = sorted([c for c in st.columns if c.startswith('stimulus_') and st[c].notna().any()],
                      key=lambda c: int(c.split('_')[1]))
    if inf_cols:
        np.savetxt(os.path.join(out_dir, 'stimulus_schedule_inference.txt'), _schedule(st, inf_cols), fmt='%g')
    n_inf = max(1, len(inf_cols))

    # Inference stimuli (row all of perturbation_inference): targets STIMk and rate effects RATEk
    pi = read_sheet(wb, 'perturbation_inference')
    is_all = (pi['sample_id'].astype(str).str.strip().str.lower() == 'all') if 'sample_id' in pi else pd.Series([], dtype=bool)
    lists, rates = {}, {}
    for _, r in pi[is_all].iterrows():
        for c in pi.columns:
            m = re.fullmatch(r'(STIM|RATE)(\d+)', str(c))
            if not m or _is_empty(r[c]):
                continue
            k, cell = int(m.group(2)), str(_clean(r[c]))
            if m.group(1) == 'STIM':
                # A gene list of gene_lists, else comma-separated genes
                genes = _values(gl, cell) if cell in gl.columns else [g.strip() for g in re.split(r'[,\s]+', cell) if g.strip()]
                lists.setdefault(k, []).extend(str(g) for g in genes)
            else:
                from .tools.perturbations import parse_rate
                rates.setdefault(k, []).extend(parse_rate(cell))
    bad = sorted(k for k in set(lists) | set(rates) if k > n_inf)
    if bad:
        print(f"[CardamomOT] Warning: perturbation_inference describes inference stimuli {bad} but "
              f"stimulus_inference_schedule has {n_inf} stimulus column(s): ignored")
    if any(lists.get(k) for k in range(1, n_inf + 1)):
        cols = [lists.get(k, []) for k in range(1, n_inf + 1)]
        n = max(len(l) for l in cols)
        write('stimulus_targets.txt', '\n'.join('\t'.join(l[i] if i < len(l) else '' for l in cols)
                                                for i in range(n)) + '\n')
    rates = {k: v for k, v in rates.items() if k <= n_inf and v}
    if rates:
        import json
        json.dump({str(k): v for k, v in rates.items()}, open(os.path.join(out_dir, 'stimulus_rates.json'), 'w'),
                  indent=1)

    ss = read_sheet(wb, 'stimulus_simulation_schedule')
    s_inf = sorted([c for c in ss.columns if c.startswith('stimulus_') and ss[c].notna().any()],
                   key=lambda c: int(c.split('_')[1]))
    s_pert = sorted([c for c in ss.columns if re.fullmatch(r'STIM\d+', c) and ss[c].notna().any()],
                    key=lambda c: int(c[4:]))
    if s_inf or s_pert:
        rows = ss.dropna(subset=s_inf + s_pert, how='all')
        if 'time' in rows and rows['time'].notna().all() and len(rows):
            rows = rows.sort_values('time')
        if not s_inf:
            # Inference stimuli not given: default schedule (0 at the first time, 1 after)
            print(f"[CardamomOT] Warning: sheet stimulus_simulation_schedule has perturbation stimuli but no inference "
                  f"stimulus column: default 0 at the first time and 1 after for the {n_inf} inference stimuli")
            inf = np.ones((len(rows), n_inf))
            inf[0] = 0
        else:
            inf = rows[s_inf].astype(float).to_numpy()
        pert = rows[s_pert].astype(float).fillna(1.0).to_numpy() if s_pert else np.zeros((len(rows), 0))
        np.savetxt(os.path.join(out_dir, 'stimulus_schedule_simulate.txt'), np.hstack([inf, pert]), fmt='%g')

    pt = read_sheet(wb, 'perturbation_simulation')
    pcols = [c for c in pt.columns if c in ('KO', 'OV') or re.fullmatch(r'(STIM|RATE)\d+', str(c))]
    if pcols and len(pt):
        lines = ['\t'.join(pcols)]
        for _, r in pt.iterrows():
            cells = ['0' if _is_empty(r[c]) else str(_clean(r[c])) for c in pcols]
            if any(c != '0' for c in cells):
                lines.append('\t'.join(cells))
        if len(lines) > 1:
            write('KO_OV_Stim_simulate.txt', '\n'.join(lines) + '\n')

    ki = pi[~is_all] if len(pi) else pi
    if 'sample_id' in ki and len(ki):
        lines = ['SAMPLE_ID\tKO\tOV']
        for _, r in ki.iterrows():
            if not _is_empty(r['sample_id']) and (not _is_empty(r.get('KO')) or not _is_empty(r.get('OV'))):
                lines.append('\t'.join(['0' if _is_empty(r.get(c)) else str(_clean(r.get(c)))
                                        for c in ('sample_id', 'KO', 'OV')]))
        if len(lines) > 1:
            write('KO_OV_inference.txt', '\n'.join(lines) + '\n')

    tm = read_sheet(wb, 'times')
    for col in ('times_to_inference', 'times_to_simulate'):
        vals = _values(tm, col)
        if vals:
            write(f'{col}.txt', '\n'.join(f'{float(v):g}' for v in vals) + '\n')

    for sheet, name, cols in [('proliferation_rates', 'proliferation_rates.txt', ['cell_type', 'net_rate_per_hour']),
                              ('population_sizes', 'population_sizes.txt', ['time', 'population_size'])]:
        df = read_sheet(wb, sheet)
        if all(c in df for c in cols):
            df = df.dropna(subset=cols)
            if len(df):
                write(name, '\n'.join(f'{_clean(a)}\t{_clean(b)}' for a, b in zip(df[cols[0]], df[cols[1]])) + '\n')

    mp = read_sheet(wb, 'model_parameters')
    if 'parameter' in mp and 'value' in mp:
        import json
        values = {str(k).strip(): _clean(v) for k, v in zip(mp['parameter'], mp['value'])
                  if not _is_empty(k) and not _is_empty(v)}
        if values:
            json.dump(values, open(os.path.join(out_dir, 'model_parameters.json'), 'w'), indent=1, default=str)

    tr = read_sheet(wb, 'transition_rates')
    if len(tr) and tr.shape[1] > 1:
        tr = tr.set_index(tr.columns[0])
        tr.index = tr.index.astype(str)
        tr.to_csv(os.path.join(out_dir, 'transition_rates.csv'))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_SYNCED = {}


def project_parameters(project):
    """{attribute: value} filled in the model_parameters sheet of the project."""
    import json
    path = os.path.join(input_dir(project), 'model_parameters.json')
    return json.load(open(path)) if os.path.exists(path) else {}


def depth_factor_used(project):
    """use_depth_factor of the project (model_parameters sheet, else the default of NetworkModel)."""
    from .model.base import NetworkModel
    m = NetworkModel(1)
    m.apply_project_parameters(project, verb=False)
    return bool(m.use_depth_factor)


def input_dir(project):
    """
    Directory of the run inputs of a project (cardamomOT/inputs/), synchronised once per process
    from Data/CardamomOT_inputs.xlsx (created if missing) and from the legacy text files of Data/.
    """
    project = os.path.abspath(project)
    if project in _SYNCED:
        return _SYNCED[project]
    from openpyxl import load_workbook
    data_dir = os.path.join(project, 'Data')
    path = os.path.join(data_dir, WORKBOOK)
    out = os.path.join(project, CACHE)
    if os.path.isdir(data_dir):
        if not os.path.exists(path):
            create_workbook(path)
            print(f"[CardamomOT] Created Data/{WORKBOOK}: optional inputs of the run (empty cells = defaults)")
        wb = load_workbook(path)
        if _migrate(wb):
            # Workbook of an earlier version: lowercase sheet names, merged sheets, new sheets
            try:
                wb.save(path)
                print(f"[CardamomOT] Data/{WORKBOOK} updated to the current sheet layout")
            except PermissionError:
                pass
        if _sync_parameter_sheet(wb['model_parameters']):
            # Layout of the current version (groups, descriptions, new parameters), values kept
            try:
                wb.save(path)
            except PermissionError:
                pass
        if import_legacy(wb, data_dir):
            try:
                wb.save(path)
            except PermissionError:
                print(f"[CardamomOT] Warning: could not save Data/{WORKBOOK} (open in another program?); "
                      f"the imported text files are used for this run")
        export(wb, out)
    else:
        os.makedirs(out, exist_ok=True)
    _SYNCED[project] = out
    return out
