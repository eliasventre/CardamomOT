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
import os
import re
import shutil

import numpy as np
import pandas as pd

WORKBOOK = 'CardamomOT_inputs.xlsx'
CACHE = os.path.join('cardamomOT', 'inputs')

# Parameters of NetworkModel listed in the Model_parameters sheet (others can be added as rows)
MODEL_PARAMETERS = [
    # gene selection
    'num_max_genes', 'n_query_genes', 'n_entropy_genes', 'max_free_params', 'network_method', 'k_in_steiner',
    'closure_min', 'min_entropy_change', 'min_nb_separation', 'literature_selection', 'literature_depth',
    'literature_resources',
    # per-cell depth
    'allow_depth_correction', 'depth_method', 'depth_by_cell_type',
    # mixture and samples
    'mean_forcing_em', 'force_basins', 'temporal_basins', 'batch_size_mixture', 'integrate_samples',
    # network inference
    'stimulus', 'prior_network_pen', 'min_n_loops', 'max_iter', 'batch_size_traj', 'n_network_fits',
    # proliferation and simulation
    'simulate_with_proliferation', 'growth_reg_source', 'simulation_stochastic',
]

# Sheet -> (columns with their description, purpose, legacy files)
SHEETS = {
    'Model_parameters': (
        [('parameter', 'Attribute of NetworkModel (CardamomOT/model/base.py); rows can be added for other attributes'),
         ('value', 'Value for this project: when filled, it overrides the default AND the options of the pipeline / CLI'),
         ('default', 'Default value of CardamomOT (for information)'),
         ('description', 'What the parameter does (comment of base.py)')],
        'Parameters of the model for this project: a filled value overrides the default and the pipeline / '
        'command-line options; empty value = default or command-line option.',
        []),
    'Gene_lists': (
        [('genes_queries', 'Genes of interest for the gene selection (select_genes_and_split -c 1), one per row'),
         ('proliferation_signatures', 'Proliferation marker genes (get_proliferation_rates), replace the built-in list'),
         ('death_signatures', 'Death marker genes (get_proliferation_rates), replace the built-in list'),
         ('senescence_signatures', 'Senescence / arrest marker genes (get_proliferation_rates), replace the built-in list')],
        'Gene lists, one gene per row in each column.',
        ['genes_queries.txt', 'proliferation_signatures', 'death_signatures', 'senescence_signatures']),
    'Stimulus_inference': (
        [('time', 'Timepoint (optional: without times, rows follow the sorted timepoints of the data)'),
         ('stimulus_1', 'Value of stimulus 1 at this timepoint (default: 0 at the first timepoint, 1 after)')],
        'Schedule of the stimuli during the inference; one row per timepoint, one column per stimulus '
        '(add stimulus_2, stimulus_3... for several stimuli).',
        ['stimulus_schedule_inference.txt', 'stimulus_schedule.txt']),
    'Stimulus_targets': (
        [('stimulus_1', 'Possible direct targets of stimulus 1, one gene per row (empty column: no constraint)')],
        'Possible targets of each stimulus (restricts the stimulus edges in the gene selection and the '
        'network); a column naming no gene of the data means no constraint.',
        ['stimulus_targets']),
    'Simulation_schedule': (
        [('time', 'Simulated timepoint (optional: without times, rows follow the sorted simulated timepoints)'),
         ('stimulus_1', 'Inference stimulus 1 during the simulations (default: its inference schedule)'),
         ('STIM1', 'Perturbation stimulus STIM1 of the Perturbations sheet (default: 0 at the first time, 1 after)')],
        'Schedules of the simulations, one row per simulated timepoint: inference stimuli (stimulus_k), '
        'then perturbation stimuli (STIMk, as in the Perturbations sheet).',
        ['stimulus_schedule_simulate.txt', 'stimulus_schedule_simul.txt']),
    'Perturbations': (
        [('KO', 'Knocked-out genes, comma-separated; GENE-X for a partial KO of X%'),
         ('OV', 'Over-expressed genes, comma-separated; GENE-X for a partial OV of X%'),
         ('STIM1', "Perturbation stimulus 1: targets followed by + (activated) or - (inhibited), e.g. CHGA+STMN2-"),
         ('comment', 'Free comment (ignored)')],
        'In-silico perturbations simulated by simulate_network_KOV, one condition per row '
        '(add STIM2, STIM3... for several perturbation stimuli; their effects add up).',
        ['KO_OV_Stim_simulate.txt', 'KO_OV_simulate.txt']),
    'KO_OV_inference': (
        [('sample_id', 'dataset_id of a sample carrying genetic perturbations'),
         ('KO', 'Genes knocked-out in this sample, comma-separated'),
         ('OV', 'Genes over-expressed in this sample, comma-separated')],
        'Known perturbations of the measured samples, used as priors of the inference.',
        ['KO_OV_inference.txt']),
    'Times': (
        [('times_to_inference', 'Inference restricted to the timepoints <= the largest value of this column'),
         ('times_to_simulate', 'Timepoints of the simulations (0 added if absent)')],
        'Timepoints, one per row.',
        ['times_to_inference.txt', 'times_to_simulate.txt']),
    'Proliferation_rates': (
        [('cell_type', 'Cell type (as in obs cell_type_proliferation, else cell_type_transition, else cell_type)'),
         ('net_rate_per_hour', 'Reference net proliferation rate (birth - death), in h^-1')],
        'Anchors of the net proliferation rates per cell type (get_proliferation_rates); every cell type '
        'needs a value.',
        ['proliferation_rates']),
    'Population_sizes': (
        [('time', 'Timepoint'), ('population_size', 'Total population size at this timepoint')],
        'Absolute population sizes, anchoring the growth estimated by optimal transport.',
        ['population_sizes']),
    'Transition_rates': (
        [('from \\ to', 'Source cell type (rows) and target cell types (header): allowed transition rates')],
        'Cell-type transition rate matrix constraining optimal transport: first column = source cell types, '
        'header = target cell types (as in obs cell_type_transition, else cell_type).',
        ['transition_rates']),
}


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


def create_workbook(path):
    """Empty workbook with the documented structure (README sheet + one sheet per input)."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment
    wb = Workbook()
    ws = wb.active
    ws.title = 'README'
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
    ws.column_dimensions['A'].width = 24
    ws.column_dimensions['B'].width = 95
    ws.column_dimensions['C'].width = 45
    for name, (columns, purpose, legacy) in SHEETS.items():
        _style_header(wb.create_sheet(name), columns)
    _fill_parameter_rows(wb['Model_parameters'])
    wb.save(path)


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


def _fill_parameter_rows(ws, values=None):
    """Rows parameter / value / default / description of MODEL_PARAMETERS."""
    from openpyxl.styles import Alignment
    docs = parameter_docs()
    for i, name in enumerate(MODEL_PARAMETERS, start=2):
        default, comment = docs.get(name, (None, ''))
        ws.cell(row=i, column=1, value=name)
        if values and name in values:
            ws.cell(row=i, column=2, value=values[name])
        ws.cell(row=i, column=3, value=repr(default) if isinstance(default, (dict, list, tuple)) or default is None
                else default)
        c = ws.cell(row=i, column=4, value=comment)
        c.alignment = Alignment(wrap_text=True, vertical='top')
    ws.column_dimensions['A'].width = 28
    ws.column_dimensions['B'].width = 14
    ws.column_dimensions['C'].width = 14
    ws.column_dimensions['D'].width = 110


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
    gl = read_sheet(wb, 'Gene_lists')
    for col, name in [('genes_queries', 'genes_queries.txt'), ('proliferation_signatures', 'proliferation_signatures'),
                      ('death_signatures', 'death_signatures'), ('senescence_signatures', 'senescence_signatures')]:
        path = _legacy_path(data_dir, name)
        if path:
            gl = _set_column(gl, col, _gene_list(path))
            note(path, 'Gene_lists')
    if imported:
        write_sheet(wb, 'Gene_lists', gl)

    # Inference schedule (rows in timepoint order, no time column)
    n_inf = None
    for name in ('stimulus_schedule_inference.txt', 'stimulus_schedule.txt'):
        path = _legacy_path(data_dir, name)
        if path:
            arr = np.loadtxt(path, ndmin=2)
            n_inf = arr.shape[1]
            df = pd.DataFrame(arr, columns=[f'stimulus_{k + 1}' for k in range(n_inf)])
            df.insert(0, 'time', None)
            write_sheet(wb, 'Stimulus_inference', df)
            note(path, 'Stimulus_inference')
            break
    if n_inf is None:
        n_inf = max(1, sum(c.startswith('stimulus_') for c in read_sheet(wb, 'Stimulus_inference').columns))

    # Stimulus targets (tab columns, or one list)
    path = _legacy_path(data_dir, 'stimulus_targets')
    if path:
        from .config import read_stimulus_targets
        cols = read_stimulus_targets(data_dir) or []
        df = pd.DataFrame()
        for k, genes in enumerate(cols, start=1):
            df = _set_column(df, f'stimulus_{k}', genes)
        write_sheet(wb, 'Stimulus_targets', df)
        note(path, 'Stimulus_targets')

    # Simulation schedule: inference stimuli, then perturbation stimuli
    for name in ('stimulus_schedule_simulate.txt', 'stimulus_schedule_simul.txt'):
        path = _legacy_path(data_dir, name)
        if path:
            arr = np.loadtxt(path, ndmin=2)
            cols = [f'stimulus_{k + 1}' if k < n_inf else f'STIM{k - n_inf + 1}' for k in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=cols)
            df.insert(0, 'time', None)
            write_sheet(wb, 'Simulation_schedule', df)
            note(path, 'Simulation_schedule')
            break

    # Perturbations
    for name in ('KO_OV_Stim_simulate.txt', 'KO_OV_simulate.txt'):
        path = _legacy_path(data_dir, name)
        if path:
            df = _raw_table(path)
            df.columns = ['STIM1' if c.upper() == 'STIM' else (c.upper() if c.upper() in ('KO', 'OV') or
                          re.fullmatch(r'STIM\d+', c.upper()) else c) for c in df.columns]
            df = df.replace({'0': None, '': None})
            write_sheet(wb, 'Perturbations', df)
            note(path, 'Perturbations')
            break

    # Perturbations of the measured samples
    path = _legacy_path(data_dir, 'KO_OV_inference.txt')
    if path:
        df = _raw_table(path)
        df.columns = ['sample_id' if c.upper() in ('SAMPLE_ID', 'DATASET_ID') else c.upper() for c in df.columns]
        write_sheet(wb, 'KO_OV_inference', df.replace({'0': None, '': None}))
        note(path, 'KO_OV_inference')

    # Timepoints
    tm = read_sheet(wb, 'Times')
    changed = False
    for col in ('times_to_inference', 'times_to_simulate'):
        path = _legacy_path(data_dir, f'{col}.txt')
        if path:
            vals = [float(l.strip()) for l in open(path) if l.strip()]
            tm = _set_column(tm, col, vals)
            note(path, 'Times')
            changed = True
    if changed:
        write_sheet(wb, 'Times', tm)

    # Two-column tables
    for sheet, name, cols in [('Proliferation_rates', 'proliferation_rates', ['cell_type', 'net_rate_per_hour']),
                              ('Population_sizes', 'population_sizes', ['time', 'population_size'])]:
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
        write_sheet(wb, 'Transition_rates', df.reset_index(drop=True))
        note(path, 'Transition_rates')
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

    gl = read_sheet(wb, 'Gene_lists')
    for col, name in [('genes_queries', 'genes_queries.txt'), ('proliferation_signatures', 'proliferation_signatures.txt'),
                      ('death_signatures', 'death_signatures.txt'), ('senescence_signatures', 'senescence_signatures.txt')]:
        genes = _values(gl, col)
        if genes:
            write(name, '\n'.join(map(str, genes)) + '\n')

    st = read_sheet(wb, 'Stimulus_inference')
    inf_cols = sorted([c for c in st.columns if c.startswith('stimulus_') and st[c].notna().any()],
                      key=lambda c: int(c.split('_')[1]))
    if inf_cols:
        np.savetxt(os.path.join(out_dir, 'stimulus_schedule_inference.txt'), _schedule(st, inf_cols), fmt='%g')
    n_inf = max(1, len(inf_cols))

    tg = read_sheet(wb, 'Stimulus_targets')
    tcols = sorted([c for c in tg.columns if c.startswith('stimulus_')], key=lambda c: int(c.split('_')[1]))
    lists = [list(map(str, _values(tg, c))) for c in tcols]
    if any(lists):
        n = max(len(l) for l in lists)
        write('stimulus_targets.txt', '\n'.join('\t'.join(l[i] if i < len(l) else '' for l in lists)
                                                for i in range(n)) + '\n')

    ss = read_sheet(wb, 'Simulation_schedule')
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
            print(f"[CardamomOT] Warning: sheet Simulation_schedule has perturbation stimuli but no inference "
                  f"stimulus column: default 0 at the first time and 1 after for the {n_inf} inference stimuli")
            inf = np.ones((len(rows), n_inf))
            inf[0] = 0
        else:
            inf = rows[s_inf].astype(float).to_numpy()
        pert = rows[s_pert].astype(float).fillna(1.0).to_numpy() if s_pert else np.zeros((len(rows), 0))
        np.savetxt(os.path.join(out_dir, 'stimulus_schedule_simulate.txt'), np.hstack([inf, pert]), fmt='%g')

    pt = read_sheet(wb, 'Perturbations')
    pcols = [c for c in pt.columns if c in ('KO', 'OV') or re.fullmatch(r'STIM\d+', str(c))]
    if pcols and len(pt):
        lines = ['\t'.join(pcols)]
        for _, r in pt.iterrows():
            cells = ['0' if _is_empty(r[c]) else str(_clean(r[c])) for c in pcols]
            if any(c != '0' for c in cells):
                lines.append('\t'.join(cells))
        if len(lines) > 1:
            write('KO_OV_Stim_simulate.txt', '\n'.join(lines) + '\n')

    ki = read_sheet(wb, 'KO_OV_inference')
    if 'sample_id' in ki and len(ki):
        lines = ['SAMPLE_ID\tKO\tOV']
        for _, r in ki.iterrows():
            if not _is_empty(r['sample_id']):
                lines.append('\t'.join(['0' if _is_empty(r.get(c)) else str(_clean(r.get(c)))
                                        for c in ('sample_id', 'KO', 'OV')]))
        if len(lines) > 1:
            write('KO_OV_inference.txt', '\n'.join(lines) + '\n')

    tm = read_sheet(wb, 'Times')
    for col in ('times_to_inference', 'times_to_simulate'):
        vals = _values(tm, col)
        if vals:
            write(f'{col}.txt', '\n'.join(f'{float(v):g}' for v in vals) + '\n')

    for sheet, name, cols in [('Proliferation_rates', 'proliferation_rates.txt', ['cell_type', 'net_rate_per_hour']),
                              ('Population_sizes', 'population_sizes.txt', ['time', 'population_size'])]:
        df = read_sheet(wb, sheet)
        if all(c in df for c in cols):
            df = df.dropna(subset=cols)
            if len(df):
                write(name, '\n'.join(f'{_clean(a)}\t{_clean(b)}' for a, b in zip(df[cols[0]], df[cols[1]])) + '\n')

    mp = read_sheet(wb, 'Model_parameters')
    if 'parameter' in mp and 'value' in mp:
        import json
        values = {str(k).strip(): _clean(v) for k, v in zip(mp['parameter'], mp['value'])
                  if not _is_empty(k) and not _is_empty(v)}
        if values:
            json.dump(values, open(os.path.join(out_dir, 'model_parameters.json'), 'w'), indent=1, default=str)

    tr = read_sheet(wb, 'Transition_rates')
    if len(tr) and tr.shape[1] > 1:
        tr = tr.set_index(tr.columns[0])
        tr.index = tr.index.astype(str)
        tr.to_csv(os.path.join(out_dir, 'transition_rates.csv'))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_SYNCED = {}


def project_parameters(project):
    """{attribute: value} filled in the Model_parameters sheet of the project."""
    import json
    path = os.path.join(input_dir(project), 'model_parameters.json')
    return json.load(open(path)) if os.path.exists(path) else {}


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
        if 'Model_parameters' not in wb.sheetnames:
            # Workbook created before the parameter sheet: add it (empty values)
            ws = wb.create_sheet('Model_parameters', 1)
            _style_header(ws, SHEETS['Model_parameters'][0])
            _fill_parameter_rows(ws)
            wb.save(path)
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
