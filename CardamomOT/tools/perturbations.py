"""
In-silico perturbations to simulate: the perturbation_simulation sheet.

Tab-separated table with a header and the columns KO, OV and optionally STIM1, STIM2... (STIM =
STIM1); one condition per row, '0' or empty for nothing; '#' lines are comments:
- KO / OV: comma-separated genes, 'GENE-X' for a partial perturbation of X% (0 < X < 100);
- STIMk: perturbation stimulus k and its targets, each gene followed by '+' (activated) or '-'
  (inhibited), e.g. 'CHGA+STMN2-' or 'CHGA+,STMN2-'. Its effect on a target is that of an
  interaction of ±(100 + sum of the |interactions| received by the gene), which dominates them
  like a KO/OV, scaled by the value of the stimulus, which follows its own schedule (column k of
  the perturbation stimuli in stimulus_simulation_schedule sheet, after the inference stimuli;
  default 0 at the first simulated time and 1 after). Several stimuli add their effects.
- RATEk: effect of perturbation stimulus k on the net proliferation rate (same schedule as STIMk,
  which may be empty): comma-separated 'TARGET:delta' entries, delta (per time unit, e.g. -0.01)
  being added to the rate R of a cell at the maximal score of TARGET, in proportion to its score:
  TARGET = a gene list of the gene_lists sheet (any extra column, e.g. ferroptosis_sensitive),
  genes joined by '+' (e.g. FTH1+FTL+TFRC), a single gene, or 'all' (every cell, score 1). The score
  of a cell is the mean over the genes of its protein level divided by the 99th percentile of the
  trajectories, clipped to [0, 1]. Applied in the branching simulation (proliferation MLP), along
  each simulated path; the population size of each condition is recorded.
"""
import os
import re

import numpy as np

FILE_NAME = 'KO_OV_Stim_simulate.txt'


def find_perturbation_file(data_dir):
    """Path of the perturbation table (sheet perturbation_simulation, exported), or None."""
    path = os.path.join(data_dir, FILE_NAME)
    return path if os.path.exists(path) else None


def parse_gene_with_pct(token):
    """'GENE-X' -> (gene, X) for 0 < X < 100, else (token, None) (keeps names like 'HIF-1A')."""
    token = token.strip()
    m = re.match(r'^(.+)-(\d+(?:\.\d+)?)$', token)
    if m and 0.0 < float(m.group(2)) < 100.0:
        return m.group(1).strip(), float(m.group(2))
    return token, None


def parse_stim(cell, genes=None):
    """
    Targets of a perturbation stimulus, [(gene, +1 / -1)]. Separated by commas, each entry ends
    with its sign; without commas the entries are split on the signs, using the gene names
    (longest match) when given so that names containing '-' are kept.
    """
    s = re.sub(r'\s+', '', str(cell))
    if s.lower() in ('', '0', 'none', 'nan'):
        return []
    if ',' in s:
        out = []
        for tok in filter(None, s.split(',')):
            if tok[-1] not in '+-':
                raise ValueError(f"STIM entry '{tok}' must end with '+' or '-'")
            out.append((tok[:-1], 1 if tok[-1] == '+' else -1))
        return out
    if genes is not None:
        upper = sorted({str(g) for g in genes}, key=len, reverse=True)
        out, i = [], 0
        while i < len(s):
            hit = next((g for g in upper if s[i:].upper().startswith(g.upper())
                        and i + len(g) < len(s) and s[i + len(g)] in '+-'), None)
            if hit is None:
                break
            out.append((s[i:i + len(hit)], 1 if s[i + len(hit)] == '+' else -1))
            i += len(hit) + 1
        if i == len(s):
            return out
    return [(g, 1 if sign == '+' else -1) for g, sign in re.findall(r'(.+?)([+-])', s)]


def parse_rate(cell):
    """Net-rate effects of a RATE cell: [(target, delta)] from 'TARGET:delta, ...'."""
    s = re.sub(r'\s+', '', str(cell))
    if s.lower() in ('', '0', 'none', 'nan'):
        return []
    out = []
    for tok in filter(None, s.split(',')):
        if ':' not in tok:
            raise ValueError(f"RATE entry '{tok}' must be TARGET:delta (e.g. ferroptosis_sensitive:-0.01)")
        target, delta = tok.rsplit(':', 1)
        out.append((target, float(delta)))
    return out


def rate_target_genes(target, genes, input_dir=None):
    """
    Genes of a RATE target among `genes` (model genes): None for 'all', else the genes of the gene list
    gene_list_<target>.txt of the run inputs (gene_lists sheet), of a '+'-joined list, or the gene itself.
    """
    if target.lower() == 'all':
        return None
    upper = {str(g).upper(): str(g) for g in genes}
    path = os.path.join(input_dir, f'gene_list_{target}.txt') if input_dir else None
    if path and os.path.exists(path):
        from ..config import read_gene_list
        names = read_gene_list(path)
    else:
        names = target.split('+')
    found = [upper[n.upper()] for n in names if n.upper() in upper]
    if not found:
        raise ValueError(f"RATE target '{target}': no gene of the model (gene list gene_list_{target}.txt "
                         f"absent or no overlap)")
    missing = len(names) - len(found)
    if missing:
        print(f"[CardamomOT] RATE target '{target}': {len(found)} genes of the model used, {missing} absent")
    return found


def load_perturbations(file_path, genes=None):
    """
    Conditions [{'KO': [(gene, pct)], 'OV': [(gene, pct)], 'STIM': {k: [(gene, sign)]},
    'RATE': {k: [(target, delta)]}, 'SCHEDULES': [names]}] of the table (SCHEDULES: 'default', scenario names
    of stimulus_simulation_schedule or 'all'; ['default'] if empty).
    """
    if file_path is None or not os.path.exists(file_path):
        raise FileNotFoundError(f"Perturbation table not found: {file_path}")
    with open(file_path) as f:
        lines = [line for line in f.read().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    header = [h.strip().upper() for h in lines[0].split('\t')]
    idx = {k: header.index(k) for k in ('KO', 'OV', 'SCHEDULES') if k in header}
    # Perturbation stimuli: STIM (= STIM1), STIM1, STIM2...
    stim_cols = {(1 if h == 'STIM' else int(h[4:])): j for j, h in enumerate(header)
                 if h == 'STIM' or re.fullmatch(r'STIM\d+', h)}
    # Net-rate effects: RATE (= RATE1), RATE1, RATE2... (schedule of the stimulus of same index)
    rate_cols = {(1 if h == 'RATE' else int(h[4:])): j for j, h in enumerate(header)
                 if h == 'RATE' or re.fullmatch(r'RATE\d+', h)}
    if not set(idx) - {'SCHEDULES'} and not stim_cols and not rate_cols:
        raise ValueError(f"{os.path.basename(file_path)} must contain a 'KO', 'OV', 'STIM' or 'RATE' column")
    combos = []
    for line in lines[1:]:
        parts = line.split('\t')
        cell = lambda k: parts[idx[k]].strip() if k in idx and idx[k] < len(parts) else ''
        gene_list = lambda c: ([] if c.lower() in ('', '0', 'none', 'nan')
                               else [parse_gene_with_pct(g) for g in c.split(',') if g.strip()])
        stims = {k: parse_stim(parts[j].strip() if j < len(parts) else '', genes) for k, j in sorted(stim_cols.items())}
        rates = {k: parse_rate(parts[j].strip() if j < len(parts) else '') for k, j in sorted(rate_cols.items())}
        sched = [x.strip() for x in cell('SCHEDULES').split(',') if x.strip() and x.strip() != '0']
        combo = dict(KO=gene_list(cell('KO')), OV=gene_list(cell('OV')), STIM={k: v for k, v in stims.items() if v},
                     RATE={k: v for k, v in rates.items() if v}, SCHEDULES=sched or ['default'])
        if combo['KO'] or combo['OV'] or combo['STIM'] or combo['RATE']:
            combos.append(combo)
    return combos


def gene_label(gene, pct):
    return f"{gene}pct{int(pct)}" if pct is not None else gene


def combo_label(combo):
    """File label of a condition: KO_<genes>_OV_<genes>[_STIM_<gene>up|dn-...]."""
    ko = '-'.join(gene_label(g, p) for g, p in combo['KO']) if combo['KO'] else 'none'
    ov = '-'.join(gene_label(g, p) for g, p in combo['OV']) if combo['OV'] else 'none'
    label = f"KO_{ko}_OV_{ov}"
    for k, targets in sorted(combo.get('STIM', {}).items()):
        label += f'_STIM{k}_' + '-'.join(f"{g}{'up' if s > 0 else 'dn'}" for g, s in targets)
    for k, effects in sorted(combo.get('RATE', {}).items()):
        label += f'_RATE{k}_' + '-'.join(f"{t.replace('+', '.')}{d:+g}" for t, d in effects)
    return label


def condition_runs(combos, scenarios):
    """
    Simulations of the conditions: [(combo, scenario, label)], one per schedule asked by the condition
    (SCHEDULES; 'default' = default schedule, label combo_label; a scenario adds _SCEN_<name>; 'all' = default
    + every scenario of `scenarios`). Unknown scenarios are ignored with a warning.
    """
    runs, unknown = [], set()
    for combo in combos:
        asked = combo.get('SCHEDULES') or ['default']
        names = (['default'] + sorted(scenarios)) if 'all' in asked else asked
        for name in dict.fromkeys(names):
            if name != 'default' and name not in scenarios:
                unknown.add(name)
                continue
            runs.append((combo, name, combo_label(combo) + ('' if name == 'default' else f'_SCEN_{name}')))
    if unknown:
        print(f"[CardamomOT] Warning: schedules {sorted(unknown)} of perturbation_simulation are not scenarios of "
              f"stimulus_simulation_schedule: ignored")
    return runs


def combo_description(combo):
    """Readable description of a condition."""
    fmt = lambda g, pct: f"{g} ({pct:g}%)" if pct is not None else g
    parts = []
    if combo['KO']:
        parts.append('KO ' + ' + '.join(fmt(g, p) for g, p in combo['KO']))
    if combo['OV']:
        parts.append('OV ' + ' + '.join(fmt(g, p) for g, p in combo['OV']))
    for k, targets in sorted(combo.get('STIM', {}).items()):
        parts.append(f'Stimulus {k}: ' + ' '.join(f"{g}{'+' if s > 0 else '-'}" for g, s in targets))
    for k, effects in sorted(combo.get('RATE', {}).items()):
        parts.append(f'Rate {k}: ' + ' '.join(f"{t} {d:+g}" for t, d in effects))
    return ' · '.join(parts)


def combo_genes(combo):
    return [g for g, _ in combo['KO'] + combo['OV'] + [x for t in combo.get('STIM', {}).values() for x in t]]


def perturbation_schedule(columns, times, k, times_ref=None):
    """
    Value of perturbation stimulus k (1-based) at each simulated time: column k of `columns` (the
    perturbation part of a simulation schedule, config.simulation_schedule), the row of the last reference
    time <= t (times_ref; None = one row per sorted simulated time), default 0 at the first time and 1 after.
    Missing rows hold the last value; extra rows are ignored with a warning. Returns a function t -> value.
    """
    times = np.sort(np.asarray(times, dtype=float))
    if columns is None or columns.shape[1] < k:
        return lambda t: 0.0 if t <= times[0] else 1.0
    vals = np.asarray(columns, dtype=float)[:, k - 1]
    ref = times if times_ref is None else np.sort(np.asarray(times_ref, dtype=float))
    if len(vals) > len(ref):
        print(f"[CardamomOT] Warning: simulation schedule has {len(vals)} rows for {len(ref)} times: "
              f"rows beyond ignored")
        vals = vals[:len(ref)]
    ref = ref[:len(vals)]
    return lambda t: float(vals[max(0, int(np.searchsorted(ref, float(t) + 1e-9, side='right')) - 1)])
