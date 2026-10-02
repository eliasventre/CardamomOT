"""
In-silico perturbations to simulate: Data/KO_OV_Stim_simulate.txt (old name KO_OV_simulate.txt).

Tab-separated table with a header and the columns KO, OV and optionally STIM1, STIM2... (STIM =
STIM1); one condition per row, '0' or empty for nothing; '#' lines are comments:
- KO / OV: comma-separated genes, 'GENE-X' for a partial perturbation of X% (0 < X < 100);
- STIMk: perturbation stimulus k and its targets, each gene followed by '+' (activated) or '-'
  (inhibited), e.g. 'CHGA+STMN2-' or 'CHGA+,STMN2-'. Its effect on a target is that of an
  interaction of ±(100 + sum of the |interactions| received by the gene), which dominates them
  like a KO/OV, scaled by the value of the stimulus, which follows its own schedule (column k of
  the perturbation stimuli in Data/stimulus_schedule_simulate.txt, after the inference stimuli;
  default 0 at the first simulated time and 1 after). Several stimuli add their effects.
"""
import os
import re

import numpy as np

FILE_NAMES = ('KO_OV_Stim_simulate.txt', 'KO_OV_simulate.txt')


def find_perturbation_file(data_dir):
    """Path of the perturbation table (new name first; old name accepted with a warning), or None."""
    for name in FILE_NAMES:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            if name != FILE_NAMES[0]:
                print(f"[CardamomOT] Warning: reading {name}; rename it {FILE_NAMES[0]}")
            return path
    return None


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


def load_perturbations(file_path, genes=None):
    """Conditions [{'KO': [(gene, pct)], 'OV': [(gene, pct)], 'STIM': {k: [(gene, sign)]}}] of the table."""
    if file_path is None or not os.path.exists(file_path):
        raise FileNotFoundError(f"Perturbation table not found: {file_path}")
    with open(file_path) as f:
        lines = [line for line in f.read().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    header = [h.strip().upper() for h in lines[0].split('\t')]
    idx = {k: header.index(k) for k in ('KO', 'OV') if k in header}
    # Perturbation stimuli: STIM (= STIM1), STIM1, STIM2...
    stim_cols = {(1 if h == 'STIM' else int(h[4:])): j for j, h in enumerate(header)
                 if h == 'STIM' or re.fullmatch(r'STIM\d+', h)}
    if not idx and not stim_cols:
        raise ValueError(f"{os.path.basename(file_path)} must contain a 'KO', 'OV' or 'STIM' column")
    combos = []
    for line in lines[1:]:
        parts = line.split('\t')
        cell = lambda k: parts[idx[k]].strip() if k in idx and idx[k] < len(parts) else ''
        gene_list = lambda c: ([] if c.lower() in ('', '0', 'none', 'nan')
                               else [parse_gene_with_pct(g) for g in c.split(',') if g.strip()])
        stims = {k: parse_stim(parts[j].strip() if j < len(parts) else '', genes) for k, j in sorted(stim_cols.items())}
        combo = dict(KO=gene_list(cell('KO')), OV=gene_list(cell('OV')), STIM={k: v for k, v in stims.items() if v})
        if combo['KO'] or combo['OV'] or combo['STIM']:
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
    return label


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
    return ' · '.join(parts)


def combo_genes(combo):
    return [g for g, _ in combo['KO'] + combo['OV'] + [x for t in combo.get('STIM', {}).values() for x in t]]


def perturbation_schedule(columns, times, k):
    """
    Value of perturbation stimulus k (1-based) at each simulated time: column k of `columns` (the
    perturbation part of stimulus_schedule_simulate.txt, config.simulation_schedule; one row per
    sorted simulated time), default 0 at the first time and 1 after. Returns a function t -> value.
    """
    times = np.sort(np.asarray(times, dtype=float))
    if columns is None or columns.shape[1] < k:
        return lambda t: 0.0 if t <= times[0] else 1.0
    vals = np.asarray(columns, dtype=float)[:, k - 1]
    if len(vals) != len(times):
        raise ValueError(f"stimulus_schedule_simulate.txt has {len(vals)} rows but {len(times)} simulated times")
    return lambda t: float(vals[max(0, int(np.searchsorted(times, t, side='right')) - 1)])
