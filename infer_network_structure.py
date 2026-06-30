"""
infer_network_structure.py
---------------------------
Gene regulatory network structure inference pipeline.

Estimates the regulatory interactions between genes by solving an optimal
transport problem that matches the temporal dynamics of the inferred
mixture model to observed transcriptomic data.

Usage:
    python infer_network_structure.py -i <project_path> -s <split>

Required input files:
    - Data/data_<split>.h5ad: count matrix with degradation rates
    - cardamom/modes.npy, proba.npy, mixture_parameters.npy: from infer_mixture.py

Output files:
    - cardamom/inter.npy: gene interaction matrix (G×G)
    - cardamom/basal.npy: basal expression parameters (G×1)
    - cardamom/alpha.npy: cellular scaling factors
"""
import sys; sys.path += ['../']
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
import getopt
import anndata as ad
import pandas as pd
import os
import pickle

def main(argv):
    """
    Infer the gene regulatory network structure from temporal scRNA-seq data.

    Args:
        argv: Command-line arguments (--input, --split).
    """
    inputfile = ''
    split = ''
    stimulus = -1.0
    prior = -1.0
    force_basins = -1
    temporal_basins = -1
    try:
        opts, args = getopt.getopt(argv, "hi:s:t:p:f:b:",
                                   ["input=", "split=", "stimulus=", "prior=",
                                    "force-basins=", "temporal-basins="])
    except getopt.GetoptError:
        print("Error: Invalid arguments. Use: infer_network_structure.py -i <project> -s <split> "
              "[--stimulus <float>] [--prior <float>] [--force-basins <int>] [--temporal-basins <int>]")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        if opt in ("-s", "--split"):
            split = '{}'.format(arg)
        if opt in ("-t", "--stimulus"):
            stimulus = float(arg)
        if opt in ("-p", "--prior"):
            prior = float(arg)
        if opt in ("-f", "--force-basins"):
            force_basins = float(arg)
        if opt in ("-b", "--temporal-basins"):
            temporal_basins = int(arg)

    p = '{}/'.format(inputfile)

    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    if os.path.exists(data_path):
        adata = ad.read_h5ad(data_path)
        print(f"[infer_network_structure] Loaded data from {data_path}")
    else:
        error_msg = (
            f"Error: Data file not found at {data_path}.\n"
            f"Ensure you have created a 'data_{split}.h5ad' file in the Data/ directory."
        )
        print(error_msg)
        raise FileNotFoundError(error_msg)

    # ─── CHECK TEMPORAL INFORMATION ──────────────────────────────────────
    try:
        times = adata.obs['time'].values
        if len(np.unique(times)) <= 1:
            raise ValueError(
                "Data must contain temporal information with at least 2 distinct timepoints."
            )
    except KeyError as e:
        error_msg = f"Error: 'time' column not found in adata.obs. {e}"
        print(error_msg)
        raise SystemExit(error_msg)

    print(f"[infer_network_structure] Starting network inference ({adata.shape[1]} genes)...")

    # ─── LOAD STIMULUS SCHEDULE (optional) ──────────────────────────────
    stim_sched = None
    sched_path = os.path.join(p, 'Data', 'stimulus_schedule.txt')
    if os.path.exists(sched_path):
        stim_sched = np.loadtxt(sched_path)
        print(f"[infer_network_structure] Loaded stimulus schedule from {sched_path}")

    # ─── DETECT n_stimuli FROM SCHEDULE ─────────────────────────────────
    _stim_arr = np.asarray(stim_sched) if stim_sched is not None else None
    n_stimuli = int(_stim_arr.shape[1]) if (_stim_arr is not None and _stim_arr.ndim == 2) else 1
    print(f"[infer_network_structure] n_stimuli detected: {n_stimuli}")

    # ─── LOAD MIXTURE PARAMETERS ────────────────────────────────────────
    try:
        model = NetworkModel_beta(adata.shape[1], n_stimuli=n_stimuli)
        if stimulus >= 0:
            model.stimulus = stimulus
        if prior >= 0:
            model.prior_network_pen = prior
        if force_basins >= 0:
            model.force_basins = force_basins
        if temporal_basins >= 0:
            model.temporal_basins = temporal_basins
        print(f"[infer_network_structure] stimulus={model.stimulus}, prior_network_pen={model.prior_network_pen}, "
              f"force_basins={model.force_basins}, temporal_basins={model.temporal_basins}")
        model.modes = np.load(os.path.join(p, 'cardamomOT', 'modes.npy'))
        with open(os.path.join(p, 'cardamomOT', 'pi_init.pkl'), "rb") as f:
            model.pi_init = pickle.load(f)
        model.proba = np.load(os.path.join(p, 'cardamomOT', 'proba.npy'))
        model.proba_init = np.load(os.path.join(p, 'cardamomOT', 'proba_init.npy'))
        model.a = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        model.n_networks = np.load(os.path.join(p, 'cardamomOT', 'n_networks.npy'))

        print(f"[infer_network_structure] Loaded mixture parameters ({model.n_networks} networks)")
    except FileNotFoundError as e:
        error_msg = (
            f"Error: Required mixture parameter files not found. "
            f"Run infer_mixture.py first. {e}"
        )
        print(error_msg)
        raise SystemExit(error_msg)

    G_tot = adata.shape[1] + model.n_stimuli
    ns = model.n_stimuli
    model.ref_network = np.ones((G_tot, G_tot, model.n_networks))
    stim_labels = ['Stimulus'] if ns == 1 else [f'Stimulus_{i}' for i in range(ns)]
    genes_list = stim_labels + [g.upper() for g in adata.var_names]
    genes_only = [g.upper() for g in adata.var_names]   # no stimulus prefix
    ref_path = os.path.join(p, 'cardamomOT', 'ref_network.csv')
    if os.path.exists(ref_path):
        ref_df = pd.read_csv(ref_path, index_col=0)
        ref_df.columns = ref_df.columns.astype(str)
        ref_df.index = ref_df.index.astype(str)
        # CSV contains only genes (no stimulus rows/cols)
        common_genes = [g for g in genes_only if g in ref_df.index]
        if common_genes:
            sub_df = ref_df.loc[common_genes, [c for c in common_genes if c in ref_df.columns]]
            print(f"[infer_network_structure] shape of ref_network gene block = {sub_df.shape}")
            ref_mat = np.abs(sub_df.to_numpy())
            row_idxs = [ns + genes_only.index(g) for g in sub_df.index]
            col_idxs = [ns + genes_only.index(g) for g in sub_df.columns]
            for n in range(model.n_networks):
                for ii, ri in enumerate(row_idxs):
                    model.ref_network[ri, col_idxs, n] = ref_mat[ii, :]
            # stimulus rows/cols are left at 1.0 and will be overridden by model.stimulus
            # inside fit_network (self.ref_network[:ns, :] = self.stimulus)

    # ─── LOAD USER-PROVIDED reference_network.csv FROM Data/ ─────────────
    # Users can place Data/reference_network.csv to override or supplement the
    # structural prior without running prepare_reference_network.py.
    data_ref_path = os.path.join(p, 'Data', 'reference_network.csv')
    if os.path.exists(data_ref_path):
        _ref_df = pd.read_csv(data_ref_path, index_col=0)
        _ref_df.columns = _ref_df.columns.astype(str).str.upper()
        _ref_df.index = _ref_df.index.astype(str).str.upper()
        _common_ref = [g for g in genes_only if g in _ref_df.index]
        if _common_ref:
            _sub = _ref_df.loc[_common_ref, [c for c in _common_ref if c in _ref_df.columns]]
            _ref_mat = np.abs(_sub.to_numpy())
            _row_idxs = [ns + genes_only.index(g) for g in _sub.index]
            _col_idxs = [ns + genes_only.index(g) for g in _sub.columns]
            for _n in range(model.n_networks):
                for _ii, _ri in enumerate(_row_idxs):
                    model.ref_network[_ri, _col_idxs, _n] = _ref_mat[_ii, :]
            print(f"[infer_network_structure] Loaded structural prior from {data_ref_path} "
                  f"({len(_common_ref)}/{len(genes_only)} genes matched).")
        else:
            # No gene names recognised — validate size then assume positional order
            _n_genes = len(genes_only)
            _r, _c = _ref_df.shape
            if (_r, _c) != (_n_genes, _n_genes) and (_r, _c) != (G_tot, G_tot):
                print(
                    f"[infer_network_structure] ERROR: Data/reference_network.csv has no "
                    f"recognisable gene names and its shape {(_r, _c)} is inconsistent with "
                    f"{_n_genes} genes (expected {_n_genes}×{_n_genes} or {G_tot}×{G_tot} "
                    f"including stimulus). Add gene names as row/column headers or provide a "
                    f"matrix of the correct size."
                )
                raise SystemExit(1)
            import warnings as _warnings
            _warnings.warn(
                f"Data/reference_network.csv has no gene names — assuming positional order "
                f"matches adata.var_names.",
                UserWarning, stacklevel=2
            )
            _vals = _ref_df.to_numpy().astype(float)
            if _vals.shape[0] == G_tot:
                # Full matrix including stimulus provided
                _block = np.abs(_vals)
            else:
                # Gene-only matrix: embed into gene block
                _block = np.zeros((G_tot, G_tot))
                _block[ns:ns + _n_genes, ns:ns + _n_genes] = np.abs(_vals)
            for _n in range(model.n_networks):
                model.ref_network[:, :, _n] = _block

    # Infer network
    model.d = np.ones((2, G_tot))
    model.d[1, model.n_stimuli:], model.d[0, model.n_stimuli:] = adata.var['d1'].values, adata.var['d0'].values

    # ─── LOAD OPTIONAL INIT / REF ARRAYS ────────────────────────────────
    def _load_gene_vec(fname):
        """Load a (G_tot,) or (G_tot, n_networks) basal vector from npy or csv."""
        npy = os.path.join(p, 'Data', fname + '.npy')
        csv = os.path.join(p, 'Data', fname + '.csv')
        if os.path.exists(npy):
            arr = np.load(npy)
            print(f"[infer_network_structure] Loaded {fname} from {npy} shape={arr.shape}")
            return arr
        if os.path.exists(csv):
            df = pd.read_csv(csv, index_col=0)
            df.index = df.index.astype(str).str.upper()
            common = [g for g in df.index if g in genes_list]
            arr = np.zeros((G_tot, df.shape[1]))
            for row_g in common:
                arr[genes_list.index(row_g), :] = df.loc[row_g].values
            arr = arr[:, 0] if arr.shape[1] == 1 else arr
            print(f"[infer_network_structure] Loaded {fname} from {csv} shape={arr.shape}")
            return arr
        return None

    def _load_gene_mat(fname):
        """Load a (G_tot, G_tot) or (G_tot, G_tot, n_networks) inter matrix from npy or csv.

        If the CSV index contains gene names (case-insensitive), values are remapped to adata
        gene order. Otherwise the file is treated as positional: a warning is emitted and the
        pipeline stops with an error if the matrix dimensions are inconsistent with the data.
        """
        import warnings as _warnings

        npy = os.path.join(p, 'Data', fname + '.npy')
        csv = os.path.join(p, 'Data', fname + '.csv')
        if os.path.exists(npy):
            arr = np.load(npy)
            print(f"[infer_network_structure] Loaded {fname} from {npy} shape={arr.shape}")
            return arr
        if os.path.exists(csv):
            df = pd.read_csv(csv, index_col=0)
            # Normalise to uppercase so lowercase gene names in the CSV still match.
            # Convert to str first in case the index/columns are numeric (float).
            df.columns = df.columns.astype(str).str.upper()
            df.index = df.index.astype(str).str.upper()
            common = [g for g in df.index if g in genes_list]
            if common:
                # Gene-named path: reorder to match adata gene order
                arr = np.zeros((G_tot, G_tot))
                for row_g in common:
                    for col_g in [c for c in df.columns if c in genes_list]:
                        arr[genes_list.index(row_g), genes_list.index(col_g)] = df.loc[row_g, col_g]
                print(f"[infer_network_structure] Loaded {fname} from {csv} (gene-named) shape={arr.shape}")
                return arr

            # ── Positional fallback: no gene names recognised ────────────────
            def _is_float(s):
                try:
                    float(str(s).strip())
                    return True
                except ValueError:
                    return False
            raw = pd.read_csv(csv, header=None, dtype=str)
            data = raw.values
            r0 = [str(v).strip() for v in data[0]]
            # Detect header row (first cell empty OR first row non-numeric)
            has_header = (not r0[0]) or not all(_is_float(v) for v in r0 if v)
            sr = 1 if has_header else 0
            # Detect index column (empty corner cell OR first col non-numeric)
            if has_header and not r0[0]:
                has_idx = True
            else:
                c0 = [str(v).strip() for v in data[sr:, 0]]
                has_idx = not all(_is_float(v) for v in c0 if v)
            sc = 1 if has_idx else 0
            vals = data[sr:, sc:].astype(float)

            # ── Size validation ──────────────────────────────────────────────
            n_genes = G_tot - ns   # expected number of gene rows/cols (no stimulus)
            if vals.shape[0] == G_tot and vals.shape[1] == G_tot:
                # Full matrix (genes + stimulus) provided positionally
                _is_full = True
            elif vals.shape[0] == n_genes and vals.shape[1] == n_genes:
                # Gene-only matrix
                _is_full = False
            else:
                print(
                    f"[infer_network_structure] ERROR: Data/{fname}.csv has no recognisable "
                    f"gene names and its shape {vals.shape} is inconsistent with {n_genes} genes "
                    f"(expected {n_genes}×{n_genes} without stimulus or {G_tot}×{G_tot} with "
                    f"stimulus). Add gene names as row/column headers or provide a matrix of the "
                    f"correct size."
                )
                raise SystemExit(1)

            _warnings.warn(
                f"Data/{fname}.csv has no gene names — assuming positional order matches "
                f"adata.var_names.",
                UserWarning, stacklevel=2
            )
            print(
                f"[infer_network_structure] WARNING: Data/{fname}.csv has no gene names — "
                f"assuming gene order matches adata.var_names."
            )

            arr = np.zeros((G_tot, G_tot))
            if _is_full:
                arr[:, :] = vals[:G_tot, :G_tot]
            else:
                # Gene-only: place at gene rows/cols (offset by ns stimulus indices)
                arr[ns:ns + n_genes, ns:ns + n_genes] = vals[:n_genes, :n_genes]
            print(f"[infer_network_structure] Loaded {fname} from {csv} (positional, assumed adata gene order) shape={arr.shape}")
            return arr
        return None

    basal_init = _load_gene_vec('basal_init')
    inter_init = _load_gene_mat('inter_init')
    basal_ref  = _load_gene_vec('basal_ref') 
    inter_ref  = _load_gene_mat('inter_ref') * 3

    # ─── PER-SAMPLE KOV PRIOR (overrides basal_ref if present) ──────────
    # Data/KO_OV_inference.txt : TSV with columns  sample_id | KO | OV
    # sample_id values must match adata.obs['dataset_id'].
    # KO genes get basal_ref = -100 (forced OFF), OV genes get +100 (forced ON).
    # Also saves basal_ref_mask.npy: bool (n_samples, G_tot) for downstream
    # simulate_network_KOV clean-basal computation.
    kov_path = os.path.join(p, 'Data', 'KO_OV_inference.txt')
    if os.path.exists(kov_path):
        try:
            df_kov = pd.read_csv(kov_path, sep='\t', dtype=str).fillna('')
            df_kov.columns = [c.strip().upper() for c in df_kov.columns]

            if 'DATASET_ID' in df_kov.columns:
                df_kov = df_kov.rename(columns={'DATASET_ID': 'SAMPLE_ID'})
            if 'SAMPLE_ID' not in df_kov.columns:
                raise ValueError("KO_OV_inference.txt must have a 'sample_id' column")

            unique_samples = (np.sort(adata.obs['dataset_id'].unique())
                              if 'dataset_id' in adata.obs else np.array(['0']))
            n_samp = len(unique_samples)
            sample_to_idx = {str(s): i for i, s in enumerate(unique_samples)}
            n_nw = model.n_networks

            basal_ref_kov = np.zeros((n_samp, G_tot, n_nw))
            basal_ref_mask = np.zeros((n_samp, G_tot), dtype=bool)

            def _parse_genes(cell):
                if not cell or cell in ('0', 'nan', 'NAN'):
                    return []
                return [g.strip().upper() for g in cell.split(',') if g.strip()]

            for _, row in df_kov.iterrows():
                sid = str(row.get('SAMPLE_ID', '')).strip()
                if sid not in sample_to_idx:
                    print(f"[infer_network_structure] Warning: sample_id '{sid}' not in dataset_id — skipped")
                    continue
                s_idx = sample_to_idx[sid]
                for gene in _parse_genes(row.get('KO', '')):
                    if gene in genes_only:
                        gi = ns + genes_only.index(gene)
                        basal_ref_kov[s_idx, gi, :] = -100.0
                        basal_ref_mask[s_idx, gi] = True
                for gene in _parse_genes(row.get('OV', '')):
                    if gene in genes_only:
                        gi = ns + genes_only.index(gene)
                        basal_ref_kov[s_idx, gi, :] = 100.0
                        basal_ref_mask[s_idx, gi] = True

            np.save(os.path.join(p, 'cardamomOT', 'basal_ref_mask'), basal_ref_mask)
            basal_ref = basal_ref_kov   # 3D (n_samples, G_tot, n_networks)
            print(f"[infer_network_structure] Loaded per-sample KOV prior from {kov_path} "
                  f"({n_samp} samples, mask nnz={basal_ref_mask.sum()})")
        except Exception as e:
            print(f"[infer_network_structure] Warning: could not load KO_OV_inference.txt: {e}")

    # ─── LOAD TRANSITION RATES (optional) ───────────────────────────────────
    transition_rates = None
    tr_path = os.path.join(p, 'Data', 'transition_rates.csv')
    if os.path.exists(tr_path):
        transition_rates = pd.read_csv(tr_path, index_col=0)
        transition_rates.index = transition_rates.index.astype(str)
        transition_rates.columns = transition_rates.columns.astype(str)
        print(f"[infer_network_structure] Loaded transition rates from {tr_path} "
              f"shape={transition_rates.shape}")

    model.fit_network(adata, intensity_prior=100, verb=1, stimulus_schedule=stim_sched,
                      basal_init=basal_init, inter_init=inter_init,
                      basal_ref=basal_ref, inter_ref=inter_ref,
                      transition_rates=transition_rates)

    # Save inferred network structure parameters
    cardamom_dir = os.path.join(p, 'cardamomOT')
    if inter_ref is not None:
        np.save(os.path.join(cardamom_dir, 'inter_ref'), inter_ref)
    if basal_ref is not None:
        np.save(os.path.join(cardamom_dir, 'basal_ref'), basal_ref)
    np.save(os.path.join(cardamom_dir, 'basal'), model.basal)
    np.save(os.path.join(cardamom_dir, 'inter'), model.inter)
    np.save(os.path.join(cardamom_dir, 'inter_t'), model.inter_t)
    np.save(os.path.join(cardamom_dir, f'inter_stim{model.stimulus}_prior{model.prior_network_pen}'), model.inter)
    np.save(os.path.join(cardamom_dir, 'basal_tmp'), model.basal_tmp)
    np.save(os.path.join(cardamom_dir, 'inter_tmp'), model.inter_tmp)
    np.save(os.path.join(cardamom_dir, 'data_prot'), model.prot)
    np.save(os.path.join(cardamom_dir, 'data_rna'), model.rna)
    np.save(os.path.join(cardamom_dir, 'data_times'), model.times_data)
    np.save(os.path.join(cardamom_dir, 'data_samples'), model.samples_data)
    np.save(os.path.join(cardamom_dir, 'proba_traj'), model.proba_traj)
    np.save(os.path.join(cardamom_dir, 'data_kon_theta'), model.kon_theta)
    np.save(os.path.join(cardamom_dir, 'data_kon_beta'), model.kon_beta)
    if model.kon_beta_harissa is not None:
        np.save(os.path.join(cardamom_dir, 'data_kon_beta_harissa'), model.kon_beta_harissa)
    np.save(os.path.join(cardamom_dir, 'alpha'), model.alpha)
    np.save(os.path.join(cardamom_dir, 'degradations'), model.d)
    if model.R_opt is not None:
        np.save(os.path.join(cardamom_dir, 'data_R_opt'), model.R_opt)

    print(f"[infer_network_structure] Successfully saved network inference results to {cardamom_dir}")


if __name__ == "__main__":
   main(sys.argv[1:])
