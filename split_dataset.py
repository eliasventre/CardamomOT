"""
split_dataset.py
----------------
Split the cells of Data/data.h5ad into train/test sets, before the gene selection (select_genes.py only
sees the train cells: no leakage of the test cells into the selection).

Usage:
    python split_dataset.py -i <project_path>
(split, train_rate, seed: model_parameters sheet; seed 0 if not set, so that the split is reproducible)

Required input files:
    - Data/data.h5ad: input count matrix, with temporal information in obs['time']
Optional:
    - times_inference (times sheet): with split = 'train', cells after its largest time are left out
    - perturbation_inference sheet: samples with remove_from_inference

Output:
    - Data/data.h5ad, with obs['split']:
        'full'  every cell, with split = 'full';
        'train' / 'test' with split = 'train': per (sample, time), train_rate of the cells (at least 100) in
                train, at most as many in test; the other cells are 'unused';
        'test'  every cell of the samples with remove_from_inference, whatever the split.
"""
import sys; sys.path += ['../']
import os
import numpy as np
import anndata as ad
from CardamomOT import NetworkModel, harmonize_obs
from CardamomOT.inputs import input_dir, removed_samples
from CardamomOT.run_options import parse_step_options, configure


def split_labels(times, samples, split, rate, max_time=None, removed=None, seed=None):
    """obs['split'] of the cells (see module docstring); removed: bool mask of the held-out samples."""
    labels = np.full(len(times), 'full' if split == 'full' else 'unused', dtype=object)
    removed = np.zeros(len(times), bool) if removed is None else removed
    if split == 'train':
        rng = np.random.default_rng(seed)
        kept = ~removed & (times <= max_time if max_time is not None else True)
        for t in np.unique(times[kept]):
            for s in np.unique(samples[kept]):
                idx = np.flatnonzero(kept & (times == t) & (samples == s))
                rng.shuffle(idx)
                n_train = max(min(100, len(idx)), int(len(idx) * float(rate)))
                labels[idx[:n_train]] = 'train'
                # Test cells of each (time, sample) capped at its number of train cells
                labels[idx[n_train:2 * n_train]] = 'test'
    labels[removed] = 'test'
    return labels


def main(argv):
    opts = parse_step_options(argv, 'split_dataset', __doc__)
    p = opts.p
    model = configure(NetworkModel(1), opts)
    split, rate = model.split, model.train_rate
    if split not in ('train', 'full'):
        print(f"[split_dataset] Error: split must be 'train' or 'full' (got {split!r})")
        sys.exit(1)
    print(f"[split_dataset] split={split}, train_rate={rate}")

    data_path = os.path.join(p, 'Data', 'data.h5ad')
    if not os.path.exists(data_path):
        print(f"[split_dataset] Error: input data file not found at {data_path}")
        sys.exit(1)
    adata = ad.read_h5ad(data_path)
    obs = ad.AnnData(obs=adata.obs.copy())
    harmonize_obs(obs)  # obs names close to the expected ones, read without renaming those of data.h5ad
    obs = obs.obs
    times = obs['time'].astype(float).values if 'time' in obs else np.zeros(adata.n_obs)
    samples = obs['dataset_id'].astype(str).values if 'dataset_id' in obs else np.zeros(adata.n_obs, str)

    # Samples held out of the inference (remove_from_inference): every cell in test
    removed = np.zeros(adata.n_obs, bool)
    if 'dataset_id' in obs:
        names, _ = removed_samples(p, present=np.unique(samples))
        removed = np.isin(samples, names)
        if removed.all():
            print("[split_dataset] Error: every sample is removed from the inference")
            sys.exit(1)
        if names:
            print(f"[split_dataset] Samples removed from the inference (all their {int(removed.sum())} cells in "
                  f"test): {names}")

    # Inference restricted to the times <= the largest of times_inference (train split only, as before)
    max_time = None
    times_file = os.path.join(input_dir(p), 'times_inference.txt')
    if split == 'train' and os.path.exists(times_file):
        with open(times_file) as f:
            max_time = max(float(line) for line in f if line.strip())
        print(f"[split_dataset] Cells after t = {max_time:g} (times_inference) left out")

    # Reproducible split (seed 0 if none): every later step depends on it
    adata.obs['split'] = split_labels(times, samples, split, rate, max_time, removed,
                                      0 if model.seed is None else model.seed)
    counts = adata.obs['split'].value_counts()
    print("[split_dataset] Cells per split: " + ', '.join(f'{k} {v}' for k, v in counts.items()))
    adata.obs['split'] = adata.obs['split'].astype('category')
    adata.write(data_path)
    print(f"[split_dataset] Saved obs['split'] to {data_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
