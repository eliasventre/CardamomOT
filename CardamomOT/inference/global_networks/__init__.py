"""
Methods building the coarse global network on which the gene selection (Steiner tree) is run.

A method is a module exposing

    build_network(adata, stim, seed=0, **params) -> C

- adata : raw counts (cells x candidate genes), obs['time'] and, if any, obs['dataset_id'];
- stim  : (n_times, n_stimuli) stimulus schedule per sorted timepoint;
- C     : ((n_stimuli + G) x (n_stimuli + G)) signed network, stimuli first, C[i, j] = effect of i on j
          (only |C| is used by the selection).

Built-in methods live in this folder (otvelo_corr, otvelo_granger). A custom method is a file
<project>/network_methods/<name>.py defining build_network, selected with
model.network_method = '<name>' (or a path to any .py file); its optional DEFAULTS dict gives
default parameters, overridden by model.network_method_params.
"""
import importlib
import importlib.util
import os

BUILTIN = {'otvelo_corr': 'CardamomOT.inference.global_networks.otvelo_corr',
           'otvelo_granger': 'CardamomOT.inference.global_networks.otvelo_granger'}


def load_network_method(name, project_path=None):
    """Module of the network method: built-in name, <project>/network_methods/<name>.py, or a .py path."""
    if name in BUILTIN:
        return importlib.import_module(BUILTIN[name])
    candidates = [name] if name.endswith('.py') else []
    if project_path is not None:
        candidates.append(os.path.join(project_path, 'network_methods', f'{name}.py'))
    for path in candidates:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(f'cardamomot_network_{os.path.basename(path)[:-3]}', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, 'build_network'):
                raise AttributeError(f"{path} must define build_network(adata, stim, seed=0, **params)")
            return module
    raise ValueError(f"Unknown network method '{name}': built-in {sorted(BUILTIN)}, "
                     f"or a file network_methods/{name}.py in the project")


def build_global_network(name, adata, stim, seed=0, params=None, project_path=None):
    """Run a network method with its DEFAULTS updated by params."""
    module = load_network_method(name, project_path)
    kwargs = dict(getattr(module, 'DEFAULTS', {}))
    kwargs.update(params or {})
    return module.build_network(adata, stim, seed=seed, **kwargs)
