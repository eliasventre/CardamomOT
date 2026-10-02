"""
Methods estimating the per-cell depth factor s_i (estimate_cell_depth.py).

A method is a module exposing

    compute(X, lib, groups, **params) -> s

- X      : (N, G) raw counts of the whole transcriptome (sparse);
- lib    : (N,) per-cell library size (top genes excluded, see depth.library);
- groups : (N,) labels of the homogeneous groups (sample, time, cell type);
- s      : (N,) positive depth factors; counts are modelled as NB(k, c / s_i), s_i = 1 for a typical cell.

Built in: group_median (default; library relative to the median of its group, so that
differences between groups stay biological) and poissonian (Fang & Pachter 2025: Poisson MLE on
the genes whose overdispersion equals the extrinsic noise, computed within groups). A custom method is <project>/depth_methods/<name>.py defining compute, selected
with model.depth_method = '<name>' (or a path to any .py file); its optional DEFAULTS dict gives
default parameters, overridden by model.depth_method_params.
"""
import importlib
import importlib.util
import os

BUILTIN = {'group_median': 'CardamomOT.inference.depth_methods.group_median',
           'poissonian': 'CardamomOT.inference.depth_methods.poissonian'}


def load_depth_method(name, project_path=None):
    """Module of the depth method: built-in name, <project>/depth_methods/<name>.py, or a .py path."""
    if name in BUILTIN:
        return importlib.import_module(BUILTIN[name])
    candidates = [name] if name.endswith('.py') else []
    if project_path is not None:
        candidates.append(os.path.join(project_path, 'depth_methods', f'{name}.py'))
    for path in candidates:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(f'cardamomot_depth_{os.path.basename(path)[:-3]}', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, 'compute'):
                raise AttributeError(f"{path} must define compute(X, lib, groups, **params)")
            return module
    raise ValueError(f"Unknown depth method '{name}' (built in: {sorted(BUILTIN)}; custom: "
                     f"<project>/depth_methods/{name}.py)")


def compute_depth(name, X, lib, groups, params=None, project_path=None):
    """Depth factors of the method, with its DEFAULTS overridden by params."""
    module = load_depth_method(name, project_path)
    kwargs = dict(getattr(module, 'DEFAULTS', {}))
    kwargs.update(params or {})
    return module.compute(X, lib, groups, **kwargs)
