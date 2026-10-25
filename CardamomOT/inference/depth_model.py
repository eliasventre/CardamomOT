"""
How the depth factor s_i of a cell enters the NB counts NB(k, c) (shape k, rate c: mean k / c at the reference
depth s = 1), parameter depth_model of NetworkModel:

- 'burst_frequency': the depth multiplies the burst frequency, NB(k s_i, c): mean s k / c, Fano factor
  (1 + c) / c independent of the depth;
- 'mean' (default, as scVI): the depth multiplies the burst size, i.e. only the mean, NB(k, c / s_i): mean s k / c,
  dispersion (shape) k independent of the depth.

Both coincide at s = 1. The setting is process-wide: it is set by the entry points that know the model
(NetworkModel methods, the mixture fit of each gene in the parallel workers, the scripts).
"""
import numpy as np

MODELS = ('burst_frequency', 'mean')
_MODEL = ['mean']  # default of NetworkModel.depth_model


def set_depth_model(model):
    """Set the depth model of the current process (None: unchanged)."""
    if model is None:
        return
    model = str(model).strip().lower()
    if model not in MODELS:
        raise ValueError(f"depth_model '{model}' (use one of {MODELS})")
    _MODEL[0] = model


def get_depth_model():
    return _MODEL[0]


def burst_frequency(model=None):
    """True if the depth multiplies the burst frequency (model, or the current setting if None)."""
    return (get_depth_model() if model is None else str(model).strip().lower()) == 'burst_frequency'


def nb_shape_rate(k, c, s=None, model=None):
    """(shape, rate) of the NB of counts at depth s (None = 1), broadcast like k * s and c / s."""
    if s is None:
        return k, c
    if burst_frequency(model):
        return np.asarray(k) * s, c
    return k, np.asarray(c) / s


def nb_numpy_params(k, c, s=None, model=None):
    """(n, p) of numpy / scipy nbinom for NB(k, c) at depth s (None = 1): n = shape, p = rate / (1 + rate)."""
    n, rate = nb_shape_rate(k, c, s, model)
    rate = np.asarray(rate, dtype=float)
    return n, rate / (1.0 + rate)
