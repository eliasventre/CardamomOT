"""
Pseudotime experiments
=====================================================================================

Datasets and tools used to study the inference of a network along a pseudotime, i.e. when
the sampling time of each cell is not known.

Contents
--------
``simulate_pseudotime_data``
    Simulated time-stamped datasets (CN5, BN8, FN8) with continuous per-cell times, written to
    ``CardamomOT/pseudotime/data/`` (see ``README.md``).
``binning``
    Pseudotime -> bins, the time labels the CardamomOT cycle expects (``bin_from_tau``).
``base_pseudotime.PseudotimeNetworkModel``
    ``NetworkModel`` subclass whose time labels are the pseudotime bins, with the NB mixture fitted
    on the bins (``fit_mixture_on_bins``).  The three ordering options come next (see ``PLAN.md``).

``PseudotimeNetworkModel`` is imported lazily: the simulation script only needs ``harissa``, while
the model pulls the whole inference stack.
"""

from .binning import BinScheme, bin_from_tau, tau_from_time

__all__ = ["simulate_pseudotime_data", "binning", "base_pseudotime", "options", "costs",
           "BinScheme", "bin_from_tau", "tau_from_time", "PseudotimeNetworkModel"]

_LAZY = {"PseudotimeNetworkModel": ("base_pseudotime", "PseudotimeNetworkModel")}


def __getattr__(name):
    """Lazy import of the model (PEP 562): ``CardamomOT.pseudotime.PseudotimeNetworkModel``."""
    if name in _LAZY:
        from importlib import import_module
        module, attr = _LAZY[name]
        return getattr(import_module(f"{__name__}.{module}"), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

