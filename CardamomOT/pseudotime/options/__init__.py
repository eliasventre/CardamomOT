"""
Ordering options of the pseudotime cycle (``PLAN.md`` §5).

Each option turns the current state of a ``PseudotimeNetworkModel`` — burst modes, protein states,
network :math:`\\Theta` — into a per-cell pseudotime ``tau`` (an *absolute* coordinate: a distance
or a step count, 0 on the roots).

* ``palantir``    — Option 3, mechanistic Palantir;
* ``partial_ot``  — Option 1, partial OT from the initial population;
* ``tsp``         — Option 2, path-dependent TSP.
"""

from .features import FEATURE_SPACES, FEATURE_WEIGHTS, feature_matrix
from .palantir_like import PalantirLike, PalantirResult
from .partial_ot import PartialOT, PartialOTResult
from .tsp import PathTSP, TSPResult

OPTIONS = {
    "palantir": PalantirLike,
    "partial_ot": PartialOT,
    "tsp": PathTSP,
}


def get_option(name: str, model):
    """Instantiate the option ``name`` with the ``*_`` parameters of ``model``."""
    if name not in OPTIONS:
        raise ValueError(f"unknown pseudotime method {name!r}; available: {sorted(OPTIONS)}")
    return OPTIONS[name].from_model(model)


__all__ = ["OPTIONS", "get_option", "PalantirLike", "PalantirResult", "PartialOT",
           "PartialOTResult", "PathTSP", "TSPResult", "feature_matrix", "FEATURE_SPACES",
           "FEATURE_WEIGHTS"]
