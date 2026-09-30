"""
Reproducible seeding for the main process and joblib workers.

Global seeding does not reach loky worker processes, so each parallel task
is seeded with a deterministic child seed derived from the model seed.
"""
import random

import numpy as np


def task_seed(seed, *keys):
    """Deterministic child seed from a base seed and integer task keys (None if seed is None)."""
    if seed is None:
        return None
    return int(np.random.SeedSequence([int(seed), *[int(k) for k in keys]]).generate_state(1)[0])


def seed_everything(seed):
    """Seed python, numpy and torch global generators (no-op if seed is None)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def seeded_call(seed, fn, *args, **kwargs):
    """Seed the current process, then call fn: wrap joblib tasks with it."""
    seed_everything(seed)
    return fn(*args, **kwargs)
