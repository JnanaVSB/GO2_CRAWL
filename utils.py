"""
Utility functions for Go2 Crawl.
"""

import random
import numpy as np


def set_global_seed(seed=None):
    """
    Seed all random number generators for reproducibility.

    If seed is None, auto-generates a random seed so every run
    is reproducible after the fact — just use the printed seed.

    Seeds: Python's random, NumPy. If PyTorch is installed,
    also seeds torch and torch.cuda.

    Parameters
    ----------
    seed : int or None
        Random seed value. If None, one is auto-generated.

    Returns
    -------
    seed : int
        The seed that was used (useful when auto-generated).
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"Global seed: {seed}")
    return seed