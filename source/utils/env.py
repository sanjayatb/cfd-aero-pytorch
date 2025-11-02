import numpy as np
import torch
import random


# NumPy 2 removed ndarray.ptp; restore for third-party libs expecting it.
if not hasattr(np.ndarray, "ptp"):
    def _ndarray_ptp(self, *args, **kwargs):
        return np.ptp(self, *args, **kwargs)

    setattr(np.ndarray, "ptp", _ndarray_ptp)


def setup_seed(seed: int):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Makes convs deterministic
    torch.backends.cudnn.benchmark = False  # Slows down but consistent
