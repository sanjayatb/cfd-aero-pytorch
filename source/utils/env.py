import numpy as np
import torch
import random


def setup_seed(seed: int):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Makes convs deterministic
    torch.backends.cudnn.benchmark = False  # Slows down but consistent
