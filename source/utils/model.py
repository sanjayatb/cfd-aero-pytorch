import numpy as np
from sklearn.metrics import r2_score

def compute_r2_score(output, target):
    """Compute R-squared score."""
    return r2_score(target, output)
    # target_mean = torch.mean(target)
    # ss_tot = torch.sum((target - target_mean) ** 2)
    # ss_res = torch.sum((target - output) ** 2)
    # r2 = 1 - ss_res / ss_tot
    # return r2

def compute_rel_l2_score(y_true, y_pred):
    """
    Computes the squared relative L2 error:
    Rel L2 = ||y_pred - y_true||^2 / ||y_true||^2
    """
    numerator = np.sum((y_pred - y_true) ** 2)
    denominator = np.sum(y_true ** 2)
    return np.sqrt(numerator / denominator) if denominator != 0 else float('inf')

