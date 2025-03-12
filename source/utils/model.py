import torch
from sklearn.metrics import r2_score

def compute_r2_score(output, target):
    """Compute R-squared score."""
    return r2_score(target, output)
    # target_mean = torch.mean(target)
    # ss_tot = torch.sum((target - target_mean) ** 2)
    # ss_res = torch.sum((target - output) ** 2)
    # r2 = 1 - ss_res / ss_tot
    # return r2
