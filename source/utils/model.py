import torch


def r2_score(output, target):
    """Compute R-squared score."""
    target_tensor = torch.tensor(target)
    output_tensor = torch.tensor(output)
    target_mean = torch.mean(target_tensor)
    ss_tot = torch.sum((target_tensor - target_mean) ** 2)
    ss_res = torch.sum((target_tensor - output_tensor) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2
