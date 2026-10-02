import torch

def noise_distribution(counts: torch.Tensor,
                       alpha: float = 0.75) -> torch.Tensor:
    weights = counts.pow(alpha)
    return weights / weights.sum()