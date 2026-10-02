import torch

def subsample_keep_probs(counts: torch.Tensor,
                         t: float = 1e-5) -> torch.Tensor:
    frequencies = counts / counts.sum()
    return torch.clamp(torch.sqrt(t / frequencies), max=1.0)