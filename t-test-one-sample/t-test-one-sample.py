import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    x = np.asarray(x, dtype=float)
    mean = np.mean(x)
    centered = x - mean
    sample_std = np.sqrt(np.sum(centered ** 2) / (x.size - 1))
    if sample_std == 0.0:
        difference = float(mean - mu0)
        if difference == 0.0:
            return 0.0
        return float(np.inf if difference > 0 else -np.inf)
    return float((mean - mu0) / (sample_std / np.sqrt(x.size)))