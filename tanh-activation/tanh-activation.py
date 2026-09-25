import numpy as np

def tanh(x: list) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.tanh(x)