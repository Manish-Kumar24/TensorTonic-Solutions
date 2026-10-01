import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
          W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    joined = np.concatenate([h_prev, x_t], axis=-1)
    return sigmoid(joined @ W_r.T + b_r).astype(np.float64)