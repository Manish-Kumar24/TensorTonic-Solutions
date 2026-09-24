import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)
    s_new = beta * s + (1 - beta) * g ** 2
    w_new = w - lr / np.sqrt(s_new + eps) * g
    return np.round(w_new, 6).tolist(), np.round(s_new, 6).tolist()