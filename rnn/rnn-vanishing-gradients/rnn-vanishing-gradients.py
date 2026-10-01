import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> np.ndarray:
    rho=np.linalg.norm(W_hh,ord=2)
    return np.power(rho,np.arange(T,dtype=np.float64)).astype(np.float64)