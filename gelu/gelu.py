import math
import numpy as np

def gelu(x: list) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    erf = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))