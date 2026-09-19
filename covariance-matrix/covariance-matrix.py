import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    X = np.asarray(X, dtype=float)
    centered = X - np.mean(X, axis=0)
    return centered.T @ centered / (X.shape[0] - 1)