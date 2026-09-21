import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    centered = X - np.mean(X, axis=0)
    covariance = centered.T @ centered / (X.shape[0] - 1)
    standard_deviation = np.sqrt(np.diag(covariance))
    denominator = np.outer(standard_deviation, standard_deviation)
    with np.errstate(divide="ignore", invalid="ignore"):
        return covariance / denominator