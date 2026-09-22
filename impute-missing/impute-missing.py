import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    result = np.asarray(X, dtype=float).copy()
    if result.ndim == 1:
        missing = np.isnan(result)
        observed = result[~missing]
        fill = 0.0 if observed.size == 0 else float(np.mean(observed) if strategy == "mean" else np.median(observed))
        result[missing] = fill
        return result
    for column_index in range(result.shape[1]):
        column = result[:, column_index]
        missing = np.isnan(column)
        observed = column[~missing]
        fill = 0.0 if observed.size == 0 else float(np.mean(observed) if strategy == "mean" else np.median(observed))
        result[missing, column_index] = fill
    return result