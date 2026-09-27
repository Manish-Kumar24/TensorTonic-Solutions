import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual_sum = np.sum((y_true - y_pred) ** 2)
    total_sum = np.sum((y_true - np.mean(y_true)) ** 2)
    if total_sum == 0:
        return 1.0 if residual_sum == 0 else 0.0
    return float(1.0 - residual_sum / total_sum)