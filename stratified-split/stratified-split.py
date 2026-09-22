import numpy as np

def stratified_split(X: list, y: list, test_size: float = 0.2, seed: int = 42) -> dict:
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        test_count = int(round(indices.size * test_size))
        if indices.size > 1:
            test_count = min(test_count, indices.size - 1)
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])
    train_indices = np.sort(np.asarray(train_indices, dtype=int))
    test_indices = np.sort(np.asarray(test_indices, dtype=int))
    return {
        "X_train": X[train_indices], "X_test": X[test_indices],
        "y_train": y[train_indices], "y_test": y[test_indices],
    }