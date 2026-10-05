import numpy as np

def bert_pooler(hidden_states: np.ndarray, W_pool: np.ndarray,
                b_pool: np.ndarray) -> np.ndarray:
    classification_states = hidden_states[:, 0, :]
    return np.tanh(classification_states @ W_pool + b_pool)