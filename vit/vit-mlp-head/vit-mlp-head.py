import numpy as np

def classification_head(encoder_output: np.ndarray,
                        W_head: np.ndarray) -> np.ndarray:
    cls_state = encoder_output[:, 0, :]
    mean = np.mean(cls_state, axis=-1, keepdims=True)
    variance = np.mean((cls_state - mean) ** 2, axis=-1, keepdims=True)
    normalized = (cls_state - mean) / np.sqrt(variance + 1e-6)
    return normalized @ W_head