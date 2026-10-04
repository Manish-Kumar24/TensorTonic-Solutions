import numpy as np

def prepend_class_token(patches: np.ndarray,
                        cls_token: np.ndarray) -> np.ndarray:
    tokens = np.broadcast_to(cls_token, (patches.shape[0], 1, patches.shape[-1]))
    return np.concatenate((tokens, patches), axis=1)