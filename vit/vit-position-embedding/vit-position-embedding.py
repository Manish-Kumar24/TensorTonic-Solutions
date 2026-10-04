import numpy as np

def add_position_embedding(patches: np.ndarray,
                           pos_embed: np.ndarray) -> np.ndarray:
    return patches + pos_embed