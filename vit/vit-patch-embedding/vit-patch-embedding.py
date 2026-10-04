import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, height, width, channels = image.shape
    grid_height = height // patch_size
    grid_width = width // patch_size
    usable = image[:, :grid_height * patch_size, :grid_width * patch_size, :]
    patches = usable.reshape(
        batch, grid_height, patch_size, grid_width, patch_size, channels
    )
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(batch, grid_height * grid_width, -1)
    return patches @ W_proj + bias