import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(0)
    keep = 1.0 - p
    mask = (rng.random(x.shape) < keep).astype(float) / keep
    return x * mask, mask