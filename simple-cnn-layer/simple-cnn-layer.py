import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    batch_size, _, height, width = x.shape
    output_channels, _, kernel_height, kernel_width = W.shape
    output_height = height - kernel_height + 1
    output_width = width - kernel_width + 1
    output = np.zeros(
        (batch_size, output_channels, output_height, output_width),
        dtype=float,
    )
    for n in range(batch_size):
        for output_channel in range(output_channels):
            for row in range(output_height):
                for column in range(output_width):
                    patch = x[
                        n,
                        :,
                        row:row + kernel_height,
                        column:column + kernel_width,
                    ]
                    output[n, output_channel, row, column] = (
                        np.sum(patch * W[output_channel]) + b[output_channel]
                    )
    return output