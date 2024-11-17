from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.signal import convolve2d

from common.operators import ConvolutionMode


def kernel_to_func(
    kernel: Callable[[Any], npt.NDArray] | npt.NDArray | sp.csr_matrix,
) -> Callable[[npt.NDArray], npt.NDArray]:
    """Convert blurring kernel to function.

    Args:
        - kernel: can be function, NDArray or sparse matrix

    Returns
    -------
        Callable function for blurring
    """
    if isinstance(kernel, np.ndarray):
        return partial(convolve2d, in2=kernel, mode=ConvolutionMode.SAME)

    if isinstance(kernel, sp.csr_matrix):
        return lambda x: kernel @ x.flatten()

    return kernel


def step_func_gen(N: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate toy 1D function.

    Args:
        N: length
    """
    x_min = -1
    x_max = 1
    xs = np.linspace(x_min, x_max, N)
    coords = np.array(
        [[-0.95, -0.6], [-0.6, -0.2], [-0.2, 0.2], [0.4, 0.6], [0.6, 1.0]],
    )
    indices = np.round((coords + 1) / 2 * (N - 1)).astype(np.int16)
    vals = np.array([1, 0.2, -0.5, 0.7, -0.7])
    ys = np.zeros(xs.shape)

    for i in range(vals.shape[0]):
        idx = np.linspace(
            indices[i][0],
            indices[i][1],
            indices[i][1] - indices[i][0] + 1,
        ).astype(np.int16)
        ys[idx] = vals[i]

    return xs, ys
