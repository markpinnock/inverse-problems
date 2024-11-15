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
