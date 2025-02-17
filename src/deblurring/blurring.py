import enum
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.stats import norm

from common.utils import kernel_to_func


@enum.unique
class GaussianType(str, enum.Enum):
    """Return type of Gaussian.

    Attributes:
        distribution: full distribution of size N from 0.01 to 0.99
        kernel: return a discrete kernel from -N // 2 to N // 2
    """

    DISTRIBUTION = "distribution"
    KERNEL = "kernel"


def gaussian(x: npt.NDArray, mu: float, sigma: float) -> npt.NDArray:
    """Create a 1D Gaussian distribution from specific x.

    Args:
    ----
        x: x values
        mu: Mean
        sigma: Standard deviation

    Returns:
    -------
        np.ndarray: 1D Gaussian distribution

    """
    exp_term = np.exp(-np.square(x - mu) / (2 * np.square(sigma)))
    return exp_term / exp_term.sum()


def gaussian_kernel_1d(
    sigma: float,
    size: int,
    gaussian_type: str = GaussianType.KERNEL,
) -> npt.NDArray[np.float64]:
    """Create a 1D Gaussian kernel.

    Args:
    ----
        sigma: Standard deviation of the Gaussian kernel
        size: Size of the kernel
        gaussian_type: `distribution` or `kernel`

    Returns:
    -------
        np.ndarray: 1D Gaussian kernel

    """
    normal_distribution = norm(loc=0.0, scale=sigma)

    if gaussian_type == GaussianType.KERNEL:
        min_x = -(size // 2)
        max_x = size // 2
    else:
        min_x = normal_distribution.ppf(0.01)
        max_x = normal_distribution.ppf(0.99)

    x = np.linspace(min_x, max_x, size)
    y = normal_distribution.pdf(x)
    return y / y.sum()


def gaussian_kernel_2d(
    sigma: float,
    size: int,
    gaussian_type: str = GaussianType.KERNEL,
) -> npt.NDArray[np.float64]:
    """Create a 2D Gaussian kernel.

    Args:
    ----
        sigma: Standard deviation of the Gaussian kernel
        size: Size of the kernel
        gaussian_type: `distribution` or `kernel`

    Returns:
    -------
        np.ndarray: 2D Gaussian kernel

    """
    normal_distribution = norm(loc=0.0, scale=sigma)

    if gaussian_type == GaussianType.KERNEL:
        min_x = -(size // 2)
        max_x = size // 2
    else:
        min_x = normal_distribution.ppf(0.01)
        max_x = normal_distribution.ppf(0.99)

    x = np.linspace(min_x, max_x, size)
    X, Y = np.meshgrid(x, x)
    xy_norm = np.sqrt(np.square(X) + np.square(Y))
    y = normal_distribution.pdf(xy_norm)
    return y / np.sum(y)


def blur_and_noise(
    arr: npt.NDArray,
    kernel: Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
    | npt.NDArray
    | sp.csr_matrix,
    theta: float,
) -> npt.NDArray[np.float64]:
    """Perform blurring using custom kernel and add noise.

    Args:
        - arr: input array (NDArray)
        - kernel: can be function, NDArray or sparse matrix

    Returns
    -------
        Blurred and noisy array
    """
    # Generate random noise
    eta = np.random.randn(*arr.shape)

    # Convolve with kernel and add noise
    return kernel_to_func(kernel)(arr) + theta * eta
