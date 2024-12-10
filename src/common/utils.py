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


def rect_function_1d(
    width: int | float,
    x_range: tuple[int | float, int | float],
    num_samples: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Create 1D rectangle/box function.

    Args:
        width: width of rectangle
        x_range: range of x values
        num_samples: number of points

    Returns:
        y-values and corresponding x-values
    """
    x_values = np.linspace(x_range[0], x_range[1], num_samples)
    y_values = np.zeros_like(x_values)
    y_values[(x_values >= -width / 2) & (x_values <= width / 2)] = 1

    return y_values, x_values


def rect_function_2d(
    x_width: int | float,
    y_width: int | float,
    x_range: tuple[int | float, int | float],
    y_range: tuple[int | float, int | float],
    num_x: int,
    num_y: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Create 2D rectangle/box function.

    Args:
        x_width: width of rectangle on x side
        y_width: width of rectangle on y side
        x_range: range of x values
        y_range: range of y values
        num_x: number of points on x side
        num_y: number of points on y side

    Returns:
        z-values and corresponding y-values and x-values
    """
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_range[0], x_range[1], num_x),
        np.linspace(y_range[0], y_range[1], num_y),
    )
    z_values = np.zeros_like(x_grid)
    z_values[
        (x_grid >= -x_width / 2)
        & (x_grid <= x_width / 2)
        & (y_grid >= -y_width / 2)
        & (y_grid <= y_width / 2)
    ] = 1

    return z_values, x_grid, y_grid


def sinc_function_1d(
    x_range: tuple[int | float, int | float],
    num_samples: int,
    normalised: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Create 1D sinc function.

    Args:
        x_range: range of x values
        num_samples: number of points
        normalised: normalised sinc function

    Returns:
        y-values and corresponding x-values
    """
    x_values = np.linspace(x_range[0], x_range[1], num_samples)
    y_values = np.sinc(x_values) if normalised else np.sinc(x_values / np.pi)

    return y_values, x_values


def sinc_function_2d(
    x_range: tuple[int | float, int | float],
    y_range: tuple[int | float, int | float],
    num_x: int,
    num_y: int,
    normalised: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Create 2D sinc function.

    Args:
        x_range: range of x values
        y_range: range of y values
        num_x: number of points on x side
        num_y: number of points on y side
        normalised: normalised sinc function

    Returns:
        z-values and corresponding y-values and x-values
    """
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_range[0], x_range[1], num_x),
        np.linspace(y_range[0], y_range[1], num_y),
    )
    z_values = (
        np.sinc(x_grid) * np.sinc(y_grid)
        if normalised
        else np.sinc(x_grid / np.pi) * np.sinc(y_grid / np.pi)
    )

    return z_values, x_grid, y_grid


def get_min_max(num_samples: int) -> tuple[float, float]:
    """Get minimum and maximum for an array.

    Notes:
        - This assumes integer spacing between each element
          and that the array is symmetrical in the odd case
          and nearly symmetrical in the even case

    Args:
        num_samples: Number of elements in the array

    Returns:
        Minimum and maximum element in the array
    """
    return -(num_samples // 2), (num_samples - 1) // 2


def rect_phantom(
    img_dims: tuple[int, int],
    phantom_dims: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """Create image with white and black regions.

    Notes:
        - rect_phantom((4, 4), (2, 2)) will create the following image:
          0 0 0 0
          0 1 0 0
          0 0 1 0
          0 0 0 0

    Args:
        img_dims: height and width of the image
        phantom_dims: height and width of the phantom
    """
    half_width_y = phantom_dims[0] // 2
    half_width_x = phantom_dims[1] // 2
    centre_y = img_dims[0] // 2
    centre_x = img_dims[1] // 2
    img = np.zeros(img_dims)
    img[centre_y - half_width_y : centre_y, centre_x - half_width_x : centre_x] = 1
    img[centre_y : centre_y + half_width_y, centre_x : centre_x + half_width_x] = 1

    return img
