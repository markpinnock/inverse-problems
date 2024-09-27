"""Sparse operators for various image processing tasks."""

import enum

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp


class ConvolutionMode(int, enum.Enum):
    FULL = "full"
    SAME = "same"
    VALID = "valid"
    PERIODIC = "periodic"


def custom_operator_1d(
    kernel: npt.NDArray[np.float32],
    matrix_size: int,
    conv_mode: str | ConvolutionMode = ConvolutionMode.SAME,
) -> sp.csr_array:
    """Create a 1D sparse convolutional matrix for a custom kernel.

    Adapted from:
        https://dsp.stackexchange.com/questions/76344/generate-the-matrix-form-of-1d-convolution-kernel

    Notes
    -----
        - The kernel is assumed to be 1D of size K with image vector length N.
        - `full` convolution mode is equivalent to full zero-padding both sides of the input (N + K - 1).
        - `same` convolution mode is equivalent to half zero-padding both sides of the input (N).
        - `valid` convolution mode is equivalent to no padding (output N - K + 1).
        - `periodic` convolution mode is equivalent to circular padding the input (N).

    Args:
        kernel: the custom 1D kernel as a numpy array
        matrix_size: the length of the vectorised image
        conv_mode: the convolution shape (full, same, valid, periodic)

    """
    if kernel.ndim != 1:
        raise ValueError("The kernel must be 1D.")

    kernel_length = kernel.shape[0]
    kernel = kernel[::-1]  # Original code ported from MATLAB (col vs. row ordering)

    # Determine convolution mode
    match conv_mode:
        case ConvolutionMode.FULL:
            row_first_idx = 0
            row_last_idx = kernel_length + matrix_size - 1
            periodic = False

        case ConvolutionMode.SAME:
            row_first_idx = kernel_length // 2
            row_last_idx = row_first_idx + matrix_size - 1
            periodic = False

        case ConvolutionMode.VALID:
            row_first_idx = kernel_length - 1
            row_last_idx = matrix_size - 1
            periodic = False

        case ConvolutionMode.PERIODIC:
            row_first_idx = kernel_length // 2
            row_last_idx = row_first_idx + matrix_size - 1
            periodic = True

    mtx_idx = 0

    i_idx = np.zeros(matrix_size * kernel_length, dtype=np.uint32)
    j_idx = np.zeros(matrix_size * kernel_length, dtype=np.uint32)
    values = np.zeros(matrix_size * kernel_length)

    # Loop through rows of matrix and allocate kernel to correct position
    for row_idx in range(matrix_size):
        for kernel_idx in range(kernel_length):
            if (
                kernel_idx + row_idx >= row_first_idx
                and kernel_idx + row_idx <= row_last_idx
            ):
                i_idx[mtx_idx] = kernel_idx + row_idx - row_first_idx
                j_idx[mtx_idx] = row_idx
                values[mtx_idx] = kernel[kernel_idx]
                mtx_idx += 1

            # Account for periodic boundary conditions if necessary
            if periodic and kernel_idx + row_idx < row_first_idx:
                i_idx[mtx_idx] = matrix_size + kernel_idx + row_idx - row_first_idx
                j_idx[mtx_idx] = row_idx
                values[mtx_idx] = kernel[kernel_idx]
                mtx_idx += 1

            elif periodic and kernel_idx + row_idx > row_last_idx:
                i_idx[mtx_idx] = kernel_idx + row_idx - row_last_idx - 1
                j_idx[mtx_idx] = row_idx
                values[mtx_idx] = kernel[kernel_idx]
                mtx_idx += 1

    return sp.csr_array((values, (j_idx, i_idx)))
