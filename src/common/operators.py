"""Sparse operators for various image processing tasks."""

import enum

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp


class ConvolutionMode(int, enum.Enum):
    """Convolution modes for creating sparse operators.

    Notes
    -----
        For a kernel of size K with image vector length N:
        - `full` convolution mode is equivalent to full zero-padding both sides of the input (N + K - 1).
        - `same` convolution mode is equivalent to half zero-padding both sides of the input (N).
        - `valid` convolution mode is equivalent to no padding (output N - K + 1).
        - `periodic` convolution mode is equivalent to circular padding the input (N).

    """

    FULL = "full"
    SAME = "same"
    VALID = "valid"
    PERIODIC = "periodic"


def identity_operator(img: npt.NDArray[np.uint8 | np.float32]) -> sp.dia_matrix:
    """Create identity operator.

    Args:
        img: The input image.

    Returns
    -------
        Sparse identity matrix

    """
    return sp.eye(np.prod(img.shape))


def dx_operator(
    img: npt.NDArray[np.uint8 | np.float32],
    conv_mode: str | ConvolutionMode = ConvolutionMode.SAME,
) -> sp.csr_matrix:
    """Create derivative operator in the x-direction.

    Args:
        img: The input image.
        conv_mode: The convolutional mode. Either "valid" or "periodic".

    Returns
    -------
        Sparse x-direction difference operator

    """
    flat_dims = np.prod(img.shape)
    vals = np.ones((2, flat_dims)) * np.array([[-1], [1]])

    if conv_mode == ConvolutionMode.VALID:
        Dx = sp.spdiags(vals, [0, 1], flat_dims - 1, flat_dims)

    elif conv_mode == ConvolutionMode.PERIODIC:
        Dx = sp.spdiags(vals, [0, 1], flat_dims, flat_dims).tolil()
        Dx[-1, 0] = 1

    else:
        raise ValueError(f"Convolution mode `{conv_mode}` currently not supported")

    return Dx.tocsr()


def custom_operator_1d(
    kernel: npt.NDArray[np.float32],
    matrix_size: int,
    conv_mode: str | ConvolutionMode = ConvolutionMode.SAME,
) -> sp.csr_array:
    """Create a 1D sparse convolutional matrix for a 1D custom kernel.

    Adapted from:
        https://dsp.stackexchange.com/questions/76344/generate-the-matrix-form-of-1d-convolution-kernel

    Args:
        kernel: the custom 1D kernel as a numpy array
        matrix_size: the length of the vectorised image
        conv_mode: the convolution shape (full, same, valid, periodic)

    Returns
    -------
        Sparse 1D operator

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
