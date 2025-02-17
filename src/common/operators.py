"""Sparse operators for various image processing tasks."""

import enum
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp


class ConvolutionMode(str, enum.Enum):
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


def identity_operator(
    img: npt.NDArray[np.uint8 | np.float32], **kwargs: Any
) -> sp.csr_matrix:
    """Create identity operator.

    Args:
        img: The input image/array.
        kwargs: Unused kwargs to maintain same interface

    Returns
    -------
        Sparse identity matrix

    """
    return sp.eye(np.prod(img.shape)).tocsr()


def dx_operator_1d(
    arr: npt.NDArray[np.uint8 | np.float32],
    conv_mode: str | ConvolutionMode = ConvolutionMode.SAME,
) -> sp.csr_matrix:
    """Create derivative operator for 1D arrays.

    Args:
        img: The input array.
        conv_mode: The convolutional mode (full, same, valid, periodic).

    Returns
    -------
        Sparse difference operator

    """
    if arr.ndim > 1:
        raise ValueError(f"The input array must be 1D: {arr.shape}")

    return custom_operator_1d(
        kernel=np.array([-1, 1]),
        arr_size=arr.shape[0],
        conv_mode=conv_mode,
    )


def dx_operator(
    img: npt.NDArray[np.uint8 | np.float32],
    conv_mode: str | ConvolutionMode = ConvolutionMode.SAME,
) -> sp.csr_matrix:
    """Create x-direction derivative operator for 2D images.

    Args:
        img: The input image.
        conv_mode: The convolutional mode (full, same, valid, periodic).

    Returns
    -------
        Sparse x-direction derivative operator

    """
    if img.ndim != 2:
        raise ValueError(f"The input array must be 2D: {img.shape}")

    if img.shape[0] != img.shape[1]:
        raise ValueError(f"Currently only square images supported: {img.shape}")

    return custom_operator_2d(
        kernel=np.array([[-1, 1]]),
        image_size=img.shape[0],
        conv_mode=conv_mode,
    )


def dy_operator(
    img: npt.NDArray[np.uint8 | np.float32],
    conv_mode: str | ConvolutionMode = ConvolutionMode.SAME,
) -> sp.csr_matrix:
    """Create y-direction derivative operator for 2D images.

    Args:
        img: The input image.
        conv_mode: The convolutional mode (full, same, valid, periodic).

    Returns
    -------
        Sparse y-direction derivative operator

    """
    if img.ndim != 2:
        raise ValueError(f"The input array must be 2D: {img.shape}")

    if img.shape[0] != img.shape[1]:
        raise ValueError(f"Currently only square kernels supported: {img.shape}")

    return custom_operator_2d(
        kernel=np.array([[-1], [1]]),
        image_size=img.shape[0],
        conv_mode=conv_mode,
    )


def derivative_operator(
    img: npt.NDArray[np.uint8 | np.float32],
    conv_mode: str | ConvolutionMode = ConvolutionMode.SAME,
) -> sp.csr_matrix:
    """Create derivative operator for a 2D image.

    Notes
    -----
        The derivative operator is the concatenation of the x and y derivative operators.

    Args:
        img: The input image.
        conv_mode: The convolutional mode (full, same, valid, periodic).

    Returns
    -------
        Sparse derivative operator

    """
    Dx = dx_operator(img, conv_mode=conv_mode)
    Dy = dy_operator(img, conv_mode=conv_mode)
    return sp.vstack([Dx, Dy])


def laplacian_operator(
    img: npt.NDArray[np.uint8 | np.float32],
    conv_mode: str | ConvolutionMode = ConvolutionMode.SAME,
) -> sp.csr_matrix:
    """Create Laplacian operator for a 2D image.

    Args:
        img: The input image.
        conv_mode: The convolutional mode: only `same` supported.

    Returns
    -------
        Sparse Laplacian operator

    """
    if conv_mode != ConvolutionMode.SAME:
        raise ValueError(f"Only `same` conv mode is supported: got {conv_mode}.")

    D = derivative_operator(img, conv_mode=conv_mode)
    return -D.T @ D


def perona_malik_operator(
    img: npt.NDArray[np.uint8 | np.float32],
    conv_mode: str | ConvolutionMode = ConvolutionMode.SAME,
) -> sp.csr_matrix:
    """Create Perona-Malik operator for a 2D image.

    Args:
        img: The input image.
        conv_mode: The convolutional mode. Either "valid" or "periodic".

    Returns
    -------
        Sparse Perona-Malik operator

    """
    Dx = dx_operator(img, conv_mode=conv_mode)
    Dy = dy_operator(img, conv_mode=conv_mode)

    img = img.reshape([-1, 1])
    img_dx = Dx @ img
    img_dy = Dy @ img

    abs_derivative = np.sqrt(np.square(img_dx) + np.square(img_dy))
    threshold = 0.5 * np.max(abs_derivative)
    gamma = np.exp(-abs_derivative / threshold)
    gamma_diag_hat = np.sqrt(sp.diags(gamma.ravel()))

    return sp.vstack([gamma_diag_hat @ Dx, gamma_diag_hat @ Dy])


def custom_operator_1d(
    kernel: npt.NDArray,
    arr_size: int,
    conv_mode: str | ConvolutionMode = ConvolutionMode.SAME,
) -> sp.csr_array:
    """Create a sparse convolutional matrix for a 1D custom kernel.

    Args:
        kernel: the custom 1D kernel as a numpy array
        arr_size: the length of the array
        conv_mode: the convolution shape (full, same, valid, periodic)

    Returns
    -------
        Sparse 1D operator

    """
    if kernel.ndim != 1:
        raise ValueError("The kernel must be 1D.")

    kernel_size = kernel.shape[0]

    # Determine convolution mode
    match conv_mode:
        case ConvolutionMode.FULL:
            input_size = arr_size + kernel_size - 1
            kernel_width = kernel_size - 1

        case ConvolutionMode.SAME:
            input_size = arr_size
            kernel_width = kernel_size // 2

        case ConvolutionMode.VALID:
            input_size = arr_size - kernel_size + 1

        case ConvolutionMode.PERIODIC:
            input_size = arr_size
            kernel_width = kernel_size // 2

    conv_matrix = sp.lil_matrix((input_size, arr_size))

    # Loop through each element in array
    for row_idx in range(input_size):
        # Loop through each element in the kernel
        for kernel_idx in range(kernel_size):
            match conv_mode:
                case ConvolutionMode.FULL:
                    # Shift column index back by full kernel width
                    col_idx = row_idx + kernel_idx - kernel_width

                    if 0 <= col_idx < arr_size:
                        conv_matrix[row_idx, col_idx] = kernel[kernel_idx]

                case ConvolutionMode.SAME:
                    # Shift column index back by half kernel width
                    col_idx = row_idx + kernel_idx - kernel_width

                    if 0 <= col_idx < arr_size:
                        conv_matrix[row_idx, col_idx] = kernel[kernel_idx]

                case ConvolutionMode.VALID:
                    # No shift necessary for valid convolution
                    col_idx = row_idx + kernel_idx
                    conv_matrix[row_idx, col_idx] = kernel[kernel_idx]

                case ConvolutionMode.PERIODIC:
                    # Compute periodic (wrapped) column index
                    col_idx = (row_idx + kernel_idx - kernel_width) % input_size
                    conv_matrix[row_idx, col_idx] = kernel[kernel_idx]

    return conv_matrix.tocsr()


def custom_operator_2d(
    kernel: npt.NDArray,
    image_size: int,
    conv_mode: str | ConvolutionMode = ConvolutionMode.SAME,
) -> sp.csr_array:
    """Create a sparse convolutional matrix for a 2D custom kernel.

    Args:
        kernel: the custom 2D kernel as a numpy array
        arr_size: the length of the array
        conv_mode: the convolution shape (full, same, valid, periodic)

    Returns
    -------
        Sparse 2D operator

    """
    if kernel.ndim != 2:
        raise ValueError("The kernel must be 2D.")

    kernel_h, kernel_w = kernel.shape

    # Determine input_h and input_w (image boundaries) and kernel radius
    match conv_mode:
        case ConvolutionMode.FULL:
            input_h = image_size + kernel_h - 1
            input_w = image_size + kernel_w - 1
            radius_h = kernel_h - 1
            radius_w = kernel_w - 1

        case ConvolutionMode.SAME:
            input_h = image_size
            input_w = image_size
            radius_h = kernel_h // 2
            radius_w = kernel_w // 2

        case ConvolutionMode.VALID:
            input_h = image_size - kernel_h + 1
            input_w = image_size - kernel_w + 1

        case ConvolutionMode.PERIODIC:
            input_h = image_size
            input_w = image_size
            radius_h = kernel_h // 2
            radius_w = kernel_w // 2

    # Initialize the convolution matrix as a sparse matrix
    conv_matrix = sp.lil_matrix((input_h * input_w, image_size**2))

    # Loop through each pixel in image
    for img_i in range(input_h):
        for img_j in range(input_w):
            # Define the row of this pixel in the output matrix
            row_idx = img_i * input_w + img_j

            # Loop through each element in the kernel
            for ker_i in range(kernel_h):
                for ker_j in range(kernel_w):
                    match conv_mode:
                        case ConvolutionMode.FULL:
                            # Shift column index back by full kernel width
                            offset_i = img_i + ker_i - radius_h
                            offset_j = img_j + ker_j - radius_w
                            col_idx = offset_i * image_size + offset_j

                            if (0 <= offset_i < image_size) and (
                                0 <= offset_j < image_size
                            ):
                                conv_matrix[row_idx, col_idx] = kernel[ker_i, ker_j]

                        case ConvolutionMode.SAME:
                            # Shift column index back by half kernel width
                            offset_i = img_i + ker_i - radius_h
                            offset_j = img_j + ker_j - radius_w

                            if (0 <= offset_i < image_size) and (
                                0 <= offset_j < image_size
                            ):
                                col_idx = offset_i * image_size + offset_j
                                conv_matrix[row_idx, col_idx] = kernel[ker_i, ker_j]

                        case ConvolutionMode.VALID:
                            # No shift necessary for valid convolution
                            offset_i = img_i + ker_i
                            offset_j = img_j + ker_j
                            col_idx = offset_i * image_size + offset_j
                            conv_matrix[row_idx, col_idx] = kernel[ker_i, ker_j]

                        case ConvolutionMode.PERIODIC:
                            # Compute periodic (wrapped) column index
                            offset_i = (img_i + ker_i - radius_h) % image_size
                            offset_j = (img_j + ker_j - radius_w) % image_size
                            col_idx = offset_i * image_size + offset_j
                            conv_matrix[row_idx, col_idx] = kernel[ker_i, ker_j]

    return conv_matrix.tocsr()
