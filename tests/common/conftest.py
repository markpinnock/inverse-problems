"""Fixtures for common unit tests."""

# mypy: disable-error-code="misc"
import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture
def valid_4_first_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [-1  1  0  0 ]
        [ 0 -1  1  0 ]
        [ 0  0 -1  1 ]

    """
    return np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]], dtype=np.float64)


@pytest.fixture
def valid_5_first_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [-1  1  0  0  0 ]
        [ 0 -1  1  0  0 ]
        [ 0  0 -1  1  0 ]
        [ 0  0  0 -1  1 ]

    """
    return np.array(
        [[-1, 1, 0, 0, 0], [0, -1, 1, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, -1, 1]],
        dtype=np.float64,
    )


@pytest.fixture
def same_4_first_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [ 1  0  0  0 ]
        [-1  1  0  0 ]
        [ 0 -1  1  0 ]
        [ 0  0 -1  1 ]

    """
    return np.array(
        [[1, 0, 0, 0], [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]],
        dtype=np.float64,
    )


@pytest.fixture
def same_5_first_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [ 1  0  0  0  0 ]
        [-1  1  0  0  0 ]
        [ 0 -1  1  0  0 ]
        [ 0  0 -1  1  0 ]
        [ 0  0  0 -1  1 ]

    """
    return np.array(
        [
            [1, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def periodic_4_first_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [ 1  0  0 -1 ]
        [-1  1  0  0 ]
        [ 0 -1  1  0 ]
        [ 0  0 -1  1 ]

    """
    return np.array(
        [[1, 0, 0, -1], [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]],
        dtype=np.float64,
    )


@pytest.fixture
def periodic_5_first_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [ 1  0  0  0 -1 ]
        [-1  1  0  0  0 ]
        [ 0 -1  1  0  0 ]
        [ 0  0 -1  1  0 ]
        [ 0  0  0 -1  1 ]

    """
    return np.array(
        [
            [1, 0, 0, 0, -1],
            [-1, 1, 0, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def full_4_first_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [ 1  0  0  0 ]
        [-1  1  0  0 ]
        [ 0 -1  1  0 ]
        [ 0  0 -1  1 ]
        [ 0  0  0 -1 ]

    """
    return np.array(
        [[1, 0, 0, 0], [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [0, 0, 0, -1]],
        dtype=np.float64,
    )


@pytest.fixture
def full_5_first_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [ 1  0  0  0  0 ]
        [-1  1  0  0  0 ]
        [ 0 -1  1  0  0 ]
        [ 0  0 -1  1  0 ]
        [ 0  0  0 -1  1 ]
        [ 0  0  0  0 -1 ]

    """
    return np.array(
        [
            [1, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1],
            [0, 0, 0, 0, -1],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def valid_4_second_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [ 1 -2  1  0 ]
        [ 0  1 -2  1 ]

    """
    return np.array([[1, -2, 1, 0], [0, 1, -2, 1]], dtype=np.float64)


@pytest.fixture
def valid_5_second_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [ 1 -2  1  0  0 ]
        [ 0  1 -2  1  0 ]
        [ 0  0  1 -2  1 ]

    """
    return np.array(
        [[1, -2, 1, 0, 0], [0, 1, -2, 1, 0], [0, 0, 1, -2, 1]],
        dtype=np.float64,
    )


@pytest.fixture
def same_4_second_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [-2  1  0  0 ]
        [ 1 -2  1  0 ]
        [ 0  1 -2  1 ]
        [ 1  0  1 -2 ]

    """
    return np.array(
        [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]],
        dtype=np.float64,
    )


@pytest.fixture
def same_5_second_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [-2  1  0  0  0 ]
        [ 1 -2  1  0  0 ]
        [ 0  1 -2  1  0 ]
        [ 0  0  1 -2  1 ]
        [ 0  0  0  1 -2 ]

    """
    return np.array(
        [
            [-2, 1, 0, 0, 0],
            [1, -2, 1, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 1, -2, 1],
            [0, 0, 0, 1, -2],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def periodic_4_second_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [-2  1  0  1 ]
        [ 1 -2  1  0 ]
        [ 0  1 -2  1 ]
        [ 1  0  1 -2 ]

    """
    return np.array(
        [[-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1, -2]],
        dtype=np.float64,
    )


@pytest.fixture
def periodic_5_second_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [-2  1  0  0  1 ]
        [ 1 -2  1  0  0 ]
        [ 0  1 -2  1  0 ]
        [ 0  0  1 -2  1 ]
        [ 1  0  0  1 -2 ]

    """
    return np.array(
        [
            [-2, 1, 0, 0, 1],
            [1, -2, 1, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 1, -2, 1],
            [1, 0, 0, 1, -2],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def full_4_second_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [ 1  0  0  0 ]
        [-2  1  0  0 ]
        [ 1 -2  1  0 ]
        [ 0  1 -2  1 ]
        [ 0  0  1 -2 ]
        [ 0  0  0  1 ]

    """
    return np.array(
        [
            [1, 0, 0, 0],
            [-2, 1, 0, 0],
            [1, -2, 1, 0],
            [0, 1, -2, 1],
            [0, 0, 1, -2],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def full_5_second_order_1d() -> npt.NDArray:
    """Return 1D convolution matrix.

    Returns
    -------
        [ 1  0  0  0  0 ]
        [-2  1  0  0  0 ]
        [ 1 -2  1  0  0 ]
        [ 0  1 -2  1  0 ]
        [ 0  0  1 -2  1 ]
        [ 0  0  0  1 -2 ]
        [ 0  0  0  0  1 ]

    """
    return np.array(
        [
            [1, 0, 0, 0, 0],
            [-2, 1, 0, 0, 0],
            [1, -2, 1, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 1, -2, 1],
            [0, 0, 0, 1, -2],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def valid_4_first_order_2d(valid_4_first_order_1d: npt.NDArray) -> npt.NDArray:
    """Return 2D convolution matrix.

    Returns
    -------
        [ B  B  0  0 ]
        [ 0  B  B  0 ]
        [ 0  0  B  B ]

    """
    block = valid_4_first_order_1d
    zero = np.zeros_like(block)
    return np.vstack(
        [
            np.hstack([block, block, zero, zero]),
            np.hstack([zero, block, block, zero]),
            np.hstack([zero, zero, block, block]),
        ],
    )


@pytest.fixture
def valid_5_first_order_2d(valid_5_first_order_1d: npt.NDArray) -> npt.NDArray:
    """Return 2D convolution matrix.

    Returns
    -------
        [ B  B  0  0  0 ]
        [ 0  B  B  0  0 ]
        [ 0  0  B  B  0 ]
        [ 0  0  0  B  B ]

    """
    block = valid_5_first_order_1d
    zero = np.zeros_like(block)
    return np.vstack(
        [
            np.hstack([block, block, zero, zero, zero]),
            np.hstack([zero, block, block, zero, zero]),
            np.hstack([zero, zero, block, block, zero]),
            np.hstack([zero, zero, zero, block, block]),
        ],
    )


@pytest.fixture
def same_4_first_order_2d(same_4_first_order_1d: npt.NDArray) -> npt.NDArray:
    """Return 2D convolution matrix.

    Returns
    -------
        [ B  0  0  0 ]
        [ B  B  0  0 ]
        [ 0  B  B  0 ]
        [ 0  0  B  B ]

    """
    block = same_4_first_order_1d
    zero = np.zeros_like(block)
    return np.vstack(
        [
            np.hstack([block, zero, zero, zero]),
            np.hstack([block, block, zero, zero]),
            np.hstack([zero, block, block, zero]),
            np.hstack([zero, zero, block, block]),
        ],
    )


@pytest.fixture
def same_5_first_order_2d(same_5_first_order_1d: npt.NDArray) -> npt.NDArray:
    """Return 2D convolution matrix.

    Returns
    -------
        [ B  0  0  0  0 ]
        [ B  B  0  0  0 ]
        [ 0  B  B  0  0 ]
        [ 0  0  B  B  0 ]
        [ 0  0  0  B  B ]

    """
    block = same_5_first_order_1d
    zero = np.zeros_like(block)
    return np.vstack(
        [
            np.hstack([block, zero, zero, zero, zero]),
            np.hstack([block, block, zero, zero, zero]),
            np.hstack([zero, block, block, zero, zero]),
            np.hstack([zero, zero, block, block, zero]),
            np.hstack([zero, zero, zero, block, block]),
        ],
    )


@pytest.fixture
def periodic_4_first_order_2d(periodic_4_first_order_1d: npt.NDArray) -> npt.NDArray:
    """Return 2D convolution matrix.

    Returns
    -------
        [ B  0  0  B ]
        [ B  B  0  0 ]
        [ 0  B  B  0 ]
        [ 0  0  B  B ]

    """
    block = periodic_4_first_order_1d
    zero = np.zeros_like(block)
    return np.vstack(
        [
            np.hstack([block, zero, zero, block]),
            np.hstack([block, block, zero, zero]),
            np.hstack([zero, block, block, zero]),
            np.hstack([zero, zero, block, block]),
        ],
    )


@pytest.fixture
def periodic_5_first_order_2d(periodic_5_first_order_1d: npt.NDArray) -> npt.NDArray:
    """Return 2D convolution matrix.

    Returns
    -------
        [ B  0  0  0  B ]
        [ B  B  0  0  0 ]
        [ 0  B  B  0  0 ]
        [ 0  0  B  B  0 ]
        [ 0  0  0  B  B ]

    """
    block = periodic_5_first_order_1d
    zero = np.zeros_like(block)
    return np.vstack(
        [
            np.hstack([block, zero, zero, zero, block]),
            np.hstack([block, block, zero, zero, zero]),
            np.hstack([zero, block, block, zero, zero]),
            np.hstack([zero, zero, block, block, zero]),
            np.hstack([zero, zero, zero, block, block]),
        ],
    )


@pytest.fixture
def full_4_first_order_2d(full_4_first_order_1d: npt.NDArray) -> npt.NDArray:
    """Return 2D convolution matrix.

    Returns
    -------
        [ B  0  0  0 ]
        [ B  B  0  0 ]
        [ 0  B  B  0 ]
        [ 0  0  B  B ]
        [ 0  0  0  B ]

    """
    block = full_4_first_order_1d
    zero = np.zeros_like(block)
    return np.vstack(
        [
            np.hstack([block, zero, zero, zero]),
            np.hstack([block, block, zero, zero]),
            np.hstack([zero, block, block, zero]),
            np.hstack([zero, zero, block, block]),
            np.hstack([zero, zero, zero, block]),
        ],
    )


@pytest.fixture
def full_5_first_order_2d(full_5_first_order_1d: npt.NDArray) -> npt.NDArray:
    """Return 2D convolution matrix.

    Returns
    -------
        [ B  0  0  0  0 ]
        [ B  B  0  0  0 ]
        [ 0  B  B  0  0 ]
        [ 0  0  B  B  0 ]
        [ 0  0  0  B  B ]
        [ 0  0  0  0  B ]

    """
    block = full_5_first_order_1d
    zero = np.zeros_like(block)
    return np.vstack(
        [
            np.hstack([block, zero, zero, zero, zero]),
            np.hstack([block, block, zero, zero, zero]),
            np.hstack([zero, block, block, zero, zero]),
            np.hstack([zero, zero, block, block, zero]),
            np.hstack([zero, zero, zero, block, block]),
            np.hstack([zero, zero, zero, zero, block]),
        ],
    )
