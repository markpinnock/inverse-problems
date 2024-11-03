"""Fixtures for common unit tests."""

# mypy: disable-error-code="misc"
import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture
def valid_4_first_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
    return np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]], dtype=np.float64)


@pytest.fixture
def valid_5_first_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
    return np.array(
        [[-1, 1, 0, 0, 0], [0, -1, 1, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, -1, 1]],
        dtype=np.float64,
    )


@pytest.fixture
def same_4_first_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
    return np.array(
        [[1, 0, 0, 0], [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]],
        dtype=np.float64,
    )


@pytest.fixture
def same_5_first_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
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
def periodic_4_first_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
    return np.array(
        [[1, 0, 0, -1], [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]],
        dtype=np.float64,
    )


@pytest.fixture
def periodic_5_first_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
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
def full_4_first_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
    return np.array(
        [[1, 0, 0, 0], [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [0, 0, 0, -1]],
        dtype=np.float64,
    )


@pytest.fixture
def full_5_first_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
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
def valid_4_second_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
    return np.array([[1, -2, 1, 0], [0, 1, -2, 1]], dtype=np.float64)


@pytest.fixture
def valid_5_second_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
    return np.array(
        [[1, -2, 1, 0, 0], [0, 1, -2, 1, 0], [0, 0, 1, -2, 1]],
        dtype=np.float64,
    )


@pytest.fixture
def same_4_second_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
    return np.array(
        [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]],
        dtype=np.float64,
    )


@pytest.fixture
def same_5_second_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
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
def periodic_4_second_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
    return np.array(
        [[-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1, -2]],
        dtype=np.float64,
    )


@pytest.fixture
def periodic_5_second_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
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
def full_4_second_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
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
def full_5_second_order_1d() -> npt.NDArray[np.float64]:
    """Return 1D convolution matrix."""
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
