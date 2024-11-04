"""Unit tests for operators."""

# mypy: disable-error-code="misc"
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from common.operators import custom_operator_1d, custom_operator_2d, identity_operator


def test_identity_operator() -> None:
    """Test identity operator."""
    img = np.random.rand(4, 4)
    operator = identity_operator(img)

    assert operator.shape == (16, 16)
    assert np.allclose(operator.toarray(), np.eye(16))
    assert np.allclose(operator @ img.flatten(), img.flatten())


@pytest.mark.parametrize(
    ("conv_mode", "arr_size", "expected_operator"),
    [
        ("valid", 4, "valid_4_first_order_1d"),
        ("valid", 5, "valid_5_first_order_1d"),
        ("same", 4, "same_4_first_order_1d"),
        ("same", 5, "same_5_first_order_1d"),
        ("periodic", 4, "periodic_4_first_order_1d"),
        ("periodic", 5, "periodic_5_first_order_1d"),
        ("full", 4, "full_4_first_order_1d"),
        ("full", 5, "full_5_first_order_1d"),
    ],
)
def test_custom_operator_1d_even(
    conv_mode: str,
    arr_size: int,
    expected_operator: npt.NDArray,
    request: Any,
) -> None:
    """Test custom 1D operator for even kernels."""
    expected_operator = request.getfixturevalue(expected_operator)
    kernel = np.array([-1, 1])

    operator = custom_operator_1d(kernel=kernel, arr_size=arr_size, conv_mode=conv_mode)
    assert np.equal(operator.toarray(), expected_operator).all()


@pytest.mark.parametrize(
    ("conv_mode", "arr_size", "expected_operator"),
    [
        ("valid", 4, "valid_4_second_order_1d"),
        ("valid", 5, "valid_5_second_order_1d"),
        ("same", 4, "same_4_second_order_1d"),
        ("same", 5, "same_5_second_order_1d"),
        ("periodic", 4, "periodic_4_second_order_1d"),
        ("periodic", 5, "periodic_5_second_order_1d"),
        ("full", 4, "full_4_second_order_1d"),
        ("full", 5, "full_5_second_order_1d"),
    ],
)
def test_custom_operator_1d_odd(
    conv_mode: str,
    arr_size: int,
    expected_operator: npt.NDArray,
    request: Any,
) -> None:
    """Test custom 1D operator for odd kernels."""
    expected_operator = request.getfixturevalue(expected_operator)
    kernel = np.array([1, -2, 1])

    operator = custom_operator_1d(kernel=kernel, arr_size=arr_size, conv_mode=conv_mode)
    assert np.equal(operator.toarray(), expected_operator).all()


@pytest.mark.parametrize(
    ("conv_mode", "img_size", "expected_operator"),
    [
        ("valid", 4, "valid_4_first_order_2d"),
        ("valid", 5, "valid_5_first_order_2d"),
        ("same", 4, "same_4_first_order_2d"),
        ("same", 5, "same_5_first_order_2d"),
        ("periodic", 4, "periodic_4_first_order_2d"),
        ("periodic", 5, "periodic_5_first_order_2d"),
        ("full", 4, "full_4_first_order_2d"),
        ("full", 5, "full_5_first_order_2d"),
    ],
)
def test_custom_operator_2d_even(
    conv_mode: str,
    img_size: int,
    expected_operator: npt.NDArray,
    request: Any,
) -> None:
    """Test custom 2D operator for odd kernels."""
    expected_operator = request.getfixturevalue(expected_operator)
    kernel = np.array([[-1, 1], [-1, 1]])

    operator = custom_operator_2d(
        kernel=kernel,
        image_size=img_size,
        conv_mode=conv_mode,
    )
    assert np.equal(operator.toarray(), expected_operator).all()
