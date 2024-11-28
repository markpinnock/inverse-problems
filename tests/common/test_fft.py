import numpy as np
import pytest

from common.fft import get_min_max, set_up_kspace


@pytest.mark.parametrize("num_samples,expected", [(4, (-2, 1)), (5, (-2, 2))])  # type: ignore[misc]
def test_get_min_max(num_samples: int, expected: tuple[int, int]) -> None:
    """Test get_min_max function"""
    assert get_min_max(num_samples) == expected


@pytest.mark.parametrize("num_samples", [8, 9])  # type: ignore[misc]
def test_set_up_kspace(num_samples: int) -> None:
    """Test set_up_kspace function."""
    # Check that k-space values are correct
    fov = 1
    _, k_values = set_up_kspace(fov, num_samples)
    k_min, k_max = get_min_max(num_samples)
    assert k_values[0] == k_min
    assert k_values[-1] == k_max

    # Check that x- and k-values match when k-space FOV passed in
    fov = 2
    x_values_1, k_values_1 = set_up_kspace(fov, num_samples)
    k_fov = k_values_1.max() - k_values_1.min()
    k_values_2, x_values_2 = set_up_kspace(k_fov, num_samples)
    assert np.isclose(x_values_1, x_values_2).all()
    assert np.isclose(k_values_1, k_values_2).all()
