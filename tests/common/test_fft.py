import numpy as np
import pytest

from common.fft import fft_1d, get_min_max, ifft_1d, set_up_kspace, sinc_function


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


@pytest.mark.parametrize("factor", [1, 2])  # type: ignore[misc]
def test_fft_1d(factor: int) -> None:
    """Test both forward and inverse 1D FFT."""
    # Odd array length case
    y, x = sinc_function(-10 * factor, 10 * factor, 101)
    ky, kx = fft_1d(y, x)
    ry, rx = ifft_1d(ky, kx)

    # Check reconstructed signal and spatial dims are correct
    assert np.isclose(np.square(ry - y).sum(), 0.0)
    assert np.isclose(np.square(rx - x).sum(), 0.0)

    # Odd array length case - note that spatial dims are not exact
    y, x = sinc_function(-10 * factor, 10 * factor, 10)
    ky, kx = fft_1d(y, x)
    ry, _ = ifft_1d(ky, kx)

    # Check reconstructed signal is correct
    assert np.isclose(np.square(ry - y).sum(), 0.0)
