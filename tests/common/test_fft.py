import numpy as np
import pytest

from common.fft import fft_1d, fft_2d, ifft_1d, ifft_2d, set_up_kspace
from common.utils import get_min_max, sinc_function_1d, sinc_function_2d


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
    y, x = sinc_function_1d(x_range=(-10 * factor, 10 * factor), num_samples=101)
    ky, kx = fft_1d(y, x)
    ry, rx = ifft_1d(ky, kx)

    # Check reconstructed signal and spatial dims are correct
    assert np.isclose(np.square(ry - y).sum(), 0.0)
    assert np.isclose(np.square(rx - x).sum(), 0.0)

    # Odd array length case - note that spatial dims are not exact
    y, x = sinc_function_1d(x_range=(-10 * factor, 10 * factor), num_samples=10)
    ky, kx = fft_1d(y, x)
    ry, _ = ifft_1d(ky, kx)

    # Check reconstructed signal is correct
    assert np.isclose(np.square(ry - y).sum(), 0.0)


@pytest.mark.parametrize("factor", [1, 2])  # type: ignore[misc]
def test_fft_2d(factor: int) -> None:
    """Test both forward and inverse 2D FFT."""
    # Odd array length case
    z, x, y = sinc_function_2d(
        x_range=(-10 * factor, 10 * factor),
        y_range=(-20, 20),
        num_x=101,
        num_y=101,
    )
    kz, kx, ky = fft_2d(z, x, y)
    rz, rx, ry = ifft_2d(kz, kx, ky)

    # Check reconstructed signal and spatial dims are correct
    assert np.isclose(np.square(rz - z).sum(), 0.0)
    assert np.isclose(np.square(rx - x).sum(), 0.0)
    assert np.isclose(np.square(ry - y).sum(), 0.0)

    # Odd array length case - note that spatial dims are not exact
    z, x, y = sinc_function_2d(
        x_range=(-10 * factor, 10 * factor),
        y_range=(-20, 20),
        num_x=100,
        num_y=100,
    )
    kz, kx, ky = fft_2d(z, x, y)
    rz, _, _ = ifft_2d(kz, kx, ky)

    # Check reconstructed signal is correct
    assert np.isclose(np.square(rz - z).sum(), 0.0)
