import numpy as np

from common.fft import set_up_kspace


def test_set_up_kspace() -> None:
    """Test set_up_kspace function."""
    # Check that k-space values are correct
    fov = 1
    num_samples = 16
    _, k_values = set_up_kspace(fov, num_samples)
    assert k_values[0] == -8
    assert k_values[-1] == 7

    # Check that x- and k-values match when k-space FOV passed in
    fov = 2
    x_values_1, k_values_1 = set_up_kspace(fov, num_samples)
    k_fov = k_values_1.max() - k_values_1.min()
    k_values_2, x_values_2 = set_up_kspace(k_fov, num_samples)
    assert np.isclose(x_values_1, x_values_2).all()
    assert np.isclose(k_values_1, k_values_2).all()
