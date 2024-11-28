import numpy as np
import numpy.typing as npt


def get_min_max(num_samples: int) -> tuple[float, float]:
    """Get minimum and maximum for an array.

    Notes:
        - This assumes integer spacing between each element
          and that the array is symmetrical in the odd case
          and nearly symmetrical in the even case

    Args:
        num_samples: Number of elements in the array

    Returns:
        Minimum and maximum element in the array
    """
    if num_samples % 2:
        return -(num_samples // 2), (num_samples // 2)
    else:
        return -num_samples / 2, num_samples / 2 - 1


def set_up_kspace(
    fov: int,
    num_samples: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Set up k-space.

    Notes:
        - This is a utility function for generating k-space from a given field of view
        - Note that k-space FOV can be passed in to return the corresponding x values

    Args:
        fov: Signal/image field of view
        num_samples: Number of samples

    Returns:
        Tuple: signal/image x values corresponding and k-space values
    """
    delta_k = 1 / fov

    # Generate array of k values
    k_min = (-num_samples / 2) * delta_k
    k_max = (num_samples / 2 - 1) * delta_k
    k_values = np.arange(k_min, k_max + delta_k, delta_k)

    # Generate array of x values
    delta_x = 1 / (k_max - k_min)
    x_min = (-num_samples / 2) * delta_x
    x_max = (num_samples / 2 - 1) * delta_x
    x_values = np.arange(x_min, x_max + delta_x, delta_x)

    return x_values, k_values
