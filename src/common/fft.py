import numpy as np
import numpy.typing as npt

from common.log import get_logger

logger = get_logger(__name__)


def rect_function_1d(
    width: int | float,
    x_range: tuple[int | float, int | float],
    num_samples: int,
    normalised: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Create 1D rectangle/box function.

    Args:
        width: width of rectangle
        x_range: range of x values
        num_samples: number of points
        normalised: scale rectangle to be of height 1

    Returns:
        y-values and corresponding x-values
    """
    x_values = np.linspace(x_range[0], x_range[1], num_samples)
    y_values = np.zeros_like(x_values)
    height = 1 / width if normalised else 1
    y_values[(x_values >= -width / 2) & (x_values <= width / 2)] = height

    return y_values, x_values


def sinc_function(
    x_min: int | float,
    x_max: int | float,
    num_samples: int,
    normalised: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    x_values = np.linspace(x_min, x_max, num_samples)
    y_values = np.sinc(x_values) if normalised else np.sinc(x_values / np.pi)

    return y_values, x_values


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
    return -(num_samples // 2), (num_samples - 1) // 2


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
    k_min, k_max = get_min_max(num_samples)
    k_min *= delta_k
    k_max *= delta_k
    k_values = np.arange(k_min, k_max + delta_k, delta_k)

    # Generate array of x values
    delta_x = 1 / (k_max - k_min)
    x_min, x_max = get_min_max(num_samples)
    x_min *= delta_x
    x_max *= delta_x
    x_values = np.arange(x_min, x_max + delta_x, delta_x)

    return x_values, k_values


def fft_1d(
    signal: npt.NDArray,
    x_values: npt.NDArray | None = None,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """Get FFT of 1D signal.

    Args:
        signal: signal values at each x_value
        x_values: x values (e.g. space or time points)

    Returns:
        Tuple: k-space values and FFT of signal
    """
    num_samples = len(signal)

    if x_values is None:
        min_x, max_x = get_min_max(num_samples)
        x_values = np.linspace(min_x, max_x, num_samples)

    if len(signal) != len(x_values):
        raise ValueError(
            f"Signal and x values must have same length: {signal.shape} vs. {x_values.shape}",
        )

    # Generate array of k-space values and FFT
    k_values = np.fft.fftshift(np.fft.fftfreq(num_samples, x_values[1] - x_values[0]))
    signal_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(signal)))

    if len(k_values) != len(signal_fft):
        raise ValueError(
            f"K-space and FFT array dims must match: {k_values.shape} vs. {signal_fft.shape}",
        )

    return signal_fft, k_values


def ifft_1d(
    signal_fft: npt.NDArray[np.complex128],
    k_values: npt.NDArray[np.float64],
) -> tuple[np.float64, np.float64]:
    """Get inverse FFT of 1D signal.

    Args:
        signal_fft: frequency values at each k-value
        k_values: k space values (i.e. frequency points)
    """
    num_samples = len(signal_fft)

    if k_values is None:
        min_k, max_k = get_min_max(num_samples)
        k_values = np.linspace(min_k, max_k, num_samples)

    if len(k_values) != len(signal_fft):
        raise ValueError(
            f"FFT and k-space dims must match: {signal_fft.shape} vs. {k_values.shape}",
        )

    x_values = np.fft.fftshift(np.fft.fftfreq(num_samples, k_values[1] - k_values[0]))
    signal = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(signal_fft)))

    if len(x_values) != len(signal):
        raise ValueError(
            f"Signal and x values must have same length: {signal.shape} vs. {x_values.shape}",
        )

    if not np.isclose(signal.imag.sum(), 0.0):
        logger.warning("Result has imaginary component")

    return signal.real, x_values
