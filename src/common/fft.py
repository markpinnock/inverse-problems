import numpy as np
import numpy.typing as npt

from common.log import get_logger
from common.utils import get_min_max

logger = get_logger(__name__)


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

    # Ensure regular grid
    dx = np.diff(x_values)
    if not np.isclose(dx, dx[0]).all():
        raise ValueError(
            "Function only defined for regular spatial/temporal coordinates",
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

    # Ensure regular grid
    dk = np.diff(k_values)
    if not np.isclose(dk, dk[0]).all():
        raise ValueError(
            "Function only defined for regular spatial/temporal frequency coordinates",
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


def fft_2d(
    image: npt.NDArray,
    x_grid: npt.NDArray | None = None,
    y_grid: npt.NDArray | None = None,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Get FFT of 2D image.

    Args:
        image: pixel intensities at each x_value
        x_grid: x grid points
        y_grid: y grid points

    Returns:
        Tuple: k-space grid values and FFT of image
    """
    num_y, num_x = image.shape

    if x_grid is None or y_grid is None:
        min_x, max_x = get_min_max(num_x)
        min_y, max_y = get_min_max(num_y)
        x_grid, y_grid = np.meshgrid(
            np.linspace(min_x, max_x, num_x),
            np.linspace(min_y, max_y, num_y),
        )

    if image.shape != x_grid.shape and image.shape != y_grid.shape:
        raise ValueError(
            f"Image and x/y grid must have same shape: {image.shape} vs. {x_grid.shape}  vs. {y_grid.shape}",
        )

    # Ensure regular grid
    dx = np.diff(x_grid)
    dy = np.diff(y_grid)
    if not np.isclose(dx, dx[0]).all() or not np.isclose(dy, dy[0]).all():
        raise ValueError("Function only defined for regular spatial coordinates")

    # Generate array of k-space values and FFT
    kx_values = np.fft.fftshift(np.fft.fftfreq(num_x, x_grid[0, 1] - x_grid[0, 0]))
    ky_values = np.fft.fftshift(np.fft.fftfreq(num_y, y_grid[1, 0] - y_grid[0, 0]))
    kx_grid, ky_grid = np.meshgrid(kx_values, ky_values)
    image_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))

    if len(ky_values) != image_fft.shape[0] or len(kx_values) != image_fft.shape[1]:
        raise ValueError(
            f"K-space and FFT array dims must match: {kx_grid.shape} vs. {ky_grid.shape} vs. {image_fft.shape}",
        )

    return image_fft, kx_grid, ky_grid


def ifft_2d(
    image_fft: npt.NDArray[np.complex128],
    kx_grid: npt.NDArray[np.float64],
    ky_grid: npt.NDArray[np.float64],
) -> tuple[np.float64, np.float64, np.float64]:
    """Get inverse FFT of 2D image.

    Args:
        image_fft: frequency values at each k-value
        kx_grid: k-space x grid points
        ky_grid: k-space y grid points
    """
    num_y, num_x = image_fft.shape

    if kx_grid is None or ky_grid is None:
        min_kx, max_kx = get_min_max(num_x)
        min_ky, max_ky = get_min_max(num_y)
        kx_grid, ky_grid = np.meshgrid(
            np.linspace(min_kx, max_kx, num_x),
            np.linspace(min_ky, max_ky, num_y),
        )

    if image_fft.shape != kx_grid.shape and image_fft.shape != ky_grid.shape:
        raise ValueError(
            f"Image and x/y grid must have same shape: {image_fft.shape} vs. {kx_grid.shape}  vs. {ky_grid.shape}",
        )

    # Ensure regular grid
    dkx = np.diff(kx_grid)
    dky = np.diff(ky_grid)
    if not np.isclose(dkx, dkx[0]).all() or not np.isclose(dky, dky[0]).all():
        raise ValueError(
            "Function only defined for regular spatial frequency coordinates",
        )

    # Generate array of spatial grid values and original image
    x_values = np.fft.fftshift(np.fft.fftfreq(num_x, kx_grid[0, 1] - kx_grid[0, 0]))
    y_values = np.fft.fftshift(np.fft.fftfreq(num_y, ky_grid[1, 0] - ky_grid[0, 0]))
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(image_fft)))

    if len(y_values) != image.shape[0] or len(x_values) != image.shape[1]:
        raise ValueError(
            f"Spatial and image array dims must match: {x_grid.shape} vs. {y_grid.shape} vs. {image.shape}",
        )

    if not np.isclose(image.imag.sum(), 0.0):
        logger.warning("Result has imaginary component")

    return image.real, x_grid, y_grid
