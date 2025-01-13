import numpy as np
import numpy.typing as npt
import pywt

from common.wavelets import thresholding_1d, thresholding_2d


def signal_l1(
    x: npt.NDArray,
    threshold: float | None = None,
    quantile: float | None = None,
) -> npt.NDArray[np.float64]:
    """Threshold signal in spatial/temporal domain.

    Args:
        x: Input signal.
        threshold: Threshold all coefficients above this value
        quantile: Threshold all coefficients above this quantile

    Returns:
        Thresholded signal
    """
    if threshold is None and quantile is None:
        raise ValueError("Please specify threshold or quantile")
    if threshold is None:
        threshold = np.quantile(np.abs(x), q=quantile)
    return pywt.threshold(x, threshold, mode="soft")


def wavelet_l1(
    x: npt.NDArray,
    threshold: float | None = None,
    quantile: float | None = None,
    end_level: int | None = None,
) -> npt.NDArray:
    """Threshold signal in wavelet domain.

    Args:
        x: Input signal.
        threshold: Threshold all coefficients above this value
        quantile: Threshold all coefficients above this quantile
        end_level: final level to threshold (inclusive)

    Returns:
        Thresholded signal
    """
    if threshold is None and quantile is None:
        raise ValueError("Please specify threshold or quantile")

    if x.ndim == 1:
        num_levels = int(np.log2(x.shape[0])) - 1
    elif x.ndim == 2:
        num_levels = int(np.log2(min(x.shape))) - 1
    else:
        raise ValueError(f"Only 1D or 2D shapes supported, got: {x.shape}")

    if x.ndim == 1:
        wavelets = pywt.wavedec(x, "haar", level=num_levels)
    else:
        wavelets = pywt.wavedec2(x, "haar", level=num_levels)

    coeffs = np.hstack([np.hstack(w).flatten() for w in wavelets[1:]])
    if threshold is None:
        threshold = np.quantile(np.abs(coeffs), q=quantile)

    if x.ndim == 1:
        thresholded_wavelets = thresholding_1d(
            wavelets=wavelets,
            threshold=threshold,
            end_level=end_level if end_level is not None else num_levels,
            mode="soft",
        )
        return pywt.waverec(thresholded_wavelets, "haar")
    else:
        thresholded_wavelets = thresholding_2d(
            wavelets=wavelets,
            threshold=threshold,
            end_level=end_level if end_level is not None else num_levels,
            mode="soft",
        )
        return pywt.waverec2(thresholded_wavelets, "haar")
