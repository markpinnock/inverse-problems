import enum

import numpy as np
import numpy.typing as npt
import pywt


@enum.unique
class ThresholdMode(str, enum.Enum):
    HARD = "hard"
    SOFT = "soft"


def thresholding_1d(
    wavelets: list[npt.NDArray[np.float64]],
    threshold: float,
    end_level: int,
    mode: str,
) -> list[npt.NDArray[np.float64]]:
    """Threshold 1D wavelet decomposition.

    Notes:
        - `end_level` specifies the levels to threshold (1-indexing)
        - If `end_level` is 5 and there are 7 levels, then levels 1-5 are thresholded

    Args:
        wavelets: list of wavelet coefficients
        threshold: threshold value
        end_level: final level to threshold (inclusive)
        mode: `hard` or `soft` thresholding

    Returns:
        Thresholded wavelets
    """
    if mode not in {m.value for m in ThresholdMode}:
        raise ValueError(f"Threshold mode not supported: {mode}")
    num_levels = len(wavelets) - 1
    wavelet_thresh = [wavelets[0]]

    for i in range(1, num_levels + 1):
        if num_levels - i < end_level:
            wavelet_thresh.append(pywt.threshold(wavelets[i], threshold, mode=mode))
        else:
            wavelet_thresh.append(wavelets[i])

    return wavelet_thresh


def thresholding_2d(
    wavelets: list[
        npt.NDArray[np.float64]
        | tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
    ],
    threshold: float,
    end_level: int,
    mode: str,
) -> list[
    npt.NDArray[np.float64]
    | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
]:
    """Threshold 2D wavelet decomposition.

    Notes:
        - `end_level` specifies the levels to threshold (1-indexing)
        - If `end_level` is 5 and there are 7 levels, then levels 1-5 are thresholded

    Args:
        wavelets: list of wavelet coefficients
        threshold: threshold value
        end_level: final level to threshold (inclusive)
        mode: `hard` or `soft` thresholding

    Returns:
        Thresholded wavelets
    """
    if mode not in {m.value for m in ThresholdMode}:
        raise ValueError(f"Threshold mode not supported: {mode}")
    num_levels = len(wavelets) - 1
    wavelet_thresh = [wavelets[0]]

    for i in range(1, num_levels + 1):
        if num_levels - i < end_level:
            wavelet_thresh.append(
                tuple(
                    pywt.threshold(wavelets[i][j], threshold, mode=mode)
                    for j in range(3)
                ),
            )
        else:
            wavelet_thresh.append(wavelets[i])

    return wavelet_thresh
