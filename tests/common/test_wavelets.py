import numpy as np

from common.wavelets import ThresholdMode, thresholding_1d


def test_thresholding_1d() -> None:
    """Test 1D wavelet thresholding function."""
    wavelets = [np.ones(2), np.ones(4), np.ones(8) * 2, np.ones(16)]

    # Test all levels thresholded
    thresholded = thresholding_1d(
        wavelets,
        threshold=1.5,
        end_level=0,
        mode=ThresholdMode.HARD,
    )
    assert [w.mean() for w in thresholded] == [1.0, 1.0, 2.0, 1.0]

    # Test different levels thresholded
    thresholded = thresholding_1d(
        wavelets,
        threshold=1.5,
        end_level=1,
        mode=ThresholdMode.HARD,
    )
    assert [w.mean() for w in thresholded] == [1.0, 1.0, 2.0, 0.0]
    thresholded = thresholding_1d(
        wavelets,
        threshold=1.5,
        end_level=2,
        mode=ThresholdMode.HARD,
    )
    assert [w.mean() for w in thresholded] == [1.0, 1.0, 2.0, 0.0]
    thresholded = thresholding_1d(
        wavelets,
        threshold=1.5,
        end_level=3,
        mode=ThresholdMode.HARD,
    )
    assert [w.mean() for w in thresholded] == [1.0, 0.0, 2.0, 0.0]

    # Test low-res signal not thresholded
    thresholded = thresholding_1d(
        wavelets,
        threshold=1.5,
        end_level=4,
        mode=ThresholdMode.HARD,
    )
    assert [w.mean() for w in thresholded] == [1.0, 0.0, 2.0, 0.0]
