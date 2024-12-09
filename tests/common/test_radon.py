import pytest

from common.radon import iradon, radon
from common.utils import rect_phantom


@pytest.mark.parametrize(  # type: ignore[misc]
    ("img_dims", "sino_dims"),
    [
        ((16, 16), (16, 8)),
        ((16, 16), (16, 12)),
        ((24, 16), (16, 8)),
        ((24, 16), (16, 12)),
    ],
)
def test_radon_iradon_dims(
    img_dims: tuple[int, int],
    sino_dims: tuple[int, int],
) -> None:
    """Test Radon and inverse Radon transform output dimensions."""
    img = rect_phantom(img_dims=(32, 32), phantom_dims=img_dims)
    sino = radon(
        img=img,
        views=sino_dims[0],
        angle=180,
        detector_count=sino_dims[1],
    )
    assert sino.shape == sino_dims

    img_hat = iradon(
        sinogram=sino,
        img_dims=img_dims,
        views=sino_dims[0],
        angle=180,
        detector_count=sino_dims[1],
    )
    assert img_hat.shape == img_dims
