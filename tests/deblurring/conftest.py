import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture  # type: ignore[misc]
def kernel() -> npt.NDArray[np.float64]:
    return np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
