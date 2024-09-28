import numpy as np

from common.operators import identity_operator


def test_identity_operator() -> None:
    """Test identity operator."""
    img = np.random.rand(4, 4)
    operator = identity_operator(img)

    assert operator.shape == (16, 16)
    assert np.allclose(operator.toarray(), np.eye(16))
    assert np.allclose(operator @ img.flatten(), img.flatten())
