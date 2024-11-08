from functools import partial

import numpy as np
import scipy.sparse as sp
from scipy.signal import convolve2d

from deblurring.solvers import GMRESSolver


def test_GMRESSolver_ATb_op() -> None:
    """Test the ATb operator for GMRESSolver."""
    np_kernel = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])
    img = np.ones((4, 4))

    # Test numpy kernel
    gmres = GMRESSolver(b=img, kernel=np_kernel)
    assert np.equal(gmres.ATb_op(), -img.flatten()).all()

    # Test sparse kernel
    sparse_kernel = -sp.eye(16).tocsr()
    gmres = GMRESSolver(b=img, kernel=sparse_kernel)
    assert np.equal(gmres.ATb_op(), -img.flatten()).all()

    # Test functional kernel
    func_kernel = partial(convolve2d, in2=np_kernel, mode="same")
    gmres = GMRESSolver(b=img, kernel=func_kernel)
    assert np.equal(gmres.ATb_op(), -img.flatten()).all()
