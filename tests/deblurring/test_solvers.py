from functools import partial

import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse as sp
from scipy.signal import convolve2d

from deblurring.solvers import GMRESSolver, LSQRSolver


@pytest.mark.parametrize(("alpha", "expected"), [(0.0, 1.0), (1.0, 0.0), (2.0, -1.0)])  # type: ignore[misc]
def test_GMRESSolver_ATA_op(
    alpha: float,
    expected: float,
    kernel: npt.NDArray[np.float64],
) -> None:
    """Test the ATA operator for GMRESSolver."""
    img = np.ones((4, 4))
    flat_dims = np.prod(img.shape)
    gmres = GMRESSolver(b=np.zeros_like(img), kernel=kernel)

    # Set up operator
    L = sp.eye(flat_dims)
    ATA = partial(gmres.ATA_op, alpha=alpha, LTL=-L.T @ L)
    out = ATA(img.flatten())

    # Test against expected value
    exp_img = expected * np.ones(flat_dims)
    assert np.equal(out, exp_img).all()


def test_GMRESSolver_ATb_op(kernel: npt.NDArray[np.float64]) -> None:
    """Test the ATb operator for GMRESSolver."""
    img = np.ones((4, 4))

    # Test numpy kernel
    gmres = GMRESSolver(b=img, kernel=kernel)
    assert np.equal(gmres.ATb_op(), -img.flatten()).all()

    # Test sparse kernel
    sparse_kernel = -sp.eye(16).tocsr()
    gmres = GMRESSolver(b=img, kernel=sparse_kernel)
    assert np.equal(gmres.ATb_op(), -img.flatten()).all()

    # Test functional kernel
    func_kernel = partial(convolve2d, in2=kernel, mode="same")
    gmres = GMRESSolver(b=img, kernel=func_kernel)
    assert np.equal(gmres.ATb_op(), -img.flatten()).all()


@pytest.mark.parametrize(("alpha"), [0.0, 2.0, 4.0])  # type: ignore[misc]
def test_LSQRSolver_A_op(alpha: float, kernel: npt.NDArray[np.float64]) -> None:
    """Test the A operator for LSQRSolver."""
    img = np.ones((4, 4))
    flat_dims = np.prod(img.shape)
    lsqr = LSQRSolver(b=np.zeros_like(img), kernel=kernel)

    # Set up operator
    L = -sp.eye(flat_dims)
    A = partial(lsqr.A_op, alpha=alpha, L=L)
    out = A(img.flatten())

    # Test against expected value
    exp_reg_term = np.sqrt(alpha) * -np.ones(flat_dims)
    assert np.equal(out[0:flat_dims], -np.ones(flat_dims)).all()
    assert np.equal(out[flat_dims:], exp_reg_term).all()
