from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.signal import convolve2d

from common.operators import ConvolutionMode


class Solver(ABC):
    """Abstract class for iterative solvers."""

    _b: npt.NDArray  # Blurred image
    _kernel: Callable[[Any], npt.NDArray]  # Forward operator

    def __init__(
        self,
        b: npt.NDArray,
        kernel: Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
        | npt.NDArray
        | sp.csr_matrix,
    ):
        """Initialise Solver class.

        Args:
            b: Blurred image
            kernel: Blurring kernel (function, numpy array or sparse matrix)

        """
        self._b = b
        self._dims = b.shape
        self._flat_dims = np.prod(self._dims)
        self._kernel = self._set_kernel(kernel)

    def _set_kernel(
        self,
        kernel: Callable[[Any], npt.NDArray] | npt.NDArray | sp.csr_matrix,
    ) -> Callable[[npt.NDArray], npt.NDArray]:
        """Convert the kernel to a callable function.

        Args:
        ----
            kernel: Blurring kernel in function, numpy array or sparse matrix form

        Returns
        -------
            Callable: Kernel as a callable function

        """
        if isinstance(kernel, np.ndarray):
            return partial(convolve2d, in2=kernel, mode=ConvolutionMode.SAME)

        if isinstance(kernel, sp.csr_matrix):
            return lambda x: kernel @ x.flatten()

        return kernel

    def _prepare(
        self,
        L: sp.csr_matrix | None = None,
        x0: npt.NDArray | None = None,
    ) -> tuple[sp.csr_matrix, npt.NDArray]:
        """Prepare regularisation matrix and initial guess.

        Args:
            L: Regularisation matrix
            x0: Initial guess

        Returns
        -------
            Tuple: Regularisation matrix and initial guess

        """
        if L is None:
            L = sp.eye(self._flat_dims)

        if x0 is None:
            x0 = np.zeros(self._flat_dims)

        return L, x0

    @abstractmethod
    def solve(
        self,
        alpha: float,
        L: sp.csr_matrix | None = None,
        x0: npt.NDArray | None = None,
        **kwargs: dict[str, Any],
    ) -> npt.NDArray:
        """Solve the inverse problem.

        Args:
            alpha: Regularisation parameter
            L: Regularisation matrix
            x0: Initial guess
            **kwargs: Additional keyword arguments

        Returns
        -------
            npt.NDArray: Solution

        """
        raise NotImplementedError

    def calc_tikhonov_term(
        self,
        x: npt.NDArray,
        L: sp.csr_matrix,
    ) -> float:
        """Calculate the Tikhonov functional.

        Args:
            x: Solution
            L: Regularisation matrix

        Returns
        -------
            float: Tikhonov functional value

        """
        return float(np.square(L @ x.reshape([-1, 1])).sum() / 2)


class GMRESSolver(Solver):
    """GMRES iterative solver."""

    def __init__(
        self,
        b: npt.NDArray,
        kernel: Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
        | npt.NDArray
        | sp.csr_matrix,
    ):
        """Initialise GMRES Solver class.

        Args:
            b: Blurred image
            kernel: Blurring kernel (function, numpy array or sparse matrix)
        """
        super().__init__(b, kernel)

    def ATA_op(
        self,
        x_flat: npt.NDArray,
        alpha: float,
        LTL: sp.csr_matrix,
    ) -> npt.NDArray:
        """ATA operator for the normal equations.

        Args:
            x_flat: Flattened current solution
            alpha: Regularisation parameter
            LTL: Regularisation matrix tranposed and multipled with self

        Returns:
            NDArray: vectorised result of (A^T A + ɑL^T L) x
        """
        x_flat = x_flat.reshape([-1, 1])  # Required to prevent OOM issues
        x = x_flat.reshape(self._dims)
        x = self._kernel(self._kernel(x))  # A^T A x
        reg_term = (LTL @ x_flat) * alpha  # ɑL^T L x

        return x.reshape([-1, 1]) + reg_term

    def ATb_op(self) -> npt.NDArray:
        """ATb operator for the normal equations.

        Returns:
            NDArray: vectorised result of A^T b
        """
        return self._kernel(self._b).reshape([-1, 1])

    def solve(
        self,
        alpha: float,
        L: sp.csr_matrix | None = None,
        x0: npt.NDArray | None = None,
        **kwargs: dict[str, Any],
    ) -> npt.NDArray:
        """Solve the inverse problem.

        Args:
            alpha: Regularisation parameter
            L: Regularisation matrix
            x0: Initial guess
            **kwargs: Additional keyword arguments

        Returns
            npt.NDArray: Solution
        """
        L, x0 = self._prepare(L, x0)
        LTL = L.T @ L

        # Set up sparse operators
        ATA = sp.linalg.LinearOperator(
            shape=(self._flat_dims, self._flat_dims),
            matvec=lambda x: self.ATA_op(x, alpha=alpha, LTL=LTL),
        )
        ATb = self.ATb_op()

        # Run GMRES
        x_hat, res = sp.linalg.gmres(A=ATA, b=ATb, x0=x0, **kwargs)

        if res == 0:
            print("Successfully converged")
            return x_hat.reshape(self._dims)

        else:
            print("Did not converge")
            return x_hat.reshape(self._dims)


class LSQRSolver(Solver):
    """LSQR iterative solver."""

    def __init__(
        self,
        b: npt.NDArray,
        kernel: Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
        | npt.NDArray
        | sp.csr_matrix,
    ):
        """Initialise LSQR Solver class.

        Args:
            b: Blurred image
            kernel: Blurring kernel (function, numpy array or sparse matrix)
        """
        super().__init__(b, kernel)

    def A_op(
        self,
        x_flat: npt.NDArray,
        alpha: float,
        L: sp.csr_matrix,
    ) -> npt.NDArray:
        """Augmented operator for least squares.

        Notes:
            - Calculates [A ] x
                         [ɑL]
        Args:
            x_flat: Flattened current solution
            alpha: Regularisation parameter
            L: Sparse regularisation matrix

        Returns:
            NDArray: result of above calculation
        """
        x_flat = x_flat.reshape([-1, 1])  # Required to prevent OOM issues
        x = x_flat.reshape(self._dims)
        x = self._kernel(x).reshape([-1, 1])  # A x
        reg_term = (L @ x_flat) * np.sqrt(alpha)  # sqrt(ɑ)L x

        return np.vstack([x, reg_term])

    def AT_op(
        self,
        b_flat: npt.NDArray,
        alpha: float,
        L: sp.csr_matrix,
    ) -> npt.NDArray:
        """Transposed augmented operator for least squares.

        Notes:
            - Calculates [A^T ɑL^T] [b]
                                    [0]
        Args:
            b_flat: Flattened blurred image
            alpha: Regularisation parameter
            L: Sparse regularisation matrix
        """
        b_flat = b_flat.reshape([-1, 1])  # Required to prevent OOM issues
        b = b_flat[0 : self._flat_dims].reshape(self._dims)
        b_zeros = b_flat[self._flat_dims :]
        b = self._kernel(b).reshape([-1, 1])  # A^T b
        reg_term = L.T @ b_zeros * np.sqrt(alpha)  # sqrt(ɑ)L^T 0

        return b + reg_term

    def solve(
        self,
        alpha: float,
        L: sp.csr_matrix | None = None,
        x0: npt.NDArray | None = None,
        **kwargs: dict[str, Any],
    ) -> npt.NDArray:
        """Solve the inverse problem.

        Args:
            alpha: Regularisation parameter
            L: Regularisation matrix
            x0: Initial guess
            **kwargs: Additional keyword arguments

        Returns
            npt.NDArray: Solution
        """
        L, x0 = self._prepare(L, x0)

        b_flat = np.reshape(self._b, [-1, 1])
        b_aug = np.vstack([b_flat, np.zeros([L.shape[0], 1])])

        A = sp.linalg.LinearOperator(
            shape=(self._flat_dims + L.shape[0], self._flat_dims),
            matvec=lambda x: self.A_op(x, alpha=alpha, L=L),
            rmatvec=lambda x: self.AT_op(x, alpha=alpha, L=L),
        )

        lsqr_output = sp.linalg.lsqr(A=A, b=b_aug, x0=x0, **kwargs)
        f_hat = lsqr_output[0]
        it = lsqr_output[2]

        print(f"Converged in {it} iterations")
        return f_hat.reshape(self._dims)
