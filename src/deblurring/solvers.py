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
        penalty_matrix: sp.csr_matrix | None = None,
        x0: npt.NDArray | None = None,
    ) -> tuple[sp.csr_matrix, npt.NDArray]:
        """Prepare the penalty matrix and initial guess.

        Args:
            penalty_matrix: Regularisation matrix
            x0: Initial guess

        Returns
        -------
            Tuple: Penalty matrix and initial guess

        """
        if penalty_matrix is None:
            penalty_matrix = sp.eye(self._flat_dims)

        if x0 is None:
            x0 = np.zeros(self._flat_dims)

        return penalty_matrix, x0

    @abstractmethod
    def solve(
        self,
        alpha: float,
        tik_func: sp.csr_matrix | None = None,
        x0: npt.NDArray | None = None,
        **kwargs: dict[str, Any],
    ) -> npt.NDArray:
        """Solve the inverse problem.

        Args:
            alpha: Regularisation parameter
            tik_func: Regularisation matrix
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
        penalty_matrix: sp.csr_matrix,
    ) -> float:
        """Calculate the Tikhonov functional.

        Args:
            x: Solution
            penalty_matrix: Regularisation matrix

        Returns
        -------
            float: Tikhonov functional value

        """
        return float(np.square(penalty_matrix @ x.reshape([-1, 1])).sum() / 2)


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
        penalty_matrix: sp.csr_matrix,
    ) -> npt.NDArray:
        """ATA operator for the normal equation.

        Notes:
            - Calculates (A^T A + alpha * L^T L) x

        Args:
            x_flat: Flattened current solution
            alpha: Regularisation parameter
            penalty_matrix: Sparse regularisation matrix
        """
        x_flat = x_flat.reshape([-1, 1])
        x = x_flat.reshape(self._dims)
        x = self._kernel(self._kernel(x))
        penalty_term = penalty_matrix @ x_flat
        penalty_term *= alpha

        return x.reshape([-1, 1]) + penalty_term

    def ATb_op(self) -> npt.NDArray:
        """ATb operator for the normal equation.

        Notes:
            - Calculates A^T b
        """
        return self._kernel(self._b).reshape([-1, 1])

    def solve(
        self,
        alpha: float,
        tik_func: sp.csr_matrix | None = None,
        x0: npt.NDArray | None = None,
        **kwargs: dict[str, Any],
    ) -> npt.NDArray:
        """Solve the inverse problem.

        Args:
            alpha: Regularisation parameter
            tik_func: Regularisation matrix
            x0: Initial guess
            **kwargs: Additional keyword arguments

        Returns
            npt.NDArray: Solution
        """
        penalty_matrix, x0 = self._prepare(tik_func, x0)
        penalty_matrix = penalty_matrix.T @ penalty_matrix

        # Set up sparse operators
        ATA = sp.linalg.LinearOperator(
            shape=(self._flat_dims, self._flat_dims),
            matvec=lambda x: self.ATA_op(x, alpha=alpha, penalty_matrix=penalty_matrix),
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
        penalty_matrix: sp.csr_matrix,
    ) -> npt.NDArray:
        """Augmented operator for least squares.

        Notes:
            - Calculates [A ]
                         [ɑL]
        Args:
            x_flat: Flattened current solution
            alpha: Regularisation parameter
            penalty_matrix: Sparse regularisation matrix
        """
        x_flat = x_flat.reshape([-1, 1])
        x = x_flat.reshape(self._dims)
        x = self._kernel(x).reshape([-1, 1])

        penalty_term = penalty_matrix @ x_flat
        penalty_term *= np.sqrt(alpha)

        return np.vstack([x, penalty_term])

    def AT_op(
        self,
        x_flat: npt.NDArray,
        alpha: float,
        penalty_matrix: sp.csr_matrix,
    ) -> npt.NDArray:
        """Transposed augmented operator for least squares.

        Notes:
            - Calculates [A^T ɑL^T]

        Args:
            x_flat: Flattened current solution
            alpha: Regularisation parameter
            penalty_matrix: Sparse regularisation matrix
        """
        x_flat = x_flat.reshape([-1, 1])
        x = x_flat[0 : self._flat_dims].reshape(self._dims)
        x = self._kernel(x).reshape([-1, 1])

        penalty_term = penalty_matrix.T @ x_flat[self._flat_dims :]
        penalty_term *= np.sqrt(alpha)

        return x + penalty_term

    def solve(
        self,
        alpha: float,
        tik_func: sp.csr_matrix | None = None,
        x0: npt.NDArray | None = None,
        **kwargs: dict[str, Any],
    ) -> npt.NDArray:
        """Solve the inverse problem.

        Args:
            alpha: Regularisation parameter
            tik_func: Regularisation matrix
            x0: Initial guess
            **kwargs: Additional keyword arguments

        Returns
            npt.NDArray: Solution
        """
        penalty_matrix, x0 = self._prepare(tik_func, x0)

        b_flat = np.reshape(self._b, [-1, 1])
        b_aug = np.vstack([b_flat, np.zeros([penalty_matrix.shape[0], 1])])

        A = sp.linalg.LinearOperator(
            shape=(self._flat_dims + penalty_matrix.shape[0], self._flat_dims),
            matvec=lambda x: self.A_op(
                x,
                alpha=alpha,
                penalty_matrix=penalty_matrix,
            ),
            rmatvec=lambda x: self.AT_op(
                x,
                alpha=alpha,
                penalty_matrix=penalty_matrix,
            ),
        )

        lsqr_output = sp.linalg.lsqr(A=A, b=b_aug, x0=x0, **kwargs)
        f_hat = lsqr_output[0]
        it = lsqr_output[2]

        print(f"Converged in {it} iterations")
        return f_hat.reshape(self._dims)
