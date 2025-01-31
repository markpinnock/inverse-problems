"""Least square type solvers: GMRES and LSQR."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Union

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from common.log import get_logger
from common.utils import kernel_to_func

logger = get_logger(__name__)

MAX_ITER = 100

OperatorType = Union[Callable[[npt.NDArray], npt.NDArray], npt.NDArray, sp.csr_matrix]


class LstSqSolver(ABC):
    """Abstract class for iterative least squares-type solvers."""

    _b: npt.NDArray  # Blurred image
    _b_dims: tuple[int, int]  # Dimensions of the blurred image
    _flat_b_dims: int  # Dimensions of the flattened blurred image
    _A: Callable[[Any], npt.NDArray]  # Forward operator
    _AT: Callable[[Any], npt.NDArray]  # Adjoint operator
    _x_dims: tuple[int, int]  # Dimensions of the solution
    _flat_x_dims: int  # Dimensions of the flattened solution

    def __init__(
        self,
        b: npt.NDArray,
        A: OperatorType,
        AT: OperatorType | None = None,
        x_dims: tuple[int, int] | None = None,
    ):
        """Initialise Solver class.

        Args:
            b: Blurred image
            A: Forward operator (function, numpy array or sparse matrix)
            AT: Adjoint operator (function, numpy array or sparse matrix)
            x_dims: Dimensions of the solution

        """
        self._b = b
        self._b_dims = b.shape
        self._A = kernel_to_func(A)
        self._AT = kernel_to_func(AT) if AT is not None else self._A
        if x_dims is None:
            self._x_dims = b.shape
        else:
            self._x_dims = x_dims
        self._flat_b_dims = np.prod(self._b_dims)
        self._flat_x_dims = np.prod(self._x_dims)

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
            L = sp.eye(self._flat_x_dims)

        if x0 is None:
            x0 = np.zeros(self._flat_x_dims)

        return L, x0

    @abstractmethod
    def solve(
        self,
        alpha: float,
        L: sp.csr_matrix | None = None,
        x0: npt.NDArray | None = None,
        verbose: bool = True,
        **kwargs: dict[str, Any],
    ) -> npt.NDArray:
        """Solve the inverse problem.

        Args:
            alpha: Regularisation parameter
            L: Regularisation matrix
            x0: Initial guess
            verbose: Print status
            **kwargs: Additional keyword arguments

        Returns
        -------
            npt.NDArray: Solution

        """
        raise NotImplementedError


class GMRESSolver(LstSqSolver):
    """GMRES iterative solver."""

    def __init__(
        self,
        b: npt.NDArray,
        A: OperatorType,
        AT: OperatorType | None = None,
        x_dims: tuple[int, int] | None = None,
    ):
        """Initialise GMRES Solver class.

        Args:
            b: Blurred image
            A: Forward operator (function, numpy array or sparse matrix)
            AT: Adjoint operator (function, numpy array or sparse matrix)
            x_dims: Dimensions of the solution
        """
        super().__init__(b, A, AT, x_dims)

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
        x = x_flat.reshape(self._x_dims)
        x = self._AT(self._A(x))  # A^T A x
        reg_term = (LTL @ x_flat) * alpha  # ɑL^T L x

        return x.reshape([-1, 1]) + reg_term

    def ATb_op(self) -> npt.NDArray:
        """ATb operator for the normal equations.

        Returns:
            NDArray: vectorised result of A^T b
        """
        return self._AT(self._b).reshape([-1, 1])

    def solve(
        self,
        alpha: float,
        L: sp.csr_matrix | None = None,
        x0: npt.NDArray | None = None,
        verbose: bool = True,
        **kwargs: dict[str, Any],
    ) -> npt.NDArray:
        """Solve the inverse problem.

        Args:
            alpha: Regularisation parameter
            L: Regularisation matrix
            x0: Initial guess
            verbose: Print status
            **kwargs: Additional keyword arguments

        Returns
            npt.NDArray: Solution
        """
        L, x0 = self._prepare(L, x0)
        LTL = L.T @ L

        # Set up sparse operators
        ATA = sp.linalg.LinearOperator(
            shape=(self._flat_x_dims, self._flat_x_dims),
            matvec=lambda x: self.ATA_op(x, alpha=alpha, LTL=LTL),
        )
        ATb = self.ATb_op()

        # Run GMRES
        x_hat, res = sp.linalg.gmres(A=ATA, b=ATb, x0=x0, **kwargs)

        if res == 0 and verbose:
            logger.info("Successfully converged")
        elif res != 0:
            logger.warning("Did not converge")

        return x_hat.reshape(self._x_dims)


class LSQRSolver(LstSqSolver):
    """LSQR iterative solver."""

    def __init__(
        self,
        b: npt.NDArray,
        A: OperatorType,
        AT: OperatorType | None = None,
        x_dims: tuple[int, int] | None = None,
    ):
        """Initialise LSQR Solver class.

        Args:
            b: Blurred image
            A: Forward operator (function, numpy array or sparse matrix)
            AT: Adjoint operator (function, numpy array or sparse matrix)
            x_dims: Dimensions of the solution
        """
        super().__init__(b, A, AT, x_dims)

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
        x = x_flat.reshape(self._x_dims)
        x = self._A(x).reshape([-1, 1])  # A x
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
        b = b_flat[0 : self._flat_b_dims].reshape(self._b_dims)
        b_zeros = b_flat[self._flat_b_dims :]
        b = self._AT(b).reshape([-1, 1])  # A^T b
        reg_term = L.T @ b_zeros * np.sqrt(alpha)  # sqrt(ɑ)L^T 0

        return b + reg_term

    def solve(
        self,
        alpha: float,
        L: sp.csr_matrix | None = None,
        x0: npt.NDArray | None = None,
        verbose: bool = True,
        **kwargs: dict[str, Any],
    ) -> npt.NDArray:
        """Solve the inverse problem.

        Args:
            alpha: Regularisation parameter
            L: Regularisation matrix
            x0: Initial guess
            verbose: Print status
            **kwargs: Additional keyword arguments for LSQR (e.g. iterlim)

        Returns
            npt.NDArray: Solution
        """
        if "iter_lim" not in kwargs:
            kwargs["iter_lim"] = MAX_ITER  # type: ignore[assignment]

        L, x0 = self._prepare(L, x0)

        b_flat = np.reshape(self._b, [-1, 1])
        b_aug = np.vstack([b_flat, np.zeros([L.shape[0], 1])])

        A = sp.linalg.LinearOperator(
            shape=(self._flat_b_dims + L.shape[0], self._flat_x_dims),
            matvec=lambda x: self.A_op(x, alpha=alpha, L=L),
            rmatvec=lambda b: self.AT_op(b, alpha=alpha, L=L),
        )

        lsqr_output = sp.linalg.lsqr(A=A, b=b_aug, x0=x0, show=verbose, **kwargs)
        f_hat = lsqr_output[0]
        it = lsqr_output[2]

        if verbose and it <= MAX_ITER:
            logger.info(f"Converged in {it} iterations")
        elif it > MAX_ITER:
            logger.warning("Did not converge")

        return f_hat.reshape(self._x_dims)
