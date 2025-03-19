"""Proximal gradient solvers e.g. ISTA and ADMM."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from common.constants import MAX_ITER, TOL
from common.log import get_logger
from common.utils import OperatorType, kernel_to_func

logger = get_logger(__name__)


class ProxGradSolver(ABC):
    """Abstract class for proximal gradient solvers."""

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
        shrinkage: Callable[..., npt.NDArray] | None,
        x0: npt.NDArray | None,
    ) -> tuple[Callable[[npt.NDArray], npt.NDArray], npt.NDArray]:
        """Prepare regularisation matrix and initial guess.

        Args:
            shrinkage: Shrinkage function
            x0: Initial guess

        Returns
        -------
            Tuple: Regularisation matrix and initial guess

        """

        def identity(x: npt.NDArray) -> npt.NDArray:
            """Identity operator when sparsity not required."""
            return x

        # Identity if shrinkage not specified, else apply parameters
        if shrinkage is None:
            shrinkage = identity

        if x0 is None:
            x0 = np.zeros_like(self._b)

        return shrinkage, x0

    @abstractmethod
    def solve(
        self,
        shrinkage_func: Callable[[npt.NDArray], npt.NDArray] | None = None,
        params: dict[str, float] | None = None,
        x0: npt.NDArray | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Solve the inverse problem.

        Args:
            shrinkage_func: Shrinkage function
            params: Parameters for the shrinkage function
            x0: Initial guess
            verbose: Print status
            **kwargs: Additional keyword arguments

        Returns
        -------
            npt.NDArray: Solution

        """
        raise NotImplementedError


class ISTASolver(ProxGradSolver):
    """Iterative shrinkage thresholding solver."""

    def solve(
        self,
        shrinkage_func: Callable[[npt.NDArray], npt.NDArray] | None = None,
        params: dict[str, float] | None = None,
        x0: npt.NDArray | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Solve the inverse problem.

        Notes:
            - Performs the update step: f = Sα,W (f - λAT (Af - g))
            - Shrinkage function Sα,W (f) converts image to e.g. wavelet domain,
              thresholds at level α and converts back to spatial domain.

        Args:
            shrinkage_func: Shrinkage function
            params: Parameters for the shrinkage function
            x0: Initial guess
            verbose: Print status
            **kwargs: Additional keyword arguments
                max_iter: Maximum number of iterations
                tol: Tolerance for convergence

        Returns
            npt.NDArray: Solution
        """
        if params is None:
            params = {}
        max_iter: int = kwargs.get("max_iter", MAX_ITER)
        tol: float = kwargs.get("tol", TOL)
        lambda_ = params.pop("lambda")

        # Scale threshold by λ (µ = αλ)
        if "threshold" in params:
            params["threshold"] *= lambda_

        shrinkage, x0 = self._prepare(shrinkage_func, x0)
        x_hat = x0.copy().flatten()

        # Run ISTA
        for it in range(max_iter):
            # Get previous residual norm
            prev_residual = self._b - self._A(x_hat)
            prev_residual_norm = np.square(prev_residual).sum()

            # Gradient step followed by shrinkage
            x_hat -= lambda_ * self._AT(self._A(x_hat) - self._b)
            x_hat = shrinkage(x_hat, **params)

            # Check for convergence
            residual = self._b - self._A(x_hat)
            residual_norm = np.square(residual).sum()
            if np.abs(residual_norm - prev_residual_norm) / prev_residual_norm < tol:
                break

        if it + 1 == max_iter:
            logger.warning("Did not converge")
        elif verbose:
            logger.info(f"Converged in {it + 1} iterations")

        return x_hat
