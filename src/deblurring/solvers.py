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
