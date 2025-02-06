from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

from common.constants import MAX_ITER, TOL
from common.log import get_logger
from common.operators import ConvolutionMode
from common.utils import OperatorType
from evaluation.eval_metrics import Metrics
from solvers.least_square import LstSqSolver

logger = get_logger(__name__)


class Tuner(ABC):
    """Abstract class for tuning regularisation hyper-parameters.

    Notes:
        This determines the optimal alpha using the discrepancy principle.
    """

    _alphas: list[float]
    _f_hats: dict[float, npt.NDArray[np.float64]]
    _metrics: Metrics
    _optimal_f_hat: npt.NDArray[np.float64]

    def __init__(
        self,
        solver: type[LstSqSolver],
        g: npt.NDArray,
        A: OperatorType,
        AT: OperatorType | None = None,
        x_dims: tuple[int, int] | None = None,
        tuning_metric: str = "discrepancy",
        use_miller: bool = False,
        noise_variance: float | None = None,
        f: npt.NDArray | None = None,
    ) -> None:
        """Initialise Tuner class.

        Args:
            g: Blurred image
            A: Forward operator (function, numpy array or sparse matrix)
            AT: Adjoint operator (function, numpy array or sparse matrix)
            solver: Solver class for deblurring
            x_dims: Dimensions of the solution
            tuning_metric: Metric used for tuning
            use_miller: Use Miller criterion for tuning
            noise_variance: Variance of noise in image if available
            f: Ground truth image if available
        """
        self._g = g
        self._kernel = A
        self._solver = solver(g, A, AT, x_dims)
        self._f = f
        self._metrics = Metrics(tuning_metric, use_miller, noise_variance, f)

    @abstractmethod
    def parameter_sweep(
        self,
        alphas: list[float],
        L_func: Callable[..., sp.csr_matrix | npt.NDArray],
        x0: npt.NDArray | None = None,
        save_imgs: bool = False,
        **kwargs: Any,
    ) -> None:
        """Perform a parameter sweep over a range of regularisation hyper-parameters.

        Args:
            alphas: List of regularisation hyper-parameters
            L_func: Sparse regularisation matrix or function returning it
            x0: Initial guess
            save_imgs: Cache deblurred images for each hyper-parameter
        """
        raise NotImplementedError

    def display_sample(
        self,
        y_slice: slice = slice(None, None),
        x_slice: slice = slice(None, None),
    ) -> None:
        """Display input, reference and deblurred images for each hyper-parameter.

        Args:
            y_slice: Slice for y-axis
            x_slice: Slice for x-axis
        """
        idx = 0
        f_hat = list(self._f_hats.items())
        num_img = len(f_hat) + 2
        nrows = np.floor(np.sqrt(num_img)).astype(int)
        ncols = np.ceil(num_img / nrows).astype(int)

        _, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        axs = axs.ravel()
        axs[0].imshow(self._g[y_slice, x_slice], cmap="gray")
        axs[0].set_title("Input")
        axs[0].axis("off")

        for idx in range(1, num_img - 1):
            alpha, img = f_hat[idx - 1]
            axs[idx].imshow(img[y_slice, x_slice], cmap="gray")
            axs[idx].set_title(f"Alpha {alpha}")
            axs[idx].axis("off")

        if self._f is not None:
            axs[-1].imshow(self._f[y_slice, x_slice], cmap="gray")
            axs[-1].set_title("Reference")
        axs[-1].axis("off")

        plt.tight_layout()
        plt.show()

    def display_metrics(self) -> None:
        """Display metrics for each regularisation hyper-parameter."""
        self._metrics.display_metrics()

    @property
    def optimal_alpha(self) -> Any:
        return self._metrics.optimal_alpha

    @property
    def optimal_f_hat(self) -> npt.NDArray[np.float64]:
        return self._optimal_f_hat

    @property
    def metrics_df(self) -> pd.DataFrame:
        return self._metrics.metrics_df

    @property
    def optimal_metrics(self) -> pd.Series:
        return self._metrics.optimal_metrics


class StandardTuner(Tuner):
    """Solver class for tuning regularisation hyper-parameters."""

    def parameter_sweep(
        self,
        alphas: list[float],
        L_func: Callable[..., sp.csr_matrix | npt.NDArray],
        x0: npt.NDArray | None = None,
        save_imgs: bool = False,
        **kwargs: Any,
    ) -> None:
        """Perform a parameter sweep over a range of regularisation hyper-parameters.

        Notes:
            - L_func should be a function returning a sparse matrix
            - These can be found in src.common.operators

        Args:
            alphas: List of regularisation hyper-parameters
            L_func: Function returning sparse regularisation matrix
            x0: Initial guess
            save_imgs: Cache deblurred images for each hyper-parameter
        """
        L = L_func(self._g, conv_mode=ConvolutionMode.PERIODIC)
        self._metrics.reset_metrics()
        self._alphas = []
        self._f_hats = {}

        # Solve for each alpha
        for alpha in alphas:
            self._alphas.append(alpha)
            f_hat = self._solver.solve(alpha=alpha, L=L, x0=x0, verbose=False)

            # Calculate metrics
            L = L_func(f_hat, conv_mode=ConvolutionMode.PERIODIC)
            residual = self._g - self._kernel(f_hat)
            self._metrics.calculate_metrics(
                alpha=alpha,
                residual=residual,
                f_hat=f_hat,
                penalty_term=L @ f_hat.reshape([-1, 1]),
            )
            if save_imgs:
                self._f_hats[alpha] = f_hat
            self._metrics.log_metrics(alpha)

        # Calculate optimal alpha and solve
        self._metrics.get_optimal_alpha()
        self._optimal_f_hat = self._solver.solve(
            alpha=self._metrics.optimal_alpha,
            L=L,
            x0=x0,
            verbose=False,
        )
        L = L_func(self._optimal_f_hat, conv_mode=ConvolutionMode.PERIODIC)
        residual = self._g - self._kernel(self._optimal_f_hat)
        self._metrics.calculate_metrics(
            alpha="optimal",
            residual=residual,
            f_hat=self._optimal_f_hat,
            penalty_term=L @ self._optimal_f_hat.reshape([-1, 1]),
        )
        self._metrics.log_metrics("optimal")


class IterativeTuner(Tuner):
    """Iterative solver class for tuning regularisation hyper-parameters."""

    def iteratively_solve(
        self,
        alpha: float,
        L_func: Callable[..., sp.csr_matrix],
        x0: npt.NDArray,
        max_iter: int,
        tol: float,
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Iteratively solve the inverse problem.

        Notes:
            - L_func should be a function returning a sparse matrix
            - These can be found in src.common.operators

        Args:
            alpha: Regularisation parameter
            L_func: Function creating sparse matrix
            x0: Initial guess
            max_iter: Maximum number of iterations
            tol: Convergence tolerance

        Returns
        -------
            Tuple: Deblurred image and number of iterations
        """
        prev_f_hat = x0.copy()

        for it in range(max_iter):
            # Get previous residual norm
            prev_residual = self._g - self._kernel(prev_f_hat)
            prev_residual_norm = np.square(prev_residual).sum()

            # Re-initialise regularisation matrix with the last predicted image
            L = L_func(prev_f_hat, conv_mode=ConvolutionMode.PERIODIC)
            f_hat = self._solver.solve(
                alpha=alpha,
                L=L,
                x0=prev_f_hat,
                verbose=False,
            )
            prev_f_hat = f_hat

            # Check for convergence
            residual = self._g - self._kernel(f_hat)
            residual_norm = np.square(residual).sum()
            if np.abs(residual_norm - prev_residual_norm) / prev_residual_norm < tol:
                break

        return f_hat, it

    def parameter_sweep(
        self,
        alphas: list[float],
        L_func: Callable[..., sp.csr_matrix | npt.NDArray],
        x0: npt.NDArray | None = None,
        save_imgs: bool = False,
        **kwargs: Any,
    ) -> None:
        """Perform a parameter sweep over a range of regularisation hyper-parameters.

        Notes:
            - L_func should be a function returning a sparse matrix
            - These can be found in src.common.operators

        Args:
            alphas: List of regularisation hyper-parameters
            L_func: Sparse regularisation matrix or function returning it
            x0: Initial guess
            save_imgs: Cache deblurred images for each hyper-parameter
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        max_iter: int = kwargs.get("max_iter", MAX_ITER)
        tol: float = kwargs.get("tol", TOL)

        self._metrics.reset_metrics()
        self._alphas = []
        self._f_hats = {}

        if x0 is None:
            x0 = self._g.copy()

        # Solve for each alpha with blurred image as starting guess
        for alpha in alphas:
            self._alphas.append(alpha)
            f_hat, it = self.iteratively_solve(
                alpha=alpha,
                L_func=L_func,
                x0=x0,
                max_iter=max_iter,
                tol=tol,
            )

            # Calculate metrics
            L = L_func(f_hat, conv_mode=ConvolutionMode.PERIODIC)
            residual = self._g - self._kernel(f_hat)
            self._metrics.calculate_metrics(
                alpha=alpha,
                residual=residual,
                f_hat=f_hat,
                penalty_term=L @ f_hat.reshape([-1, 1]),
            )
            if save_imgs:
                self._f_hats[alpha] = f_hat

            if it + 1 == max_iter:
                logger.warning(
                    f"Alpha {alpha}: Maximum iterations reached",
                )
            else:
                self._metrics.log_metrics(alpha)

        # Get optimal alpha and iteratively solve
        self._metrics.get_optimal_alpha()
        self._optimal_f_hat, _ = self.iteratively_solve(
            alpha=self._metrics.optimal_alpha,
            L_func=L_func,
            x0=x0,
            max_iter=max_iter,
            tol=tol,
        )
        L = L_func(self._optimal_f_hat, conv_mode=ConvolutionMode.PERIODIC)
        residual = self._g - self._kernel(self._optimal_f_hat)
        self._metrics.calculate_metrics(
            alpha="optimal",
            residual=residual,
            f_hat=self._optimal_f_hat,
            penalty_term=L @ self._optimal_f_hat.reshape([-1, 1]),
        )
        self._metrics.log_metrics("optimal")
