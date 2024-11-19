from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

from common.log import get_logger
from common.operators import ConvolutionMode
from deblurring.solvers import Solver
from evaluation.eval_metrics import calc_dicrepancy_principle, calc_miller_criterion

logger = get_logger(__name__)


class Tuner(ABC):
    """Abstract class for tuning regularisation hyper-parameters.

    Notes:
        This determines the optimal alpha using the discrepancy principle.
    """

    _L: sp.csr_matrix
    _alphas: list[float]
    _f_hats: dict[float, npt.NDArray[np.float64]]
    _metrics: pd.DataFrame
    _optimal_alpha: float
    _optimal_f_hat: npt.NDArray[np.float64]

    def __init__(
        self,
        g: npt.NDArray,
        kernel: Callable[[npt.NDArray], npt.NDArray] | npt.NDArray | sp.csr_matrix,
        solver: Solver,
        noise_variance: float | None = None,
        f: npt.NDArray | None = None,
        miller: bool = False,
    ) -> None:
        """Initialise Tuner class.

        Args:
            g: Blurred image
            kernel: Blurring kernel (function, numpy array or sparse matrix)
            solver: Solver class for deblurring
            noise_variance: Variance of noise in image if available
            f: Ground truth image if available
            miller: Use Miller criterion for tuning
        """
        self._g = g
        self._kernel = kernel
        self._solver = solver(g, kernel)
        self._f = f
        self._noise_variance = noise_variance
        self._discrepancy_func = (
            calc_miller_criterion if miller else calc_dicrepancy_principle
        )

    def reset_metrics(self) -> None:
        """Reset metrics dataframe."""
        metric_columns = [
            "MSE",
            "pSNR",
            "SSIM",
            "residual_norm",
            "discrepancy",
            "tikhonov",
        ]
        self._metrics = pd.DataFrame(columns=metric_columns)

    @abstractmethod
    def parameter_sweep(
        self,
        alphas: list[float],
        L: sp.csr_matrix | Callable[[npt.NDArray, str], sp.csr_matrix],
        save_imgs: bool = False,
        **kwargs: Any,
    ) -> None:
        """Perform a parameter sweep over a range of regularisation hyper-parameters.

        Args:
            alphas: List of regularisation hyper-parameters
            L: Sparse regularisation matrix or function returning it
            save_imgs: Cache deblurred images for each hyper-parameter
        """
        raise NotImplementedError

    def get_optimal_alpha(self) -> None:
        """Find the optimal regularisation hyper-parameter."""
        discrepancy_vals = self._metrics.loc[:, "discrepancy"].values
        idx1 = np.argwhere((discrepancy_vals * np.roll(discrepancy_vals, -1)) < 0.0)[0][
            0
        ]
        idx2 = np.argwhere((discrepancy_vals * np.roll(discrepancy_vals, 1)) < 0.0)[1][
            0
        ]
        y1 = discrepancy_vals[idx1]
        y2 = discrepancy_vals[idx2]
        x1 = self._alphas[idx1]
        x2 = self._alphas[idx2]
        ratio = abs(y2 / y1)
        self._optimal_alpha = (x2 + ratio * x1) / (ratio + 1)

    def calculate_metrics(
        self,
        alpha: float | str,
        f_hat: npt.NDArray[np.float64],
    ) -> None:
        """Calculate metrics for a given regularisation hyper-parameter.

        Args:
            alpha: Regularisation hyper-parameter
            f_hat: Deblurred image
        """
        residual = self._g.reshape([-1, 1]) - self._kernel(f_hat).reshape([-1, 1])
        self._metrics.loc[alpha, "residual_norm"] = (residual.T @ residual).squeeze()
        self._metrics.loc[alpha, "tikhonov"] = self._solver.calc_tikhonov_term(
            x=f_hat,
            L=self._L,
        )

        if self._noise_variance is not None:
            self._metrics.loc[alpha, "discrepancy"] = self._discrepancy_func(
                residual,
                self._noise_variance,
            )

        if self._f is not None:
            self._metrics.loc[alpha, "MSE"] = mean_squared_error(self._f, f_hat)
            self._metrics.loc[alpha, "pSNR"] = peak_signal_noise_ratio(
                self._f,
                f_hat,
                data_range=1.0,
            )
            self._metrics.loc[alpha, "SSIM"] = structural_similarity(
                self._f,
                f_hat,
                data_range=1.0,
            )

    def display_metrics(self) -> None:
        """Display metrics for each regularisation hyper-parameter."""
        # Plot discrepancy values against alpha values
        plt.subplot(2, 1, 1)
        plt.semilogx(self._alphas, self._metrics.loc[:, "discrepancy"].iloc[:-1])
        plt.axhline(0, ls="--", c="k")
        plt.axvline(self._optimal_alpha, ls="--", c="k")
        plt.xlabel("Alpha")
        plt.ylabel("Discrepancy")
        plt.title("Discrepancy principle")

        # Plot L-curve (residual norm against Tikhonov term)
        plt.subplot(2, 1, 2)
        plt.loglog(
            self._metrics.loc[:, "tikhonov"].iloc[:-1],
            self._metrics.loc[:, "residual_norm"].iloc[:-1],
        )
        plt.plot(
            self._metrics.loc["optimal", "tikhonov"],
            self._metrics.loc["optimal", "residual_norm"],
            "k+",
        )
        plt.xlabel("Tikhonov Term")
        plt.ylabel("Residual norm")
        plt.title("L-Curve")
        plt.tight_layout()
        plt.show()

    def display_sample(self) -> None:
        """Display input, reference and deblurred images for each hyper-parameter."""
        idx = 0
        f_hat = list(self._f_hats.items())
        num_img = len(f_hat) + 2
        nrows = np.floor(np.sqrt(num_img)).astype(int)
        ncols = np.ceil(num_img / nrows).astype(int)

        _, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        axs = axs.ravel()
        axs[0].imshow(self._g[50:150, 80:180], cmap="gray")
        axs[0].set_title("Input")
        axs[0].axis("off")

        for idx in range(1, num_img - 1):
            alpha, img = f_hat[idx - 1]
            axs[idx].imshow(img[50:150, 80:180], cmap="gray")
            axs[idx].set_title(f"Alpha {alpha}")
            axs[idx].axis("off")

        if self._f is not None:
            axs[-1].imshow(self._f[50:150, 80:180], cmap="gray")
            axs[-1].set_title("Reference")
        axs[-1].axis("off")

        plt.tight_layout()
        plt.show()

    @property
    def optimal_alpha(self) -> float:
        return self._optimal_alpha

    @property
    def optimal_f_hat(self) -> npt.NDArray[np.float64]:
        return self._optimal_f_hat

    @property
    def metrics(self) -> pd.DataFrame:
        return self._metrics

    @property
    def optimal_metrics(self) -> pd.Series:
        return self._metrics.loc["optimal"]

    def __repr__(self) -> str:
        return repr(self._metrics)

    def __str__(self) -> str:
        return str(self._metrics)


class StandardTuner(Tuner):
    """Solver class for tuning regularisation hyper-parameters."""

    def parameter_sweep(
        self,
        alphas: list[float],
        L: sp.csr_matrix | Callable[[npt.NDArray, str], sp.csr_matrix],
        save_imgs: bool = False,
        **kwargs: Any,
    ) -> None:
        """Perform a parameter sweep over a range of regularisation hyper-parameters.

        Args:
            alphas: List of regularisation hyper-parameters
            L: Sparse regularisation matrix or function returning it
            save_imgs: Cache deblurred images for each hyper-parameter
        """
        if not isinstance(L, sp.csr_matrix):
            raise ValueError(
                f"L must be of type `scipy.sparse.csr_matrix`, got: {type(L)}",
            )

        self.reset_metrics()
        self._L = L
        self._alphas = []
        self._f_hats = {}

        for alpha in alphas:
            # Solve for this alpha
            self._alphas.append(alpha)  # ADD X0!
            f_hat = self._solver.solve(alpha=alpha, L=L, verbose=False)

            # Calculate metrics
            self.calculate_metrics(alpha=alpha, f_hat=f_hat)
            if save_imgs:
                self._f_hats[alpha] = f_hat

            logger.info(
                f"Alpha {alpha}: DP {self._metrics.loc[alpha, "discrepancy"]}",
            )

        self.get_optimal_alpha()
        self._optimal_f_hat = self._solver.solve(
            alpha=self._optimal_alpha,
            L=self._L,
            verbose=False,
        )
        self.calculate_metrics(alpha="optimal", f_hat=self._optimal_f_hat)
        logger.info(
            f"Optimal alpha {alpha}: DP {self.optimal_metrics["discrepancy"]}",
        )


class IterativeTuner(Tuner):
    """Iterative solver class for tuning regularisation hyper-parameters."""

    def iteratively_solve(
        self,
        alpha: float,
        L: Callable[[npt.NDArray, str], sp.csr_matrix],
        prev_f_hat: npt.NDArray,
        max_iter: int,
        tol: float,
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Iteratively solve the inverse problem.

        Args:
            alpha: Regularisation parameter
            L: Regularisation matrix creation function
            prev_f_hat: Previous deblurred image
            max_iter: Maximum number of iterations
            tol: Convergence tolerance

        Returns
        -------
            Tuple: Deblurred image and number of iterations
        """
        for it in range(max_iter):
            # Re-initialise regularisation matrix with the last predicted image
            self._L = L(prev_f_hat, conv_mode=ConvolutionMode.PERIODIC)  # type: ignore[call-arg]
            f_hat = self._solver.solve(
                alpha=alpha,
                L=self._L,
                x0=prev_f_hat.flatten(),
                verbose=False,
            )
            prev_f_hat = f_hat

            # Check for convergence
            residual = self._g.reshape([-1, 1]) - self._kernel(f_hat).reshape([-1, 1])
            residual_norm = (residual.T @ residual).squeeze()

            if residual_norm <= tol * np.square(self._g).sum():
                break

        return f_hat, it

    def parameter_sweep(
        self,
        alphas: list[float],
        L: sp.csr_matrix | Callable[[npt.NDArray, str], sp.csr_matrix],
        save_imgs: bool = False,
        **kwargs: Any,
    ) -> None:
        """Perform a parameter sweep over a range of regularisation hyper-parameters.

        Args:
            alphas: List of regularisation hyper-parameters
            L: Sparse regularisation matrix or function returning it
            save_imgs: Cache deblurred images for each hyper-parameter
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        if isinstance(L, sp.csr_matrix):
            raise ValueError(
                f"L must be a function returning a sparse matrix, got: {type(L)}",
            )
        max_iter: int = kwargs.get("max_iter", 20)
        tol: float = kwargs.get("tol", 1e-8)

        self.reset_metrics()
        self._alphas = []
        self._f_hats = {}

        for alpha in alphas:
            # Solve for this alpha with blurred image as starting guess
            self._alphas.append(alpha)
            f_hat, it = self.iteratively_solve(
                alpha=alpha,
                L=L,
                prev_f_hat=self._g,
                max_iter=max_iter,
                tol=tol,
            )

            # Calculate metrics
            self.calculate_metrics(alpha=alpha, f_hat=f_hat)
            if save_imgs:
                self._f_hats[alpha] = f_hat

            logger.info(
                f"Alpha {alpha}: DP {self._metrics.loc[alpha, "discrepancy"]}, iterations {it + 1}",
            )

        # Get optimal alpha and iteratively solve
        self.get_optimal_alpha()
        self._optimal_f_hat, _ = self.iteratively_solve(
            self._optimal_alpha,
            L,
            self._g,
            max_iter,
            tol,
        )

        # Calculate metrics
        self.calculate_metrics(alpha="optimal", f_hat=self._optimal_f_hat)
        logger.info(
            f"Optimal alpha {alpha}: DP {self.optimal_metrics["discrepancy"]}",
        )
