"""Articles to read.
https://brendt.wohlberg.net/publications/pdf/lin-2010-upre.pdf
https://pages.stat.wisc.edu/~wahba/ftp1/oldie/golub.heath.wahba.pdf
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

from common.log import get_logger

logger = get_logger(__name__)


class Metrics:
    """Class for storing and calculating evaluation metrics.

    Attributes:
        _metrics: DataFrame for storing metrics
    """

    _metrics_df: pd.DataFrame

    def __init__(
        self,
        tuning_metric: str,
        use_miller: bool = False,
        noise_variance: float | None = None,
        f: npt.NDArray | None = None,
    ) -> None:
        """Initialise Metrics class.

        Args:
            tuning_metric: Metric used for tuning
            use_miller: Use Miller criterion
            noise_variance: Estimated noise variance
            f: True image
        """
        self._tuning_metric = tuning_metric
        self._use_miller = use_miller
        self._noise_variance = noise_variance
        self._f = f
        self.reset_metrics()

        if tuning_metric in ["MSE", "pSNR", "SSIM"] and f is None:
            raise ValueError("No reference image provided for quality metrics")
        if tuning_metric == "discrepancy" and noise_variance is None:
            raise ValueError("No noise variance provided for discrepancy principle")

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
        self._metrics_df = pd.DataFrame(columns=metric_columns)

    def calc_tikhonov_penalty(
        self,
        penalty_term: npt.NDArray[np.float64],
    ) -> float:
        """Calculate the Tikhonov functional.

        Args:
            penalty_term: Penalty term: Lx

        Returns
        -------
            float: Tikhonov functional value

        """
        return float(np.square(penalty_term).sum() / 2)

    def calc_dicrepancy_principle(self, residual: npt.NDArray[np.float64]) -> float:
        """Calculate the discrepancy principle (non-Miller method).

        Args:
            residual: Residual image
        """
        dim = np.prod(residual.shape)
        residual_norm = np.square(residual).sum()
        return float(residual_norm / dim - self._noise_variance)

    def calc_miller_criterion(
        self,
        residual: npt.NDArray[np.float64],
        penalty_term: npt.NDArray[np.float64],
    ) -> float:
        """Calculate the Miller discrepancy principle.

        Args:
            residual: Residual image
            penalty_term: Penalty term: Lx
        """
        residual = residual.reshape([-1, 1])
        residual_norm = np.square(residual).sum() / (2 * self._noise_variance)  # type: ignore[operator]

        return float(residual_norm - penalty_term)

    def calc_discrepancy(
        self,
        residual: npt.NDArray[np.float64],
        penalty_term: npt.NDArray[np.float64],
    ) -> float:
        """Calculate the discrepancy principle.

        Args:
            residual: Residual image
            penalty_term: Penalty term: Lx
        """
        return (
            self.calc_dicrepancy_principle(residual)
            if self._use_miller is not None
            else self.calc_miller_criterion(residual, penalty_term)
        )

    def calculate_metrics(
        self,
        alpha: float | str,
        residual: npt.NDArray[np.float64],
        f_hat: npt.NDArray[np.float64],
        penalty_term: npt.NDArray[np.float64],
    ) -> None:
        """Calculate metrics for a given regularisation hyper-parameter.

        Args:
            alpha: Regularisation hyper-parameter
            f_hat: Reconstructed image
            residual: Residual image
            penalty_term: Penalty term: Lx
            L: Regularisation matrix
        """
        self._metrics_df.loc[alpha, "residual_norm"] = np.square(residual).sum()
        self._metrics_df.loc[alpha, "tikhonov"] = self.calc_tikhonov_penalty(
            penalty_term=penalty_term,
        )

        if self._noise_variance is not None:
            self._metrics_df.loc[alpha, "discrepancy"] = self.calc_discrepancy(
                residual=residual,
                penalty_term=penalty_term,
            )

        if self._f is not None:
            self._metrics_df.loc[alpha, "MSE"] = mean_squared_error(self._f, f_hat)
            self._metrics_df.loc[alpha, "pSNR"] = peak_signal_noise_ratio(
                self._f,
                f_hat,
                data_range=1.0,
            )
            self._metrics_df.loc[alpha, "SSIM"] = structural_similarity(
                self._f,
                f_hat,
                data_range=1.0,
            )

    def get_optimal_discrepancy(self) -> None:
        """Find the optimal regularisation hyper-parameter using the discrepancy principle."""
        discrepancy_vals = self._metrics_df.loc[:, "discrepancy"].values
        idx1 = np.argwhere((discrepancy_vals * np.roll(discrepancy_vals, -1)) < 0.0)[0][
            0
        ]
        idx2 = np.argwhere((discrepancy_vals * np.roll(discrepancy_vals, 1)) < 0.0)[1][
            0
        ]
        y1 = discrepancy_vals[idx1]
        y2 = discrepancy_vals[idx2]
        x1 = self._metrics_df.index[idx1]
        x2 = self._metrics_df.index[idx2]
        ratio = abs(y2 / y1)
        self._optimal_alpha = (x2 + ratio * x1) / (ratio + 1)

    def get_optimal_metric(self, metric: str = "MSE") -> None:
        """Find the optimal regularisation hyper-parameter using the discrepancy principle."""
        if metric == "SSIM":
            optimal_alpha_idx = self._metrics_df.loc[:, metric].values.argmax()
        else:
            optimal_alpha_idx = self._metrics_df.loc[:, metric].values.argmin()
        self._optimal_alpha = self._metrics_df.index[optimal_alpha_idx]

    def get_optimal_alpha(self) -> None:
        """Find the optimal regularisation hyper-parameter."""
        if self._noise_variance is not None:
            self.get_optimal_discrepancy()
        elif self._f is not None:
            self.get_optimal_metric("SSIM")
        else:
            raise ValueError("No metric available to determine optimal alpha")

    def log_metrics(self, alpha: float | str) -> None:
        """Log metrics for each regularisation hyper-parameter."""
        if self._tuning_metric in ["MSE", "pSNR", "SSIM"]:
            logger.info(
                f"Alpha {alpha}: {self._tuning_metric} {self._metrics_df.loc[alpha, self._tuning_metric]}",
            )
        elif self._tuning_metric == "discrepancy":
            logger.info(
                f"Alpha {alpha}: discrepancy {self._metrics_df.loc[alpha, 'discrepancy']}",
            )
        else:
            logger.warning("No metrics available")

    def display_metrics(self) -> None:
        """Display metrics for each regularisation hyper-parameter."""
        # Plot discrepancy values against alpha values
        plt.subplot(2, 1, 1)
        plt.semilogx(
            self._metrics_df.index[:-1],
            self._metrics_df.loc[:, "discrepancy"].iloc[:-1],
        )
        plt.axhline(0, ls="--", c="k")
        plt.axvline(self._optimal_alpha, ls="--", c="k")
        plt.xlabel("Alpha")
        plt.ylabel("Discrepancy")
        plt.title("Discrepancy principle")

        # Plot L-curve (residual norm against Tikhonov term)
        plt.subplot(2, 1, 2)
        plt.loglog(
            self._metrics_df.loc[:, "tikhonov"].iloc[:-1],
            self._metrics_df.loc[:, "residual_norm"].iloc[:-1],
        )
        plt.plot(
            self._metrics_df.loc["optimal", "tikhonov"],
            self._metrics_df.loc["optimal", "residual_norm"],
            "k+",
        )
        plt.xlabel("Tikhonov Term")
        plt.ylabel("Residual norm")
        plt.title("L-Curve")
        plt.tight_layout()
        plt.show()

    @property
    def optimal_alpha(self) -> Any:
        return self._optimal_alpha

    @property
    def metrics_df(self) -> pd.DataFrame:
        return self._metrics_df

    @property
    def optimal_metrics(self) -> pd.Series:
        return self._metrics_df.loc["optimal"]

    def __repr__(self) -> str:
        return repr(self._metrics_df)

    def __str__(self) -> str:
        return str(self._metrics_df)
