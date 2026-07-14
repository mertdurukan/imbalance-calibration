from collections.abc import Callable

import numpy as np

from src.config import BOOTSTRAP_N


def auroc(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
    raise NotImplementedError


def auprc(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
    raise NotImplementedError


def brier(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
    raise NotImplementedError


def cal_slope(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
    raise NotImplementedError


def cal_intercept(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
    raise NotImplementedError


def ece(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
    raise NotImplementedError


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
    raise NotImplementedError


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[..., float],
    n: int = BOOTSTRAP_N,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high) — percentile bootstrap over ITEMS."""
    raise NotImplementedError
