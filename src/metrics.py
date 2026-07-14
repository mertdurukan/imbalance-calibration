"""Metric implementations — exact definitions from METRICS.md.

Every constant comes from src/config.py; there are no numeric literals in the
metric logic itself.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import (
    BOOTSTRAP_N,
    CAL_CLIP_EPS,
    CI_LOWER_PERCENTILE,
    CI_UPPER_PERCENTILE,
    ECE_N_BINS,
)


def auroc(y_true: np.ndarray, y_prob: np.ndarray, **kwargs: object) -> float:
    """AUROC (METRICS.md §1): sklearn.metrics.roc_auc_score."""
    return float(roc_auc_score(y_true, y_prob))


def auprc(y_true: np.ndarray, y_prob: np.ndarray, **kwargs: object) -> float:
    """AUPRC (METRICS.md §1): average precision, NOT trapezoidal PR area."""
    return float(average_precision_score(y_true, y_prob))


def brier(y_true: np.ndarray, y_prob: np.ndarray, **kwargs: object) -> float:
    """Brier score (METRICS.md §1): mean((p - y)**2)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def _clip_logit(y_prob: np.ndarray) -> np.ndarray:
    """Linear predictor L = logit(p) with p clipped to [eps, 1 - eps]."""
    p = np.asarray(y_prob, dtype=float)
    p = np.clip(p, CAL_CLIP_EPS, 1.0 - CAL_CLIP_EPS)
    return np.log(p / (1.0 - p))


def cal_slope(y_true: np.ndarray, y_prob: np.ndarray, **kwargs: object) -> float:
    """Calibration slope (METRICS.md §2).

    Fit y ~ a + b·L by unregularized logistic regression, where L = logit(p).
    Returns the slope b.
    """
    y = np.asarray(y_true, dtype=float)
    logit = _clip_logit(y_prob)
    design = sm.add_constant(logit, has_constant="add")
    model = sm.GLM(y, design, family=sm.families.Binomial())
    result = model.fit()
    return float(result.params[1])


def cal_intercept(y_true: np.ndarray, y_prob: np.ndarray, **kwargs: object) -> float:
    """Calibration intercept / calibration-in-the-large (METRICS.md §2).

    Fit y ~ a + offset(L): slope fixed at 1 via the offset, only an intercept
    column free. Returns the intercept a.
    """
    y = np.asarray(y_true, dtype=float)
    logit = _clip_logit(y_prob)
    design = np.ones((y.shape[0], 1), dtype=float)
    model = sm.GLM(y, design, family=sm.families.Binomial(), offset=logit)
    result = model.fit()
    return float(result.params[0])


def ece(y_true: np.ndarray, y_prob: np.ndarray, **kwargs: object) -> float:
    """Expected Calibration Error with equal-MASS bins (METRICS.md §3)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    order = np.argsort(p)
    y_sorted = y[order]
    p_sorted = p[order]
    bins_y = np.array_split(y_sorted, ECE_N_BINS)
    bins_p = np.array_split(p_sorted, ECE_N_BINS)
    total = 0.0
    for yb, pb in zip(bins_y, bins_p):
        if len(yb) == 0:
            continue
        acc = float(np.mean(yb))
        conf = float(np.mean(pb))
        total += (len(yb) / n) * abs(acc - conf)
    return float(total)


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, pt: float, **kwargs: object) -> float:
    """Net Benefit at decision threshold pt (METRICS.md §4).

    NB(pt) = TP/n - (FP/n) * (pt / (1 - pt)); positives are p >= pt.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    predicted_positive = p >= pt
    tp = float(np.sum(predicted_positive & (y == 1)))
    fp = float(np.sum(predicted_positive & (y == 0)))
    odds = pt / (1.0 - pt)
    return float(tp / n - (fp / n) * odds)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[..., float],
    n: int = BOOTSTRAP_N,
    seed: int = 0,
    **metric_kwargs: object,
) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high) — percentile bootstrap over ITEMS.

    The point estimate is the metric on the FULL sample (not the bootstrap mean).
    Any extra keyword arguments are forwarded to ``metric_fn`` on both the
    point-estimate call and every bootstrap resample (e.g. ``pt`` for
    :func:`net_benefit`).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    point = float(metric_fn(y, p, **metric_kwargs))

    n_items = len(y)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, n_items, size=n_items)
        estimates[i] = metric_fn(y[idx], p[idx], **metric_kwargs)

    ci_low = float(np.percentile(estimates, CI_LOWER_PERCENTILE))
    ci_high = float(np.percentile(estimates, CI_UPPER_PERCENTILE))
    return point, ci_low, ci_high
