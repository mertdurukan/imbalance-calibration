"""Known-answer tests for src.metrics — METRICS.md §§2–5.

These tests encode the exact metric definitions. They must fail with
NotImplementedError until src/metrics.py is implemented.
"""

from __future__ import annotations

import numpy as np

from src.config import BOOTSTRAP_N, ECE_N_BINS
from src.metrics import (
    bootstrap_ci,
    brier,
    cal_intercept,
    cal_slope,
    ece,
    net_benefit,
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _make_true_logistic(
    n: int = 100_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y, true_p, true_logit) from METRICS.md §2 known-answer setup."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n)
    true_logit = 0.5 * x - 2.0
    true_p = _sigmoid(true_logit)
    y = rng.binomial(1, true_p).astype(float)
    return y, true_p, true_logit


def _ece_equal_mass(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """Local reference: METRICS.md §3 equal-mass ECE."""
    order = np.argsort(p)
    y_sorted = y[order]
    p_sorted = p[order]
    bins_y = np.array_split(y_sorted, n_bins)
    bins_p = np.array_split(p_sorted, n_bins)
    n = len(y)
    total = 0.0
    for yb, pb in zip(bins_y, bins_p):
        if len(yb) == 0:
            continue
        acc = float(np.mean(yb))
        conf = float(np.mean(pb))
        total += (len(yb) / n) * abs(acc - conf)
    return total


def _ece_equal_width(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """Local reference: equal-WIDTH bins (the common incorrect alternative)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    total = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        nb = int(mask.sum())
        if nb == 0:
            continue
        acc = float(np.mean(y[mask]))
        conf = float(np.mean(p[mask]))
        total += (nb / n) * abs(acc - conf)
    return total


# ---------------------------------------------------------------------------
# §2 Calibration slope & intercept
# ---------------------------------------------------------------------------


def test_cal_slope_perfect() -> None:
    y, p, _ = _make_true_logistic(n=100_000, seed=0)
    slope = cal_slope(y, p)
    assert abs(slope - 1.0) < 0.05


def test_cal_intercept_perfect() -> None:
    y, p, _ = _make_true_logistic(n=100_000, seed=1)
    intercept = cal_intercept(y, p)
    assert abs(intercept - 0.0) < 0.05


def test_cal_slope_overextreme() -> None:
    """Feed sigmoid(2.0 * true_logit) — over-extreme predictions; slope << 1."""
    y, _, true_logit = _make_true_logistic(n=100_000, seed=2)
    p_miscal = _sigmoid(2.0 * true_logit)
    slope = cal_slope(y, p_miscal)
    # Doubling the true logit implies a true calibration slope of ~0.5, so a
    # correct implementation must land well below 1. The previous 0.9 threshold
    # was so loose it would pass even for a nearly-correct (barely over-extreme)
    # model; 0.65 actually pins down the expected ~0.5.
    assert slope < 0.65


def test_cal_slope_intercept_finite_at_extremes() -> None:
    """Clipping guard (METRICS.md §2): exact 0.0 / 1.0 probs must not blow up.

    logit(0) = -inf and logit(1) = +inf, so without the [1e-6, 1 - 1e-6] clip the
    logistic regressions would receive non-finite inputs. Both metrics must return
    finite values.
    """
    y, _, true_logit = _make_true_logistic(n=100_000, seed=8)
    p = _sigmoid(true_logit)
    p[0] = 0.0
    p[1] = 1.0
    slope = cal_slope(y, p)
    intercept = cal_intercept(y, p)
    assert np.isfinite(slope)
    assert np.isfinite(intercept)


def test_cal_intercept_overestimation() -> None:
    """Feed sigmoid(true_logit + 1.0) — systematic overestimation; intercept << 0."""
    y, _, true_logit = _make_true_logistic(n=100_000, seed=3)
    p_miscal = _sigmoid(true_logit + 1.0)
    intercept = cal_intercept(y, p_miscal)
    assert intercept < -0.5


# ---------------------------------------------------------------------------
# §3 ECE — equal-MASS bins
# ---------------------------------------------------------------------------


def test_ece_perfect() -> None:
    rng = np.random.default_rng(4)
    n = 100_000
    p = rng.uniform(0.05, 0.95, size=n)
    y = rng.binomial(1, p).astype(float)
    value = ece(y, p)
    assert value < 0.01


def test_ece_constant_half_balanced() -> None:
    n = 10_000
    y = np.array([0.0] * (n // 2) + [1.0] * (n // 2))
    # LOAD-BEARING SHUFFLE: y is built sorted (all 0s then all 1s). Because p is
    # constant, np.argsort is stable and preserves that order, so equal-mass bins
    # would come out perfectly class-separated (acc_b ∈ {0, 1}) and ECE ≈ 0.5 —
    # an artifact of the label ordering, not the calibration. METRICS.md §3's
    # "balanced 50/50" means labels distributed at random across bins. Shuffling
    # with a fixed seed restores that. Do NOT remove this.
    rng = np.random.default_rng(101)
    rng.shuffle(y)
    p = np.full(n, 0.5)
    value = ece(y, p)
    assert abs(value - 0.0) < 0.02


def test_ece_constant_high_balanced() -> None:
    n = 10_000
    y = np.array([0.0] * (n // 2) + [1.0] * (n // 2))
    # LOAD-BEARING SHUFFLE: see test_ece_constant_half_balanced. Without shuffling,
    # the sorted labels + constant p + stable argsort separate the classes across
    # bins and ECE collapses to an ordering artifact instead of the intended 0.4.
    # Do NOT remove this.
    rng = np.random.default_rng(102)
    rng.shuffle(y)
    p = np.full(n, 0.9)
    value = ece(y, p)
    assert abs(value - 0.4) < 0.02


def test_ece_equal_mass_not_equal_width() -> None:
    """Skewed p: equal-mass ECE must match the quantile reference, not equal-width."""
    rng = np.random.default_rng(5)
    n = 30_000
    p = rng.beta(0.3, 4.0, size=n)  # heavily skewed toward 0
    # Miscalibrate: labels flip relative to a threshold so |acc - conf| is non-trivial
    y = (p < 0.15).astype(float)

    value = ece(y, p)

    mass_ref = _ece_equal_mass(y, p, ECE_N_BINS)
    width_ref = _ece_equal_width(y, p, ECE_N_BINS)

    # Equal-mass partition: bin counts differ by at most 1
    order = np.argsort(p)
    bin_sizes = [len(b) for b in np.array_split(order, ECE_N_BINS)]
    assert max(bin_sizes) - min(bin_sizes) <= 1

    # Equal-mass and equal-width must meaningfully disagree on this skewed set
    assert abs(mass_ref - width_ref) > 0.02

    # Implementation must match equal-mass, not equal-width
    assert abs(value - mass_ref) < 1e-9
    assert abs(value - width_ref) > 0.02


# ---------------------------------------------------------------------------
# §4 Net Benefit
# ---------------------------------------------------------------------------


def test_net_benefit_hand_computed() -> None:
    """Hand-computed 10-item example at pt=0.2.

    y     : [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]   # event_rate = 0.4
    p     : [0.9, 0.8, 0.7, 0.3, 0.6, 0.5, 0.1, 0.05, 0.02, 0.01]
    pt    : 0.2
    predicted_positive = (p >= 0.2) → indices 0,1,2,3,4,5
      TP = sum(pred & y==1) = 4   (items 0,1,2,3)
      FP = sum(pred & y==0) = 2   (items 4,5)
      n  = 10
      odds = pt/(1-pt) = 0.2/0.8 = 0.25
      NB = TP/n - (FP/n)*odds
         = 4/10 - (2/10)*0.25
         = 0.4 - 0.05
         = 0.35
    """
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    p = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.5, 0.1, 0.05, 0.02, 0.01])
    pt = 0.2
    expected = 0.35

    value = net_benefit(y, p, pt=pt)
    assert abs(value - expected) < 1e-9


def test_net_benefit_treat_all_reference() -> None:
    """Treat-all: predicting everyone positive → NB = event_rate - (1-er)*(pt/(1-pt))."""
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)  # event_rate = 0.3
    p = np.ones_like(y)  # always predict positive
    pt = 0.2
    event_rate = float(np.mean(y))
    expected = event_rate - (1.0 - event_rate) * (pt / (1.0 - pt))

    value = net_benefit(y, p, pt=pt)
    assert abs(value - expected) < 1e-9


def test_net_benefit_random_below_treat_all() -> None:
    """A random-noise model has NB below treat-all at a low threshold."""
    rng = np.random.default_rng(6)
    n = 20_000
    event_rate = 0.2
    y = rng.binomial(1, event_rate, size=n).astype(float)
    p = rng.uniform(0.0, 1.0, size=n)  # uninformative scores
    pt = 0.05

    nb_model = net_benefit(y, p, pt=pt)

    er = float(np.mean(y))
    nb_treat_all = er - (1.0 - er) * (pt / (1.0 - pt))
    assert nb_model < nb_treat_all


# ---------------------------------------------------------------------------
# §5 bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_shape_and_point() -> None:
    rng = np.random.default_rng(7)
    n = 500
    p = rng.uniform(0.1, 0.9, size=n)
    y = rng.binomial(1, p).astype(float)

    point, lo, hi = bootstrap_ci(y, p, brier, n=BOOTSTRAP_N, seed=0)

    assert lo < point < hi
    # Point estimate must equal the metric on the full sample (not bootstrap mean)
    full = brier(y, p)
    assert abs(point - full) < 1e-9
