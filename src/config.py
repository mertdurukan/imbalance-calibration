from typing import Final

SEEDS: Final[list[int]] = [0, 1, 2, 3, 4]
N_FOLDS: Final[int] = 5
N_JOBS: Final[int] = 8            # M4 Pro; capped to avoid memory pressure

# Dataset selection criteria (PREREG §4.1)
MIN_MINORITY_RATE: Final[float] = 0.01
MAX_MINORITY_RATE: Final[float] = 0.20
MIN_N_ROWS: Final[int] = 2_000
MAX_N_ROWS: Final[int] = 200_000
MAX_MISSING_RATE: Final[float] = 0.30
# PREREG §4.1 specified "the first 10 datasets", but the pre-registered pool
# (OpenML-CC18 ∪ tag:imbalanced) mechanically yields only 8 datasets meeting the
# criteria (the `imbalanced` tag is empty on OpenML). The pool was NOT broadened
# and no threshold was relaxed, to preserve the anti-cherry-picking guarantee.
# See DEVIATIONS.md 2026-07-14 "N_DATASETS: 10 -> 8 (pool exhausted)".
N_DATASETS: Final[int] = 8

# Dataset selection source pool (PREREG §4.1): the OpenML-CC18 benchmark suite
# (suite 99) plus datasets carrying the OpenML tag `imbalanced`.
OPENML_CC18_SUITE_ID: Final[int] = 99
OPENML_IMBALANCED_TAG: Final[str] = "imbalanced"
N_BINARY_CLASSES: Final[int] = 2

MODELS: Final[list[str]] = ["logreg", "xgboost", "mlp"]
CONDITIONS: Final[list[str]] = ["none", "rus", "ros", "smote"]
# "none_threshold" is NOT a fit condition — it reuses "none" predictions.
# It is applied at analysis time only. See METRICS.md §4.

ECE_N_BINS: Final[int] = 15
NET_BENEFIT_THRESHOLDS: Final[list[float]] = [0.05, 0.10, 0.20]  # + event rate, computed per dataset
BOOTSTRAP_N: Final[int] = 2_000

# Metric implementation constants (METRICS.md §§2, 5)
CAL_CLIP_EPS: Final[float] = 1e-6          # clip p to [eps, 1 - eps] before logit
CI_LOWER_PERCENTILE: Final[float] = 2.5    # 95% percentile bootstrap CI, lower bound
CI_UPPER_PERCENTILE: Final[float] = 97.5   # 95% percentile bootstrap CI, upper bound
