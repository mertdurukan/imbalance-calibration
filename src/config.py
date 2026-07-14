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
N_DATASETS: Final[int] = 10

MODELS: Final[list[str]] = ["logreg", "xgboost", "mlp"]
CONDITIONS: Final[list[str]] = ["none", "rus", "ros", "smote"]
# "none_threshold" is NOT a fit condition — it reuses "none" predictions.
# It is applied at analysis time only. See METRICS.md §4.

ECE_N_BINS: Final[int] = 15
NET_BENEFIT_THRESHOLDS: Final[list[float]] = [0.05, 0.10, 0.20]  # + event rate, computed per dataset
BOOTSTRAP_N: Final[int] = 2_000
