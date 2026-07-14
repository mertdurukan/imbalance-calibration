# SPEC — Implementation Contract

This document is the single source of truth for implementation. `PREREG.md` defines
*what* the experiment is; this defines *how the code is shaped*. Cursor: implement
exactly these signatures. Do not restructure.

---

## 1. Repository layout (create exactly this)

```
.
├── PREREG.md                  # frozen — do not edit
├── SPEC.md                    # this file
├── METRICS.md                 # exact metric formulas
├── TASKS.md                   # ordered implementation tasks
├── DEVIATIONS.md              # append-only log
├── .cursorrules
├── environment.yml
├── Makefile
├── datasets.txt               # generated ONCE by task 1, then frozen & committed
├── src/
│   ├── __init__.py
│   ├── config.py              # all constants. no magic numbers elsewhere.
│   ├── datasets.py            # OpenML selection + loading
│   ├── models.py              # model factory (fixed hyperparams)
│   ├── conditions.py          # resampling condition → imblearn Pipeline
│   ├── metrics.py             # metric implementations (see METRICS.md)
│   ├── runner.py              # the experiment loop, cached & resumable
│   └── analyze.py             # results → tables + figures
├── tests/
│   ├── test_leakage.py        # contract: no resampling outside train fold
│   ├── test_metrics.py        # contract: metrics match known-answer cases
│   └── test_config.py         # contract: no tuning, seeds fixed
└── results/
    ├── cells/                 # one parquet per cell (cache)
    ├── results.parquet        # concatenated
    └── figures/
```

---

## 2. `src/config.py` — frozen constants

```python
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
```

**Rule:** every number used anywhere in `src/` comes from here. A literal number in
`runner.py` or `models.py` is a bug.

---

## 3. Module contracts (exact signatures)

### `src/datasets.py`
```python
def select_datasets() -> list[int]:
    """Apply PREREG §4.1 criteria against the OpenML API.
    Returns the first N_DATASETS OpenML dataset IDs sorted ASCENDING by ID.
    MUST be deterministic. MUST be run once; result written to datasets.txt.
    """

def load_dataset(dataset_id: int) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (X, y) with y in {0,1}, 1 = minority class.
    Categorical columns are one-hot encoded. Missing values: median (numeric) /
    most-frequent (categorical) imputation, fitted INSIDE the CV pipeline, never here.
    This function returns RAW features; imputation is a pipeline step.
    """
```

### `src/models.py`
```python
def make_model(name: str, seed: int) -> BaseEstimator:
    """Fixed hyperparameters. PREREG §4.2. No tuning. Ever.

    logreg : LogisticRegression(max_iter=5000, random_state=seed)
    xgboost: XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, random_state=seed,
                           eval_metric="logloss", tree_method="hist")
             NOTE: scale_pos_weight is NOT set. It is out of scope (PREREG §6).
    mlp    : MLPClassifier(hidden_layer_sizes=(64,32), early_stopping=True,
                           max_iter=500, random_state=seed)
    """
```

### `src/conditions.py`
```python
def make_pipeline(model_name: str, condition: str, seed: int) -> ImbPipeline:
    """Builds: [imputer] -> [scaler] -> [resampler | passthrough] -> [model]

    The resampler MUST be an imblearn step so it is applied to TRAIN FOLDS ONLY.
    condition -> resampler:
        "none"  -> None (no resampling step)
        "rus"   -> RandomUnderSampler(sampling_strategy=1.0, random_state=seed)
        "ros"   -> RandomOverSampler(sampling_strategy=1.0, random_state=seed)
        "smote" -> SMOTE(k_neighbors=5, sampling_strategy=1.0, random_state=seed)

    Scaler: StandardScaler for logreg/mlp; passthrough for xgboost.
    """
```

### `src/metrics.py`
See `METRICS.md`. Every function has the signature:
```python
def <metric>(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float: ...
```
Plus:
```python
def bootstrap_ci(y_true, y_prob, metric_fn, n: int = BOOTSTRAP_N, seed: int = 0
                 ) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high) — percentile bootstrap over ITEMS."""
```

### `src/runner.py`
```python
def cell_id(dataset_id: int, model: str, condition: str, seed: int, fold: int) -> str:
    """Stable hash-free identifier: f'{dataset_id}__{model}__{condition}__s{seed}__f{fold}'"""

def run_cell(dataset_id: int, model: str, condition: str, seed: int, fold: int) -> pd.DataFrame:
    """Fits ONE pipeline on ONE train fold, predicts on the held-out fold,
    returns a ONE-ROW dataframe matching the schema in §4.
    On exception: returns a one-row frame with status='failed' and the exception text.
    NEVER raises. NEVER skips.
    """

def run_all(force: bool = False) -> None:
    """Iterates the full grid. Skips cells whose parquet already exists in
    results/cells/ unless force=True. Writes results/results.parquet at the end.
    """
```

---

## 4. Output schema — `results/results.parquet` (FROZEN)

One row per (dataset × model × condition × seed × fold). Columns, exactly:

| column | type | notes |
|---|---|---|
| `dataset_id` | int | OpenML ID |
| `dataset_name` | str | |
| `n_rows` | int | |
| `event_rate` | float | minority class prevalence in the FULL dataset |
| `model` | str | one of MODELS |
| `condition` | str | one of CONDITIONS |
| `seed` | int | |
| `fold` | int | 0..4 |
| `status` | str | `"ok"` or `"failed"` |
| `error` | str \| null | exception text if failed |
| `fit_seconds` | float | |
| `auroc` | float | |
| `auprc` | float | |
| `brier` | float | |
| `ece` | float | 15 equal-mass bins |
| `cal_slope` | float | see METRICS.md §2 |
| `cal_intercept` | float | see METRICS.md §2 |
| `nb_at_eventrate` | float | Net Benefit at threshold = event_rate |
| `nb_at_005` / `nb_at_010` / `nb_at_020` | float | |
| `y_prob_path` | str | relative path to saved per-item probabilities (parquet) |

**`y_prob_path` is mandatory.** Saving raw per-item predictions means every metric can be
recomputed later without refitting 3,000 models. This is the single highest-value line in
this spec — do not skip it.

---

## 5. Caching & resumability

- `run_cell` writes `results/cells/{cell_id}.parquet` immediately after computing.
- `run_all` checks existence first and skips.
- Consequence: the run can be interrupted at any time and resumed. This is required,
  not optional — a 3,000-fit run WILL be interrupted.

---

## 6. `make` targets

| target | does |
|---|---|
| `make setup` | create env from `environment.yml` |
| `make test` | `pytest tests/ -x -q` |
| `make datasets` | run `select_datasets()`, write `datasets.txt` (refuses to overwrite if it exists) |
| `make pilot` | run 1 dataset × 1 model × all conditions × 1 seed — end-to-end smoke test |
| `make reproduce` | `make test && python -m src.runner && python -m src.analyze` |

**`make reproduce` must be the only command a reader needs.**
