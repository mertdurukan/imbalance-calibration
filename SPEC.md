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
├── requirements.txt           # human-readable intent (direct deps, pinned)
├── requirements.lock.txt      # exact reproducible env (pip freeze output)
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

    "missing-value rate <= MAX_MISSING_RATE" is defined CELL-LEVEL:
        missing_rate = NumberOfMissingValues / (NumberOfInstances * NumberOfFeatures)
    (a proportion of data cells, NOT a proportion of affected rows). This
    definitional lock resolves a gap in PREREG §4.1 — see DEVIATIONS.md 2026-07-14
    "Ambiguity resolution: definition of missing-value rate".
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
| `make setup` | create `.venv` with `python3.11 -m venv .venv`, upgrade pip, then `pip install -r requirements.txt` |
| `make verify` | import-check the core deps (sklearn, imblearn, xgboost, statsmodels, pandas, openml) |
| `make test` | `pytest tests/ -x -q` |
| `make datasets` | run `select_datasets()`, write `datasets.txt` (refuses to overwrite if it exists) |
| `make pilot` | run 1 dataset × 1 model × all conditions × 1 seed — end-to-end smoke test |
| `make reproduce` | `make test && python -m src.runner && python -m src.analyze` |

**`make reproduce` must be the only command a reader needs.**

### 6.1 Dependencies: two files, two purposes

The environment is pinned in two committed files (see DEVIATIONS.md 2026-07-14 — conda was
replaced by venv+pip on the target machine):

| file | purpose | how it is produced |
|---|---|---|
| `requirements.txt` | **human-readable intent** — the direct dependencies the study needs, each pinned to an exact version | maintained by hand |
| `requirements.lock.txt` | **exact reproducible environment** — the full transitive closure, byte-for-byte what was installed | `pip freeze > requirements.lock.txt` |

- `make setup` installs from `requirements.txt` (the intent).
- To reproduce the *exact* environment used for a result, install from the lock file
  instead: `./.venv/bin/pip install -r requirements.lock.txt`.
- Any change to `requirements.txt` MUST be followed by regenerating `requirements.lock.txt`
  in the same commit, so the two never drift.

---

## Changelog

Append-only. Records changes to this SPEC. Deviations from `PREREG.md` still go in
`DEVIATIONS.md`; this log is for the implementation contract itself.

- **2026-07-14** — Added `requirements.lock.txt` (output of `pip freeze`) alongside
  `requirements.txt`, and documented both in §6.1. `requirements.txt` is the
  human-readable intent; `requirements.lock.txt` is the exact reproducible environment.
- **2026-07-14** — Pinned `scipy==1.13.1` in `requirements.txt` (and regenerated
  `requirements.lock.txt`). The lock file had resolved `scipy==1.17.1`, but SciPy removed
  `scipy._lib._util._lazywhere` in 1.14, which `statsmodels==0.14.2` imports, so
  `import statsmodels.api` failed. 1.13.1 is the last pre-1.14 release and is compatible
  with `statsmodels==0.14.2` and `numpy==1.26.4`. This is an environment/build fix, not a
  design deviation — no estimand, metric, or hyperparameter changed.
- **2026-07-14** — corrected test_ece_equal_mass_not_equal_width (original construction made
  both binnings agree at ECE ~0.85 and could not distinguish them); fixed bootstrap_ci to
  forward metric kwargs. No experimental results existed at the time of either change.
- **2026-07-14** — locked the definition of "missing-value rate" in the `select_datasets`
  contract (§3) as CELL-LEVEL: `NumberOfMissingValues / (NumberOfInstances *
  NumberOfFeatures)`. PREREG §4.1 left the denominator unspecified; this records the
  resolution so it cannot drift. See DEVIATIONS.md 2026-07-14 "Ambiguity resolution:
  definition of missing-value rate". No experimental results existed at the time.
