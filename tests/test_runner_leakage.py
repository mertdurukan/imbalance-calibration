"""Runner-level leakage contract test — the guard pipeline tests CANNOT provide.

.cursorrules prohibition #2, PREREG §4.3, SPEC §§3-5. See the TODO at the top of
``tests/test_leakage.py``: the leak we actually fear is resampling applied to the
FULL dataset BEFORE the CV split. That bug would live in ``src/runner.py``, not in
``make_pipeline``, so NO pipeline-level test can detect it. This is the real guard.

Strategy — a sentinel-id probe:

  * Give every row a unique integer sentinel id in a feature column.
  * Partition the ids by the SAME fold split ``run_cell`` uses: every TRAINING row
    gets an id < 10_000, every VALIDATION (held-out) row gets an id >= 10_000.
  * Drive the real ``run_cell`` fit path with ``load_dataset`` and ``make_pipeline``
    monkeypatched to (a) hand it the synthetic sentinel dataset and (b) splice a
    passthrough probe in just before the final estimator, which records the exact
    ``X`` the model is trained on (post-resampling, inside the pipeline).
  * SMOTE interpolates only between TRAINING minority points, so the maximum sentinel
    id reaching the model can exceed 10_000 ONLY IF a held-out row entered the
    resampler — i.e. the train/validation split was violated. Assert the max < 10_000.

The probe sits after the pipeline's ``StandardScaler`` (order is imputer -> scaler ->
resampler -> model), so the sentinel it sees is standardized; we invert that single
column through the fitted scaler to recover raw ids before asserting.

``src/runner.py`` is still a stub (``run_cell`` raises ``NotImplementedError``), so
this test currently ERRORS. That is the expected pre-implementation state (TASKS.md
Task 4: test BEFORE runner). It must PASS once a non-leaky ``run_cell`` exists and
FAIL on a leaky one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

from src import config

# The ORIGINAL factory, captured at import time. The monkeypatched ``make_pipeline``
# below delegates to this so it can splice in the probe without recursing into itself.
from src.conditions import make_pipeline as _real_make_pipeline

SENTINEL_COL: str = "sentinel_id"
SENTINEL_BOUNDARY: int = 10_000
FAKE_DATASET_ID: int = 999_999  # load_dataset is patched; this never hits OpenML.

# Module-level sinks so captures survive even if the pipeline were cloned (an
# imblearn ``clone`` copies constructor params, not module state). ``run_cell`` is
# specified to fit ONE pipeline directly (SPEC §3), so no clone is expected — but
# this keeps the probe robust regardless.
_CAPTURED_SENTINEL_COLUMNS: list[np.ndarray] = []
_CAPTURED_PIPELINE: dict[str, object] = {}


def _make_imbalanced_with_sentinel(
    n: int = 1000,
    event_rate: float = 0.05,
    n_features: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) as ``load_dataset`` returns them (DataFrame, Series), 1 = minority.

    Built by construction (not sampled) so the event rate is EXACT and the fold
    arithmetic is deterministic. A mean shift on the positive class gives SMOTE
    something learnable to interpolate. The sentinel column is added by the caller
    AFTER the fold split is known, so leave it as a placeholder here.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(n * event_rate))
    n_neg = n - n_pos
    y = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    feats = rng.normal(size=(n, n_features))
    feats[y == 1] += 1.0
    perm = rng.permutation(n)
    feats, y = feats[perm], y[perm]

    columns = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(feats, columns=columns)
    X[SENTINEL_COL] = 0  # placeholder; filled per-fold by the caller
    return X, pd.Series(y, name="target")


class _SentinelProbe(BaseEstimator, TransformerMixin):
    """Identity transformer that records ONE column of the ``X`` it is fitted on.

    Spliced in just before the final estimator, its ``fit`` receives exactly the
    ``(X, y)`` the model trains on — i.e. POST-resampling inside the imblearn
    pipeline. ``transform`` is the identity, so it never perturbs data and never
    resamples at predict time.
    """

    def __init__(self, sentinel_col: int) -> None:
        self.sentinel_col = sentinel_col

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "_SentinelProbe":
        col = np.asarray(X)[:, self.sentinel_col].astype(float)
        _CAPTURED_SENTINEL_COLUMNS.append(col.copy())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


def test_runner_no_leakage(monkeypatch, tmp_path) -> None:
    """The maximum sentinel id reaching the model must stay strictly below the
    train/validation boundary — proof that resampling never saw the held-out fold.
    """
    _CAPTURED_SENTINEL_COLUMNS.clear()
    _CAPTURED_PIPELINE.clear()

    seed = config.SEEDS[0]
    fold = 0

    X, y = _make_imbalanced_with_sentinel(n=1000, event_rate=0.05, seed=seed)

    # Replicate the SAME fold split run_cell is contracted to use (StratifiedKFold,
    # shuffle=True, random_state=seed — the canonical split shared with test_leakage).
    # StratifiedKFold ignores feature VALUES (only y and row count), so assigning the
    # sentinel column afterwards does not change which rows land in train vs val.
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=seed
    )
    train_idx, val_idx = list(skf.split(X, y))[fold]

    # TRAIN ids strictly below the boundary; VALIDATION ids strictly at/above it.
    sentinel = np.empty(len(y), dtype=np.int64)
    sentinel[train_idx] = np.arange(len(train_idx))
    sentinel[val_idx] = SENTINEL_BOUNDARY + np.arange(len(val_idx))
    X[SENTINEL_COL] = sentinel
    sentinel_idx = X.columns.get_loc(SENTINEL_COL)

    assert sentinel[train_idx].max() < SENTINEL_BOUNDARY
    assert sentinel[val_idx].min() >= SENTINEL_BOUNDARY

    def _fake_load_dataset(dataset_id: int) -> tuple[pd.DataFrame, pd.Series]:
        return X.copy(), y.copy()

    def _wrapped_make_pipeline(model_name: str, condition: str, seed: int):
        pipe = _real_make_pipeline(model_name, condition, seed)
        probe = _SentinelProbe(sentinel_col=sentinel_idx)
        pipe.steps.insert(len(pipe.steps) - 1, ("sentinel_probe", probe))
        _CAPTURED_PIPELINE["pipe"] = pipe
        return pipe

    # Patch at the source modules AND at the runner's namespace (raising=False, since
    # run_cell is not implemented yet), so the injection works whichever import style
    # the runner ends up using.
    monkeypatch.setattr("src.datasets.load_dataset", _fake_load_dataset, raising=False)
    monkeypatch.setattr("src.runner.load_dataset", _fake_load_dataset, raising=False)
    monkeypatch.setattr(
        "src.conditions.make_pipeline", _wrapped_make_pipeline, raising=False
    )
    monkeypatch.setattr(
        "src.runner.make_pipeline", _wrapped_make_pipeline, raising=False
    )

    from src.runner import run_cell

    # run_cell is contracted to NEVER raise; it must FIT (invoking our probe) before
    # doing anything that could mark the cell failed. Write to a pytest tmp_path so the
    # test artifact (FAKE_DATASET_ID) never lands in the real results/ tree (SPEC §1).
    run_cell(FAKE_DATASET_ID, "logreg", "smote", seed, fold, results_dir=tmp_path)

    assert _CAPTURED_SENTINEL_COLUMNS, (
        "the model was never fitted — the probe recorded nothing. run_cell must fit "
        "the pipeline built by make_pipeline on the train fold."
    )

    pipe = _CAPTURED_PIPELINE.get("pipe")
    assert pipe is not None, "make_pipeline was never called by run_cell"
    scaler = pipe.named_steps["scaler"]
    assert hasattr(scaler, "mean_"), (
        "the StandardScaler was never fitted — run_cell must fit the pipeline object "
        "returned by make_pipeline (SPEC §3: 'fits ONE pipeline on ONE train fold')."
    )

    # The probe captured the scaled sentinel column; invert that single column back to
    # raw ids through the fitted scaler.
    scale = float(scaler.scale_[sentinel_idx])
    mean = float(scaler.mean_[sentinel_idx])
    max_raw_id = max(
        float((col * scale + mean).max()) for col in _CAPTURED_SENTINEL_COLUMNS
    )

    assert max_raw_id < SENTINEL_BOUNDARY, (
        f"a validation-fold row reached the model's training data: max sentinel id "
        f"{max_raw_id:.1f} >= {SENTINEL_BOUNDARY}. Resampling saw the held-out fold "
        "— the CV split was violated (resampling applied before/across the split)."
    )
