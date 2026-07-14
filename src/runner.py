"""The experiment loop — cached, resumable, and leakage-free (SPEC §§3-5).

``run_cell`` fits ONE pipeline on ONE train fold and evaluates on the held-out
fold. Resampling lives INSIDE the imblearn pipeline (built by
``src.conditions.make_pipeline``), so it is applied to the TRAIN FOLD ONLY —
never to the full dataset, never across the split (.cursorrules #2, PREREG §4.3).

``load_dataset`` and ``make_pipeline`` are imported as module-level names so the
runner-level leakage contract test (``tests/test_runner_leakage.py``) can
monkeypatch them on ``src.runner``.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config, metrics
from src.conditions import make_pipeline
from src.datasets import load_dataset

# The REAL scientific output tree. Every write path is derived from a
# ``results_dir`` argument that DEFAULTS to this — tests pass a pytest tmp_path
# instead, so a test can never write into the real results directory (SPEC §1).
RESULTS_DIR: Path = Path("results")
CELLS_DIR: Path = RESULTS_DIR / "cells"
# Per-item probabilities live in their OWN directory, NOT in cells/. Sharing the
# cells/ glob namespace silently mixed y_prob files with cell results; separating
# them makes results/cells/*.parquet match EXACTLY one file per cell (SPEC §1, §4).
YPROB_DIR: Path = RESULTS_DIR / "yprob"
RESULTS_PATH: Path = RESULTS_DIR / "results.parquet"
DATASETS_FILE: Path = Path("datasets.txt")


def _cells_dir(results_dir: Path) -> Path:
    """Cell-result parquet directory under ``results_dir``."""
    return results_dir / "cells"


def _yprob_dir(results_dir: Path) -> Path:
    """Per-item probability parquet directory under ``results_dir``."""
    return results_dir / "yprob"

logger = logging.getLogger(__name__)

# (dataset_id, column) facts already disclosed, so an all-missing column is
# logged ONCE rather than on every one of the ~3,000 cells that touch it.
_LOGGED_ALL_NAN: set[tuple[int, str]] = set()

# The FROZEN output schema (SPEC §4). Column order is fixed here.
SCHEMA_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "dataset_name",
    "n_rows",
    "event_rate",
    "model",
    "condition",
    "seed",
    "fold",
    "status",
    "error",
    "fit_seconds",
    "auroc",
    "auprc",
    "brier",
    "ece",
    "cal_slope",
    "cal_intercept",
    "nb_at_eventrate",
    "nb_at_005",
    "nb_at_010",
    "nb_at_020",
    "y_prob_path",
)


def cell_id(dataset_id: int, model: str, condition: str, seed: int, fold: int) -> str:
    """Stable hash-free identifier: f'{dataset_id}__{model}__{condition}__s{seed}__f{fold}'"""
    return f"{dataset_id}__{model}__{condition}__s{seed}__f{fold}"


def _dataset_name(dataset_id: int) -> str:
    """Human-readable OpenML name for the results table.

    ``load_dataset`` (called first in ``run_cell``) has already downloaded and
    locally cached the dataset object, so this is a cheap metadata read for real
    datasets. It is intentionally NOT wrapped in its own try/except: if it fails
    the whole cell is honestly recorded as ``failed`` rather than silently
    degraded.
    """
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=False,
        download_qualities=False,
        download_features_meta_data=False,
    )
    return str(dataset.name)


def _save_y_prob(
    cid: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    yprob_dir: Path = YPROB_DIR,
) -> str:
    """Persist per-item held-out (y_true, y_prob) so every metric can be
    recomputed later without refitting (SPEC §4: y_prob_path is mandatory).

    Writes to ``{yprob_dir}/{cid}.parquet`` — a directory SEPARATE from the
    cell-results directory, so ``results/cells/*.parquet`` never matches a y_prob
    file. Returns the path (relative to the repo root for the default location).
    """
    yprob_dir.mkdir(parents=True, exist_ok=True)
    path = yprob_dir / f"{cid}.parquet"
    pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=int),
            "y_prob": np.asarray(y_prob, dtype=float),
        }
    ).to_parquet(path, index=False)
    return path.as_posix()


def _write_cell(cid: str, row: dict[str, object], cells_dir: Path = CELLS_DIR) -> pd.DataFrame:
    """Write the one-row cell result to ``{cells_dir}/{cid}.parquet`` and return it."""
    cells_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=list(SCHEMA_COLUMNS))
    df.to_parquet(cells_dir / f"{cid}.parquet", index=False)
    return df


def _log_all_nan_columns(dataset_id: int, X: pd.DataFrame) -> None:
    """Disclose (once per dataset/column) any feature that is entirely missing.

    A fully all-NaN column cannot be median-imputed, so ``SimpleImputer`` drops it
    (its default ``keep_empty_features=False``). We neither drop it ourselves nor
    hide the fact (.cursorrules #3): OpenML dataset 38 ``sick`` carries an
    all-missing ``TBG`` column. It is logged here and belongs in the paper's
    limitations. The matching SimpleImputer ``UserWarning`` is suppressed at fit
    time only to keep the ~3,000-cell run readable — the fact is recorded, not
    silenced.
    """
    for col in X.columns:
        if not X[col].isna().all():
            continue
        key = (int(dataset_id), str(col))
        if key in _LOGGED_ALL_NAN:
            continue
        _LOGGED_ALL_NAN.add(key)
        logger.warning(
            "dataset %d: feature %r is entirely missing (all-NaN) and is dropped "
            "by the median SimpleImputer; recorded as a data limitation, not "
            "silently excluded.",
            int(dataset_id),
            str(col),
        )


def run_cell(
    dataset_id: int,
    model: str,
    condition: str,
    seed: int,
    fold: int,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Fits ONE pipeline on ONE train fold, predicts on the held-out fold,
    returns a ONE-ROW dataframe matching the schema in §4.
    On exception: returns a one-row frame with status='failed' and the exception text.
    NEVER raises. NEVER skips.

    ``results_dir`` DEFAULTS to the real ``results/`` tree; tests pass a pytest
    tmp_path so they never contaminate the scientific output directory (SPEC §1).
    Cell results are written to ``{results_dir}/cells/`` and per-item
    probabilities to ``{results_dir}/yprob/`` (separate glob namespaces).
    """
    cid = cell_id(dataset_id, model, condition, seed, fold)
    cells_dir = _cells_dir(results_dir)
    yprob_dir = _yprob_dir(results_dir)

    dataset_name: str = ""
    n_rows: float = float("nan")
    event_rate: float = float("nan")
    status: str = "ok"
    error: str | None = None
    fit_seconds: float = float("nan")
    auroc_v: float = float("nan")
    auprc_v: float = float("nan")
    brier_v: float = float("nan")
    ece_v: float = float("nan")
    cal_slope_v: float = float("nan")
    cal_intercept_v: float = float("nan")
    nb_eventrate: float = float("nan")
    nb_005: float = float("nan")
    nb_010: float = float("nan")
    nb_020: float = float("nan")
    y_prob_path: str | None = None

    try:
        # RAW features; imputation/scaling/resampling are all pipeline steps.
        X, y = load_dataset(dataset_id)
        _log_all_nan_columns(dataset_id, X)
        n_rows = int(len(X))
        # event_rate: minority (class 1) prevalence in the FULL dataset (SPEC §4).
        event_rate = float(np.mean(np.asarray(y, dtype=float)))

        # The canonical split: StratifiedKFold(shuffle=True, random_state=seed).
        skf = StratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=seed
        )
        train_idx, val_idx = list(skf.split(X, y))[fold]

        # Resampling lives INSIDE this pipeline -> applied to the train fold ONLY.
        pipe = make_pipeline(model, condition, seed)

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            # All-NaN columns are dropped by SimpleImputer (disclosed once via
            # _log_all_nan_columns above); silence ONLY that specific, expected
            # warning so it does not flood the ~3,000-cell run. e.g. dataset 38
            # `sick` column TBG. No other warning is suppressed.
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values",
                category=UserWarning,
            )
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        fit_seconds = float(time.perf_counter() - t0)

        # Held-out fold: samplers are pass-through at transform/predict time, so
        # the validation set keeps its original size and class balance.
        p_val = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        y_val = np.asarray(y.iloc[val_idx], dtype=int)

        auroc_v = metrics.auroc(y_val, p_val)
        auprc_v = metrics.auprc(y_val, p_val)
        brier_v = metrics.brier(y_val, p_val)
        ece_v = metrics.ece(y_val, p_val)
        cal_slope_v = metrics.cal_slope(y_val, p_val)
        cal_intercept_v = metrics.cal_intercept(y_val, p_val)

        t005, t010, t020 = config.NET_BENEFIT_THRESHOLDS
        nb_eventrate = metrics.net_benefit(y_val, p_val, pt=event_rate)
        nb_005 = metrics.net_benefit(y_val, p_val, pt=t005)
        nb_010 = metrics.net_benefit(y_val, p_val, pt=t010)
        nb_020 = metrics.net_benefit(y_val, p_val, pt=t020)

        dataset_name = _dataset_name(dataset_id)
        y_prob_path = _save_y_prob(cid, y_val, p_val, yprob_dir)
    except Exception as exc:  # NEVER raise, NEVER skip (.cursorrules #3).
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        # A failed cell carries no (partial/misleading) numbers.
        fit_seconds = float("nan")
        auroc_v = auprc_v = brier_v = ece_v = float("nan")
        cal_slope_v = cal_intercept_v = float("nan")
        nb_eventrate = nb_005 = nb_010 = nb_020 = float("nan")
        y_prob_path = None

    row: dict[str, object] = {
        "dataset_id": int(dataset_id),
        "dataset_name": dataset_name,
        "n_rows": n_rows,
        "event_rate": event_rate,
        "model": model,
        "condition": condition,
        "seed": int(seed),
        "fold": int(fold),
        "status": status,
        "error": error,
        "fit_seconds": fit_seconds,
        "auroc": auroc_v,
        "auprc": auprc_v,
        "brier": brier_v,
        "ece": ece_v,
        "cal_slope": cal_slope_v,
        "cal_intercept": cal_intercept_v,
        "nb_at_eventrate": nb_eventrate,
        "nb_at_005": nb_005,
        "nb_at_010": nb_010,
        "nb_at_020": nb_020,
        "y_prob_path": y_prob_path,
    }
    return _write_cell(cid, row, cells_dir)


def _read_dataset_ids() -> list[int]:
    """The frozen experiment datasets, in file order (datasets.txt, SPEC §1)."""
    lines = DATASETS_FILE.read_text().strip().splitlines()
    return [int(line) for line in lines if line.strip()]


def run_all(force: bool = False, results_dir: Path = RESULTS_DIR) -> None:
    """Iterates the full grid. Skips cells whose parquet already exists in
    ``{results_dir}/cells/`` unless force=True. Writes
    ``{results_dir}/results.parquet`` at the end.
    """
    dataset_ids = _read_dataset_ids()
    cells_dir = _cells_dir(results_dir)
    frames: list[pd.DataFrame] = []

    for dataset_id in dataset_ids:
        for model in config.MODELS:
            for condition in config.CONDITIONS:
                for seed in config.SEEDS:
                    for fold in range(config.N_FOLDS):
                        cid = cell_id(dataset_id, model, condition, seed, fold)
                        cell_path = cells_dir / f"{cid}.parquet"
                        if cell_path.exists() and not force:
                            frames.append(pd.read_parquet(cell_path))
                            continue
                        frames.append(
                            run_cell(dataset_id, model, condition, seed, fold, results_dir)
                        )

    results_dir.mkdir(parents=True, exist_ok=True)
    results = pd.concat(frames, ignore_index=True)
    results = results[list(SCHEMA_COLUMNS)]
    results.to_parquet(results_dir / "results.parquet", index=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        ids = Path("datasets.txt").read_text().strip().splitlines()
        dataset_id = int(ids[0])
        for condition in config.CONDITIONS:
            for fold in range(config.N_FOLDS):
                run_cell(dataset_id, "xgboost", condition, config.SEEDS[0], fold)
    else:
        run_all()
