"""Results-directory hygiene contract tests (SPEC §1, §4).

Two defects, both silent (wrong answers, no error), are guarded here:

  1. A test that invokes ``run_cell`` must NEVER write into the real ``results/``
     tree. A leaked test artifact (the synthetic FAKE_DATASET_ID) would otherwise
     appear in the scientific results and be reported in the paper.

  2. Per-item probability files must live in ``results/yprob/``, NOT
     ``results/cells/``. When they shared the ``cells/`` glob namespace,
     ``glob("results/cells/*.parquet")`` matched BOTH cell results and y_prob
     files, silently mixing them for any downstream reader.

These are engineering contracts, decided BEFORE any experimental result was used;
see DEVIATIONS.md 2026-07-14 "Results-directory hygiene".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.runner import CELLS_DIR, cell_id, run_cell

# Distinct from the leakage test's 999_999 so a stray artifact is unambiguous.
FAKE_DATASET_ID: int = 424_242


def _synthetic(
    n: int = 400, event_rate: float = 0.2, n_features: int = 6, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """A small, learnable imbalanced dataset shaped like ``load_dataset`` output."""
    rng = np.random.default_rng(seed)
    n_pos = int(round(n * event_rate))
    n_neg = n - n_pos
    y = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    feats = rng.normal(size=(n, n_features))
    feats[y == 1] += 1.0
    perm = rng.permutation(n)
    feats, y = feats[perm], y[perm]
    X = pd.DataFrame(feats, columns=[f"f{i}" for i in range(n_features)])
    return X, pd.Series(y, name="target")


def _patch_runner(monkeypatch) -> None:
    """Feed run_cell a synthetic dataset + name so it never touches OpenML."""
    X, y = _synthetic()
    monkeypatch.setattr(
        "src.runner.load_dataset", lambda dataset_id: (X.copy(), y.copy()), raising=False
    )
    monkeypatch.setattr(
        "src.runner._dataset_name", lambda dataset_id: "synthetic", raising=False
    )


def test_run_cell_never_writes_to_real_results(tmp_path, monkeypatch) -> None:
    """run_cell under a tmp output dir leaves results/cells/ completely untouched."""
    _patch_runner(monkeypatch)

    before = (
        sorted(p.name for p in CELLS_DIR.iterdir()) if CELLS_DIR.exists() else []
    )

    df = run_cell(
        FAKE_DATASET_ID, "logreg", "none", config.SEEDS[0], 0, results_dir=tmp_path
    )
    assert (df["status"] == "ok").all(), df["error"].tolist()

    after = (
        sorted(p.name for p in CELLS_DIR.iterdir()) if CELLS_DIR.exists() else []
    )
    assert before == after, (
        "run_cell with results_dir=tmp_path modified the real results/cells/ "
        "directory — a test must never write into the scientific output tree."
    )

    cid = cell_id(FAKE_DATASET_ID, "logreg", "none", config.SEEDS[0], 0)
    assert (tmp_path / "cells" / f"{cid}.parquet").exists()
    assert (tmp_path / "yprob" / f"{cid}.parquet").exists()


def test_cells_glob_returns_one_file_per_cell_never_yprob(tmp_path, monkeypatch) -> None:
    """glob(cells/*.parquet) returns EXACTLY one file per cell and no y_prob file."""
    _patch_runner(monkeypatch)

    n_cells = 0
    for condition in config.CONDITIONS:
        df = run_cell(
            FAKE_DATASET_ID, "logreg", condition, config.SEEDS[0], 0, results_dir=tmp_path
        )
        assert (df["status"] == "ok").all(), df["error"].tolist()
        n_cells += 1

    cells = sorted((tmp_path / "cells").glob("*.parquet"))
    yprob = sorted((tmp_path / "yprob").glob("*.parquet"))

    assert len(cells) == n_cells, (
        f"cells/*.parquet matched {len(cells)} files for {n_cells} cells — the glob "
        "namespace is contaminated (y_prob files must live in yprob/)."
    )
    assert len(yprob) == n_cells

    for p in cells:
        assert not p.name.endswith(".yprob.parquet"), (
            f"{p.name} is a y_prob file matched by the cells glob."
        )

    # Every cell's y_prob_path must point into yprob/, never cells/.
    for p in cells:
        y_prob_path = pd.read_parquet(p)["y_prob_path"].iloc[0]
        assert "/yprob/" in y_prob_path, y_prob_path
        assert "/cells/" not in y_prob_path, y_prob_path
