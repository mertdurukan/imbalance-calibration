"""Leakage contract tests — .cursorrules prohibition #2, PREREG §4.3, SPEC §3.

These encode the single most common (and study-invalidating) error in the
imbalance literature: resampling that touches the validation data. Resampling
MUST live inside an ``imblearn.pipeline.Pipeline`` so it is applied to TRAIN
FOLDS ONLY.

`make_pipeline` is not implemented yet, so every test here fails with
NotImplementedError. That is the expected pre-implementation state (TASKS.md
Task 3: test BEFORE pipeline).
"""

# NOT COVERED HERE — MUST BE COVERED IN TASK 4 (runner):
# The leak we actually fear is resampling applied to the FULL dataset BEFORE the CV
# split. That bug would live in runner.py, not in make_pipeline, and NO pipeline-level
# test can detect it. Task 4 must add test_runner_no_leakage: attach a sentinel id
# column (train ids < 10_000, validation ids >= 10_000), run run_cell, and assert via a
# probe that the maximum sentinel id reaching the model is < 10_000. SMOTE interpolates
# only between training minority points, so a validation id can only appear if the split
# was violated. This is the real guard. Do not consider leakage covered until it exists.

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

from src import config
from src.conditions import make_pipeline

RESAMPLER_CLASSES: tuple[type, ...] = (SMOTE, RandomOverSampler, RandomUnderSampler)


def _make_imbalanced(
    n: int = 1000,
    event_rate: float = 0.05,
    n_features: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with EXACTLY round(n * event_rate) positives, 1 = minority.

    Built by construction (not sampled) so the event rate is exact and the fold
    arithmetic in the tests below is deterministic. A mean shift on the positive
    class gives SMOTE something learnable to interpolate.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(n * event_rate))
    n_neg = n - n_pos
    y = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    X = rng.normal(size=(n, n_features))
    X[y == 1] += 1.0
    perm = rng.permutation(n)
    return X[perm], y[perm]


def _resampler_steps(pipe: ImbPipeline) -> list[BaseEstimator]:
    """Steps that are resamplers (imblearn samplers expose ``fit_resample``)."""
    return [est for _, est in pipe.steps if hasattr(est, "fit_resample")]


class _ClassBalanceProbe(BaseEstimator, TransformerMixin):
    """Passthrough transformer that records the class balance of `y` at fit time.

    Placed immediately before the final estimator, its ``fit`` receives exactly
    the ``(X, y)`` the model is trained on — i.e. POST-resampling inside an
    imblearn pipeline. ``transform`` is an identity, so it never perturbs data
    and never sees ``y`` at predict time (predict does not resample).
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "_ClassBalanceProbe":
        y_arr = np.asarray(y)
        self.n_seen_: int = int(y_arr.shape[0])
        self.n_pos_seen_: int = int(np.sum(y_arr == 1))
        self.n_neg_seen_: int = int(np.sum(y_arr == 0))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


def test_resampler_is_inside_pipeline() -> None:
    """smote/rus/ros pipelines carry the correct sampler AS a pipeline step."""
    seed = config.SEEDS[0]
    expected: dict[str, type] = {
        "smote": SMOTE,
        "rus": RandomUnderSampler,
        "ros": RandomOverSampler,
    }
    for condition, sampler_cls in expected.items():
        pipe = make_pipeline("logreg", condition, seed)
        assert isinstance(pipe, ImbPipeline), (
            f"{condition}: expected imblearn Pipeline, got {type(pipe)!r}"
        )
        samplers = _resampler_steps(pipe)
        assert len(samplers) == 1, (
            f"{condition}: expected exactly one resampler step, found {len(samplers)}"
        )
        assert isinstance(samplers[0], sampler_cls), (
            f"{condition}: resampler is {type(samplers[0])!r}, expected {sampler_cls!r}"
        )


def test_no_resampling_on_none() -> None:
    """The 'none' pipeline has NO resampler step at all (not even a passthrough)."""
    seed = config.SEEDS[0]
    pipe = make_pipeline("logreg", "none", seed)
    assert isinstance(pipe, ImbPipeline)

    samplers = _resampler_steps(pipe)
    assert samplers == [], (
        f"'none' must contain no resampler step, found {[type(s) for s in samplers]}"
    )
    for name, est in pipe.steps:
        assert not isinstance(est, RESAMPLER_CLASSES), (
            f"'none' step {name!r} is a resampler ({type(est)!r})"
        )
    step_names = [name for name, _ in pipe.steps]
    assert "resampler" not in step_names, (
        f"'none' must not carry a (passthrough) 'resampler' step: {step_names}"
    )


def test_sampler_skipped_at_predict_time() -> None:
    """CRITICAL: an imblearn Pipeline applies samplers at FIT time only.

    Samplers must never run at transform/predict time — otherwise inference
    would fabricate or drop rows. We fit a SMOTE pipeline (which oversamples the
    training data, so the model sees MORE rows than the input) and then confirm
    prediction returns exactly one row per input row, including on a tiny slice.
    A pipeline that resampled at predict time would fail this.
    """
    X, y = _make_imbalanced(n=1000, event_rate=0.05, seed=config.SEEDS[0])
    pipe = make_pipeline("logreg", "smote", config.SEEDS[0])

    probe = _ClassBalanceProbe()
    pipe.steps.insert(len(pipe.steps) - 1, ("balance_probe", probe))
    pipe.fit(X, y)

    # At FIT time the model saw MORE rows than n (SMOTE oversampled).
    assert probe.n_seen_ > len(y), (
        f"model saw {probe.n_seen_} rows, not more than n={len(y)} "
        "— oversampling did not happen inside the pipeline"
    )
    # At PREDICT time the sampler is skipped: exactly one prediction per input row.
    assert len(pipe.predict(X)) == len(y), (
        f"predict returned {len(pipe.predict(X))} rows for {len(y)} inputs "
        "— a sampler ran at predict time"
    )
    assert pipe.predict_proba(X).shape[0] == len(y), (
        f"predict_proba returned {pipe.predict_proba(X).shape[0]} rows for "
        f"{len(y)} inputs — a sampler ran at predict time"
    )
    # Predicting on a 7-row slice returns exactly 7 rows — no resampling on inference.
    assert len(pipe.predict(X[:7])) == 7, (
        f"predict returned {len(pipe.predict(X[:7]))} rows for a 7-row slice "
        "— a sampler ran at predict time"
    )


def test_class_balance_after_resample() -> None:
    """After SMOTE inside the pipeline the model trains on BALANCED data, while
    the validation fold keeps the ORIGINAL imbalance.

    A probe transformer inserted just before the final estimator captures the
    class balance the model actually sees (post-resampling on the train fold).
    The held-out fold is never resampled, so it must retain the ~5% event rate.
    """
    X, y = _make_imbalanced(n=1000, event_rate=0.05, seed=config.SEEDS[0])
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEEDS[0]
    )
    train_idx, val_idx = next(iter(skf.split(X, y)))
    X_train, y_train = X[train_idx], y[train_idx]
    X_val = X[val_idx]

    pipe = make_pipeline("logreg", "smote", config.SEEDS[0])

    probe = _ClassBalanceProbe()
    pipe.steps.insert(len(pipe.steps) - 1, ("balance_probe", probe))
    pipe.fit(X_train, y_train)

    # TRAINING data reaching the model is balanced 1:1 (SMOTE sampling_strategy=1.0)
    assert probe.n_pos_seen_ == probe.n_neg_seen_, (
        f"model saw imbalanced training data: {probe.n_neg_seen_} neg vs "
        f"{probe.n_pos_seen_} pos — SMOTE was not applied to the train fold"
    )
    # SMOTE oversamples: the model sees MORE rows than the raw train fold.
    assert probe.n_seen_ > len(train_idx), (
        f"model saw {probe.n_seen_} rows, not more than the {len(train_idx)} raw "
        "train rows — oversampling did not happen inside the pipeline"
    )

    # VALIDATION data is never inflated: one prediction per held-out row.
    preds = pipe.predict(X_val)
    assert len(preds) == len(val_idx), (
        f"predicted {len(preds)} items for a {len(val_idx)}-row validation fold "
        "— resampling leaked into validation"
    )
