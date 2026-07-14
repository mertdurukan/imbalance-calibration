"""Config / no-tuning contract tests — .cursorrules prohibition #1, PREREG §4.2, §6.

These encode the "no tuning, seeds fixed" guarantee that makes the study a valid
comparison of CONDITIONS rather than a leaderboard. Each test must be able to FAIL
on a real violation:

- if someone freezes a different seed set, changes the grid dimensions, or edits
  ECE binning -> test_seeds_frozen fails;
- if someone wraps a model or a pipeline step in GridSearchCV / RandomizedSearchCV
  / a halving search / anything subclassing BaseSearchCV, or hands it a param_grid
  / param_distributions / cv -> test_models_have_no_tuning /
  test_pipelines_have_no_tuning fail;
- if any RNG is left unseeded (random_state=None) or ignores the requested seed ->
  test_models_are_seeded fails;
- if a search library is imported into the fit path -> test_no_search_imports fails;
- if xgboost silently turns on class weighting via scale_pos_weight ->
  test_xgboost_no_scale_pos_weight fails (PREREG §6 puts weighting out of scope).
"""

from __future__ import annotations

from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
)
from sklearn.model_selection._search import BaseSearchCV

from src import config
from src.conditions import make_pipeline
from src.models import make_model

# Every sklearn hyperparameter-search estimator. BaseSearchCV is the common base,
# so isinstance against it catches subclasses too; the concrete classes are listed
# to make the intent (and any failure message) explicit.
SEARCH_ESTIMATOR_CLASSES: tuple[type, ...] = (
    BaseSearchCV,
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)

# Parameter keys that only exist to configure a hyperparameter search.
TUNING_PARAM_KEYS: tuple[str, ...] = ("param_grid", "param_distributions", "cv")

# Source files that must stay free of any search / tuning machinery (the fit path).
_SRC_DIR: Path = Path(__file__).resolve().parent.parent / "src"
FIT_PATH_SOURCES: tuple[Path, ...] = (
    _SRC_DIR / "models.py",
    _SRC_DIR / "conditions.py",
    _SRC_DIR / "runner.py",
)

# Substrings that betray a hyperparameter search in the source text.
FORBIDDEN_SOURCE_SUBSTRINGS: tuple[str, ...] = (
    "GridSearchCV",
    "RandomizedSearchCV",
    "Optuna",
    "optuna",
    "param_grid",
    "param_distributions",
)

TEST_SEED: int = 7


def _pipeline_estimators(pipe: object) -> list[BaseEstimator]:
    """Every estimator step of an (imblearn) pipeline, in order.

    Walks ``pipe.steps`` so the assertions cover imputer/scaler/resampler/model,
    not just the final estimator.
    """
    return [est for _, est in pipe.steps]


def _assert_not_a_search(est: object, where: str) -> None:
    """Fail if ``est`` is a hyperparameter-search estimator or carries tuning params."""
    assert not isinstance(est, SEARCH_ESTIMATOR_CLASSES), (
        f"{where}: {type(est)!r} is a hyperparameter-search estimator "
        "(no tuning is allowed — .cursorrules #1, PREREG §4.2)"
    )
    if hasattr(est, "get_params"):
        params = est.get_params(deep=False)
        for key in TUNING_PARAM_KEYS:
            assert key not in params, (
                f"{where}: {type(est).__name__} exposes tuning param {key!r} "
                "— no hyperparameter search is allowed (.cursorrules #1)"
            )


def test_seeds_frozen() -> None:
    """The pre-registered constants defining the grid and ECE binning are frozen."""
    assert config.SEEDS == [0, 1, 2, 3, 4], config.SEEDS
    assert config.N_FOLDS == 5, config.N_FOLDS
    assert config.N_DATASETS == 8, config.N_DATASETS
    assert config.ECE_N_BINS == 15, config.ECE_N_BINS


def test_models_have_no_tuning() -> None:
    """No model factory returns a search estimator or a tuning-configured one."""
    for name in config.MODELS:
        model = make_model(name, 0)
        _assert_not_a_search(model, f"make_model({name!r}, 0)")


def test_pipelines_have_no_tuning() -> None:
    """No pipeline step, across all MODELS x CONDITIONS, is a search estimator."""
    for name in config.MODELS:
        for condition in config.CONDITIONS:
            pipe = make_pipeline(name, condition, 0)
            _assert_not_a_search(
                pipe, f"make_pipeline({name!r}, {condition!r}, 0)"
            )
            for est in _pipeline_estimators(pipe):
                _assert_not_a_search(
                    est,
                    f"make_pipeline({name!r}, {condition!r}, 0) step "
                    f"{type(est).__name__}",
                )


def test_models_are_seeded() -> None:
    """Every model, and every seed-bearing pipeline step, uses the requested seed.

    No estimator exposing ``random_state`` may leave it ``None`` (an unseeded RNG
    destroys reproducibility — .cursorrules "no implicit RNG").
    """
    for name in config.MODELS:
        model = make_model(name, TEST_SEED)
        assert model.get_params()["random_state"] == TEST_SEED, (
            f"make_model({name!r}, {TEST_SEED}) has random_state="
            f"{model.get_params()['random_state']!r}, expected {TEST_SEED}"
        )

    for name in config.MODELS:
        for condition in config.CONDITIONS:
            pipe = make_pipeline(name, condition, TEST_SEED)
            seeded_steps = 0
            for est in _pipeline_estimators(pipe):
                params = est.get_params(deep=False) if hasattr(est, "get_params") else {}
                if "random_state" not in params:
                    continue
                seeded_steps += 1
                assert params["random_state"] is not None, (
                    f"make_pipeline({name!r}, {condition!r}, {TEST_SEED}) step "
                    f"{type(est).__name__} has random_state=None (unseeded RNG)"
                )
                assert params["random_state"] == TEST_SEED, (
                    f"make_pipeline({name!r}, {condition!r}, {TEST_SEED}) step "
                    f"{type(est).__name__} has random_state={params['random_state']!r}, "
                    f"expected {TEST_SEED}"
                )
            # The model step always exposes random_state, so at least one step must.
            assert seeded_steps >= 1, (
                f"make_pipeline({name!r}, {condition!r}, {TEST_SEED}) exposes no "
                "seeded step — the model must carry random_state"
            )


def test_no_search_imports() -> None:
    """The fit-path source files contain no search / tuning machinery, verbatim."""
    for path in FIT_PATH_SOURCES:
        source = path.read_text()
        for needle in FORBIDDEN_SOURCE_SUBSTRINGS:
            assert needle not in source, (
                f"{path.name} contains forbidden search/tuning token {needle!r} "
                "(.cursorrules #1: no hyperparameter tuning anywhere)"
            )


def test_xgboost_no_scale_pos_weight() -> None:
    """xgboost must not silently enable class weighting (PREREG §6, out of scope).

    If ``scale_pos_weight`` were set, the "no correction" arm would not be a
    baseline — it would already be a correction.
    """
    params = make_model("xgboost", 0).get_params()
    assert params.get("scale_pos_weight") is None, (
        "xgboost has scale_pos_weight="
        f"{params.get('scale_pos_weight')!r}; class weighting is out of scope "
        "(PREREG §6) — it must be absent or None"
    )
