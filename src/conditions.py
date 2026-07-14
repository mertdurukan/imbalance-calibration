from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src import config
from src.models import make_model


def _make_resampler(condition: str, seed: int) -> BaseSampler | None:
    """condition -> imblearn sampler (or None for 'none'). PREREG §4.3, SPEC §3.

    'none' returns None so NO resampler step exists in the pipeline — not a
    passthrough. Samplers balance minority:majority to 1:1.
    """
    if condition == "none":
        return None
    if condition == "rus":
        return RandomUnderSampler(
            sampling_strategy=config.RESAMPLE_SAMPLING_STRATEGY,
            random_state=seed,
        )
    if condition == "ros":
        return RandomOverSampler(
            sampling_strategy=config.RESAMPLE_SAMPLING_STRATEGY,
            random_state=seed,
        )
    if condition == "smote":
        return SMOTE(
            k_neighbors=config.SMOTE_K_NEIGHBORS,
            sampling_strategy=config.RESAMPLE_SAMPLING_STRATEGY,
            random_state=seed,
        )
    raise ValueError(
        f"unknown condition: {condition!r}; expected one of {config.CONDITIONS}"
    )


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
    steps: list[tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy=config.IMPUTER_STRATEGY)),
    ]

    # Scaler for logreg/mlp; OMITTED entirely for xgboost (trees are scale-invariant).
    if model_name in ("logreg", "mlp"):
        steps.append(("scaler", StandardScaler()))
    elif model_name != "xgboost":
        raise ValueError(
            f"unknown model name: {model_name!r}; expected one of {config.MODELS}"
        )

    # Resampler as an imblearn step so it runs at FIT time only. For 'none' the
    # step is omitted entirely — never added as a passthrough.
    resampler = _make_resampler(condition, seed)
    if resampler is not None:
        steps.append(("resampler", resampler))

    steps.append(("model", make_model(model_name, seed)))

    return ImbPipeline(steps)
