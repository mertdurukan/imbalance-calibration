from imblearn.pipeline import Pipeline as ImbPipeline


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
    raise NotImplementedError
