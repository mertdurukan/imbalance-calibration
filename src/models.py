from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src import config


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
    if name == "logreg":
        return LogisticRegression(
            max_iter=config.LOGREG_MAX_ITER,
            random_state=seed,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=config.XGB_N_ESTIMATORS,
            learning_rate=config.XGB_LEARNING_RATE,
            max_depth=config.XGB_MAX_DEPTH,
            subsample=config.XGB_SUBSAMPLE,
            colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
            random_state=seed,
            eval_metric=config.XGB_EVAL_METRIC,
            tree_method=config.XGB_TREE_METHOD,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=config.MLP_HIDDEN_LAYER_SIZES,
            early_stopping=config.MLP_EARLY_STOPPING,
            max_iter=config.MLP_MAX_ITER,
            random_state=seed,
        )
    raise ValueError(f"unknown model name: {name!r}; expected one of {config.MODELS}")
