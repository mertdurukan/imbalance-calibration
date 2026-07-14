from sklearn.base import BaseEstimator


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
    raise NotImplementedError
