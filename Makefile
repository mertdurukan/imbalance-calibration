PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: setup verify test datasets pilot reproduce

setup:
	python3.11 -m venv .venv && $(PIP) install --upgrade pip && $(PIP) install -r requirements.txt

# Functional smoke test, NOT a bare import check. Importing a library does not prove it
# works: `import statsmodels` succeeded while `statsmodels.api` was broken by an
# incompatible scipy. This target actually exercises the code paths the study depends on.
define VERIFY_PY
import numpy as np
import statsmodels.api as sm
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

rng = np.random.default_rng(0)
X = rng.normal(size=(20, 3))
y = np.array([1.0] * 8 + [0.0] * 12)
offset = rng.normal(size=20)

# 1. statsmodels GLM(Binomial) WITH offset= (the calibration-intercept code path)
sm.GLM(y, np.ones((20, 1)), family=sm.families.Binomial(), offset=offset).fit()

# 2. XGBClassifier fit + predict_proba
proba = XGBClassifier(n_estimators=10, eval_metric="logloss").fit(X, y).predict_proba(X)
assert proba.shape == (20, 2)

# 3. imblearn Pipeline with SMOTE, exercised via fit_resample (train-fold resampling)
X_res, y_res = Pipeline([("smote", SMOTE(k_neighbors=5, random_state=0))]).fit_resample(X, y)
assert len(X_res) == len(y_res) > len(X)

print("verify ok")
endef
export VERIFY_PY

verify:
	$(PYTHON) -c "$$VERIFY_PY"

test:
	$(PYTHON) -m pytest tests/ -x -q

datasets:
	$(PYTHON) -m src.datasets

pilot:
	$(PYTHON) -m src.runner pilot

reproduce: test
	$(PYTHON) -m src.runner
	$(PYTHON) -m src.analyze
