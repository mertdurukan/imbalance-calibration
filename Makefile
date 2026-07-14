PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: setup verify test datasets pilot reproduce

setup:
	python3.11 -m venv .venv && $(PIP) install --upgrade pip && $(PIP) install -r requirements.txt

verify:
	$(PYTHON) -c "import sklearn, imblearn, xgboost, statsmodels, pandas, openml; print('imports ok')"

test:
	$(PYTHON) -m pytest tests/ -x -q

datasets:
	$(PYTHON) -m src.datasets

pilot:
	$(PYTHON) -m src.runner pilot

reproduce: test
	$(PYTHON) -m src.runner
	$(PYTHON) -m src.analyze
