import pandas as pd


def cell_id(dataset_id: int, model: str, condition: str, seed: int, fold: int) -> str:
    """Stable hash-free identifier: f'{dataset_id}__{model}__{condition}__s{seed}__f{fold}'"""
    raise NotImplementedError


def run_cell(dataset_id: int, model: str, condition: str, seed: int, fold: int) -> pd.DataFrame:
    """Fits ONE pipeline on ONE train fold, predicts on the held-out fold,
    returns a ONE-ROW dataframe matching the schema in §4.
    On exception: returns a one-row frame with status='failed' and the exception text.
    NEVER raises. NEVER skips.
    """
    raise NotImplementedError


def run_all(force: bool = False) -> None:
    """Iterates the full grid. Skips cells whose parquet already exists in
    results/cells/ unless force=True. Writes results/results.parquet at the end.
    """
    raise NotImplementedError


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from src.config import CONDITIONS, N_FOLDS, SEEDS

    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        ids = Path("datasets.txt").read_text().strip().splitlines()
        dataset_id = int(ids[0])
        for condition in CONDITIONS:
            for fold in range(N_FOLDS):
                run_cell(dataset_id, "xgboost", condition, SEEDS[0], fold)
    else:
        run_all()
