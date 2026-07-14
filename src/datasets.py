import sys
from pathlib import Path

import pandas as pd


def select_datasets() -> list[int]:
    """Apply PREREG §4.1 criteria against the OpenML API.
    Returns the first N_DATASETS OpenML dataset IDs sorted ASCENDING by ID.
    MUST be deterministic. MUST be run once; result written to datasets.txt.
    """
    raise NotImplementedError


def load_dataset(dataset_id: int) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (X, y) with y in {0,1}, 1 = minority class.
    Categorical columns are one-hot encoded. Missing values: median (numeric) /
    most-frequent (categorical) imputation, fitted INSIDE the CV pipeline, never here.
    This function returns RAW features; imputation is a pipeline step.
    """
    raise NotImplementedError


if __name__ == "__main__":
    out = Path("datasets.txt")
    if out.exists():
        sys.exit("datasets.txt already exists; refusing to overwrite")
    ids = select_datasets()
    out.write_text("\n".join(map(str, ids)) + "\n")
