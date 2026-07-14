"""OpenML dataset selection and loading (PREREG §4.1, SPEC §3).

Selection is MECHANICAL and deterministic: the candidate pool and the numeric
filters are fixed, and the result is sorted by OpenML dataset ID ascending. No
dataset is ever hand-picked, and none is excluded because it errors, is slow, or
looks unusual — if it meets the criteria, it is IN.
"""

from __future__ import annotations

import sys
from pathlib import Path

import openml
import pandas as pd

from src.config import (
    MAX_MINORITY_RATE,
    MAX_MISSING_RATE,
    MAX_N_ROWS,
    MIN_MINORITY_RATE,
    MIN_N_ROWS,
    N_BINARY_CLASSES,
    N_DATASETS,
    OPENML_CC18_SUITE_ID,
    OPENML_IMBALANCED_TAG,
)

DATASETS_FILE = Path("datasets.txt")


def _candidate_ids() -> set[int]:
    """PREREG §4.1 candidate pool: OpenML-CC18 suite ∪ datasets tagged `imbalanced`.

    Both arms are tabular by construction (CC18 is a tabular benchmark; the
    `imbalanced` tag is queried directly), so no separate image/text exclusion is
    required.
    """
    suite = openml.study.get_suite(OPENML_CC18_SUITE_ID)
    cc18_ids = {int(x) for x in suite.data}

    tagged = openml.datasets.list_datasets(
        tag=OPENML_IMBALANCED_TAG, output_format="dataframe"
    )
    tagged_ids = {
        int(x) for x in (tagged["did"] if "did" in tagged.columns else tagged.index)
    }
    return cc18_ids | tagged_ids


def candidate_table() -> pd.DataFrame:
    """The mechanical candidate pool (PREREG §4.1), filtered and sorted by ID.

    Pool: OpenML-CC18 suite ∪ datasets tagged `imbalanced`, filtered with the
    frozen numeric criteria:
      - exactly 2 target classes
      - NumberOfInstances in [MIN_N_ROWS, MAX_N_ROWS]
      - minority class rate in [MIN_MINORITY_RATE, MAX_MINORITY_RATE]
      - missing-value rate <= MAX_MISSING_RATE, where the rate is CELL-LEVEL:
        NumberOfMissingValues / (NumberOfInstances * NumberOfFeatures)
        (see DEVIATIONS.md 2026-07-14 "definition of missing-value rate").
    Sorted by dataset ID ascending. Returns EVERY qualifying dataset (no
    truncation), so the rule's output is auditable.

    Columns: did, name, n_rows, n_features, minority_rate, missing_rate.
    """
    candidates = _candidate_ids()

    catalog = openml.datasets.list_datasets(output_format="dataframe")
    if "did" not in catalog.columns:
        catalog = catalog.reset_index()
    df = catalog[catalog["did"].isin(candidates)].copy()

    for col in (
        "did",
        "MinorityClassSize",
        "NumberOfInstances",
        "NumberOfClasses",
        "NumberOfFeatures",
        "NumberOfMissingValues",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    minority_rate = df["MinorityClassSize"] / df["NumberOfInstances"]
    # "missing-value rate" = CELL-LEVEL fraction of data cells that are missing
    # (see DEVIATIONS.md 2026-07-14 "definition of missing-value rate").
    missing_rate = df["NumberOfMissingValues"] / (
        df["NumberOfInstances"] * df["NumberOfFeatures"]
    )

    keep = (
        (df["NumberOfClasses"] == N_BINARY_CLASSES)
        & (df["NumberOfInstances"] >= MIN_N_ROWS)
        & (df["NumberOfInstances"] <= MAX_N_ROWS)
        & (minority_rate >= MIN_MINORITY_RATE)
        & (minority_rate <= MAX_MINORITY_RATE)
        & (missing_rate <= MAX_MISSING_RATE)
    )

    result = df.loc[keep].copy()
    result["minority_rate"] = minority_rate.loc[keep]
    result["missing_rate"] = missing_rate.loc[keep]
    result = result.rename(
        columns={"NumberOfInstances": "n_rows", "NumberOfFeatures": "n_features"}
    )
    result = result[["did", "name", "n_rows", "n_features", "minority_rate", "missing_rate"]]
    result = result.sort_values("did").reset_index(drop=True)
    result["did"] = result["did"].astype(int)
    result["n_rows"] = result["n_rows"].astype(int)
    result["n_features"] = result["n_features"].astype(int)
    return result


def select_datasets() -> list[int]:
    """Apply PREREG §4.1 criteria against the OpenML API.

    Returns the first N_DATASETS OpenML dataset IDs sorted ASCENDING by ID.
    MUST be deterministic. MUST be run once; result written to datasets.txt.
    """
    return candidate_table()["did"].tolist()[:N_DATASETS]


def load_dataset(dataset_id: int) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (X, y) with y in {0,1}, 1 = minority class.

    Categorical columns are one-hot encoded. Missing values are NOT touched here:
    imputation (median for numeric, most-frequent for categorical) is a pipeline
    step so it is fitted INSIDE the CV folds only. This function returns RAW
    features.
    """
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    X, y, categorical_indicator, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )

    class_counts = y.value_counts()
    minority_label = class_counts.idxmin()
    y_binary = (y == minority_label).astype(int)
    y_binary.name = y.name

    categorical_cols = [
        col for col, is_cat in zip(X.columns, categorical_indicator) if is_cat
    ]
    X_encoded = pd.get_dummies(X, columns=categorical_cols, dummy_na=False)
    return X_encoded, y_binary


if __name__ == "__main__":
    if DATASETS_FILE.exists():
        print(
            f"error: {DATASETS_FILE} already exists; refusing to overwrite. "
            "It is frozen once written (anti-cherry-picking guarantee).",
            file=sys.stderr,
        )
        sys.exit(1)

    candidates = candidate_table()
    with pd.option_context(
        "display.max_rows", None, "display.width", 200, "display.max_colwidth", 60
    ):
        print(
            f"Full candidate pool (mechanical, BEFORE truncation): "
            f"{len(candidates)} datasets passed all filters"
        )
        print(candidates.to_string(index=False))

    ids = candidates["did"].tolist()[:N_DATASETS]
    DATASETS_FILE.write_text("\n".join(map(str, ids)) + "\n")
    print(f"\nwrote first {len(ids)} dataset ids (by ascending ID) to {DATASETS_FILE}")
