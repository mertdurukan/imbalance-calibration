"""scripts/diagnose_mlp.py — DIAGNOSIS ONLY (Task 7).

Read-only post-hoc diagnosis of the H1 FAIL for the ``mlp`` model. Reads ONLY the
frozen ``results/results.parquet`` and the saved ``results/yprob/`` files. Refits
nothing, changes no configuration / hyperparameter / model, and drops no cell from the
pre-registered analysis.

The pre-registered H1 result stands exactly as reported in Table 1
(``results/tables/table1_h1_discrimination.csv``). Report 5 below recomputes the mlp
H1 contrast under a POST-HOC exclusion; it is labelled EXPLORATORY and is NOT part of
the pre-registration.

Every reported point estimate carries a 95% interval (2.5/97.5 percentiles of the
replicate distribution), consistent with METRICS.md §5 and src/analyze.py. Long output
is written to ``results/diagnostics/`` (.cursorrules); stdout gets a short summary only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CI_LOWER_PERCENTILE, CI_UPPER_PERCENTILE

RESULTS_PATH: Path = Path("results") / "results.parquet"
DIAG_DIR: Path = Path("results") / "diagnostics"

# One replicate = one (dataset, seed, fold) cell, matching src/analyze.py.
REPLICATE_KEYS: tuple[str, ...] = ("dataset_id", "seed", "fold")

# Post-hoc DIAGNOSTIC thresholds. These are not pre-registered estimands; they are
# descriptive cut-points used only to COUNT and LABEL replicates in this diagnosis.
NEAR_CHANCE_AUROC: float = 0.60  # "near-chance" / collapsed-baseline flag
BACKWARDS_SLOPE: float = 0.0  # cal_slope < 0 ⇒ predictions ordered backwards

# H1 pre-registered threshold, reproduced from src/analyze.py for the exploratory
# recompute in Report 5 only.
H1_DELTA_AUROC_THRESHOLD: float = 0.01
CORRECTIONS: tuple[str, ...] = ("rus", "ros", "smote")


def summarize(values: np.ndarray) -> tuple[float, float, float, int]:
    """Mean and 2.5/97.5 percentile interval over the FINITE values only.

    Returns ``(mean, ci_low, ci_high, n)``; mirrors ``src.analyze.summarize`` so the
    diagnostic intervals are computed identically to the pre-registered tables.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(finite))
    lo = float(np.percentile(finite, CI_LOWER_PERCENTILE))
    hi = float(np.percentile(finite, CI_UPPER_PERCENTILE))
    return mean, lo, hi, n


def _fmt(mean: float, lo: float, hi: float, decimals: int = 4) -> str:
    """Format a point estimate with its 95% interval: ``mean [lo, hi]``."""
    if not np.isfinite(mean):
        return "nan"
    return f"{mean:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _write(df: pd.DataFrame, stem: str, title: str, notes: list[str]) -> None:
    """Persist a diagnostic table as both CSV (raw) and Markdown (formatted)."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIAG_DIR / f"{stem}.csv", index=False)
    lines: list[str] = [f"# {title}", ""]
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("| " + " | ".join("---" for _ in df.columns) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.to_list()) + " |")
    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"> {note}")
    lines.append("")
    (DIAG_DIR / f"{stem}.md").write_text("\n".join(lines))


def _mlp(results: pd.DataFrame) -> pd.DataFrame:
    """The ``ok`` mlp rows (the frozen run has no non-ok cells; .cursorrules #3)."""
    return results[(results["model"] == "mlp") & (results["status"] == "ok")].copy()


# --------------------------------------------------------------------------------
# Report 1 — distribution of mlp AUROC by condition; near-chance mlp/none count
# --------------------------------------------------------------------------------
def report1_auroc_distribution(results: pd.DataFrame) -> pd.DataFrame:
    """Distribution of mlp AUROC per condition + count of near-chance mlp/none cells.

    Reports the AUROC replicate distribution (mean [95%], min, max) for every mlp
    condition, then counts mlp/none replicates with AUROC < NEAR_CHANCE_AUROC, broken
    down by (dataset_id, event_rate).
    """
    mlp = _mlp(results)

    dist_rows: list[dict[str, object]] = []
    for cond in ["none", *CORRECTIONS]:
        a = mlp[mlp["condition"] == cond]["auroc"].to_numpy()
        mean, lo, hi, n = summarize(a)
        finite = a[np.isfinite(a)]
        dist_rows.append(
            {
                "condition": cond,
                "n": n,
                "auroc_mean_[95%]": _fmt(mean, lo, hi),
                "auroc_min": f"{float(np.min(finite)):.4f}" if finite.size else "nan",
                "auroc_max": f"{float(np.max(finite)):.4f}" if finite.size else "nan",
                f"n_auroc_lt_{NEAR_CHANCE_AUROC:g}": int(
                    np.sum(finite < NEAR_CHANCE_AUROC)
                ),
            }
        )
    dist = pd.DataFrame(dist_rows)
    _write(
        dist,
        "report1_mlp_auroc_distribution",
        "Report 1 — mlp AUROC distribution by condition",
        [
            f"n_auroc_lt_{NEAR_CHANCE_AUROC:g}: count of replicates with "
            f"AUROC < {NEAR_CHANCE_AUROC:g} (post-hoc near-chance flag; not "
            "pre-registered).",
            "95% interval = 2.5/97.5 percentiles of the replicate distribution.",
        ],
    )

    none = mlp[mlp["condition"] == "none"].copy()
    none["near_chance"] = none["auroc"] < NEAR_CHANCE_AUROC
    grp = (
        none.groupby(["dataset_id", "dataset_name", "event_rate"])
        .agg(
            n_replicates=("auroc", "size"),
            n_near_chance=("near_chance", "sum"),
            auroc_min=("auroc", "min"),
            auroc_max=("auroc", "max"),
        )
        .reset_index()
        .sort_values("event_rate")
    )
    grp["event_rate"] = grp["event_rate"].map(lambda v: f"{v:.4f}")
    grp["auroc_min"] = grp["auroc_min"].map(lambda v: f"{v:.4f}")
    grp["auroc_max"] = grp["auroc_max"].map(lambda v: f"{v:.4f}")
    grp["n_near_chance"] = grp["n_near_chance"].astype(int)
    _write(
        grp,
        "report1_mlp_none_near_chance_by_dataset",
        f"Report 1 — mlp/none replicates with AUROC < {NEAR_CHANCE_AUROC:g}, "
        "by dataset & event rate",
        [
            f"n_near_chance = replicates (of n_replicates) with AUROC < "
            f"{NEAR_CHANCE_AUROC:g}. Sorted by event_rate ascending.",
        ],
    )
    return grp


# --------------------------------------------------------------------------------
# Report 2 — mlp/none cal_slope < 0 (backwards predictions)
# --------------------------------------------------------------------------------
def report2_backwards_slope(results: pd.DataFrame) -> pd.DataFrame:
    """Count mlp/none replicates with cal_slope < 0, by (dataset_id, event_rate)."""
    mlp = _mlp(results)

    dist_rows: list[dict[str, object]] = []
    for cond in ["none", *CORRECTIONS]:
        s = mlp[mlp["condition"] == cond]["cal_slope"].to_numpy()
        mean, lo, hi, n = summarize(s)
        finite = s[np.isfinite(s)]
        dist_rows.append(
            {
                "condition": cond,
                "n": n,
                "cal_slope_mean_[95%]": _fmt(mean, lo, hi, 3),
                "cal_slope_min": f"{float(np.min(finite)):.3f}" if finite.size else "nan",
                "cal_slope_max": f"{float(np.max(finite)):.3f}" if finite.size else "nan",
                "n_slope_lt_0": int(np.sum(finite < BACKWARDS_SLOPE)),
            }
        )
    dist = pd.DataFrame(dist_rows)
    _write(
        dist,
        "report2_mlp_cal_slope_distribution",
        "Report 2 — mlp cal_slope distribution by condition",
        [
            "n_slope_lt_0: count of replicates with cal_slope < 0 (backwards "
            "ordering; post-hoc flag, not pre-registered).",
            "95% interval = 2.5/97.5 percentiles of the replicate distribution.",
        ],
    )

    none = mlp[mlp["condition"] == "none"].copy()
    none["backwards"] = none["cal_slope"] < BACKWARDS_SLOPE
    grp = (
        none.groupby(["dataset_id", "dataset_name", "event_rate"])
        .agg(
            n_replicates=("cal_slope", "size"),
            n_slope_lt_0=("backwards", "sum"),
            cal_slope_min=("cal_slope", "min"),
            cal_slope_max=("cal_slope", "max"),
        )
        .reset_index()
        .sort_values("event_rate")
    )
    grp["event_rate"] = grp["event_rate"].map(lambda v: f"{v:.4f}")
    grp["cal_slope_min"] = grp["cal_slope_min"].map(lambda v: f"{v:.3f}")
    grp["cal_slope_max"] = grp["cal_slope_max"].map(lambda v: f"{v:.3f}")
    grp["n_slope_lt_0"] = grp["n_slope_lt_0"].astype(int)
    _write(
        grp,
        "report2_mlp_none_backwards_by_dataset",
        "Report 2 — mlp/none replicates with cal_slope < 0, by dataset & event rate",
        [
            "n_slope_lt_0 = replicates (of n_replicates) with cal_slope < 0. "
            "Sorted by event_rate ascending.",
        ],
    )
    return grp


# --------------------------------------------------------------------------------
# Report 3 — mlp/none AUROC vs dataset event_rate
# --------------------------------------------------------------------------------
def report3_auroc_vs_event_rate(results: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset mlp/none AUROC (mean [95%]) alongside the dataset event rate."""
    none = _mlp(results)
    none = none[none["condition"] == "none"]

    rows: list[dict[str, object]] = []
    for (did, name, er), sub in none.groupby(
        ["dataset_id", "dataset_name", "event_rate"]
    ):
        mean, lo, hi, n = summarize(sub["auroc"].to_numpy())
        rows.append(
            {
                "dataset_id": int(did),
                "dataset_name": name,
                "event_rate": f"{float(er):.4f}",
                "n": n,
                "mlp_none_auroc_mean_[95%]": _fmt(mean, lo, hi),
                f"n_auroc_lt_{NEAR_CHANCE_AUROC:g}": int(
                    (sub["auroc"] < NEAR_CHANCE_AUROC).sum()
                ),
            }
        )
    tab = pd.DataFrame(rows).sort_values("event_rate").reset_index(drop=True)
    _write(
        tab,
        "report3_mlp_none_auroc_vs_event_rate",
        "Report 3 — mlp/none AUROC vs dataset event rate",
        [
            "One row per dataset, sorted by event_rate ascending (most imbalanced "
            "first). 95% interval = 2.5/97.5 percentiles over the 25 seed×fold "
            "replicates.",
        ],
    )
    return tab


# --------------------------------------------------------------------------------
# Report 4 — for broken mlp/none replicates, the paired ros/smote replicate
# --------------------------------------------------------------------------------
def report4_correction_on_broken(results: pd.DataFrame) -> pd.DataFrame:
    """For each mlp/none replicate with AUROC < NEAR_CHANCE_AUROC, list the AUROC and
    cal_slope of the SAME (dataset, seed, fold) under none / rus / ros / smote.

    Pairing is on (dataset_id, seed, fold) — the pre-registered replicate keys — so the
    ros/smote value shown is the exact matched fit, not an average.
    """
    mlp = _mlp(results)
    wide_auroc = mlp.pivot_table(
        index=list(REPLICATE_KEYS) + ["dataset_name", "event_rate"],
        columns="condition",
        values="auroc",
    ).reset_index()
    wide_slope = mlp.pivot_table(
        index=list(REPLICATE_KEYS),
        columns="condition",
        values="cal_slope",
    ).reset_index()

    broken = wide_auroc[wide_auroc["none"] < NEAR_CHANCE_AUROC].merge(
        wide_slope, on=list(REPLICATE_KEYS), suffixes=("_auroc", "_slope")
    )
    broken = broken.sort_values(["event_rate", "dataset_id", "seed", "fold"])

    rows: list[dict[str, object]] = []
    for r in broken.itertuples(index=False):
        d = r._asdict()
        rows.append(
            {
                "dataset_id": int(d["dataset_id"]),
                "dataset_name": d["dataset_name"],
                "event_rate": f"{float(d['event_rate']):.4f}",
                "seed": int(d["seed"]),
                "fold": int(d["fold"]),
                "auroc_none": f"{d['none_auroc']:.4f}",
                "auroc_rus": f"{d['rus_auroc']:.4f}",
                "auroc_ros": f"{d['ros_auroc']:.4f}",
                "auroc_smote": f"{d['smote_auroc']:.4f}",
                "slope_none": f"{d['none_slope']:.3f}",
                "slope_ros": f"{d['ros_slope']:.3f}",
                "slope_smote": f"{d['smote_slope']:.3f}",
            }
        )
    detail = pd.DataFrame(rows)
    _write(
        detail,
        "report4_correction_on_broken_replicates",
        f"Report 4 — matched corrections on broken mlp/none replicates "
        f"(none AUROC < {NEAR_CHANCE_AUROC:g})",
        [
            "Each row is ONE (dataset, seed, fold). Columns show the AUROC/cal_slope of "
            "that SAME replicate under each condition (paired on dataset_id, seed, "
            "fold).",
            "Sorted by event_rate ascending, then dataset_id, seed, fold.",
        ],
    )

    # Paired summary over the broken subset: Δ(correction − none) AUROC.
    summ_rows: list[dict[str, object]] = []
    for corr in CORRECTIONS:
        d = (broken[f"{corr}_auroc"] - broken["none_auroc"]).to_numpy()
        mean, lo, hi, n = summarize(d)
        summ_rows.append(
            {
                "contrast": f"{corr} - none  (broken subset only)",
                "n_pairs": n,
                "delta_auroc_mean_[95%]": _fmt(mean, lo, hi),
            }
        )
    summary = pd.DataFrame(summ_rows)
    _write(
        summary,
        "report4_correction_on_broken_summary",
        f"Report 4 (summary) — paired ΔAUROC on broken mlp/none replicates "
        f"(none AUROC < {NEAR_CHANCE_AUROC:g})",
        [
            "Paired ΔAUROC = AUROC(correction) − AUROC(none) within each broken "
            "(dataset, seed, fold) replicate, then summarised. Descriptive only; this "
            "is a post-hoc subset, not a pre-registered estimand.",
        ],
    )
    return detail


# --------------------------------------------------------------------------------
# Report 5 — EXPLORATORY H1 recompute excluding broken mlp/none replicates
# --------------------------------------------------------------------------------
def report5_exploratory_h1(results: pd.DataFrame) -> pd.DataFrame:
    """Recompute the mlp H1 paired ΔAUROC contrast EXCLUDING replicates whose mlp/none
    AUROC < NEAR_CHANCE_AUROC.

    EXPLORATORY — post-hoc exclusion, NOT pre-registered, reported for diagnosis only.
    The pre-registered H1 result stands as reported in Table 1.
    """
    mlp = _mlp(results)
    none = mlp[mlp["condition"] == "none"][list(REPLICATE_KEYS) + ["auroc"]]

    excluded_keys = none[none["auroc"] < NEAR_CHANCE_AUROC][list(REPLICATE_KEYS)]
    n_total = len(none)
    n_excluded = len(excluded_keys)

    none_kept = none[none["auroc"] >= NEAR_CHANCE_AUROC].rename(
        columns={"auroc": "auroc_none"}
    )

    rows: list[dict[str, object]] = []
    for corr in CORRECTIONS:
        c = mlp[mlp["condition"] == corr][list(REPLICATE_KEYS) + ["auroc"]].rename(
            columns={"auroc": "auroc_corr"}
        )
        merged = none_kept.merge(c, on=list(REPLICATE_KEYS))
        d = (merged["auroc_corr"] - merged["auroc_none"]).to_numpy()
        mean, lo, hi, n = summarize(d)
        passed = np.isfinite(mean) and abs(mean) < H1_DELTA_AUROC_THRESHOLD
        rows.append(
            {
                "model": "mlp",
                "contrast": f"{corr} - none",
                "n_pairs_kept": n,
                "delta_auroc_mean_[95%]": _fmt(mean, lo, hi),
                "H1_recomputed (|ΔAUROC|<0.01)": "PASS" if passed else "FAIL",
            }
        )
    tab = pd.DataFrame(rows)
    _write(
        tab,
        "report5_EXPLORATORY_h1_excluding_broken",
        "Report 5 — EXPLORATORY mlp H1 recompute (post-hoc exclusion of broken "
        "mlp/none replicates)",
        [
            "EXPLORATORY — post-hoc exclusion, not pre-registered, reported for "
            "diagnosis only. The pre-registered H1 result stands as reported in "
            "Table 1 (results/tables/table1_h1_discrimination.csv).",
            f"Exclusion rule: drop the (dataset, seed, fold) replicates whose mlp/none "
            f"AUROC < {NEAR_CHANCE_AUROC:g}. Excluded {n_excluded} of {n_total} "
            f"mlp/none replicates.",
            "Paired ΔAUROC = AUROC(correction) − AUROC(none) within each kept "
            "replicate, then summarised (2.5/97.5 percentile interval).",
        ],
    )
    return tab


def main() -> None:
    """Run all five read-only diagnostics; write results/diagnostics/, print a summary."""
    results = pd.read_parquet(RESULTS_PATH)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    mlp_none = _mlp(results)
    mlp_none = mlp_none[mlp_none["condition"] == "none"]
    n_none = len(mlp_none)
    n_near_chance = int((mlp_none["auroc"] < NEAR_CHANCE_AUROC).sum())
    n_backwards = int((mlp_none["cal_slope"] < BACKWARDS_SLOPE).sum())

    report1_auroc_distribution(results)
    report2_backwards_slope(results)
    report3_auroc_vs_event_rate(results)
    report4_correction_on_broken(results)
    report5_exploratory_h1(results)

    print(
        f"diagnose_mlp: {n_none} mlp/none replicates; "
        f"{n_near_chance} with AUROC < {NEAR_CHANCE_AUROC:g}; "
        f"{n_backwards} with cal_slope < 0. "
        f"Tables -> {DIAG_DIR}/ (report1..report5)."
    )


if __name__ == "__main__":
    main()
