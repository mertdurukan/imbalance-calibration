"""scripts/verify_mlp_mechanism.py — MLP early-stopping mechanism check (read-only).

Post-hoc, read-only DIAGNOSIS of *why* the pre-registered ``mlp/none`` condition
collapses on the most imbalanced datasets (Table 1 H1 FAIL, and the near-chance /
backwards-slope replicates catalogued in ``scripts/diagnose_mlp.py`` Reports 1-4).

This script does NOT touch the pre-registered analysis. It:
  * reads the frozen ``results/results.parquet`` only to SELECT which cells to inspect
    (the mlp/none replicates whose frozen AUROC < NEAR_CHANCE_AUROC — the same 17 the
    diagnosis already flags) — nothing is dropped or re-weighted;
  * re-fits those exact cells as an EXACT REPRODUCTION: identical dataset, identical
    StratifiedKFold(shuffle=True, random_state=seed) split and fold index, identical
    frozen pipeline from ``src.conditions.make_pipeline`` (same imputer/scaler/model,
    same hyperparameters from ``src.config`` — NO hyperparameter is changed);
  * records, per replicate, MLPClassifier internals not stored in the frozen schema:
    ``n_iter_`` (iterations run before early stopping), ``best_validation_score_``
    (MLP's internal accuracy on its own validation split), and the held-out
    predicted-probability distribution (min/max/mean/std);
  * reproduces the held-out AUROC and checks it equals the frozen value, proving the
    reproduction is exact.

It fits fresh, in-memory pipelines and DISCARDS them. It writes nothing into
``results/cells/``, ``results/yprob/``, or ``results/results.parquet``. Output goes to
``results/diagnostics/`` (.cursorrules), stdout gets a short summary only.

Nothing here feeds the pre-registered H1/H2/H3 tables. The pre-registered result stands
exactly as computed by ``src/analyze.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config, metrics
from src.conditions import make_pipeline
from src.datasets import load_dataset

RESULTS_PATH: Path = Path("results") / "results.parquet"
DIAG_DIR: Path = Path("results") / "diagnostics"

REPLICATE_KEYS: tuple[str, ...] = ("dataset_id", "seed", "fold")

# Same near-chance flag used by scripts/diagnose_mlp.py to identify the broken cells.
NEAR_CHANCE_AUROC: float = 0.60

# High-event-rate control dataset (PREREG pool): jm1, event rate ~0.19.
CONTROL_DATASET_ID: int = 1053

# Reproduction is "exact" if the recomputed AUROC matches the frozen value to here.
AUROC_MATCH_TOL: float = 1e-9


def summarize(values: np.ndarray) -> tuple[float, float, float, int]:
    """Mean and 2.5/97.5 percentile interval over the FINITE values only.

    Mirrors ``src.analyze.summarize`` / ``scripts.diagnose_mlp.summarize`` so every
    summary interval here is computed identically to the pre-registered tables.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(finite))
    lo = float(np.percentile(finite, config.CI_LOWER_PERCENTILE))
    hi = float(np.percentile(finite, config.CI_UPPER_PERCENTILE))
    return mean, lo, hi, n


def _fmt(mean: float, lo: float, hi: float, decimals: int = 3) -> str:
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


def _refit_replicate(
    dataset_id: int, condition: str, seed: int, fold: int
) -> dict[str, float]:
    """EXACT reproduction of one frozen cell's fit; returns MLP internals + prob stats.

    This mirrors ``src.runner.run_cell`` line-for-line for the fit path (same
    StratifiedKFold, same fold index, same ``make_pipeline`` for model='mlp'), then
    reads attributes the frozen schema does not store. The fitted pipeline is
    discarded; nothing is cached. No hyperparameter is altered.
    """
    X, y = load_dataset(dataset_id)
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=seed)
    train_idx, val_idx = list(skf.split(X, y))[fold]

    pipe = make_pipeline("mlp", condition, seed)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Skipping features without any observed values",
            category=UserWarning,
        )
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])

    mlp = pipe.named_steps["model"]
    p_val = pipe.predict_proba(X.iloc[val_idx])[:, 1]
    y_val = np.asarray(y.iloc[val_idx], dtype=int)

    best_val = getattr(mlp, "best_validation_score_", float("nan"))
    return {
        "n_iter": float(mlp.n_iter_),
        "best_validation_score": float(best_val)
        if best_val is not None
        else float("nan"),
        "prob_min": float(np.min(p_val)),
        "prob_max": float(np.max(p_val)),
        "prob_mean": float(np.mean(p_val)),
        "prob_std": float(np.std(p_val)),
        "auroc_repro": float(metrics.auroc(y_val, p_val)),
    }


def _broken_keys(results: pd.DataFrame) -> pd.DataFrame:
    """The frozen mlp/none replicates with AUROC < NEAR_CHANCE_AUROC (the broken set).

    Selection is derived from the frozen data, not hardcoded; ``auroc_frozen`` is
    carried so the reproduction can be checked against it.
    """
    mlp_none = results[
        (results["model"] == "mlp")
        & (results["condition"] == "none")
        & (results["status"] == "ok")
    ]
    broken = mlp_none[mlp_none["auroc"] < NEAR_CHANCE_AUROC][
        list(REPLICATE_KEYS) + ["dataset_name", "event_rate", "auroc"]
    ].rename(columns={"auroc": "auroc_frozen"})
    return broken.sort_values(["event_rate", "dataset_id", "seed", "fold"]).reset_index(
        drop=True
    )


def report6_broken_none(broken: pd.DataFrame) -> pd.DataFrame:
    """Refit the 17 broken mlp/none replicates; record n_iter_, best_validation_score_,
    prob distribution, and reproduced-vs-frozen AUROC."""
    rows: list[dict[str, object]] = []
    for r in broken.itertuples(index=False):
        info = _refit_replicate(int(r.dataset_id), "none", int(r.seed), int(r.fold))
        match = abs(info["auroc_repro"] - float(r.auroc_frozen)) <= AUROC_MATCH_TOL
        rows.append(
            {
                "dataset_id": int(r.dataset_id),
                "dataset_name": r.dataset_name,
                "event_rate": f"{float(r.event_rate):.4f}",
                "seed": int(r.seed),
                "fold": int(r.fold),
                "n_iter_": int(info["n_iter"]),
                "best_val_score_": f"{info['best_validation_score']:.4f}",
                "prob_min": f"{info['prob_min']:.4f}",
                "prob_max": f"{info['prob_max']:.4f}",
                "prob_mean": f"{info['prob_mean']:.4f}",
                "prob_std": f"{info['prob_std']:.6f}",
                "auroc_frozen": f"{float(r.auroc_frozen):.4f}",
                "auroc_repro": f"{info['auroc_repro']:.4f}",
                "exact_match": "yes" if match else "NO",
            }
        )
    detail = pd.DataFrame(rows)
    _write(
        detail,
        "report6_mlp_none_broken_refit",
        "Report 6 — EXACT REFIT of the 17 broken mlp/none replicates "
        f"(frozen AUROC < {NEAR_CHANCE_AUROC:g})",
        [
            "Exact reproduction: same dataset, same StratifiedKFold(shuffle=True, "
            "random_state=seed) split & fold, same frozen pipeline & hyperparameters "
            "(src.config). No hyperparameter changed; fits are in-memory and discarded.",
            "n_iter_ = MLP iterations run before early stopping fired. "
            "best_val_score_ = MLPClassifier.best_validation_score_ (accuracy on its "
            "own internal early-stopping validation split, monitored because "
            "early_stopping=True per PREREG §4.2).",
            "prob_* = distribution of held-out predicted P(class=1). auroc_repro vs "
            "auroc_frozen: exact_match=yes confirms the reproduction reproduces the "
            "frozen cell.",
        ],
    )
    return detail


def report7_matched_smote(broken: pd.DataFrame) -> pd.DataFrame:
    """Refit the 17 mlp/SMOTE replicates matched on (dataset, seed, fold) to the broken
    mlp/none set; record n_iter_ (and the same internals) for contrast."""
    rows: list[dict[str, object]] = []
    for r in broken.itertuples(index=False):
        info = _refit_replicate(int(r.dataset_id), "smote", int(r.seed), int(r.fold))
        rows.append(
            {
                "dataset_id": int(r.dataset_id),
                "dataset_name": r.dataset_name,
                "event_rate": f"{float(r.event_rate):.4f}",
                "seed": int(r.seed),
                "fold": int(r.fold),
                "n_iter_": int(info["n_iter"]),
                "best_val_score_": f"{info['best_validation_score']:.4f}",
                "prob_min": f"{info['prob_min']:.4f}",
                "prob_max": f"{info['prob_max']:.4f}",
                "prob_mean": f"{info['prob_mean']:.4f}",
                "prob_std": f"{info['prob_std']:.6f}",
                "auroc_repro": f"{info['auroc_repro']:.4f}",
            }
        )
    detail = pd.DataFrame(rows)
    _write(
        detail,
        "report7_mlp_smote_matched_refit",
        "Report 7 — EXACT REFIT of the 17 matched mlp/SMOTE replicates "
        "(same dataset/seed/fold as Report 6)",
        [
            "Same (dataset, seed, fold) keys as the broken mlp/none set, condition "
            "SMOTE. Exact reproduction of the frozen mlp/smote fits; same frozen "
            "hyperparameters. n_iter_ is the contrast quantity vs Report 6.",
            "best_val_score_ here is MLP accuracy on its internal validation split of "
            "the SMOTE-BALANCED (1:1) training fold, so it is not comparable in level "
            "to Report 6's (which validates on an imbalanced split); reported for "
            "completeness.",
        ],
    )
    return detail


def report8_control_jm1(results: pd.DataFrame) -> pd.DataFrame:
    """Refit all mlp/none replicates on the high-event-rate control dataset (jm1);
    record n_iter_ and the same internals."""
    ctrl = results[
        (results["model"] == "mlp")
        & (results["condition"] == "none")
        & (results["dataset_id"] == CONTROL_DATASET_ID)
        & (results["status"] == "ok")
    ][list(REPLICATE_KEYS) + ["dataset_name", "event_rate", "auroc"]].rename(
        columns={"auroc": "auroc_frozen"}
    )
    ctrl = ctrl.sort_values(["seed", "fold"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for r in ctrl.itertuples(index=False):
        info = _refit_replicate(int(r.dataset_id), "none", int(r.seed), int(r.fold))
        match = abs(info["auroc_repro"] - float(r.auroc_frozen)) <= AUROC_MATCH_TOL
        rows.append(
            {
                "dataset_id": int(r.dataset_id),
                "dataset_name": r.dataset_name,
                "event_rate": f"{float(r.event_rate):.4f}",
                "seed": int(r.seed),
                "fold": int(r.fold),
                "n_iter_": int(info["n_iter"]),
                "best_val_score_": f"{info['best_validation_score']:.4f}",
                "prob_min": f"{info['prob_min']:.4f}",
                "prob_max": f"{info['prob_max']:.4f}",
                "prob_mean": f"{info['prob_mean']:.4f}",
                "prob_std": f"{info['prob_std']:.6f}",
                "auroc_frozen": f"{float(r.auroc_frozen):.4f}",
                "auroc_repro": f"{info['auroc_repro']:.4f}",
                "exact_match": "yes" if match else "NO",
            }
        )
    detail = pd.DataFrame(rows)
    _write(
        detail,
        "report8_mlp_none_jm1_control",
        f"Report 8 — CONTROL: exact refit of mlp/none on high-event-rate dataset "
        f"jm1 ({CONTROL_DATASET_ID}, event rate ~0.19)",
        [
            "All mlp/none replicates for the highest-event-rate dataset in the pool, "
            "exact reproduction (same split/fold/hyperparameters). Serves as the "
            "high-prevalence control for n_iter_.",
        ],
    )
    return detail


def _summary(
    broken_none: pd.DataFrame, matched_smote: pd.DataFrame, jm1: pd.DataFrame
) -> pd.DataFrame:
    """Group-level n_iter_ / prob_std summaries with 95% intervals (.cursorrules #6)."""

    def _num(df: pd.DataFrame, col: str) -> np.ndarray:
        return df[col].astype(float).to_numpy()

    rows: list[dict[str, object]] = []
    for label, df in (
        ("mlp/none broken (n=17)", broken_none),
        ("mlp/SMOTE matched (n=17)", matched_smote),
        (f"mlp/none jm1 control (n={len(jm1)})", jm1),
    ):
        ni = _num(df, "n_iter_")
        ps = _num(df, "prob_std")
        bv = _num(df, "best_val_score_")
        ni_m, ni_lo, ni_hi, n = summarize(ni)
        ps_m, ps_lo, ps_hi, _ = summarize(ps)
        bv_m, bv_lo, bv_hi, _ = summarize(bv)
        rows.append(
            {
                "group": label,
                "n": n,
                "n_iter_ (mean [95%])": _fmt(ni_m, ni_lo, ni_hi, 1),
                "n_iter_ min": int(np.min(ni)),
                "n_iter_ max": int(np.max(ni)),
                "prob_std (mean [95%])": _fmt(ps_m, ps_lo, ps_hi, 6),
                "best_val_score_ (mean [95%])": _fmt(bv_m, bv_lo, bv_hi, 4),
            }
        )
    summ = pd.DataFrame(rows)
    _write(
        summ,
        "report9_mlp_mechanism_summary",
        "Report 9 — mechanism summary: n_iter_, prob_std, best_validation_score_",
        [
            "Group-level means with 95% intervals (2.5/97.5 percentiles of the "
            "replicate distribution), matching the pre-registered tables' convention "
            "(METRICS.md §5). Descriptive only.",
            "This is a diagnostic; it does not enter the pre-registered H1/H2/H3 "
            "analysis, which stands exactly as computed by src/analyze.py.",
        ],
    )
    return summ


def main() -> None:
    """Run the read-only MLP mechanism check; write results/diagnostics/, print summary."""
    results = pd.read_parquet(RESULTS_PATH)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    broken = _broken_keys(results)
    broken_none = report6_broken_none(broken)
    matched_smote = report7_matched_smote(broken)
    jm1 = report8_control_jm1(results)
    _summary(broken_none, matched_smote, jm1)

    n_broken = len(broken_none)
    n_match = int((broken_none["exact_match"] == "yes").sum()) + int(
        (jm1["exact_match"] == "yes").sum()
    )
    n_checked = len(broken_none) + len(jm1)
    print(
        f"verify_mlp_mechanism: refit {n_broken} broken mlp/none + {len(matched_smote)} "
        f"matched mlp/smote + {len(jm1)} jm1 control (mlp/none). "
        f"Exact-AUROC-match {n_match}/{n_checked}. "
        f"Tables -> {DIAG_DIR}/ (report6..report9)."
    )


if __name__ == "__main__":
    main()
