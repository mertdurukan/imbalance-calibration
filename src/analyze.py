"""src/analyze.py — the pre-registered analysis (Task 6).

This module produces ONLY the pre-registered H1/H2/H3 estimands (PREREG §3, §4.5,
§4.6). The three main tables map 1:1 to the three hypotheses. No other cut of the
data appears here; anything else would be exploratory and is out of scope for this
module (see PREREG §4.6: "the pre-specified estimands are the H1–H3 contrasts only").

Statistics (METRICS.md §5):
- Every reported number carries a 95% interval. For across-replicate summaries the
  interval is the 2.5/97.5 percentiles of the replicate distribution — reported
  DESCRIPTIVELY, not as a test. CV folds are NOT independent, so NO t-test is run
  across folds.
- The primary contrast (Table 1, and the H3 verdict) is a PAIRED difference computed
  WITHIN each (dataset, seed, fold) replicate, then summarised. Unpaired means are
  never compared.
- Cells with status != "ok" are REPORTED (status audit + per-group ok counts), never
  silently dropped (.cursorrules #3). In the frozen run there are none.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import metrics
from src.config import (
    CI_LOWER_PERCENTILE,
    CI_UPPER_PERCENTILE,
    CONDITIONS,
    MODELS,
    NET_BENEFIT_THRESHOLDS,
)

RESULTS_DIR: Path = Path("results")
RESULTS_PATH: Path = RESULTS_DIR / "results.parquet"
TABLES_DIR: Path = RESULTS_DIR / "tables"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# One replicate = one (dataset, seed, fold) cell. Per (model, condition) there are
# N_datasets * N_seeds * N_folds of them. Pairing for the primary contrast is on
# exactly these keys (METRICS.md §5).
REPLICATE_KEYS: tuple[str, ...] = ("dataset_id", "seed", "fold")

# Fit conditions vs the reference. "none_threshold" is NOT a fit; it reuses the
# "none" predictions at a shifted decision threshold and is materialised at analysis
# time (METRICS.md §4.1).
REFERENCE_CONDITION: str = "none"
CORRECTIONS: tuple[str, ...] = ("rus", "ros", "smote")
NONE_THRESHOLD: str = "none_threshold"

# H1 pre-registered threshold (PREREG §3): |ΔAUROC| < 0.01 → "no meaningful
# improvement".
H1_DELTA_AUROC_THRESHOLD: float = 0.01

# Net-Benefit decision thresholds for the TABLE (PREREG §4.5): {event rate, 0.05,
# 0.10, 0.20}. event_rate is per-dataset and handled separately.
NB_FIXED_THRESHOLDS: tuple[float, ...] = tuple(NET_BENEFIT_THRESHOLDS)

# Decision-curve x-axis sweep for FIGURE 1 (a presentation choice, not a
# pre-registered estimand — the pre-registered NB points are in Table 3).
NB_SWEEP: np.ndarray = np.linspace(0.01, 0.50, 50)


# --------------------------------------------------------------------------------
# summary helpers
# --------------------------------------------------------------------------------
def summarize(values: np.ndarray) -> tuple[float, float, float, int]:
    """Mean and 2.5/97.5 percentile interval of a replicate distribution.

    Returns ``(mean, ci_low, ci_high, n)`` over the FINITE values only. ``n`` is the
    number of finite replicates that entered the summary, so any non-finite / missing
    replicate is visible rather than silently absorbed.
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


def _fmt(mean: float, lo: float, hi: float, decimals: int) -> str:
    """Format a point estimate with its 95% interval: ``mean [lo, hi]``."""
    if not np.isfinite(mean):
        return "nan"
    return f"{mean:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _write_table(df: pd.DataFrame, stem: str, title: str, notes: list[str]) -> None:
    """Persist a table as both CSV (raw numbers) and Markdown (formatted)."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{stem}.csv", index=False)

    lines: list[str] = [f"# {title}", ""]
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.to_list()) + " |")
    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"> {note}")
    lines.append("")
    (TABLES_DIR / f"{stem}.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------------
# status audit — report, never drop (.cursorrules #3)
# --------------------------------------------------------------------------------
def status_audit(results: pd.DataFrame) -> pd.DataFrame:
    """Count cells by status and list every failure verbatim.

    Downstream summaries aggregate only ``status == "ok"`` rows, but the counts and
    error strings are written out so that a non-ok cell is always disclosed.
    """
    counts = results["status"].value_counts().to_dict()
    n_total = int(len(results))
    n_ok = int(counts.get("ok", 0))
    rows: list[dict[str, object]] = [
        {"item": "total_cells", "value": str(n_total)},
        {"item": "status_ok", "value": str(n_ok)},
        {"item": "status_failed", "value": str(n_total - n_ok)},
    ]
    failed = results[results["status"] != "ok"]
    for _, r in failed.iterrows():
        rows.append(
            {
                "item": (
                    f"FAILED {r['dataset_id']}/{r['model']}/{r['condition']}"
                    f"/s{r['seed']}/f{r['fold']}"
                ),
                "value": str(r.get("error", "")),
            }
        )
    audit = pd.DataFrame(rows)
    _write_table(
        audit,
        "status_audit",
        "Status audit (all cells accounted for; none dropped)",
        [
            "Every cell in results.parquet is counted here. Summaries below aggregate "
            "only status=='ok' replicates; any failure is listed above, never removed "
            "(.cursorrules #3).",
        ],
    )
    return audit


# --------------------------------------------------------------------------------
# TABLE 1 — H1: discrimination (paired ΔAUROC, ΔAUPRC vs `none`)
# --------------------------------------------------------------------------------
def table1_discrimination(results: pd.DataFrame) -> pd.DataFrame:
    """Paired ΔAUROC and ΔAUPRC of each correction vs `none`, per model class.

    The difference is formed WITHIN each (dataset, seed, fold) replicate first
    (metric(correction) − metric(none)), THEN summarised across replicates
    (METRICS.md §5). PASS/FAIL is stated against the pre-registered H1 threshold
    |mean ΔAUROC| < 0.01 (PREREG §3).
    """
    ok = results[results["status"] == "ok"]
    records: list[dict[str, object]] = []
    for model in MODELS:
        m = ok[ok["model"] == model]
        none = m[m["condition"] == REFERENCE_CONDITION][
            list(REPLICATE_KEYS) + ["auroc", "auprc"]
        ]
        for corr in CORRECTIONS:
            c = m[m["condition"] == corr][
                list(REPLICATE_KEYS) + ["auroc", "auprc"]
            ]
            merged = none.merge(
                c, on=list(REPLICATE_KEYS), suffixes=("_none", "_corr")
            )
            d_auroc = (merged["auroc_corr"] - merged["auroc_none"]).to_numpy()
            d_auprc = (merged["auprc_corr"] - merged["auprc_none"]).to_numpy()
            ar_mean, ar_lo, ar_hi, n = summarize(d_auroc)
            pr_mean, pr_lo, pr_hi, _ = summarize(d_auprc)
            passed = np.isfinite(ar_mean) and abs(ar_mean) < H1_DELTA_AUROC_THRESHOLD
            records.append(
                {
                    "model": model,
                    "contrast": f"{corr} - none",
                    "n_pairs": n,
                    "delta_auroc_mean": ar_mean,
                    "delta_auroc_lo": ar_lo,
                    "delta_auroc_hi": ar_hi,
                    "delta_auprc_mean": pr_mean,
                    "delta_auprc_lo": pr_lo,
                    "delta_auprc_hi": pr_hi,
                    "H1_pass": "PASS" if passed else "FAIL",
                }
            )
    raw = pd.DataFrame.from_records(records)

    display = pd.DataFrame(
        {
            "model": raw["model"],
            "contrast": raw["contrast"],
            "n": raw["n_pairs"],
            "ΔAUROC (mean [95%])": [
                _fmt(r.delta_auroc_mean, r.delta_auroc_lo, r.delta_auroc_hi, 4)
                for r in raw.itertuples()
            ],
            "ΔAUPRC (mean [95%])": [
                _fmt(r.delta_auprc_mean, r.delta_auprc_lo, r.delta_auprc_hi, 4)
                for r in raw.itertuples()
            ],
            "H1 (|ΔAUROC|<0.01)": raw["H1_pass"],
        }
    )
    _write_table(
        display,
        "table1_h1_discrimination",
        "Table 1 — H1 (discrimination): paired ΔAUROC / ΔAUPRC vs `none`",
        [
            "Paired within each (dataset, seed, fold): difference is metric(correction) "
            "− metric(none), computed per replicate, then summarised. Unpaired means "
            "are never compared (METRICS.md §5).",
            "95% interval = 2.5/97.5 percentiles of the replicate distribution "
            "(descriptive; CV folds are not independent, so no t-test is run).",
            "H1 column: PASS iff |mean ΔAUROC| < 0.01 (PREREG §3). The full interval is "
            "shown so the reader can apply the directional falsification criterion "
            "(improvement ≥ 0.01 with a 95% interval excluding zero).",
        ],
    )
    raw.to_csv(TABLES_DIR / "table1_h1_discrimination.csv", index=False)
    return display


# --------------------------------------------------------------------------------
# TABLE 2 — H2: calibration (slope, intercept, ECE, Brier per model × condition)
# --------------------------------------------------------------------------------
def table2_calibration(results: pd.DataFrame) -> pd.DataFrame:
    """cal_slope, cal_intercept, ECE, Brier per model × condition, mean + 95%.

    PASS/FAIL per metric is stated for each CORRECTION against the `none` reference in
    the direction H2 predicts (PREREG §3 / task): slope pushed AWAY from 1.0,
    intercept pushed AWAY from 0.0, ECE RAISED. H2 makes no directional prediction for
    Brier, so Brier is reported without a PASS/FAIL verdict.
    """
    ok = results[results["status"] == "ok"]
    metric_cols = ("cal_slope", "cal_intercept", "ece", "brier")

    # Per-(model, condition) summaries and reference means for the verdicts.
    summ: dict[tuple[str, str, str], tuple[float, float, float, int]] = {}
    for model in MODELS:
        for cond in CONDITIONS:
            sub = ok[(ok["model"] == model) & (ok["condition"] == cond)]
            for col in metric_cols:
                summ[(model, cond, col)] = summarize(sub[col].to_numpy())

    records: list[dict[str, object]] = []
    display_rows: list[dict[str, object]] = []
    for model in MODELS:
        for cond in CONDITIONS:
            slope = summ[(model, cond, "cal_slope")]
            inter = summ[(model, cond, "cal_intercept")]
            ece_s = summ[(model, cond, "ece")]
            brier = summ[(model, cond, "brier")]

            if cond == REFERENCE_CONDITION:
                v_slope = v_inter = v_ece = "ref"
            else:
                ref_slope = summ[(model, REFERENCE_CONDITION, "cal_slope")][0]
                ref_inter = summ[(model, REFERENCE_CONDITION, "cal_intercept")][0]
                ref_ece = summ[(model, REFERENCE_CONDITION, "ece")][0]
                v_slope = (
                    "PASS"
                    if abs(slope[0] - 1.0) > abs(ref_slope - 1.0)
                    else "FAIL"
                )
                v_inter = (
                    "PASS" if abs(inter[0]) > abs(ref_inter) else "FAIL"
                )
                v_ece = "PASS" if ece_s[0] > ref_ece else "FAIL"

            records.append(
                {
                    "model": model,
                    "condition": cond,
                    "n": slope[3],
                    "cal_slope_mean": slope[0],
                    "cal_slope_lo": slope[1],
                    "cal_slope_hi": slope[2],
                    "cal_intercept_mean": inter[0],
                    "cal_intercept_lo": inter[1],
                    "cal_intercept_hi": inter[2],
                    "ece_mean": ece_s[0],
                    "ece_lo": ece_s[1],
                    "ece_hi": ece_s[2],
                    "brier_mean": brier[0],
                    "brier_lo": brier[1],
                    "brier_hi": brier[2],
                    "H2_slope": v_slope,
                    "H2_intercept": v_inter,
                    "H2_ece": v_ece,
                }
            )
            display_rows.append(
                {
                    "model": model,
                    "condition": cond,
                    "n": slope[3],
                    "cal_slope (mean [95%])": _fmt(*slope[:3], 3),
                    "cal_intercept (mean [95%])": _fmt(*inter[:3], 3),
                    "ECE (mean [95%])": _fmt(*ece_s[:3], 4),
                    "Brier (mean [95%])": _fmt(*brier[:3], 4),
                    "H2 slope→away 1.0": v_slope,
                    "H2 intercept→away 0.0": v_inter,
                    "H2 ECE↑": v_ece,
                }
            )
    raw = pd.DataFrame.from_records(records)
    display = pd.DataFrame(display_rows)
    _write_table(
        display,
        "table2_h2_calibration",
        "Table 2 — H2 (calibration): slope, intercept, ECE, Brier per model × condition",
        [
            "Absolute per-condition metrics. Mean + 95% interval (2.5/97.5 percentiles "
            "of the replicate distribution; descriptive, no t-test — folds not "
            "independent).",
            "Perfect calibration: slope = 1.0, intercept = 0.0. H2 predicts corrections "
            "push slope AWAY from 1.0, intercept AWAY from 0.0, and RAISE ECE.",
            "PASS/FAIL (corrections only) is vs the `none` reference (row 'ref') in "
            "each model: slope PASS iff |mean−1| > |none−1|; intercept PASS iff |mean| > "
            "|none|; ECE PASS iff mean > none. Brier has no pre-registered direction "
            "under H2 and carries no verdict.",
        ],
    )
    raw.to_csv(TABLES_DIR / "table2_h2_calibration.csv", index=False)
    return display


# --------------------------------------------------------------------------------
# Net Benefit recomputed from saved y_prob files (METRICS.md §4.1)
# --------------------------------------------------------------------------------
def _net_benefit_per_cell(results: pd.DataFrame) -> pd.DataFrame:
    """Recompute Net Benefit from the saved per-item y_prob files at analysis time.

    For every ok cell of a FIT condition, loads ``y_prob_path`` and evaluates NB at
    each pre-registered threshold {event_rate, 0.05, 0.10, 0.20}. The `none_threshold`
    arm (METRICS.md §4.1) is then materialised by copying the `none` cells' NB — it is
    NOT a separate fit; it reuses the `none` predictions at a shifted decision
    threshold, so at each tabulated threshold it equals `none` by construction.

    Returns one row per (dataset_id, model, condition, seed, fold) with NB columns.
    """
    thresholds = ("eventrate",) + tuple(
        f"{int(t * 1000):03d}" for t in NB_FIXED_THRESHOLDS
    )
    rows: list[dict[str, object]] = []
    ok = results[results["status"] == "ok"]
    for r in ok.itertuples():
        yp = pd.read_parquet(r.y_prob_path)
        y = yp["y_true"].to_numpy()
        p = yp["y_prob"].to_numpy()
        pts = (float(r.event_rate),) + NB_FIXED_THRESHOLDS
        rec: dict[str, object] = {
            "dataset_id": int(r.dataset_id),
            "dataset_name": r.dataset_name,
            "event_rate": float(r.event_rate),
            "model": r.model,
            "condition": r.condition,
            "seed": int(r.seed),
            "fold": int(r.fold),
        }
        for label, pt in zip(thresholds, pts):
            rec[f"nb_{label}"] = metrics.net_benefit(y, p, pt=pt)
        rows.append(rec)
    nb = pd.DataFrame.from_records(rows)

    # none_threshold: reuse the `none` predictions at the shifted threshold.
    none_rows = nb[nb["condition"] == REFERENCE_CONDITION].copy()
    none_rows["condition"] = NONE_THRESHOLD
    return pd.concat([nb, none_rows], ignore_index=True)


# --------------------------------------------------------------------------------
# TABLE 3 — H3: decisions (Net Benefit incl. none_threshold) + verdict
# --------------------------------------------------------------------------------
def table3_decisions(nb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Net Benefit per model × condition at each pre-registered threshold, plus the
    paired H3 verdict.

    The main table reports mean + 95% NB for every condition INCLUDING
    `none_threshold`. The verdict table forms the PAIRED difference
    (none_threshold − correction) within each (dataset, seed, fold) and states
    PASS/FAIL: H3 predicts `none + threshold shift` matches or beats every correction
    (PREREG §3); it is falsified where a correction beats `none_threshold` with a 95%
    interval excluding zero.
    """
    nb_labels = [("eventrate", "NB@eventrate"), *[
        (f"{int(t * 1000):03d}", f"NB@{t:.2f}") for t in NB_FIXED_THRESHOLDS
    ]]
    conditions_ordered = [REFERENCE_CONDITION, *CORRECTIONS, NONE_THRESHOLD]

    # --- main NB table (means per model × condition) ---
    main_records: list[dict[str, object]] = []
    main_display: list[dict[str, object]] = []
    for model in MODELS:
        for cond in conditions_ordered:
            sub = nb[(nb["model"] == model) & (nb["condition"] == cond)]
            rec: dict[str, object] = {"model": model, "condition": cond}
            disp: dict[str, object] = {"model": model, "condition": cond}
            n_val = 0
            for key, pretty in nb_labels:
                mean, lo, hi, n = summarize(sub[f"nb_{key}"].to_numpy())
                n_val = n
                rec[f"{key}_mean"] = mean
                rec[f"{key}_lo"] = lo
                rec[f"{key}_hi"] = hi
                disp[f"{pretty} (mean [95%])"] = _fmt(mean, lo, hi, 4)
            rec["n"] = n_val
            disp["n"] = n_val
            main_records.append(rec)
            main_display.append(disp)
    raw_main = pd.DataFrame.from_records(main_records)
    display_main = pd.DataFrame(main_display)
    # move n next to identifiers for readability
    cols = ["model", "condition", "n"] + [
        c for c in display_main.columns if c not in ("model", "condition", "n")
    ]
    display_main = display_main[cols]
    _write_table(
        display_main,
        "table3_h3_decisions",
        "Table 3 — H3 (decisions): Net Benefit per model × condition",
        [
            "Net Benefit recomputed at analysis time from the saved y_prob files "
            "(METRICS.md §4.1). Thresholds: {event rate, 0.05, 0.10, 0.20}.",
            "`none_threshold` reuses the `none` predictions at a shifted decision "
            "threshold; it is NOT a separate fit, so at each tabulated threshold it "
            "equals `none` by construction (METRICS.md §4.1). It is listed to make the "
            "H3 contrast explicit; the verdict is in table3_h3_verdict.",
            "Mean + 95% interval (2.5/97.5 percentiles of the replicate distribution; "
            "descriptive, no t-test — folds not independent).",
        ],
    )
    raw_main.to_csv(TABLES_DIR / "table3_h3_decisions.csv", index=False)

    # --- paired H3 verdict: (none_threshold − correction) per model × threshold ---
    verdict_records: list[dict[str, object]] = []
    verdict_display: list[dict[str, object]] = []
    for model in MODELS:
        m = nb[nb["model"] == model]
        nt = m[m["condition"] == NONE_THRESHOLD]
        for key, pretty in nb_labels:
            for corr in CORRECTIONS:
                c = m[m["condition"] == corr]
                merged = nt.merge(
                    c,
                    on=list(REPLICATE_KEYS),
                    suffixes=("_nt", "_corr"),
                )
                diff = (
                    merged[f"nb_{key}_nt"] - merged[f"nb_{key}_corr"]
                ).to_numpy()
                mean, lo, hi, n = summarize(diff)
                # H3 falsified for this cell iff the correction BEATS none_threshold
                # (diff < 0) with a 95% interval excluding zero (hi < 0).
                falsified = np.isfinite(hi) and hi < 0.0
                verdict_records.append(
                    {
                        "model": model,
                        "threshold": pretty,
                        "contrast": f"none_threshold - {corr}",
                        "n_pairs": n,
                        "diff_mean": mean,
                        "diff_lo": lo,
                        "diff_hi": hi,
                        "H3_pass": "FAIL" if falsified else "PASS",
                    }
                )
                verdict_display.append(
                    {
                        "model": model,
                        "threshold": pretty,
                        "contrast": f"none_threshold − {corr}",
                        "n": n,
                        "ΔNB (mean [95%])": _fmt(mean, lo, hi, 4),
                        "H3 (nt ≥ corr)": "FAIL" if falsified else "PASS",
                    }
                )
    raw_verdict = pd.DataFrame.from_records(verdict_records)
    display_verdict = pd.DataFrame(verdict_display)
    _write_table(
        display_verdict,
        "table3_h3_verdict",
        "Table 3 (H3 verdict): paired (none_threshold − correction) Net Benefit",
        [
            "Paired within each (dataset, seed, fold): ΔNB = NB(none_threshold) − "
            "NB(correction) per replicate, then summarised (METRICS.md §5).",
            "Positive ΔNB means `none + threshold shift` is at least as good as the "
            "correction. 95% interval = 2.5/97.5 percentiles (descriptive; no t-test).",
            "H3 PASS iff the correction does NOT beat `none_threshold` with a 95% "
            "interval excluding zero (i.e. not hi < 0). H3 is falsified where a "
            "correction beats threshold-shifting on Net Benefit with a CI excluding "
            "zero (PREREG §3).",
        ],
    )
    raw_verdict.to_csv(TABLES_DIR / "table3_h3_verdict.csv", index=False)
    return display_main, display_verdict


# --------------------------------------------------------------------------------
# FIGURE 1 / S1 — decision curves (all conditions + treat-all + treat-none)
# --------------------------------------------------------------------------------
YProbCache = dict[tuple[int, str, str], list[tuple[np.ndarray, np.ndarray]]]

# Shared colour scheme for the fit conditions across both decision-curve figures.
CONDITION_COLORS: dict[str, str] = {
    "none": "black",
    "rus": "tab:blue",
    "ros": "tab:green",
    "smote": "tab:red",
}


def _load_yprob_cache(results: pd.DataFrame) -> YProbCache:
    """Preload held-out (y_true, y_prob) per (dataset, model, condition) once.

    Both decision-curve figures consume the same predictions, so the ~2400 saved
    y_prob files are read a single time and shared.
    """
    ok = results[results["status"] == "ok"]
    cache: YProbCache = {}
    for r in ok.itertuples():
        key = (int(r.dataset_id), r.model, r.condition)
        yp = pd.read_parquet(r.y_prob_path)
        cache.setdefault(key, []).append(
            (yp["y_true"].to_numpy(), yp["y_prob"].to_numpy())
        )
    return cache


def _nb_curve_matrix(
    y_probs: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Per-cell Net-Benefit curves over NB_SWEEP: shape (n_cells, len(NB_SWEEP))."""
    per_cell = np.empty((len(y_probs), NB_SWEEP.size), dtype=float)
    for i, (y, p) in enumerate(y_probs):
        for j, pt in enumerate(NB_SWEEP):
            per_cell[i, j] = metrics.net_benefit(y, p, pt=float(pt))
    return per_cell


def _mean_nb_curve(y_probs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Mean Net-Benefit curve over the NB_SWEEP, averaged across replicate cells."""
    return np.nanmean(_nb_curve_matrix(y_probs), axis=0)


def figure_s1_decision_curves_by_dataset(
    results: pd.DataFrame, cache: YProbCache | None = None
) -> Path:
    """Supplementary Figure S1: decision curves for every (dataset, model).

    All 8 datasets × 3 models are shown as small multiples so no dataset is
    cherry-picked. This is the full, unpooled companion to the condensed Figure 1
    that appears in the paper body. The `none` curve IS the threshold-shifted
    `none_threshold` arm (a decision curve sweeps the threshold), so it is not drawn
    twice. Mean curves are plotted; the pre-registered NB points with 95% intervals
    are in Table 3.
    """
    ok = results[results["status"] == "ok"]
    dataset_ids = sorted(ok["dataset_id"].unique())
    names = {int(r.dataset_id): r.dataset_name for r in ok.itertuples()}
    if cache is None:
        cache = _load_yprob_cache(results)

    n_rows = len(dataset_ids)
    n_cols = len(MODELS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), squeeze=False
    )

    for i, did in enumerate(dataset_ids):
        event_rate = float(ok[ok["dataset_id"] == did]["event_rate"].iloc[0])
        # treat-all reference (METRICS.md §4); treat-none is NB = 0.
        odds = NB_SWEEP / (1.0 - NB_SWEEP)
        treat_all = event_rate - (1.0 - event_rate) * odds
        for j, model in enumerate(MODELS):
            ax = axes[i][j]
            for cond in CONDITIONS:
                cells = cache.get((did, model, cond))
                if not cells:
                    continue
                curve = _mean_nb_curve(cells)
                label = "none (=none_threshold)" if cond == "none" else cond
                ax.plot(
                    NB_SWEEP, curve, color=CONDITION_COLORS[cond], lw=1.4, label=label
                )
            ax.plot(NB_SWEEP, treat_all, color="gray", ls="--", lw=1.0, label="treat-all")
            ax.axhline(0.0, color="gray", ls=":", lw=1.0, label="treat-none")
            ax.axvline(event_rate, color="purple", ls="-.", lw=0.8, alpha=0.6)
            ax.set_ylim(bottom=min(-0.02, float(np.nanmin(treat_all))))
            if j == 0:
                ax.set_ylabel(f"{names[did]}\n({did})\nNet Benefit", fontsize=8)
            if i == 0:
                ax.set_title(model, fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("threshold probability", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0 and j == n_cols - 1:
                ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        "Figure S1 — Decision curves by dataset (all conditions + treat-all/treat-none). "
        "Purple dash-dot = event rate. Mean over 25 seed×fold replicates; "
        "95% intervals for the pre-registered thresholds are in Table 3.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "figure_S1_decision_curves_by_dataset.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure1_decision_curves(
    results: pd.DataFrame, cache: YProbCache | None = None
) -> Path:
    """Condensed Figure 1 for the paper body: one panel per model (logreg / xgboost /
    mlp), pooling the decision curve across all 8 datasets.

    For each (model, condition) the Net-Benefit curve is computed per replicate cell
    over the threshold sweep and POOLED across all 8 datasets (each contributes 25
    seed×fold cells → 200 cells). The line is the mean across those cells and the band
    is the 2.5/97.5 percentiles of the cell distribution — the same descriptive
    interval convention as the tables (METRICS.md §5); it is NOT a test.

    treat-all and treat-none reference lines are kept (METRICS.md §4). Because event
    rate differs across datasets, the treat-all line is the mean of the eight
    per-dataset treat-all curves (equal weight per dataset). The unpooled,
    per-dataset curves are in Figure S1 so no dataset is hidden by the pooling.
    """
    ok = results[results["status"] == "ok"]
    dataset_ids = sorted(ok["dataset_id"].unique())
    if cache is None:
        cache = _load_yprob_cache(results)

    # Mean treat-all across datasets (each dataset weighted equally).
    odds = NB_SWEEP / (1.0 - NB_SWEEP)
    treat_all_stack = []
    for did in dataset_ids:
        er = float(ok[ok["dataset_id"] == did]["event_rate"].iloc[0])
        treat_all_stack.append(er - (1.0 - er) * odds)
    treat_all_mean = np.mean(np.vstack(treat_all_stack), axis=0)

    n_cols = len(MODELS)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols, 4.2), squeeze=False)

    y_floor = min(-0.02, float(np.nanmin(treat_all_mean)))
    for j, model in enumerate(MODELS):
        ax = axes[0][j]
        for cond in CONDITIONS:
            # Pool the per-cell curves across every dataset for this (model, cond).
            cells: list[tuple[np.ndarray, np.ndarray]] = []
            for did in dataset_ids:
                cells.extend(cache.get((did, model, cond), []))
            if not cells:
                continue
            mat = _nb_curve_matrix(cells)
            mean_curve = np.nanmean(mat, axis=0)
            lo = np.nanpercentile(mat, CI_LOWER_PERCENTILE, axis=0)
            hi = np.nanpercentile(mat, CI_UPPER_PERCENTILE, axis=0)
            label = "none (=none_threshold)" if cond == "none" else cond
            ax.plot(
                NB_SWEEP, mean_curve, color=CONDITION_COLORS[cond], lw=1.6, label=label
            )
            ax.fill_between(
                NB_SWEEP, lo, hi, color=CONDITION_COLORS[cond], alpha=0.12, lw=0
            )
        ax.plot(
            NB_SWEEP, treat_all_mean, color="gray", ls="--", lw=1.1, label="treat-all"
        )
        ax.axhline(0.0, color="gray", ls=":", lw=1.1, label="treat-none")
        ax.set_ylim(bottom=y_floor)
        ax.set_title(model, fontsize=11)
        ax.set_xlabel("threshold probability", fontsize=9)
        if j == 0:
            ax.set_ylabel("Net Benefit (pooled across 8 datasets)", fontsize=9)
        ax.tick_params(labelsize=8)
        if j == n_cols - 1:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Figure 1 — Decision curves pooled across all 8 datasets, per model.\n"
        "Line = mean over 200 dataset×seed×fold replicate curves; band = 2.5/97.5 "
        "percentiles (descriptive).\n"
        "treat-all = mean over datasets; treat-none = 0. Per-dataset curves: Figure S1.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "figure1_decision_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------
# FIGURE 2 — calibration curves for ONE representative dataset, all 4 conditions
# --------------------------------------------------------------------------------
def _representative_dataset(results: pd.DataFrame) -> int:
    """Mechanically pick the representative dataset: the upper-median event rate.

    Chosen by a fixed rule (not by inspecting calibration) to avoid cherry-picking:
    sort datasets by event rate ascending, take index len//2.
    """
    er = (
        results[["dataset_id", "event_rate"]]
        .drop_duplicates()
        .sort_values("event_rate")
        .reset_index(drop=True)
    )
    return int(er.loc[len(er) // 2, "dataset_id"])


def _reliability(y: np.ndarray, p: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-MASS reliability points (conf, acc), consistent with ECE (METRICS.md §3)."""
    order = np.argsort(p)
    ys = y[order]
    ps = p[order]
    conf = np.array([b.mean() for b in np.array_split(ps, n_bins) if len(b) > 0])
    acc = np.array([b.mean() for b in np.array_split(ys, n_bins) if len(b) > 0])
    return conf, acc


def figure2_calibration_curves(results: pd.DataFrame) -> tuple[Path, int]:
    """Calibration (reliability) curves for one representative dataset, all 4
    conditions, one panel per model.

    Held-out predictions are pooled across the 25 seed×fold replicates and binned into
    equal-MASS bins, matching the ECE definition (METRICS.md §3). The diagonal is
    perfect calibration.
    """
    from src.config import ECE_N_BINS

    ok = results[results["status"] == "ok"]
    did = _representative_dataset(results)
    name = ok[ok["dataset_id"] == did]["dataset_name"].iloc[0]
    sub = ok[ok["dataset_id"] == did]

    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.4 * len(MODELS), 4.2), squeeze=False)
    colors = {"none": "black", "rus": "tab:blue", "ros": "tab:green", "smote": "tab:red"}

    for j, model in enumerate(MODELS):
        ax = axes[0][j]
        ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.0, label="perfect")
        for cond in CONDITIONS:
            cells = sub[(sub["model"] == model) & (sub["condition"] == cond)]
            ys: list[np.ndarray] = []
            ps: list[np.ndarray] = []
            for r in cells.itertuples():
                yp = pd.read_parquet(r.y_prob_path)
                ys.append(yp["y_true"].to_numpy())
                ps.append(yp["y_prob"].to_numpy())
            if not ys:
                continue
            y = np.concatenate(ys)
            p = np.concatenate(ps)
            conf, acc = _reliability(y, p, ECE_N_BINS)
            ax.plot(conf, acc, marker="o", ms=3, lw=1.3, color=colors[cond], label=cond)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("mean predicted probability", fontsize=8)
        if j == 0:
            ax.set_ylabel("observed frequency", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"Figure 2 — Calibration curves, representative dataset {name} ({did}), "
        f"all 4 conditions. Equal-mass bins (n={ECE_N_BINS}), pooled over 25 "
        f"seed×fold replicates.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "figure2_calibration_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, did


# --------------------------------------------------------------------------------
def main() -> None:
    """Produce the pre-registered H1/H2/H3 tables and the figures.

    Reads results/results.parquet and the saved results/yprob/ files; writes
    results/tables/ (markdown + csv) and results/figures/. Prints only a short
    completion summary — full output goes to results/ (.cursorrules).
    """
    results = pd.read_parquet(RESULTS_PATH)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    status_audit(results)
    table1_discrimination(results)
    table2_calibration(results)

    nb = _net_benefit_per_cell(results)
    table3_decisions(nb)

    cache = _load_yprob_cache(results)
    fig1 = figure1_decision_curves(results, cache)
    fig_s1 = figure_s1_decision_curves_by_dataset(results, cache)
    fig2, rep_did = figure2_calibration_curves(results)

    n_ok = int((results["status"] == "ok").sum())
    print(
        f"analyze: {len(results)} cells ({n_ok} ok). "
        f"Tables -> {TABLES_DIR}/ ; Figures -> {fig1.name}, {fig_s1.name}, {fig2.name} "
        f"(representative dataset {rep_did})."
    )


if __name__ == "__main__":
    main()
