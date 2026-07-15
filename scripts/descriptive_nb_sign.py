"""scripts/descriptive_nb_sign.py — DESCRIPTIVE sign-of-Net-Benefit table (read-only).

Descriptive summary of the SIGN of the pre-registered Net Benefit. This is NOT a new
hypothesis test and NOT a new metric: it re-uses the exact pre-registered NB values —
recomputed by ``src.analyze._net_benefit_per_cell`` from the saved ``results/yprob/``
files at the pre-registered thresholds {event rate, 0.05, 0.10, 0.20} (METRICS.md §4.1),
the same values that feed Table 3 — and merely counts how many replicates fall on the
wrong side of zero.

Two descriptive quantities per model × condition × threshold, in one table:
  (a) how many of the 200 replicates have NB < 0 (the model is WORSE than treating
      nobody at that threshold), with `none` reported as the reference row; and
  (b) the paired, load-bearing count/fraction where NB(correction) < 0 WHILE
      NB(none) > 0 on the SAME replicate — the correction turned a useful model into a
      harmful one.

Read-only: refits nothing, changes no configuration/hyperparameter/metric, drops no cell.
Long output goes to ``results/tables/`` (.cursorrules); stdout gets a short summary only.
Every reported fraction carries a 95% interval (.cursorrules #6): a percentile bootstrap
over the replicate indicators (config.BOOTSTRAP_N resamples, seed config.SEEDS[0]),
reported DESCRIPTIVELY — the 200 replicates share 8 datasets and CV folds are not
independent (METRICS.md §5), so this interval is not a significance test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.analyze import (
    CORRECTIONS,
    NB_FIXED_THRESHOLDS,
    REFERENCE_CONDITION,
    REPLICATE_KEYS,
    _net_benefit_per_cell,
)
from src.config import CONDITIONS, MODELS

RESULTS_PATH: Path = Path("results") / "results.parquet"
TABLES_DIR: Path = Path("results") / "tables"

TABLE_HEADER: str = (
    "DESCRIPTIVE — sign of the pre-registered Net Benefit. Not a new hypothesis "
    "test; no PASS/FAIL verdict is assigned."
)

# NB column keys / pretty labels, matching src.analyze._net_benefit_per_cell.
NB_LABELS: tuple[tuple[str, str], ...] = (
    ("eventrate", "NB@eventrate"),
    *[(f"{int(t * 1000):03d}", f"NB@{t:.2f}") for t in NB_FIXED_THRESHOLDS],
)

# Placeholder for the paired-harmful columns on the `none` reference row: the flip
# "NB(none) < 0 while NB(none) > 0" is impossible by construction, so it is not a
# number to report — it is marked n/a rather than a misleading 0.
NA_MARK: str = "—"


def _proportion_ci(indicators: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Fraction of True plus a percentile-bootstrap 95% interval over the replicates.

    Returns ``(fraction, ci_low, ci_high)``. The point estimate is the observed
    fraction on the full set; the interval is the 2.5/97.5 percentiles of the
    bootstrap resample fractions. Descriptive only (replicates are not independent).
    """
    ind = np.asarray(indicators, dtype=float)
    n = ind.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    frac = float(np.mean(ind))
    rng = np.random.default_rng(seed)
    resamples = np.empty(config.BOOTSTRAP_N, dtype=float)
    for i in range(config.BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        resamples[i] = float(np.mean(ind[idx]))
    lo = float(np.percentile(resamples, config.CI_LOWER_PERCENTILE))
    hi = float(np.percentile(resamples, config.CI_UPPER_PERCENTILE))
    return frac, lo, hi


def _fmt_frac(frac: float, lo: float, hi: float) -> str:
    """Format a fraction with its 95% interval: ``frac [lo, hi]``."""
    if not np.isfinite(frac):
        return "nan"
    return f"{frac:.4f} [{lo:.4f}, {hi:.4f}]"


def _write(df: pd.DataFrame, stem: str, title: str, notes: list[str]) -> None:
    """Persist a table as both CSV (raw numbers) and Markdown (formatted)."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{stem}.csv", index=False)
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
    (TABLES_DIR / f"{stem}.md").write_text("\n".join(lines))


def table4_negative_net_benefit(nb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sign of the pre-registered Net Benefit, per model × condition × threshold.

    One row per (model, condition, threshold). Two descriptive quantities per row:

    (a) count/fraction of replicates with NB(condition) < 0 (worse than treating
        nobody). `none` is reported as its own reference row.
    (b) the paired count/fraction of replicates where NB(condition) < 0 WHILE
        NB(none) > 0 on the SAME (dataset, seed, fold) replicate — the correction
        turned a useful model into a harmful one. On the `none` row this is not a
        meaningful quantity (impossible by construction) and is marked n/a.

    Returns ``(raw, display)``: raw numeric CSV frame and formatted markdown frame.
    """
    seed = config.SEEDS[0]
    raw_rows: list[dict[str, object]] = []
    display_rows: list[dict[str, object]] = []
    for model in MODELS:
        m = nb[nb["model"] == model]
        none = m[m["condition"] == REFERENCE_CONDITION]
        for cond in CONDITIONS:
            c = m[m["condition"] == cond]
            # Paired join to `none` on the replicate keys for part (b).
            merged = none.merge(
                c, on=list(REPLICATE_KEYS), suffixes=("_none", "_cond")
            )
            for key, pretty in NB_LABELS:
                # ---- part (a): NB(condition) < 0 ----
                vals = c[f"nb_{key}"].to_numpy(dtype=float)
                finite_a = np.isfinite(vals)
                neg = vals < 0.0
                n_a = int(finite_a.sum())
                n_neg = int(np.sum(neg & finite_a))
                frac_a, lo_a, hi_a = _proportion_ci(
                    neg[finite_a].astype(float), seed
                )

                # ---- part (b): NB(condition) < 0 WHILE NB(none) > 0, paired ----
                if cond == REFERENCE_CONDITION:
                    # Reference row: the flip is impossible by construction.
                    n_pairs = n_a
                    n_flip: int | float = 0
                    frac_b = lo_b = hi_b = float("nan")
                    n_flip_disp = NA_MARK
                    frac_b_disp = NA_MARK
                else:
                    none_nb = merged[f"nb_{key}_none"].to_numpy(dtype=float)
                    cond_nb = merged[f"nb_{key}_cond"].to_numpy(dtype=float)
                    finite_b = np.isfinite(none_nb) & np.isfinite(cond_nb)
                    flip = finite_b & (cond_nb < 0.0) & (none_nb > 0.0)
                    n_pairs = int(finite_b.sum())
                    n_flip = int(flip.sum())
                    frac_b, lo_b, hi_b = _proportion_ci(
                        flip[finite_b].astype(float), seed
                    )
                    n_flip_disp = str(n_flip)
                    frac_b_disp = _fmt_frac(frac_b, lo_b, hi_b)

                raw_rows.append(
                    {
                        "model": model,
                        "condition": cond,
                        "threshold": pretty,
                        "n": n_a,
                        "n_nb_lt_0": n_neg,
                        "frac_nb_lt_0": frac_a,
                        "frac_nb_lt_0_lo": lo_a,
                        "frac_nb_lt_0_hi": hi_a,
                        "n_pairs": n_pairs,
                        "n_corr_lt_0_and_none_gt_0": n_flip,
                        "frac_corr_lt_0_and_none_gt_0": frac_b,
                        "frac_corr_lt_0_and_none_gt_0_lo": lo_b,
                        "frac_corr_lt_0_and_none_gt_0_hi": hi_b,
                    }
                )
                display_rows.append(
                    {
                        "model": model,
                        "condition": cond,
                        "threshold": pretty,
                        "n": n_a,
                        "(a) n_NB<0": n_neg,
                        "(a) frac_NB<0 [95%]": _fmt_frac(frac_a, lo_a, hi_a),
                        "(b) n_corr<0 & none>0": n_flip_disp,
                        "(b) frac_corr<0 & none>0 [95%]": frac_b_disp,
                    }
                )
    raw = pd.DataFrame.from_records(raw_rows)
    display = pd.DataFrame.from_records(display_rows)
    _write(
        display,
        "table4_negative_net_benefit",
        TABLE_HEADER,
        [
            "NB is the pre-registered Net Benefit recomputed from the saved y_prob "
            "files at the pre-registered thresholds {event rate, 0.05, 0.10, 0.20} "
            "(the same values as Table 3, METRICS.md §4.1). NB < 0 means WORSE than "
            "treating nobody at that threshold.",
            "(a) counts replicates with NB(condition) < 0. `none` is included as the "
            "reference row so each correction can be read against the uncorrected "
            "model at the same (model, threshold).",
            "(b) is PAIRED within each (dataset, seed, fold): replicates where the "
            "correction's NB < 0 (harmful) WHILE `none`'s NB > 0 (useful) on the SAME "
            "replicate — the correction turned a useful model into a harmful one at "
            "that threshold. On the `none` reference row this quantity is impossible "
            f"by construction and is marked '{NA_MARK}'.",
            "n = replicates per (model, condition) = 8 datasets × 5 seeds × 5 folds = "
            "200 (n_pairs likewise for the paired part (b)).",
            "95% interval = percentile bootstrap over the 200 replicate indicators "
            "(2000 resamples, seed config.SEEDS[0]). Descriptive only: replicates "
            "share 8 datasets and CV folds are not independent (METRICS.md §5), so "
            "this is not a significance test.",
            "This is a DESCRIPTIVE report of the sign of a pre-registered quantity; no "
            "PASS/FAIL verdict is assigned (unlike the pre-registered H1–H3 tables).",
        ],
    )
    raw.to_csv(TABLES_DIR / "table4_negative_net_benefit.csv", index=False)
    return raw, display


def main() -> None:
    """Write the descriptive sign-of-Net-Benefit table; print a short summary."""
    results = pd.read_parquet(RESULTS_PATH)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    nb = _net_benefit_per_cell(results)
    # Restrict to the fit conditions; `none_threshold` equals `none` by construction
    # (METRICS.md §4.1) and would only duplicate the reference.
    nb = nb[nb["condition"].isin(CONDITIONS)]

    table4_negative_net_benefit(nb)

    print(
        "descriptive_nb_sign: wrote table4_negative_net_benefit to "
        f"{TABLES_DIR}/ (descriptive; sign of the pre-registered Net Benefit, "
        "not a new hypothesis test; no PASS/FAIL verdict)."
    )


if __name__ == "__main__":
    main()
