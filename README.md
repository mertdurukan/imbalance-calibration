# Class-imbalance corrections in modern tabular ML

[![CI](https://img.shields.io/github/actions/workflow/status/mertdurukan/imbalance-calibration/ci.yml?branch=main&label=CI)](https://github.com/mertdurukan/imbalance-calibration/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A pre-registered study of whether random undersampling (RUS), random oversampling (ROS),
and SMOTE help logistic regression, XGBoost, and an MLP on imbalanced tabular data. Across
**8 OpenML datasets** (event rate 5–19%), 3 models, 4 resampling conditions, and 5×5
stratified CV (2,400 fits, all completed), the corrections buy **no discrimination**, they
**degrade calibration**, and any decision-analytic benefit is **reproduced for free by
shifting the decision threshold**. The one apparent exception (the MLP seeming to gain
AUROC from resampling) is a diagnosed artifact of an early-stopping baseline that collapses
to near-chance on the most imbalanced data, not a real benefit of the correction. The design
was frozen in [`PREREG.md`](PREREG.md) before any model was fit; every departure is logged in
[`DEVIATIONS.md`](DEVIATIONS.md).

## Headline results

Each hypothesis was stated with a falsification criterion before data (`PREREG.md` §3).
All estimates carry a 95% interval; see the tables in [`results/tables/`](results/tables/).

| Hypothesis | Verdict | What the data show |
| --- | --- | --- |
| **H1**: no meaningful AUROC gain (\|ΔAUROC\| < 0.01) | **Holds for logreg & XGBoost; fails for MLP** | logreg and XGBoost: all corrections within ±0.01 AUROC of `none` (Table 1). MLP: ROS +0.055, SMOTE +0.049, RUS +0.018, but see the mechanism below. |
| **H2**: corrections damage calibration | **Holds (26 of 27 verdict cells)** | Corrections push calibration slope away from 1.0, intercept away from 0.0, and raise ECE in every model×condition cell except one (XGBoost+SMOTE intercept). RUS is the most damaging (Table 2). |
| **H3**: `none` + threshold shift ≥ every correction on Net Benefit | **Holds (all 36 verdict cells)** | No correction beats simple threshold-shifting on Net Benefit with a 95% interval excluding zero, at any threshold in {event rate, 0.05, 0.10, 0.20} (Table 3). |

### The MLP "gain" is a collapsing baseline, not a benefit

The MLP is fit with `early_stopping=True` (frozen in `PREREG.md` §4.2), which monitors
**validation accuracy**. On the most imbalanced datasets, a majority-class predictor already
maximizes accuracy, so training halts after ~12 iterations at ~0.94 validation accuracy but
**near-chance AUROC**. **17 of 200** `mlp/none` replicates have AUROC < 0.6, concentrated in
the two lowest-event-rate datasets (`wilt` 13/25, `ozone-level-8hr` 4/25). Resampling
rebalances the training fold, so accuracy no longer favors the majority predictor: training
runs longer (~31 iterations), the predicted-probability spread widens (std 0.07 → 0.23), and
AUROC rises. The MLP H1 "fail" therefore measures an **un-collapsed baseline**, not a genuine
effect of resampling; on the high-event-rate control (`jm1`, 19%) no replicate collapses.
See `results/diagnostics/report6`–`report9` and `results/figures/figure3_mlp_mechanism.png`.

### RUS can make a model actively harmful

Beyond degrading calibration, **random undersampling** decalibrates predictions enough to
push **Net Benefit below zero** (worse than treating nobody) at operating thresholds
(descriptive, `results/tables/table4_negative_net_benefit.md`). At threshold 0.20, RUS yields
negative Net Benefit in 41% of logreg replicates (82/200) and 59% of MLP replicates
(118/200); for the MLP, 89/200 replicates flipped from useful (`none` NB > 0) to harmful
(`rus` NB < 0) on the same replicate. ROS is milder and SMOTE rarely goes negative; XGBoost
is robust except RUS at 0.20 (23/200).

## Reproduce

Requires **Python 3.11**. From the repository root:

```bash
make setup      # create ./.venv and install pinned requirements
make verify     # functional smoke test of the key code paths
make reproduce  # runs pytest, then the full experiment, then the analysis
```

`make reproduce` runs the contract tests in `tests/` (which encode the study's leakage and
anti-cherry-picking guards), executes all fits via `src/runner.py`, and regenerates every
table and figure via `src/analyze.py`. Runs are **resumable and cached per cell**: completed
cells in `results/cells/` are not recomputed, so re-running does not repeat finished work.
Data are pulled from OpenML on first run (dataset IDs in [`datasets.txt`](datasets.txt)).

## Scope and limits

- **8 datasets, not 10.** The pre-registered pool mechanically yields only 8 qualifying
  datasets (`DEVIATIONS.md`); the pool was not broadened, to preserve the anti-cherry-picking
  guarantee. Generalization is correspondingly limited.
- **Tabular data only.** Findings do not extend to deep models on images/text.
- **Fixed hyperparameters, by design.** The estimand is the *difference between conditions*,
  not peak performance; results are not claimed to hold under per-condition tuning. The MLP
  early-stopping behavior above is itself a consequence of the frozen specification.
- **Class-weighting** (e.g. `scale_pos_weight`) is out of scope for v1.
- Intervals are descriptive percentile ranges over CV replicates, which are not independent;
  no significance tests are run (`METRICS.md` §5).

## Read more

- **Full write-up:** [`paper.md`](paper.md)
- **Pre-registration (frozen):** [`PREREG.md`](PREREG.md) · **Deviations:** [`DEVIATIONS.md`](DEVIATIONS.md)
- **Metrics & formulas:** [`METRICS.md`](METRICS.md) · **Implementation spec:** [`SPEC.md`](SPEC.md)
- **Result tables:** [`results/tables/`](results/tables/) · **Figures:** [`results/figures/`](results/figures/) · **Diagnostics:** [`results/diagnostics/`](results/diagnostics/)
